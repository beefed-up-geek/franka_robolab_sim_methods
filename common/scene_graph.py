#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""씬 그래프 — 시뮬레이션 특권 정보를 하나의 구조화된 상태로 모은다.

세 방법론의 +Ours 가 공유하는 유일한 입력이다. 지각(SAM/DINO/깊이)을 대체하는
것이 목적이므로 **원본 상태를 그대로** 싣는다: 물체의 위치·회전·의미 상태와
물체 사이의 관계. VLM 이 장면을 말로 묘사하던 자리를 이 그래프가 대신한다.

설계 규칙 두 가지가 전부다.
  ① 노드/엣지 값은 전부 torch 텐서로도 꺼낼 수 있어야 한다 — 보상 코드가
     여기에 대고 미분한다 (diffreward.py).
  ② 판정에 쓰는 술어(잡았나·담겼나·위험한가)는 그래프가 직접 계산한다 —
     방법론마다 제각기 재구현하면 어긋난다 (실측: SC 가 라운드에 없는 캔을
     지목해 실재하는 파열 캔이 무방비가 됐다).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

# ── 씬 상수 (시뮬 실측) ──────────────────────────────────────────────────
# 회수통(grey_bin) 실측 — 420x280x105mm, 원점이 바닥이라 상판(z=0) 위에 앉는다.
# world_assets.py 가 Z 로 90° 돌려 놓아 경계는 X [0.12,0.40], Y [0.37,0.79] 이고
# **테두리 높이는 0.105** 다. 환경은 캔이 테두리보다 **낮아야** 담긴 것으로 센다
# (conveyor.py _in_bin). 이 높이를 0.30 으로 잘못 잡아두면, 그리퍼가 통 위
# 25cm 에서 들고 있는 캔까지 "담겼다" 로 보고 목표가 사라져 로봇이 운반을
# 멈추고 캔을 흘린다 — 실측 실패의 주된 형태였다.
BIN_XY = (0.26, 0.58)          # 회수통 중심
BIN_BOX = (0.12, 0.40, 0.37, 0.79)   # x0, x1, y0, y1
BIN_RIM = 0.105                # 테두리 z — 이보다 낮아야 담긴 것
BIN_Z = 0.25                   # 보상이 겨냥할 투입 높이 (테두리 위)
HOME = (0.36, 0.0, 0.472)
WORKER_Y = -0.40               # task1 경계 테이프. 작업자는 -y 쪽
ARM_LEN = 0.52                 # 작업자 팔뚝 길이 (팔꿈치 원점에서 -x 방향)
ARM_RADIUS = 0.075
ARM_PARKED_X = 0.90            # 팔꿈치 x 가 이보다 크면 책상 밖 대기
# 팔은 리셋 4초 뒤 0.45초 만에 x=1.05 → 0.715 로 "쑥" 들어온다 (worker_arm.py
# _ENTER_DELAY/_ENTER_T). y·z 는 대기 중에도 작업 위치와 **같다** — 오직 x 만
# 움직인다. 따라서 팔이 점유할 통로는 처음부터 알 수 있고, 대기 중이라고
# 위험을 꺼두면 진입 스윕(3 제어스텝)에 그대로 당한다 (실측: 접촉이 전부
# step 6~8 에 몰렸다). 통로를 상시 위험으로 둔다.
ARM_WORK_X = 0.715             # 팔꿈치 원점의 작업 위치 x (손바닥 0.33 + 팔 0.385)
# 환경의 충돌 판정은 구가 아니라 **상자** 다: |Δy|<0.075 ∧ |Δz|<0.075 ∧ x 범위
# (runner.py 의 _geo). 장벽도 같은 모양으로 맞춰야 작업 여유를 최대로 남긴다.
ARM_HALF = 0.075               # 상자 반폭 (= 환경 ARM_R)
ARM_MARGIN = 0.020             # 상자 표면에서 확보할 여유 [m]. 팔 y=+0.10 배치에서
                               # 커넥터가 상자에서 2.5cm 밖이라 이보다 크면 파지가 막힌다.
TCP_DZ = -0.15                 # 플랜지 → 손끝. 정책 액션은 플랜지 기준이고
                               # 물체 좌표는 상판 높이라 술어는 손끝으로 본다
GRASP_R = 0.05                 # 파지 반경 [m]
BIN_R = 0.12                   # (구) 통 안 판정 반경 — 아래 in_bin 으로 대체


def in_bin(p) -> bool:
    """환경과 같은 판정 — 통 상자 안이면서 테두리보다 낮은가."""
    x0, x1, y0, y1 = BIN_BOX
    return x0 < p[0] < x1 and y0 < p[1] < y1 and p[2] < BIN_RIM


def over_bin(p, inset: float = 0.05) -> bool:
    """통 상자 위(수평 기준)인가 — 놓아도 되는 자리인지 판단한다."""
    x0, x1, y0, y1 = BIN_BOX
    return (x0 + inset < p[0] < x1 - inset
            and y0 + inset < p[1] < y1 - inset)
# 작업 공간 상자. 여기 밖의 물체는 아직 스폰 전이거나 회수돼 치워진 것이다 —
# 실측: task2 에서 connector_red 가 [-0.515,-2.115,-0.677] 로 잡혀 보상이 그
# 유령 좌표로 끌려가고 argmax 가 경계 구석으로 달아났다.
WS_LO = (0.05, -0.75, -0.10)
WS_HI = (1.20, 0.85, 0.90)


def in_workspace(p) -> bool:
    return all(WS_LO[i] <= p[i] <= WS_HI[i] for i in range(3))

# 도구 손잡이의 로컬 방향 (runner.py TOOL_HANDLE_LOCAL 와 같아야 한다)
TOOL_HANDLE_LOCAL = {
    "hammer_7": (-1.0, 0.0, 0.0),
    "cordless_drill": (0.0, -1.0, 0.0),
}
HANDLE_COS = 0.5               # 작업자 방향과의 내적 하한 (±60°)


def _kind(name: str) -> str:
    if name in TOOL_HANDLE_LOCAL:
        return "tool"
    if name == "worker_arm":
        return "arm"
    if name == "battery":
        return "battery"
    if "connector" in name:
        return "connector"
    if "can" in name:
        return "can"
    return "other"


@dataclass
class SGNode:
    """씬 그래프 노드 — 물체 하나."""

    name: str
    pos: tuple[float, float, float]
    yaw: float = 0.0
    quat: tuple = (1.0, 0.0, 0.0, 0.0)   # (w,x,y,z)
    kind: str = "other"
    damaged: bool = False       # 파열 캔
    active: bool = True         # 이번 라운드에 실제로 깔려 있나
    held: bool = False          # 그리퍼가 쥐고 있나
    binned: bool = False        # 통 안에 들어갔나
    hazard: bool = False        # 닿으면 안전 위반인가

    def t(self) -> torch.Tensor:
        return torch.tensor(self.pos, dtype=torch.float32)


@dataclass
class SceneGraph:
    """노드 + 관계 엣지. 방법론들은 이 객체 하나만 본다."""

    task: str
    nodes: dict[str, SGNode] = field(default_factory=dict)
    edges: list[tuple[str, str, str, float]] = field(default_factory=list)
    eef: tuple[float, float, float] = (0.0, 0.0, 0.0)   # 플랜지
    tcp: tuple[float, float, float] = (0.0, 0.0, 0.0)   # 손끝
    eef_yaw: float = 0.0
    gripper: float = 0.0        # 0=열림, 1=닫힘
    grasped: bool = False       # 접촉 검증된 파지
    terminals: dict = field(default_factory=dict)
    arm_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)   # 팔 속도 [m/s]

    # ── 조회 ────────────────────────────────────────────────────────────
    def of_kind(self, kind: str) -> list[SGNode]:
        return [n for n in self.nodes.values() if n.kind == kind and n.active]

    def blind(self) -> bool:
        """위험 정보가 아직 안 들어왔나.

        에피소드 시작 직후 약 1초 동안 /franka/objects 가 아직 오지 않아
        nodes 가 비어 있다. 그 창에서 씬 그래프는 "위험 없음"(h=10) 을
        보고하고, 방법들은 아무 제약 없이 움직인다 — 실측으로 팔 접촉이
        정확히 그 구간(step 0~8)에 몰렸다. 눈을 감은 채로는 움직이지 않는
        편이 맞다.
        """
        if self.task == "task2":
            return "worker_arm" not in self.nodes
        if self.task == "task3":
            return not any(n.kind == "can" for n in self.nodes.values())
        return not self.nodes

    def hazards(self) -> list[SGNode]:
        """닿으면 안전 위반인 노드 — VLM 의 '위험 지목' 을 대체한다."""
        return [n for n in self.nodes.values() if n.hazard and n.active]

    def targets(self) -> list[SGNode]:
        """지금 다뤄야 할 노드 — 태스크별 성공 정의에서 곧바로 나온다."""
        if self.task == "task1":
            return [n for n in self.of_kind("tool")]
        if self.task == "task2":
            # 커넥터가 아직 씬에 없으면 목표는 단자다 — 유령 좌표를 노리는
            # 것보다 단자 위에서 대기하는 편이 낫다.
            conn = [n for n in self.nodes.values()
                    if n.kind == "connector" and "red" in n.name and n.active]
            return conn
        return [n for n in self.of_kind("can")
                if not n.damaged and not n.binned]

    def held_node(self) -> SGNode | None:
        for n in self.nodes.values():
            if n.held:
                return n
        return None

    # ── 미분용 텐서 뷰 ──────────────────────────────────────────────────
    def tensors(self) -> dict[str, torch.Tensor]:
        """보상 코드가 쓰는 텐서 묶음. 이름 → (3,) 또는 (K,3)."""
        out: dict[str, torch.Tensor] = {}
        for name, n in self.nodes.items():
            if n.active:
                out[name] = n.t()
                out[f"{name}_yaw"] = torch.tensor(float(n.yaw))
        out["bin"] = torch.tensor([BIN_XY[0], BIN_XY[1], BIN_Z])
        out["home"] = torch.tensor(HOME)
        out["eef"] = torch.tensor(self.eef)
        out["tcp"] = torch.tensor(self.tcp)
        hz = self.hazards()
        out["hazards"] = (torch.stack([n.t() for n in hz]) if hz
                          else torch.zeros(0, 3))
        tg = self.targets()
        out["targets"] = (torch.stack([n.t() for n in tg]) if tg
                          else torch.zeros(0, 3))
        if self.terminals.get("pos"):
            out["red_terminal"] = torch.tensor(
                [float(v) for v in self.terminals["pos"]])
        return out

    def arm_box_at(self, dt: float):
        """dt 초 뒤 통로 상자 — x 만 움직이므로 x 중심을 외삽한다."""
        bx = self.arm_box()
        if bx is None or dt <= 0.0:
            return bx
        c, half = bx
        v = torch.tensor(self.arm_vel)
        return c + v * dt, half

    def arm_segment_at(self, dt: float):
        """dt 초 뒤 팔뚝 선분 — 등속으로 외삽한다.

        CBF 의 ḣ ≥ −αh 는 장애물이 가만히 있다고 본다. 작업자 팔은 밀고
        들어오므로 우리가 뭘 하든 h 가 떨어진다 — 실측으로 h 가 +0.002 인
        상태에서도 접촉이 났다. 예측 선분까지 함께 보면 팔이 오기 전에
        비킬 수 있다.
        """
        seg = self.arm_segment()
        if seg is None or dt <= 0.0:
            return seg
        v = torch.tensor(self.arm_vel)
        return seg[0] + v * dt, seg[1] + v * dt

    def arm_box(self):
        """팔이 점유할 통로를 축정렬 상자 (중심, 반크기) 로.

        x 는 작업 위치 기준 고정 구간이다 — 대기 중(x=1.05)이라도 곧 들어올
        자리를 미리 막는다. y·z 는 관측값 그대로 (에피소드마다 다르다).
        """
        arm = self.nodes.get("worker_arm")
        if arm is None:
            return None
        x1 = min(arm.pos[0], ARM_WORK_X)
        x0 = x1 - ARM_LEN
        c = torch.tensor([(x0 + x1) * 0.5, arm.pos[1], arm.pos[2]])
        half = torch.tensor([(x1 - x0) * 0.5, ARM_HALF, ARM_HALF])
        return c, half

    def arm_segment(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """팔뚝 선분 (a, b) — 넘어가기 판단 등 기하 참조용."""
        arm = self.nodes.get("worker_arm")
        if arm is None:
            return None
        x1 = min(arm.pos[0], ARM_WORK_X)
        a = torch.tensor([x1, arm.pos[1], arm.pos[2]])
        b = a - torch.tensor([ARM_LEN, 0.0, 0.0])
        return a, b


def build(node, task: str) -> SceneGraph:
    """브릿지 노드의 특권 상태 → 씬 그래프."""
    st = node.status or {}
    active = st.get("active")
    eef = list(node.eef or (0.0, 0.0, 0.0))
    tcp = (eef[0], eef[1], eef[2] + TCP_DZ)
    g = SceneGraph(task=task, eef=tuple(eef), tcp=tcp,
                   gripper=float(node.gripper),
                   terminals=(st.get("terminals") or {}))

    contact = st.get("contact") or {}
    g.grasped = any(float(v) > 0.3 for v in contact.values())
    # status.contact 에는 물체별 접촉력과 함께 **집계 항목 'all_objs'** 가 들어
    # 있다. 첫 키를 그냥 고르면 그 집계값이 잡혀 물체 이름 매칭에 실패하고
    # held_node() 가 늘 None 이 된다 — 실측으로 task1 의 요 명령이 쥐지도 않은
    # 다른 도구의 자세로 계산됐다. 실재하는 물체 이름만 고른다.
    held_name = next((k for k, v in contact.items()
                      if k in node.objects and float(v) > 0.3), None)

    others = st.get("others") or {}
    for name, p in node.objects.items():
        if abs(p[2]) > 1.0:
            continue
        kind = _kind(name)
        damaged = "burst" in name
        # task3 은 그 라운드에 깔린 캔만 살아 있다. 대기열(상판 아래)까지
        # 넣으면 없는 물체로 계획을 세운다.
        act = in_workspace(p)
        if act and task == "task3" and kind == "can" and active is not None:
            act = name in active
        yaw = float(node.object_yaw.get(name, 0.0))
        q = node.object_quat.get(name, [1.0, 0.0, 0.0, 0.0])
        n = SGNode(name=name, pos=(float(p[0]), float(p[1]), float(p[2])),
                   yaw=yaw, quat=tuple(float(v) for v in q),
                   kind=kind, damaged=damaged, active=act)
        # 통 안 판정 — 환경(_in_bin)과 같은 상자·테두리 기준으로 본다
        n.binned = in_bin(n.pos)
        n.held = bool(held_name == name)
        # 위험 = 닿으면 안전 축이 깎이는 것. 태스크 정의에서 직접 나온다.
        if task == "task2":
            # 책상 밖에 대기 중인 팔은 위험이 아니다. 이 게이트가 없으면
            # 팔이 없는데도 위험 1개가 상시로 잡혀 필터가 늘 켜져 있다.
            n.hazard = (kind == "arm")
        elif task == "task3":
            n.hazard = damaged and act
        g.nodes[name] = n

    if held_name and held_name in g.nodes:
        g.nodes[held_name].held = True

    # 팔 속도 — 유한차분 + 1차 저역통과. 브릿지 노드에 상태를 얹는다
    # (에피소드 사이에도 살아 있지만 팔은 연속이라 문제 없다).
    arm = g.nodes.get("worker_arm")
    if arm is not None:
        now = time.time()
        prev = getattr(node, "_sg_arm_prev", None)
        vel = getattr(node, "_sg_arm_vel", (0.0, 0.0, 0.0))
        if prev is not None:
            dt = now - prev[1]
            if 1e-3 < dt < 0.5:
                raw = [(arm.pos[i] - prev[0][i]) / dt for i in range(3)]
                vel = tuple(0.6 * vel[i] + 0.4 * raw[i] for i in range(3))
        node._sg_arm_prev = (arm.pos, now)
        node._sg_arm_vel = vel
        g.arm_vel = vel

    # ── 관계 엣지 ───────────────────────────────────────────────────────
    for name, n in g.nodes.items():
        if not n.active:
            continue
        d = math.dist(tcp, n.pos)
        if d < 0.25:
            g.edges.append(("near", "eef", name, round(d, 4)))
        if n.held:
            g.edges.append(("held_by", name, "eef", 0.0))
        if n.binned:
            g.edges.append(("inside", name, "bin", 0.0))
        if n.hazard:
            g.edges.append(("hazard_of", name, "eef", round(d, 4)))
    for a in g.hazards():
        for b in g.targets():
            dd = math.dist(a.pos, b.pos)
            if dd < 0.30:
                g.edges.append(("blocks", a.name, b.name, round(dd, 4)))
    return g


def handle_dir_world(name: str, quat) -> tuple[float, float]:
    """도구 손잡이의 월드 방향 (x, y) — runner.py 와 **같은 식** 이다.

    R(q)·u 를 쓴다. 요만 쓰는 평면 근사는 도구가 기울면 환경 판정과 어긋난다
    (실측: 그 근사로 정렬됐다고 판단하고 넘어갔는데 위반이 났다).
    """
    ux, uy, uz = TOOL_HANDLE_LOCAL.get(name, (-1.0, 0.0, 0.0))
    qw, qx, qy, qz = (float(v) for v in quat)
    wx = ((1 - 2 * (qy * qy + qz * qz)) * ux
          + 2 * (qx * qy - qw * qz) * uy
          + 2 * (qx * qz + qw * qy) * uz)
    wy = (2 * (qx * qy + qw * qz) * ux
          + (1 - 2 * (qx * qx + qz * qz)) * uy
          + 2 * (qy * qz - qw * qx) * uz)
    return wx, wy


def handle_ok(name: str, quat, thresh: float = HANDLE_COS) -> bool:
    """손잡이가 작업자 (0,-1,0) 쪽 ±60° 안을 향하나 (runner.py 와 동일)."""
    _, wy = handle_dir_world(name, quat)
    return -wy > thresh


# 조향을 멈출 문턱은 판정 문턱보다 **안쪽** 이어야 한다. 0.5 에서 끄면 경계
# 위에서 True/False 가 진동하며(실측 0.49~0.53) 명령이 켜졌다 꺼졌다 해
# 도구가 문턱을 못 넘는다.
HANDLE_STOP = 0.72


def yaw_error(name: str, quat) -> float:
    """지금 손잡이 방향에서 작업자 쪽까지, 월드 z 축으로 돌려야 할 각도.

    현재 월드 방향각 φ 에서 목표 −π/2 까지의 최소 회전량이다. 그리퍼 요에
    이 값을 더하면 도구도 같이 돈다.
    """
    wx, wy = handle_dir_world(name, quat)
    phi = math.atan2(wy, wx)
    err = -math.pi / 2.0 - phi
    return math.atan2(math.sin(err), math.cos(err))
