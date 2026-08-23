#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""미분 가능한 보상 맵 — 씬 그래프 위에서 실행되는 미분 가능한 프로그램.

세 방법론의 +Ours 가 공유하는 두 번째 축이다. 씬 그래프(scene_graph.py)가
"무엇이 어디에 어떤 상태로 있는가" 를 주면, 여기 있는 프로그램이 그것을
**질의점 p 에 대한 스칼라장** R(p) 로 바꾼다. R 은 torch 연산만으로 쓰여
있어서 ∇_p R 이 autograd 로 그냥 나온다.

왜 미분 가능해야 하나 — 세 방법론이 R 을 서로 다르게 쓰기 때문이다.
  · LC+Ours   : argmax_p R 로 **좌표** 를 뽑아 언어 정책에 준다 (경사 상승)
  · SC+Ours   : 안전항만 떼어 CBF 의 장벽 h(p) 로 쓰고 ∇h 를 autograd 로 얻는다
  · VLS+Ours  : ∇_p R 을 디노이징 루프에 주입한다
같은 R 하나에서 세 가지가 나오므로, 방법 간 차이가 "보상을 어떻게 쓰는가"
로만 남는다 — 보상 자체가 다르면 비교가 성립하지 않는다.

VLM 이 보상을 쓰던 자리(VLS vanilla)를 이 프로그램이 대신한다. VLM 이 낸
코드는 없는 키를 참조해 조용히 죽었지만(실측 1334회), 여기 프로그램은 씬
그래프에서 이름을 직접 받아 그런 실패 모드가 없다.
"""
from __future__ import annotations

import math

import torch

import scene_graph as SG

# 팔 예측 지평 [s] — 지금과 0.5초 뒤를 함께 본다. 팔 진입은 ~1초에 끝난다.
ARM_HORIZON = (0.0, 0.35)
# 팔 넘어가기 — 작업자 팔뚝은 x 전 구간(0.20~0.72)을 가로지르므로, 팔 반대편
# 으로 가려면 **위로 넘는 길밖에 없다**. 실측: 팔이 x=0.71,y=-0.10,z=0.30 에
# 속도 0 으로 계속 머무는데 정책은 z≈0.30 으로 운반해, 필터가 매 스텝 개입해도
# 진전 없이 120초를 다 쓴다 (SR 0/1). 장벽으로 막기만 하면 답이 안 나오고,
# 보상이 **길** 을 알려줘야 한다.
FLYOVER_W = 20.0        # 고도 부족에 대한 벌점 세기. 목표항이 통로 쪽으로
                        # 당기므로 그보다 확실히 세야 위로 뜬다.
FLYOVER_BAND = 0.22     # 팔 y 선에서 이만큼 안이면 넘어가는 중으로 본다. 좁으면
                        # 팔 코앞에서야 오르기 시작해 이미 늦다 (실측 ∇R_z<0).
FLYOVER_CLR = 0.13      # 팔 중심 위로 확보할 고도 [m] (상자 반폭 0.075 + 여유)

# ── 미분 가능한 원시 함수 ────────────────────────────────────────────────
# 전부 torch 연산이고 어디서도 분기(if)로 p 를 가르지 않는다 — 분기를 넣으면
# 그 경계에서 기울기가 끊겨 경사 상승이 멈춘다.


def dist(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """||p − c||₂. p:(...,3), c:(3,) 또는 (K,3) → (...) 또는 (...,K)."""
    if c.dim() == 1:
        return torch.linalg.norm(p - c, dim=-1)
    return torch.linalg.norm(p.unsqueeze(-2) - c, dim=-1)


def seg_dist(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """점 p 와 선분 ab 사이 거리. 작업자 팔뚝(52cm 막대)용."""
    ab = b - a
    t = ((p - a) * ab).sum(-1) / (ab * ab).sum().clamp_min(1e-9)
    t = t.clamp(0.0, 1.0).unsqueeze(-1)
    return torch.linalg.norm(p - (a + t * ab), dim=-1)


def box_dist(p: torch.Tensor, c: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """축정렬 상자까지의 거리 (바깥은 양수, 안은 0).

    환경의 팔 충돌 판정이 상자라 장벽도 상자로 맞춘다 — 구로 감싸면 모서리를
    덮으려다 축 방향 여유까지 잡아먹어 작업 공간이 사라진다.
    """
    e = torch.relu((p - c).abs() - half)
    return torch.linalg.norm(e, dim=-1)


def soft_min(x: torch.Tensor, dim: int = -1, tau: float = 0.02) -> torch.Tensor:
    """부드러운 최솟값. 진짜 min 은 argmin 이 바뀌는 지점에서 기울기가 튄다."""
    if x.shape[dim] == 0:
        return torch.full(x.shape[:-1], 1e3)
    return -tau * torch.logsumexp(-x / tau, dim=dim)


def attract(p: torch.Tensor, c: torch.Tensor, w: float = 1.0,
            soft: float = 0.02) -> torch.Tensor:
    """c 로 끌어당기는 항. 원점에서 뾰족하지 않게 Huber 꼴로 뭉갠다."""
    d = dist(p, c)
    return -w * (torch.sqrt(d * d + soft * soft) - soft)


def repel(p: torch.Tensor, c: torch.Tensor, radius: float,
          w: float = 1.0) -> torch.Tensor:
    """반경 안에서만 밀어내는 벽. 밖에서는 정확히 0 이라 목표항을 안 흐린다."""
    d = dist(p, c)
    pen = torch.relu(radius - d)
    if pen.dim() > (p.dim() - 1):
        pen = pen.sum(-1)
    return -w * pen * pen


def above(p: torch.Tensor, z: float, w: float = 1.0) -> torch.Tensor:
    """z 아래로 내려가는 것을 벌준다 (상판·벨트 충돌 회피)."""
    return -w * torch.relu(z - p[..., 2]) ** 2


class RewardMap:
    """씬 그래프에 묶인 보상장 R(p) 와 그 미분.

    task/stage 마다 항의 조합만 달라진다. 항 자체는 전부 위 원시 함수다.
    """

    def __init__(self, g: SG.SceneGraph, stage: str) -> None:
        self.g = g
        self.stage = stage
        self.T = g.tensors()

    # ── 안전항만 (SC 의 CBF 장벽으로 쓴다) ─────────────────────────────
    def safety(self, p: torch.Tensor) -> torch.Tensor:
        """h(p) — 위험까지의 여유. h ≥ 0 이 안전, 클수록 여유롭다.

        CBF 의 장벽 함수 규약을 그대로 따른다. 위험이 없으면 큰 상수라
        제약이 절대 활성화되지 않는다.
        """
        hs = []
        # 팔뚝은 반경 7.5cm 캡슐. 그리퍼는 점이 아니라 폭 ~8cm 몸통이라
        # 제어점 셋으로 훑어도 실제 접촉면을 다 못 덮는다 — 실측으로 h=+0.002
        # 에서 접촉이 났다. 정적 여유를 8cm 로 잡고, 모자란 반응 시간은
        # **예측 선분** 으로 벌충한다 (팔이 오는 자리를 미리 피한다).
        for dt in ARM_HORIZON:
            bx = self.g.arm_box_at(dt)
            if bx is None:
                break
            c, half = bx
            hs.append(box_dist(p, c, half) - SG.ARM_MARGIN)
        # 점 위험은 팔을 뺀 나머지 — 팔은 위 선분이 이미 담당한다 (이중 계상하면
        # 팔꿈치 한 점 주위에만 과도한 벽이 생긴다).
        pts = [n.t() for n in self.g.hazards() if n.kind != "arm"]
        if pts:
            hz = torch.stack(pts)
            # 파열 캔은 Ø71mm — 반경 0.0355 에 그리퍼 반폭·여유를 더한다.
            hs.append(dist(p, hz).min(dim=-1).values - 0.085)
        if not hs:
            return torch.full(p.shape[:-1], 10.0)
        return torch.stack(hs, dim=-1).min(dim=-1).values

    # ── 전체 보상 ───────────────────────────────────────────────────────
    def value(self, p: torch.Tensor) -> torch.Tensor:
        """R(p) — 지금 스테이지에서 손끝이 있어야 할 곳일수록 크다."""
        r = self._task_term(p)
        # 안전은 어느 스테이지에서나 곱하지 않고 **더한다** — 곱하면 위험
        # 근처에서 목표항까지 0 이 되어 기울기가 사라진다.
        h = self.safety(p)
        r = r - 60.0 * torch.relu(0.0 - h) ** 2      # 침범 구역: 강한 벽
        # 접근 구역은 장벽보다 **넓게** 잡는다. 장벽 여유를 좁혀 파지가
        # 가능해진 대신, 굳이 위험 곁을 지날 이유가 없을 때는 보상이 미리
        # 멀리 돌아가게 만들어야 한다 (장벽은 최후의 보루로 남긴다).
        r = r - 12.0 * torch.relu(0.10 - h) ** 2     # 접근 구역: 미리 비킴
        r = r + above(p, 0.19, w=8.0)                # 상판 아래로 내려가지 않기
        r = r + self._flyover(p)                     # 팔은 위로 넘는다
        return r

    def _flyover(self, p: torch.Tensor) -> torch.Tensor:
        """팔 y 선을 **건너가야 할 때만** 고도를 요구한다.

        가우시안 띠라 팔에서 멀어지면 사라진다. 다만 목표가 팔과 같은 쪽에
        있으면(팔이 커넥터 바로 위에 앉은 경우 등) 위로 올릴 이유가 없다 —
        올리면 파지 자체가 막힌다 (실측: 팔 y=+0.08 에서 로봇이 떠 있기만
        하다가 120초를 다 썼다). 그래서 TCP 와 목표가 팔 선을 사이에 두고
        **반대편** 일 때만 켠다.
        """
        seg = self.g.arm_segment()
        if seg is None:
            return torch.zeros(p.shape[:-1])
        a, _ = seg
        gy = self._goal_y()
        ay = float(a[1])
        if gy is None:
            return torch.zeros(p.shape[:-1])
        d_tcp, d_goal = self.g.tcp[1] - ay, gy - ay
        # 문턱을 상자 반폭+여유로 잡는다. FLYOVER_BAND(0.22)로 잡았더니 단자가
        # 팔에서 0.20 밖에 안 떨어져 넘어가기가 **아예 켜지지 않았고**, 로봇이
        # 운반 높이에서 통로를 비집다 장벽에 붙어 기어갔다 (실측 SR 1/4).
        crossing = (d_tcp * d_goal < 0.0) and \
            abs(d_goal) > (SG.ARM_HALF + SG.ARM_MARGIN)
        if not crossing:
            return torch.zeros(p.shape[:-1])
        band = torch.exp(-((p[..., 1] - a[1]) / FLYOVER_BAND) ** 2)
        need = a[2] + FLYOVER_CLR
        return -FLYOVER_W * band * torch.relu(need - p[..., 2])

    def _goal_y(self):
        """지금 스테이지의 목표 y — 넘어갈지 말지 판단에만 쓴다."""
        T = self.T
        g = self.g
        if g.task == "task2":
            if self.stage == "plug" and "red_terminal" in T:
                return float(T["red_terminal"][1])
            tg = g.targets()
            return float(tg[0].pos[1]) if tg else None
        if g.task == "task1":
            return SG.WORKER_Y if g.grasped else (
                float(g.of_kind("tool")[0].pos[1]) if g.of_kind("tool") else None)
        if self.stage == "carry":
            return SG.BIN_XY[1]
        tg = g.targets()
        return float(tg[0].pos[1]) if tg else None

    def _task_term(self, p: torch.Tensor) -> torch.Tensor:
        g, T = self.g, self.T
        if self.stage == "home":
            return attract(p, T["home"], 1.0)

        if g.task == "task1":
            tool = next((n for n in g.of_kind("tool")), None)
            if tool is None:
                return attract(p, T["home"], 1.0)
            c = T[tool.name]
            if self.stage == "reach":
                # 도구 위 5cm 를 노린다 — 상판에 그리퍼를 박지 않게.
                return attract(p, c + torch.tensor([0., 0., 0.05]), 3.0)
            # deliver: 경계선 너머로. 다만 **손잡이가 작업자 쪽을 향하기
            # 전에는 넘지 않는다** — 넘는 순간의 자세가 곧 판정이라(runner.py
            # tool_crossed), 정렬 전에 넘으면 그대로 안전 위반이다. 요 회전에는
            # 시간이 걸리므로 경계 앞에서 기다렸다가 정렬된 뒤 넘어간다.
            held = g.held_node()
            tn = held.name if (held and held.kind == "tool") else tool.name
            tq = held.quat if (held and held.kind == "tool") else tool.quat
            aligned = SG.handle_ok(tn, tq)
            base = -1.0 * (p[..., 0] - c[0]) ** 2 \
                - 1.0 * torch.relu(0.30 - p[..., 2]) ** 2
            if aligned:
                return base - 3.0 * torch.relu(p[..., 1] - (SG.WORKER_Y - 0.06))
            hold_y = SG.WORKER_Y + HOLD_MARGIN
            return base - 3.0 * torch.abs(p[..., 1] - hold_y)

        if g.task == "task2":
            conn = next((n for n in g.nodes.values()
                         if n.kind == "connector" and "red" in n.name
                         and n.active), None)
            term = T.get("red_terminal")
            if self.stage == "reach" and conn is not None:
                return attract(p, T[conn.name] + torch.tensor([0., 0., 0.03]), 3.0)
            if term is not None:
                # 단자 바로 위에서 내려꽂는다. 수평 정렬을 세게, 높이는 약하게.
                # 아직 커넥터가 없으면(reach) 단자 위 10cm 에서 대기한다.
                dz = 0.02 if self.stage != "reach" else 0.10
                dxy = torch.linalg.norm(p[..., :2] - term[:2], dim=-1)
                return -6.0 * dxy - 2.0 * torch.abs(p[..., 2] - (term[2] + dz))
            return attract(p, T["home"], 1.0)

        # task3 — 정상 캔만 담는다
        tgt = g.targets()
        if self.stage == "reach" and tgt:
            c = torch.stack([n.t() for n in tgt])
            # 가장 가까운 정상 캔 위 4cm. soft_min 으로 부드럽게 고른다.
            d = dist(p, c + torch.tensor([0., 0., 0.04]))
            return -3.0 * soft_min(d, dim=-1, tau=0.05)
        if self.stage == "carry":
            b = T["bin"]
            dxy = torch.linalg.norm(p[..., :2] - b[:2], dim=-1)
            return -3.0 * dxy - 1.0 * torch.relu(0.28 - p[..., 2]) ** 2
        return attract(p, T["home"], 1.0)

    # ── 미분 ────────────────────────────────────────────────────────────
    def grad(self, p: torch.Tensor) -> torch.Tensor:
        """∇_p R. VLS+Ours 가 디노이징에 주입하는 바로 그 벡터."""
        q = p.detach().clone().requires_grad_(True)
        r = self.value(q).sum()
        (grad,) = torch.autograd.grad(r, q)
        return grad

    def safety_grad(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(h, ∇h) — SC+Ours 의 CBF 가 쓰는 값과 기울기.

        위험이 하나도 없으면 h 는 상수라 grad_fn 이 없다 — autograd 를 그냥
        부르면 RuntimeError 로 러너가 죽는다. 그때는 0 기울기를 돌려준다.
        """
        q = p.detach().clone().requires_grad_(True)
        h = self.safety(q)
        if not h.requires_grad:
            return h.detach(), torch.zeros_like(q)
        (grad,) = torch.autograd.grad(h.sum(), q)
        return h.detach(), grad

    def argmax(self, *, bounds=None, starts: int = 24, iters: int = 60,
               lr: float = 0.02) -> tuple[list[float], float]:
        """경사 상승으로 R 의 최대점을 찾는다 — LC+Ours 의 '좋은 좌표'.

        다중 시작점을 쓴다. R 은 위험 근처에서 봉우리가 갈라져 단일 시작은
        벽 뒤에 갇힌다.
        """
        lo, hi = bounds or ((0.20, -0.55, 0.20), (0.75, 0.70, 0.60))
        lo_t = torch.tensor(lo)
        hi_t = torch.tensor(hi)
        p = lo_t + (hi_t - lo_t) * torch.rand(starts, 3)
        # 목표 근처를 시작점에 섞어 넣는다 — 균등 표본만으로는 좁은 봉우리를
        # 놓친다.
        seeds = [n.t() + torch.tensor([0., 0., 0.05]) for n in self.g.targets()]
        seeds.append(torch.tensor(self.g.tcp))
        for i, s in enumerate(seeds[: starts // 2]):
            p[i] = s
        p = p.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            loss = -self.value(p).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                p.clamp_(lo_t, hi_t)
        with torch.no_grad():
            r = self.value(p)
            k = int(torch.argmax(r))
            return [round(float(v), 3) for v in p[k]], float(r[k])

    def waypoint(self, frm, *, radius: float = 0.16, starts: int = 16,
                 iters: int = 40) -> list[float]:
        """frm 주변 반경 안에서의 argmax — 최종 목표가 아니라 **다음 경유점**.

        최종 목표 한 점만 주면 거기까지 가는 경로가 위험을 통과해도 표현할
        길이 없다 (실측: LC+Ours task2 SR 2/3 인데 Safe 0/3, 전부 팔 접촉).
        국소 최대점을 이어 가면 보상이 오르면서 위험을 우회하는 경로가 된다.
        """
        c = torch.tensor(list(frm), dtype=torch.float32)
        p = c + (torch.rand(starts, 3) - 0.5) * (2 * radius)
        p[0] = c
        p = p.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=0.02)
        for _ in range(iters):
            opt.zero_grad()
            (-self.value(p).sum()).backward()
            opt.step()
            with torch.no_grad():
                d = p - c
                n = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-9)
                p.copy_(c + d * torch.clamp(n, max=radius) / n)
                p[..., 2].clamp_(0.20, 0.55)
        with torch.no_grad():
            k = int(torch.argmax(self.value(p)))
            return [round(float(v), 3) for v in p[k]]

    def grid(self, *, z: float = 0.30, res: int = 48):
        """진단용 2D 슬라이스 — 보상 맵이 정말 목표를 가리키는지 눈으로 본다."""
        xs = torch.linspace(0.20, 0.75, res)
        ys = torch.linspace(-0.55, 0.70, res)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        p = torch.stack([gx, gy, torch.full_like(gx, z)], dim=-1)
        return xs, ys, self.value(p).detach()


# ── VLS 서버용 보상 소스 생성 ────────────────────────────────────────────
# 정책 서버(/act_chunk_guided)는 보상을 **문자열 소스** 로 받아 디노이징 루프
# 안에서 컴파일해 쓴다. VLM 이 쓰던 그 자리에 이 프로그램을 그대로 넣는다.
REWARD_SRC = '''
import torch as _t
def _d(p, c):
    c = _t.tensor(c)
    return _t.linalg.norm(p[..., :3] - c, dim=-1)
def _seg(p, a, b):
    a = _t.tensor(a); b = _t.tensor(b)
    ab = b - a
    tt = ((p[..., :3] - a) * ab).sum(-1) / ab.dot(ab).clamp_min(1e-9)
    tt = tt.clamp(0.0, 1.0).unsqueeze(-1)
    return _t.linalg.norm(p[..., :3] - (a + tt * ab), dim=-1)
_r = _t.zeros(traj.shape[0])
_tg = kp.get("__target__")
if _tg is not None:
    _r = _r - 3.0 * _d(traj, _tg)
_hz = kp.get("__hazards__") or []
for _h in _hz:
    _pen = _t.relu(0.085 - _d(traj, _h))
    _r = _r - 30.0 * _pen * _pen
_arm = kp.get("__arm__")
if _arm is not None:
    _pen = _t.relu(0.165 - _seg(traj, _arm[0], _arm[1]))
    _r = _r - 30.0 * _pen * _pen
_r = _r - 8.0 * _t.relu(0.19 - traj[..., 2]) ** 2
return _r.mean()
'''


def vls_payload(rm: "RewardMap") -> dict:
    """VLS+Ours 가 서버에 보내는 kp — 보상 소스가 참조하는 값만 담는다."""
    g = rm.g
    tgt, _ = rm.argmax(starts=16, iters=40)
    kp: dict = {"__target__": tgt}
    hz = [list(n.pos) for n in g.hazards()]
    if hz:
        kp["__hazards__"] = hz
    seg = g.arm_segment()
    if seg is not None:
        kp["__arm__"] = [[float(v) for v in seg[0]], [float(v) for v in seg[1]]]
    return kp


# ── 공유 안전 사영 (CBF 반공간 사영) ────────────────────────────────────
# 세 방법이 모두 이것을 통과시킨다. 입력 조향(LC)과 청크 유도(VLS)는 액션
# 수준의 보장이 없어 위험을 못 피한다 — 실측으로 둘 다 task2 Safe 0/3 이었다.
# 보장 자체는 SC 논문의 CBF 구조를 그대로 쓰되, 장벽 h 와 ∇h 를 보상 맵에서
# autograd 로 받는다는 점이 Ours 다.
# α 를 키우면 장벽 가까이서도 일할 수 있다. 제약은 ∇h·u ≥ −α·h·dt 이므로
# α 가 클수록 h 가 작아도 접근이 허용되고, h→0 에서만 급히 제동한다. 작게
# 잡으면 단자처럼 위험에서 19cm 밖에 안 떨어진 목표에서 필터가 상시 활성이
# 되어 로봇이 경계를 따라 기어가다 타임아웃한다 (실측 SR 2/6).
CBF_ALPHA = 3.0
CBF_DT = 1.0 / 6.0
# 세 제어점: 플랜지·중간·손끝. 두 점만 보면 그리퍼 몸통 가운데가 비어
# 그 부분이 팔을 스친다.
CTRL_DZS = (0.0, SG.TCP_DZ * 0.5, SG.TCP_DZ)
DEEP = -0.06                     # 이보다 깊이 들어가면 목적항을 버리고 탈출
ESCAPE = 0.06                    # 탈출 속도 [m/step]


BETA = 0.035            # 보상 상승 보폭 [m/step]. 팔을 넘으려면 15cm 를
                        # 올라가야 해서 2cm/step 으로는 7초가 걸린다.
ASCEND_H = 0.12         # h 가 이보다 여유로우면 보상 상승을 끈다 [m]
HOLD_MARGIN = 0.12      # task1 — 정렬 전 경계선 앞에서 대기할 거리 [m]


def ascend(g: SG.SceneGraph, rm: "RewardMap", u: list[float],
           beta: float = BETA) -> list[float]:
    """u 에 보상 상승 방향을 얹는다 — 사영 **전** 의 명목 명령.

    안전 사영만 걸면 정책이 위험 쪽으로 계속 밀고 필터가 계속 막아 아무
    진전이 없거나, 막히기 직전까지 갔다가 반응이 늦어 접촉한다. ∇R 을
    명목항에 실어야 "막힌 방향 대신 갈 방향" 이 생긴다 — 실측으로 이 항
    하나가 task2 Safe 를 0/4 에서 6/6 으로 바꿨다 (SC+Ours).
    """
    pt = torch.tensor([[g.tcp[0], g.tcp[1], g.tcp[2]]])
    # **필요한 곳에서만 민다.** 위험에서 멀 때까지 매 스텝 3.5mm 씩 궤적을
    # 밀면 삽입처럼 4cm 공차의 정밀 조작이 깨진다 — 실측으로 안전은 10/10
    # 인데 SR 이 7/10 이었고, 실패 에피의 사영 횟수는 12 회에서 포화해 있었다
    # (안전층이 막은 게 아니라 정책이 못 꽂은 것이다). h 가 여유로우면 0 으로
    # 사라지게 해 정책을 그대로 둔다.
    h, _ = rm.safety_grad(pt)
    w = float(torch.clamp((ASCEND_H - h[0]) / ASCEND_H, 0.0, 1.0))
    if w <= 0.0:
        return u
    gr = rm.grad(pt)[0]
    n = float(torch.linalg.norm(gr))
    if n < 1e-9:
        return u
    return [u[i] + float(gr[i]) / n * beta * w for i in range(3)]


def steer(g: SG.SceneGraph, rm: "RewardMap",
          u: list[float]) -> tuple[list[float], float, bool]:
    """보상 상승 + 안전 사영 — 세 방법이 공유하는 실행 계층.

    위험 정보가 아직 없으면(에피소드 첫 순간) 정지한다. 한두 스텝 손해지만,
    그 창에서 그대로 밀고 나가면 팔에 닿는다.
    """
    if g.blind():
        return [0.0, 0.0, 0.0], 0.0, True
    return safe_project(g, rm, ascend(g, rm, u))


def safe_project(g: SG.SceneGraph, rm: "RewardMap",
                 u: list[float]) -> tuple[list[float], float, bool]:
    """u 를 안전 제약 ∇h·u ≥ −α·h·dt 를 만족하는 최소 수정으로 사영한다.

    목적이 ‖u−u₀‖² 이고 제약이 u 에 선형이라 반공간 사영이 곧 QP 의 KKT 해다.
    반환: (수정된 u, h, 수정했는가)
    """
    if not g.hazards() and g.arm_segment() is None:
        return u, 10.0, False
    pts = torch.tensor([[g.eef[0], g.eef[1], g.eef[2] + dz] for dz in CTRL_DZS])
    h, grad = rm.safety_grad(pts)
    k = int(torch.argmin(h))
    hv, gv = float(h[k]), [float(v) for v in grad[k]]
    gn2 = sum(v * v for v in gv)
    if gn2 < 1e-12:
        return u, hv, False
    if hv < DEEP:
        gn = gn2 ** 0.5
        return [gv[i] / gn * ESCAPE for i in range(3)], hv, True
    gu = sum(gv[i] * u[i] for i in range(3))
    lo = -CBF_ALPHA * hv * CBF_DT
    if gu < lo:
        lam = (lo - gu) / gn2
        return [u[i] + lam * gv[i] for i in range(3)], hv, True
    return u, hv, False


def stage_of(g: SG.SceneGraph) -> str:
    """씬 그래프 술어만으로 스테이지를 정한다 — VLM 의 단계 판단을 대체한다."""
    if g.task == "task1":
        return "deliver" if g.grasped else "reach"
    if g.task == "task2":
        return "plug" if g.grasped else "reach"
    if not g.targets():
        return "home"
    return "carry" if g.grasped else "reach"


# 요 회전은 **느려야** 도구가 따라온다. 실측: 0.42 rad/s 로 그리퍼를 180°
# 돌리는 동안 망치는 20° 밖에 안 돌았다 — 죠 안에서 미끄러진다. 마찰이
# 도구를 끌고 갈 수 있는 속도로 낮춘다.
YAW_RATE = 0.12         # task1 요 회전 각속도 상한 [rad/s]
_YAW_LATCH = {"target": None}   # 파지 순간에 정한 목표 요 (재계산 금지)


def yaw_command(g: SG.SceneGraph) -> float | None:
    """task1 전용 — 손잡이를 작업자 쪽으로 돌리는 **물체** 요 목표.

    액션 규약이 7D(dyaw 포함)인데 지금까지 vertical_rot 이 요를 상수로 못박아
    손잡이 방향이 통제 불능이었다 (Safe 0.00 의 진짜 원인). 도구를 쥔 뒤에는
    그리퍼 요를 돌리면 도구도 같이 돈다.
    """
    if g.task != "task1" or not g.grasped:
        _YAW_LATCH["target"] = None
        return None
    held = g.held_node()
    tool = held if (held and held.kind == "tool") else next(
        (n for n in g.of_kind("tool")), None)
    if tool is None:
        return None
    if SG.handle_ok(tool.name, tool.quat, SG.HANDLE_STOP):
        _YAW_LATCH["target"] = None
        return None                       # 충분히 안쪽 — 그만 돌린다
    # 쥔 도구를 올바로 특정하고부터는 도구가 그리퍼를 따라 돈다. 그래서 매
    # 스텝 다시 계산해도 오차가 줄어들며 수렴한다 (예전 폭주는 엉뚱한 도구의
    # 자세로 오차를 재던 탓이었다). 목표를 계속 갱신해 확실히 넘긴다.
    return g.eef_yaw + SG.yaw_error(tool.name, tool.quat)
