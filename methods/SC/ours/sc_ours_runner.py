#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SC + Ours — 씬 그래프가 위험을 지목하고, 보상 맵의 미분이 CBF 장벽이 된다.

vanilla SC 의 파이프라인은 ① VLM 이 "가장 위험한 장애물" 하나를 지목 → ②
타원체로 모델링 → ③ CBF-QP 최소 수정이었다. ①이 병목이었다: VLM 은 한 번에
하나만 지목하고, 그 라운드에 없는 물체를 지목하기도 했다 (실측: burst_can 만
지목해 fig_can_burst 가 무방비).

여기서는 ①·②를 통째로 바꾼다.
  ① 위험 지목 → 씬 그래프의 hazard 술어. 태스크 정의에서 직접 나오므로
     빠뜨리거나 지어내는 일이 없고, **전부** 를 동시에 본다.
  ② 타원체 손계산 → 보상 맵의 안전장 h(p) 와 autograd 로 얻은 ∇h.
     팔은 점이 아니라 선분으로 들어가고, 여러 위험은 min 으로 합쳐진다.
  ③ CBF-QP 최소 수정은 논문 그대로 둔다 — 이 층이 논문의 기여다.

QP 는 vanilla 와 같은 닫힌형을 쓴다. 목적이 ‖u−u_vla‖² 이고 제약이 u 에
선형이라 반공간 사영이 곧 KKT 해다:
    u* = u + ((−αh·dt − ∇h·u)/‖∇h‖²)·∇h        (제약 위반 시)
차이는 ∇h 를 손으로 미분한 타원체 식이 아니라 보상 맵에서 autograd 로
받는다는 것뿐이다 — 그래서 장벽 모양을 바꿔도 코드가 그대로다.

task1 은 요(yaw) 채널로 손잡이를 작업자 쪽으로 돌린다. 손잡이 방향은 CBF 로
막을 수 있는 종류의 위반이 아니다 (금지 구역이 아니라 자세의 문제라서).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/workspace/methods")
sys.path.insert(0, "/workspace/methods/common")
from common.bridge import (CAMS, TASK_TEXT, MethodsBridge,       # noqa: E402
                           run_episodes)
import scene_graph as SG                                          # noqa: E402
import diffreward as DR                                           # noqa: E402

import rclpy                                                      # noqa: E402
import torch                                                      # noqa: E402

SERVER = "http://127.0.0.1:8010"
ALPHA = 1.0             # class-K 계수 [1/s] — 작을수록 일찍 제동
DT = 1.0 / 6.0
CTRL_DZS = (0.0, SG.TCP_DZ)   # 두 제어점: 플랜지와 손끝
ESCAPE_SPEED = 0.08     # 장벽 안쪽으로 들어갔을 때 탈출 속도 [m/step]
# QP 의 목적항을 u_vla 가 아니라 u_vla + β·∇R 로 둔다. 안전 필터만 얹으면
# 정책이 위험한 물체를 계속 잡으려 하고 필터가 계속 막아 아무 진전이 없다
# (실측: SC vanilla task3 SR 0.00, SCo 첫 에피 binned_ok 0). 보상 기울기를
# 목적항에 실어야 "막힌 방향 대신 갈 방향" 이 생긴다. CBF 제약은 그대로라
# 안전 보장은 약해지지 않는다 — 논문의 최소수정 구조도 유지된다.
BETA = 0.020            # 보상 상승 보폭 [m/step]
BETA_MAX = 0.030        # 그 항의 상한 [m/step]


def post(url, payload, timeout=30.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class DiffSafetyLayer:
    """보상 맵의 안전장을 장벽으로 쓰는 CBF-QP 필터."""

    def __init__(self) -> None:
        self.filtered = 0
        self.h_min = float("inf")

    def nominal(self, g: SG.SceneGraph, rm: DR.RewardMap,
                u: list[float]) -> list[float]:
        """u_vla 에 보상 상승 방향을 얹은 명목 명령 — QP 의 목적항."""
        p = torch.tensor([[g.tcp[0], g.tcp[1], g.tcp[2]]])
        gr = rm.grad(p)[0]
        n = float(torch.linalg.norm(gr))
        if n < 1e-9:
            return u
        step = min(BETA, BETA_MAX)
        d = [float(gr[i]) / n * step for i in range(3)]
        return [u[i] + d[i] for i in range(3)]

    def filter(self, g: SG.SceneGraph, rm: DR.RewardMap,
               u: list[float]) -> list[float]:
        u, hv, fixed = DR.steer(g, rm, u)
        self.h_min = min(self.h_min, hv)
        if fixed:
            self.filtered += 1
        return u

    def _unused(self, g, rm, u):
        if not g.hazards() and g.arm_segment() is None:
            return u
        # 두 제어점 중 **가장 위험한 쪽** 의 제약만 걸어 순차 사영한다.
        pts = torch.tensor([[g.eef[0], g.eef[1], g.eef[2] + dz]
                            for dz in CTRL_DZS])
        h, grad = rm.safety_grad(pts)
        k = int(torch.argmin(h))
        hv = float(h[k])
        gv = [float(v) for v in grad[k]]
        self.h_min = min(self.h_min, hv)
        gn2 = sum(v * v for v in gv)
        if gn2 < 1e-12:
            return u
        if hv < 0.0:
            # 장벽 안쪽 — QP 의 전제(안전 집합 내부)가 이미 깨졌다. 목적항을
            # 버리고 ∇h 방향(가장 빨리 빠져나오는 쪽)으로 최대 속도.
            gn = math.sqrt(gn2)
            self.filtered += 1
            return [gv[i] / gn * ESCAPE_SPEED for i in range(3)]
        gu = sum(gv[i] * u[i] for i in range(3))
        lo = -ALPHA * hv * DT
        if gu < lo:
            lam = (lo - gu) / gn2
            self.filtered += 1
            return [u[i] + lam * gv[i] for i in range(3)]
        return u


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("task1", "task2", "task3"))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="/workspace/methods/results/raw")
    ap.add_argument("--tag", default="SCo")
    args = ap.parse_args()

    rclpy.init()
    node = MethodsBridge("abs")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    while not node.ready():
        time.sleep(0.2)

    layer = DiffSafetyLayer()
    queue: list[list[float]] = []
    fails = {"n": 0}
    text = TASK_TEXT[args.task]

    def on_start(node, ep):
        # 서버가 아직 안 떴을 수 있다 — 죽지 말고 기다린다.
        for _ in range(60):
            try:
                post(f"{SERVER}/reset", {}, timeout=5.0)
                break
            except Exception:                          # noqa: BLE001
                time.sleep(2.0)
        queue.clear()
        layer.filtered = 0
        layer.h_min = float("inf")
        time.sleep(0.5)

    def on_event(node, e):
        if e.get("type") == "arm_collision":
            print("[SCo] !! 팔 접촉 step=%d h_min=%+.3f" % (
                on_event.step, layer.h_min), flush=True)
    on_event.step = 0

    def act(node, step):
        on_event.step = step
        g = SG.build(node, args.task)
        g.eef_yaw = node.eef_yaw()
        rm = DR.RewardMap(g, DR.stage_of(g))
        if not queue:
            imgs = {c: base64.b64encode(node.images[c]).decode() for c in CAMS}
            try:
                res = post(f"{SERVER}/act_chunk", {
                    "state": node.eef + [node.gripper], "images": imgs,
                    "task": text, "k": 1}, timeout=30.0)
                queue.extend(res["chunks"][0][:8])
            except Exception as e:                     # noqa: BLE001
                fails["n"] += 1
                if fails["n"] <= 3:
                    print(f"[SCo] 청크 실패({type(e).__name__}: {e})", flush=True)
                return
        a = queue.pop(0)
        u = layer.filter(g, rm, node.abs_to_delta(a))
        node.send_delta(u, a[3] > 0.5, yaw=DR.yaw_command(g), yaw_limit=DR.YAW_RATE)
        if args.task == "task1" and (step + 1) % 20 == 0:
            _h = g.held_node()
            _t = _h if (_h and _h.kind == "tool") else next(
                (n for n in g.of_kind("tool")), None)
            if _t is not None:
                _wx, _wy = SG.handle_dir_world(_t.name, _t.quat)
                _yc = DR.yaw_command(g)
                print("[SCo] t1 step%d 파지%d(%s) eefyaw=%+.2f %s손잡이="
                      "(%+.2f,%+.2f) ok=%s 명령=%s y=%+.3f" % (
                          step + 1, 1 if g.grasped else 0,
                          (_h.name if _h else str(list(((node.status or {}).get("contact") or {}).keys()))), g.eef_yaw, _t.name,
                          _wx, _wy, SG.handle_ok(_t.name, _t.quat),
                          "-" if _yc is None else "%+.2f" % _yc, _t.pos[1]),
                      flush=True)
        _t1dbg = 1
        if (step + 1) % 60 == 0:
            _a = g.nodes.get("worker_arm")
            _dbg = ("팔없음" if _a is None else
                    "팔[%.2f,%.2f,%.2f]v[%+.2f,%+.2f]%s" % (
                        _a.pos[0], _a.pos[1], _a.pos[2],
                        g.arm_vel[0], g.arm_vel[1],
                        "위험" if _a.hazard else "대기"))
            print(f"[SCo]   step{step + 1}: h_min={layer.h_min:+.3f} "
                  f"수정 {layer.filtered}회 위험 {len(g.hazards())}개",
                  _dbg, flush=True)

    run_episodes(node, args.task, args.episodes, act,
                 timeout=args.timeout, on_episode_start=on_start,
                 on_event=on_event,
                 out_jsonl=f"{args.out}/{args.tag}_{args.task}.jsonl",
                 tag=f"SCo/{args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
