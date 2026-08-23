#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LC + Ours — 미분 가능한 보상 맵이 좌표를 계산해 언어 정책에 준다.

vanilla LC 는 VLM 이 장면을 보고 "grasp the can at [0.52, -0.20]" 같은
grounded point 명령을 만들었다. 그 좌표가 좋은 좌표라는 보장이 없다는 것이
문제였다 — VLM 은 파열 캔과 정상 캔을 이름으로 구분하긴 해도, 파지점이
위험에서 얼마나 떨어져야 하는지를 수치로 풀지 못한다 (실측 task2 SR 0.42).

여기서는 그 자리를 씬 그래프 위의 보상 맵으로 대체한다:
  ① 특권 상태 → 씬 그래프 (물체 위치·회전·상태·관계)
  ② 스테이지별 미분 가능한 보상장 R(p) 를 구성
  ③ **경사 상승으로 argmax_p R** 을 풀어 좌표를 얻는다
  ④ 그 좌표를 grounded point 명령으로 만들어 task*_lang 정책에 준다
VLM 호출은 0 회다 — 논문의 "상위 지휘가 좌표를 준다" 구조는 그대로 두고,
지휘자만 언어 모델에서 미분 가능한 프로그램으로 바꾼 것이다.

task1 은 여기에 더해 **요(yaw) 명령** 을 쓴다. 액션 규약이 7D 인데 지금까지
자세가 상수로 고정돼 손잡이 방향이 통제 불능이었다 — 보상 맵이 필요한 요를
계산하고 브릿지의 요 채널로 내보낸다.
"""
from __future__ import annotations

import argparse
import base64
import json
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

SERVER = "http://127.0.0.1:8010"
EXEC_STEPS = 8          # 청크 16 중 실행할 스텝 — 자주 다시 보고 다시 고른다
REPLAN_EVERY = 1        # 매 청크마다 다시 푼다 — 경유점이 매끄럽게 따라간다
WP_RADIUS = 0.25        # 경유점 탐색 반경 [m]. 너무 좁으면 목표가 진동한다
Z_ATOMIC = 0.06         # 경유점이 이만큼 위/아래면 원자 동작으로 바꾼다


def post(url, payload, timeout=30.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ── 좌표 → 언어 명령 ─────────────────────────────────────────────────────
# task*_lang 정책이 학습 때 본 어휘를 그대로 쓴다. 어휘를 벗어나면 정책이
# 명령을 오해한다 (vanilla LC 의 STYLE_GUIDE 가 경고하던 그 지점).
def command_for(g: SG.SceneGraph, stage: str, p) -> str:
    # 원자 동작("move up")은 쓰지 않는다 — 안전은 올랐지만 정책이 과업
    # 맥락을 잃어 SR 이 3/4 에서 2/5 로 떨어졌다 (실측). 높이 회피는 실행
    # 계층(steer)이 맡고, 언어 채널은 **무엇을 집을지 고르는 데** 만 쓴다.
    xy = f"[{p[0]:.2f}, {p[1]:.2f}]"
    if g.task == "task1":
        if stage == "reach":
            return f"grasp the hammer at {xy}"
        return "pass the hammer"
    if g.task == "task2":
        if stage == "reach":
            return f"grasp the red connector at {xy}"
        return TASK_TEXT["task2"]
    if stage == "reach":
        return f"reach for the can at {xy}"
    if stage == "carry":
        return f"put the can in the bin at [{SG.BIN_XY[0]:.2f}, {SG.BIN_XY[1]:.2f}]"
    return "return to the home position"


class Planner:
    """보상 맵으로 좌표와 명령을 뽑는다 — VLM 지휘자의 자리."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.command = TASK_TEXT[task]
        self.target = None
        self.stage = "reach"
        self.n = 0

    def update(self, node):
        g = SG.build(node, self.task)
        g.eef_yaw = node.eef_yaw()
        self.stage = DR.stage_of(g)
        rm = DR.RewardMap(g, self.stage)
        # **경유점** 을 낸다 — 최종 목표 한 점은 우회를 표현하지 못한다.
        # 위험이 없으면 전역 argmax 와 같은 곳으로 수렴한다.
        if self.n % REPLAN_EVERY == 0 or self.target is None:
            # 목표 선택은 **전역** argmax 다 — 경유점은 명령을 진동시킨다.
            self.target, _ = rm.argmax(starts=20, iters=50)
        self.n += 1
        self.command = command_for(g, self.stage, self.target)
        return self.command, DR.yaw_command(g), g, rm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("task1", "task2", "task3"))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="/workspace/methods/results/raw")
    ap.add_argument("--tag", default="LCo")
    args = ap.parse_args()

    rclpy.init()
    node = MethodsBridge("abs")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    while not node.ready():
        time.sleep(0.2)

    plan = Planner(args.task)
    queue: list[list[float]] = []
    fails = {"n": 0}
    stats = {"fixed": 0, "h": 10.0}

    def on_start(node, ep):
        # 서버가 아직 안 떴을 수 있다 — 죽지 말고 기다린다.
        for _ in range(60):
            try:
                post(f"{SERVER}/reset", {}, timeout=5.0)
                break
            except Exception:                          # noqa: BLE001
                time.sleep(2.0)
        queue.clear()
        plan.target = None
        plan.n = 0
        stats["fixed"] = 0
        stats["h"] = 10.0
        time.sleep(0.5)

    def act(node, step):
        cmd, yaw, g, rm = plan.update(node)
        if not queue:
            imgs = {c: base64.b64encode(node.images[c]).decode() for c in CAMS}
            try:
                res = post(f"{SERVER}/act_chunk", {
                    "state": node.eef + [node.gripper], "images": imgs,
                    "task": cmd, "k": 1}, timeout=30.0)
                queue.extend(res["chunks"][0][:EXEC_STEPS])
            except Exception as e:                     # noqa: BLE001
                fails["n"] += 1
                if fails["n"] <= 3:
                    print(f"[LCo] 청크 실패({type(e).__name__}: {e})", flush=True)
                return
        a = queue.pop(0)
        d, h, fixed = DR.steer(g, rm, node.abs_to_delta(a))
        if fixed:
            stats["fixed"] += 1
        stats["h"] = min(stats["h"], h)
        node.send_delta(d, a[3] > 0.5, yaw=yaw, yaw_limit=DR.YAW_RATE)
        if (step + 1) % 60 == 0:
            _a = g.nodes.get("worker_arm")
            _dbg = ("팔없음" if _a is None else
                    "팔[%.2f,%.2f,%.2f]v[%+.2f,%+.2f]%s" % (
                        _a.pos[0], _a.pos[1], _a.pos[2],
                        g.arm_vel[0], g.arm_vel[1],
                        "위험" if _a.hazard else "대기"))
            print(f"[LCo]   step{step + 1}: {plan.stage} → {plan.target} "
                  f"h={stats['h']:+.3f} 사영 {stats['fixed']} "
                  f"\"{cmd}\"" + (f" yaw={yaw:+.2f}" if yaw is not None else ""),
                  _dbg, flush=True)

    run_episodes(node, args.task, args.episodes, act,
                 timeout=args.timeout, on_episode_start=on_start,
                 out_jsonl=f"{args.out}/{args.tag}_{args.task}.jsonl",
                 tag=f"LCo/{args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
