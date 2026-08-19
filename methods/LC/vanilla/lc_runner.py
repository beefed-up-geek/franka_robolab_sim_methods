#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LC — Steerable Policies (arXiv:2602.13193) 의 VLM 상위 지휘 구성.

저수준은 steering command 로 학습한 우리 lang 모델(task*_lang)이고, 상위는
off-the-shelf VLM(gemini-3.7-flash) 의 in-context reasoning 이다 — 논문
Fig.4(b) 의 구성 그대로: VLM 이 관측·이력에서 (1) 장면을 해석하고 (2) 다음에
할 일을 정하고 (3) 가장 적절한 **추상화 수준**(task/subtask/motion/point)의
명령을 골라 내리면, 저수준 정책이 N 스텝 동안 그 명령을 따른다.

VLM 호출 예산: 에피소드당 30회 이하 (사용자 제약). 재지시는 주기(N스텝)
| 그리퍼 상태 변화 | 시뮬 이벤트에서 일어나되 예산이 다하면 마지막 명령을
유지한다. 명령 이력과 실행 피드백(EEF·파지 상태)을 프롬프트에 되먹인다 —
논문의 in-context corrective steering.
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
from common.bridge import (CAMS, TASK_TEXT, MethodsBridge,  # noqa: E402
                           run_episodes)
from common.vlm import VLM                                  # noqa: E402

import rclpy                                                # noqa: E402

SERVER = "http://127.0.0.1:8010"
REQUERY_STEPS = 48        # 재지시 주기 [스텝] — 6Hz 로 8초. 120초 에피소드에
                          # 최대 15회 + 이벤트 몇 회 ≈ 예산 30 안쪽.

GOALS = {
    "task1": "Deliver the hammer to the worker across the yellow tape "
             "(y < -0.40). Deliver tools with the handle pointing toward the "
             "worker (-y direction) for safe handover.",
    "task2": "Plug the red charging connector into the battery positive "
             "terminal. If a worker's arm is in the workspace, avoid touching "
             "it while carrying the connector.",
    "task3": "Pick up only NORMAL (intact) cans from the conveyor and put "
             "them in the grey bin. Cans with bulged/burst lids are damaged — "
             "never touch or pick them.",
}

STYLE_GUIDE = """Command styles the low-level policy understands (choose the most
appropriate abstraction each time — this is the key steering decision):
1. task level      e.g. "pass the hammer" / "Plug the red charging connector into the battery positive terminal" / "Pick up the cans from the conveyor and put them in the bin"
2. subtask level   e.g. "reach for the hammer" / "grasp the can" / "lift the red connector" / "carry the can to the bin" / "return to the home position"
3. atomic motion   e.g. "move left" / "move right" / "move forward" / "move backward" / "move up" / "move down" / "close the gripper" / "open the gripper"
4. grounded point  e.g. "reach for the can at [0.52, -0.20]" / "grasp the hammer at [0.53, 0.10]" / "put the can in the bin at [0.26, 0.58]"
Coordinates are world-frame meters with 2 decimals. +x forward, +y left toward
the conveyor outlet, worker zone at y < -0.40 (task1) . The bin is at [0.26, 0.58]."""


class Supervisor:
    """비동기 재지시 — VLM 이 생각하는 몇 초 동안 정책은 현재 명령을 계속
    따른다 (논문의 계층 구조와 동일: 저수준은 N 스텝마다만 재지휘된다).
    동기로 부르면 6Hz 제어가 수 초씩 얼어 타임아웃만 까먹는다."""

    def __init__(self, vlm: VLM, task: str) -> None:
        self.vlm = vlm
        self.task = task
        self.command = TASK_TEXT[task]      # 예산 소진·실패 시의 기본 명령
        self.history: list[str] = []
        self.last_step = -10**9
        self.last_grip = None
        self._busy = False

    def scene_text(self, node) -> str:
        objs = {n: [round(v, 2) for v in p] for n, p in node.objects.items()
                if abs(p[2]) < 1.0}
        eef = [round(v, 3) for v in (node.eef or [0, 0, 0])]
        return (f"EEF position: {eef}, gripper "
                f"{'closed' if node.gripper > 0.5 else 'open'}\n"
                f"Object world positions (x,y,z): {json.dumps(objs)}")

    def requery(self, node, reason: str, wait: bool = False) -> None:
        """wait=True 면 동기(에피소드 시작 시), 아니면 백그라운드 스레드."""
        if self.vlm.used >= self.vlm.budget or self._busy:
            return
        self._busy = True
        if wait:
            self._requery_sync(node, reason)
        else:
            threading.Thread(target=self._requery_sync,
                             args=(node, reason), daemon=True).start()

    def _requery_sync(self, node, reason: str) -> None:
        img_f = node.images.get("front")
        img_w = node.images.get("wrist")
        imgs = [i for i in (img_f, img_w) if i]
        hist = "\n".join(self.history[-6:]) or "(none yet)"
        prompt = (
            f"You are the high-level reasoner controlling a steerable robot "
            f"policy.\nOverall goal: {GOALS[self.task]}\n\n{STYLE_GUIDE}\n\n"
            f"Current scene (front camera, then wrist camera):\n"
            f"{self.scene_text(node)}\n"
            f"Recent commands and outcomes:\n{hist}\n"
            f"Re-query reason: {reason}\n\n"
            "Think: (1) what is the current state of the task? (2) what should "
            "the robot do next? (3) which command abstraction is most reliable "
            "for it? Prefer grounded point commands when a specific object "
            "must be selected among several; prefer subtask commands for "
            "in-distribution moves; use atomic motions for small corrections.\n"
            'JSON: {"reasoning": "...", "command": "..."}')
        try:
            d = self.vlm.call_json(prompt, images=imgs, max_tokens=700)
            if d and isinstance(d.get("command"), str) and d["command"].strip():
                self.command = d["command"].strip()
                self.history.append(f"[{time.strftime('%H:%M:%S')}] {self.command}")
                print(f"[LC] 지시({reason}, {self.vlm.used}/{self.vlm.budget}): "
                      f"{self.command}", flush=True)
        finally:
            self._busy = False

    def maybe_requery(self, node, step: int) -> None:
        grip = node.gripper > 0.5
        if step - self.last_step >= REQUERY_STEPS:
            self.last_step = step
            self.requery(node, "periodic")
        elif self.last_grip is not None and grip != self.last_grip:
            self.last_step = step
            self.requery(node, "gripper state changed")
        self.last_grip = grip


def post(url, payload, timeout=20.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("task1", "task2", "task3"))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="/workspace/methods/results/raw")
    args = ap.parse_args()

    rclpy.init()
    node = MethodsBridge("abs")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    t0 = time.time()
    while not node.ready() and time.time() - t0 < 30:
        time.sleep(0.1)
    if not node.ready():
        print("[LC] 토픽이 오지 않습니다.", flush=True)
        return 1

    vlm = VLM(budget=30, log_path=f"{args.out}/LC_{args.task}_vlm.jsonl")
    sup = Supervisor(vlm, args.task)

    def on_start(node, ep):
        post(f"{SERVER}/reset", {})
        vlm.reset_budget()
        sup.command = TASK_TEXT[args.task]
        sup.history.clear()
        sup.last_step = -10**9
        sup.last_grip = None
        time.sleep(0.5)
        sup.requery(node, "episode start", wait=True)
        sup.last_step = 0             # 다음 주기 재지시는 REQUERY_STEPS 뒤

    def on_event(node, e):
        et = e.get("type")
        if et in ("arm_collision", "burst_touched"):
            sup.requery(node, f"safety event: {et}")
        if et == "trio_spawn":
            sup.requery(node, "new cans placed")

    def act(node, step):
        sup.maybe_requery(node, step)
        imgs = {c: base64.b64encode(node.images[c]).decode() for c in CAMS}
        res = post(f"{SERVER}/act", {"state": node.eef + [node.gripper],
                                     "images": imgs, "task": sup.command})
        node.send(res["action"])

    run_episodes(node, args.task, args.episodes, act,
                 on_episode_start=on_start, on_event=on_event,
                 timeout=args.timeout,
                 out_jsonl=f"{args.out}/LC_{args.task}.jsonl", tag=f"LC/{args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
