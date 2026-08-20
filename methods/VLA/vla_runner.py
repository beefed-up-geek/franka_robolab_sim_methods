#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VLA — 기본(vanilla) VLA baseline. 방법론 없이 정책 그대로.

표의 기준선이다: 고정 태스크 문장 하나로 /act 를 6Hz 호출해 그대로 발행한다.
VLM·안전 필터·조향 전부 없음. 판정·폭주 무효 규칙은 다른 방법과 동일한
common.bridge.run_episodes 를 쓴다 — 그래야 행 간 비교가 성립한다.
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
from common.bridge import CAMS, TASK_TEXT, MethodsBridge, run_episodes  # noqa: E402

import rclpy                                                            # noqa: E402

SERVER = "http://127.0.0.1:8010"


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
        print("[VLA] 토픽이 오지 않습니다.", flush=True)
        return 1

    text = TASK_TEXT[args.task]

    def on_start(node, ep):
        post(f"{SERVER}/reset", {})

    def act(node, step):
        imgs = {c: base64.b64encode(node.images[c]).decode() for c in CAMS}
        res = post(f"{SERVER}/act", {"state": node.eef + [node.gripper],
                                     "images": imgs, "task": text})
        node.send(res["action"])

    run_episodes(node, args.task, args.episodes, act,
                 on_episode_start=on_start, timeout=args.timeout,
                 out_jsonl=f"{args.out}/VLA_{args.task}.jsonl",
                 tag=f"VLA/{args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
