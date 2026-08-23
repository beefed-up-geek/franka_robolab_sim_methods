#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VLS + Ours — 보상 맵의 미분을 디노이징 루프에 주입한다.

vanilla VLS 는 VLM 이 보상 함수를 **코드로 써서** 주고, 서버가 K 입자를 뽑아
그 보상으로 리샘플링했다. 두 가지가 문제였다.
  · VLM 이 쓴 코드가 씬에 없는 키를 참조해 조용히 죽는다 (실측 1334회 =
    에피소드 전체 무유도). 여기서는 보상이 씬 그래프에서 직접 나오므로
    그런 실패 모드가 없다.
  · 리샘플링만으로는 정책이 만든 후보 밖으로 나갈 수 없다. 후보가 전부
    파열 캔을 스치면 고를 것이 없다.

여기서는 **미분** 을 쓴다. 보상 맵 R(p) 가 미분 가능하므로 ∇_p R 을 디노이징
중간 상태에 직접 얹을 수 있다 (서버의 /act_chunk_guided). 다만 강도는
보수적으로 둔다 — 정지된 flow-matching 정책을 매니폴드 밖으로 밀면 정밀
조작이 깨진다는 것을 VLS_authentic 에서 측정했다 (λ 상한 고착 → 파지 실패).
그래서 여기서는 λ 를 적응시키지 않고 **변위 상한 안에서 고정 강도** 로 민다.

보상 자체는 세 방법이 공유한다 (common/diffreward.py). 방법 간 차이가
"같은 보상을 어떻게 쓰는가" 로만 남아야 비교가 성립한다.
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
B_PARTICLES = 6
MCMC_STEPS = 2          # 디노이징 스텝당 보상 기울기 갱신 횟수
GUIDANCE_LR = 0.10
RBF_WEIGHT = 0.02
FK_TEMP = 1.0
LAMBDA = 1.0            # 고정 — 적응형 λ 는 상한에 고착해 해로웠다 (실측)
MAX_DEV = 0.02          # 한 청크의 총 유도 변위 상한 [정규화 단위]
EXEC_STEPS = 8


def post(url, payload, timeout=60.0):
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
    ap.add_argument("--tag", default="VLSo")
    ap.add_argument("--max-dev", type=float, default=MAX_DEV)
    ap.add_argument("--guide-off", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = MethodsBridge("abs")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    while not node.ready():
        time.sleep(0.2)

    queue: list[list[float]] = []
    text = TASK_TEXT[args.task]
    stats = {"guided": 0, "fell_back": 0, "fixed": 0, "h": 10.0}

    def on_start(node, ep):
        # 서버가 아직 안 떴을 수 있다 — 죽지 말고 기다린다.
        for _ in range(60):
            try:
                post(f"{SERVER}/reset", {}, timeout=5.0)
                break
            except Exception:                          # noqa: BLE001
                time.sleep(2.0)
        queue.clear()
        stats["guided"] = stats["fell_back"] = stats["fixed"] = 0
        stats["h"] = 10.0
        time.sleep(0.5)

    def act(node, step):
        g = SG.build(node, args.task)
        g.eef_yaw = node.eef_yaw()
        stage = DR.stage_of(g)
        rm = DR.RewardMap(g, stage)
        if not queue:
            imgs = {c: base64.b64encode(node.images[c]).decode() for c in CAMS}
            chunk = None
            if not args.guide_off:
                try:
                    res = post(f"{SERVER}/act_chunk_guided", {
                        "state": node.eef + [node.gripper], "images": imgs,
                        "task": text, "n_particles": B_PARTICLES,
                        "reward_src": DR.REWARD_SRC,
                        "kp": DR.vls_payload(rm), "lam": LAMBDA,
                        "mcmc_steps": MCMC_STEPS, "guidance_lr": GUIDANCE_LR,
                        "rbf_weight": RBF_WEIGHT, "fk_temp": FK_TEMP,
                        "tcp_dz": SG.TCP_DZ, "max_dev": args.max_dev})
                    chunk = res["chunk"]
                    stats["guided"] += 1
                except Exception as e:                 # noqa: BLE001
                    stats["fell_back"] += 1
                    if stats["fell_back"] <= 3:
                        print(f"[VLSo] 유도 실패({type(e).__name__}: {e}) — "
                              f"vanilla 대체", flush=True)
            if chunk is None:
                try:
                    chunk = post(f"{SERVER}/act_chunk", {
                        "state": node.eef + [node.gripper], "images": imgs,
                        "task": text, "k": 1}, timeout=30.0)["chunks"][0]
                except Exception as e:                 # noqa: BLE001
                    stats["fell_back"] += 1
                    if stats["fell_back"] <= 3:
                        print(f"[VLSo] 청크 실패({type(e).__name__}: {e})",
                              flush=True)
                    return
            queue.extend(chunk[:EXEC_STEPS])
        a = queue.pop(0)
        d, h, fixed = DR.steer(g, rm, node.abs_to_delta(a))
        if fixed:
            stats["fixed"] += 1
        stats["h"] = min(stats["h"], h)
        node.send_delta(d, a[3] > 0.5, yaw=DR.yaw_command(g), yaw_limit=DR.YAW_RATE)
        if (step + 1) % 60 == 0:
            _a = g.nodes.get("worker_arm")
            _dbg = ("팔없음" if _a is None else
                    "팔[%.2f,%.2f,%.2f]v[%+.2f,%+.2f]%s" % (
                        _a.pos[0], _a.pos[1], _a.pos[2],
                        g.arm_vel[0], g.arm_vel[1],
                        "위험" if _a.hazard else "대기"))
            print(f"[VLSo]   step{step + 1}: {stage} 유도 {stats['guided']}"
                  f"/폴백 {stats['fell_back']} h={stats['h']:+.3f} "
                  f"사영 {stats['fixed']}", _dbg, flush=True)

    run_episodes(node, args.task, args.episodes, act,
                 timeout=args.timeout, on_episode_start=on_start,
                 out_jsonl=f"{args.out}/{args.tag}_{args.task}.jsonl",
                 tag=f"VLSo/{args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
