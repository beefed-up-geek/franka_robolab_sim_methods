#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SC — VLSA/AEGIS (arXiv:2512.11891) 의 플러그앤플레이 안전 제약(CBF) 층.

vanilla abs VLA 위에 학습 없이 얹는다. 논문 파이프라인:
  ① VLM 이 지시문+장면에서 "가장 위험한 장애물" 하나를 지목 (의미 판단)
  ② 장애물을 타원체로 모델링, EEF 도 타원체 — 부호 있는 거리 h(x)
  ③ CBF-QP: VLA 의 명령 u 를 h 의 forward invariance 를 지키는 최소 수정
     u* = argmin ||u − u_vla||²  s.t.  ḣ ≥ −α h        (논문 eq.2)

이 구현의 충실/대체 지점 (methods/SC/vanilla/README.md 에 상세):
  · 충실 — VLM 의미 지목(①), 타원체 CBF-QP 최소 수정(③: 제약이 u 에 선형이라
    반공간 사영의 닫힌형이 곧 QP 최적해다)
  · 대체 — ② 의 GroundingDINO+깊이 융합+MVEE 를 시뮬 특권 상태(정확한 물체
    자세 + 실측 치수)로 치환. 시뮬리는 지각이 병목이 아니고, 논문의 기여인
    "의미 지목 → 기하 제약 → 최소 수정" 사슬은 그대로 남는다.

VLM 호출: 에피소드당 지목 1회 + 장면 변화(리셋·라운드) 시 재지목 — 30회
한도의 한참 아래다.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
import threading
import urllib.request

sys.path.insert(0, "/workspace/methods")
from common.bridge import (CAMS, ENVS, TASK_TEXT, MethodsBridge,  # noqa: E402
                           run_episodes)
from common.vlm import VLM                                        # noqa: E402

import rclpy                                                      # noqa: E402

# ── 장애물 카탈로그 — 시뮬 실측 치수의 타원체 반축 [m] ────────────────────
# worker_arm: 팔꿈치 원점에서 -X 로 0.52m 뻗는 반경 0.075 캡슐 → 타원체.
#   중심은 원점에서 -0.26m, 반축 (0.335, 0.075, 0.075).
# 캔(파열): Ø71mm, 파열 높이 83mm → 반축 (0.05, 0.05, 0.06).
ARM_C_OFF = (-0.26, 0.0, 0.0)
ARM_SEMI = (0.335, 0.075, 0.075)
CAN_SEMI = (0.05, 0.05, 0.06)
# EEF 는 환경 판정과 같은 **2제어점**(플랜지, 손끝 −15cm)으로 본다. 한 점에
# z 팽창을 몰아넣으면 팔 아래 19cm 의 정상 통과 경로까지 위험으로 오인해
# (실측 h=−0.72) 필터가 그래스프를 계속 방해했다.
EEF_INFLATE = (0.05, 0.05, 0.05)
CTRL_DZS = (0.0, -0.15)
ALPHA = 1.0          # class-K 계수 [1/s] — 작을수록 일찍 제동한다
DT = 1.0 / 6.0
MARGIN = 0.02        # 반축 추가 여유 [m] — 실현률·팔 진입 속도를 흡수
ARM_PARKED_X = 0.90  # 팔꿈치 원점 x 가 이보다 크면 책상 밖 대기 — 필터 끔.
                     # 파크 x=1.05, 진입 완료 x=0.715 (hand 0.33 + L 0.385) —
                     # 그 사이. 환경도 parked 상태에선 충돌 판정을 안 한다.

SERVER = "http://127.0.0.1:8010"


def post(url, payload, timeout=20.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class SafetyLayer:
    """지목된 장애물 목록에 대한 타원체 CBF 필터."""

    def __init__(self) -> None:
        self.obstacles: list[str] = []      # 물체 이름 (worker_arm 포함)
        self.filtered = 0                   # 이번 에피소드 수정 횟수
        self.h_min = float("inf")           # 진단 — 이번 에피소드 최소 h

    def geometry(self, node, name: str):
        pos = node.objects.get(name)
        if pos is None:
            return None
        if name == "worker_arm":
            if pos[0] > ARM_PARKED_X:      # 책상 밖 대기 — 장애물 아님
                return None
            c = [pos[0] + ARM_C_OFF[0], pos[1], pos[2]]
            semi = ARM_SEMI
        else:
            c = list(pos)
            semi = CAN_SEMI
        return c, [semi[i] + EEF_INFLATE[i] + MARGIN for i in range(3)]

    def filter(self, node, u: list[float]) -> list[float]:
        """CBF-QP 닫힌형 — 제약 하나당 반공간 사영, 여러 개면 순차 사영.

        h(p) = ||D⁻¹(p−c)||² − 1  (팽창 타원체 레벨셋, 안전 ⇔ h ≥ 0)
        ḣ = ∇h·(u/dt)  ⇒  제약  g·u ≥ −α·h·dt,  g = 2 D⁻²(p−c)
        목적이 ||u−u_vla||² 이고 제약이 선형이라, 위반 시 최적해는
        u* = u + ((−αh·dt − g·u)/||g||²)·g  — 이것이 곧 QP 의 KKT 해다.
        """
        if not self.obstacles or node.eef is None:
            return u
        for name in self.obstacles:
            geo = self.geometry(node, name)
            if geo is None:
                continue
            c, d = geo
            # 두 제어점 중 **가장 위험한 쪽**의 제약을 적용한다 (순차 사영).
            h, g = None, None
            for dz in CTRL_DZS:
                p = [node.eef[0], node.eef[1], node.eef[2] + dz]
                q = [(p[i] - c[i]) / d[i] for i in range(3)]
                hh = sum(v * v for v in q) - 1.0
                if h is None or hh < h:
                    h = hh
                    g = [2.0 * q[i] / d[i] for i in range(3)]
            self.h_min = min(self.h_min, h)
            if h < 0.0:
                # 확장 타원체 안 — QP 의 전제(안전 집합 내부 유지)가 이미
                # 깨졌다. 목적항을 버리고 최대 속도로 밖을 향한다 (실제
                # 충돌 반경은 이보다 훨씬 안쪽이라 이 단계는 '경고 구역').
                gn = math.sqrt(sum(v * v for v in g)) or 1e-9
                u = [g[i] / gn * 0.08 for i in range(3)]
                self.filtered += 1
                continue
            gu = sum(g[i] * u[i] for i in range(3))
            lo = -ALPHA * h * DT
            if gu < lo:
                gg = sum(v * v for v in g) or 1e-9
                lam = (lo - gu) / gg
                u = [u[i] + lam * g[i] for i in range(3)]
                self.filtered += 1
        return u


def identify(vlm: VLM, node, task: str, layer: SafetyLayer) -> None:
    """논문 ① — VLM 이 장면에서 가장 위험한 비로봇 장애물을 지목한다."""
    candidates = [n for n in node.names if n not in ("battery", "generator")]
    img = node.images.get("front")
    prompt = (
        f"You are a safety assessment module for a robot arm.\n"
        f"Task instruction: \"{TASK_TEXT[task]}\"\n"
        f"Objects in the scene: {candidates}\n"
        "Identify the objects that are safety-critical obstacles the robot's\n"
        "gripper must NOT touch while performing the task. Rules:\n"
        "- A worker's arm (worker_arm) is always critical if present.\n"
        "- Damaged/burst cans must not be touched (contamination risk), but\n"
        "  normal cans that the task requires grasping are NOT obstacles.\n"
        "- Objects the task explicitly manipulates are not obstacles.\n"
        "- If nothing is safety-critical, return an empty list.\n"
        'JSON: {"obstacles": ["name", ...]}')
    d = vlm.call_json(prompt, images=[img] if img else [], max_tokens=400)
    if d and isinstance(d.get("obstacles"), list):
        layer.obstacles = [n for n in d["obstacles"] if n in node.names]
    else:
        layer.obstacles = []      # VLM 실패 — 필터 없이 vanilla 로 진행
    print(f"[SC] 장애물 지목: {layer.obstacles or '없음'} "
          f"(VLM {vlm.used}/{vlm.budget})", flush=True)


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
        print("[SC] 토픽이 오지 않습니다.", flush=True)
        return 1

    vlm = VLM(budget=30, log_path=f"{args.out}/SC_{args.task}_vlm.jsonl")
    layer = SafetyLayer()
    text = TASK_TEXT[args.task]

    def on_start(node, ep):
        post(f"{SERVER}/reset", {})   # 정책 서버 액션 큐 비우기 (run_policy 동일)
        vlm.reset_budget()
        layer.filtered = 0
        time.sleep(0.5)               # 물체 자세 갱신 여유
        identify(vlm, node, args.task, layer)

    def on_event(node, e):
        # 라운드 재배치(task3)면 파열 캔 구성이 바뀐다 — 재지목
        if e.get("type") == "trio_spawn" and vlm.used < vlm.budget:
            identify(vlm, node, args.task, layer)
        # 팔 진입 — 에피소드 시작 지목 때는 팔이 화면 밖(파크)이라 VLM 이
        # 못 봤을 수 있다. 장면이 실제로 위험해진 순간 다시 평가한다 (논문의
        # 실시간 안전 평가에 해당).
        if e.get("type") == "arm_enter" and vlm.used < vlm.budget:
            identify(vlm, node, args.task, layer)

    def act(node, step):
        imgs = {c: base64.b64encode(node.images[c]).decode() for c in CAMS}
        res = post(f"{SERVER}/act", {"state": node.eef + [node.gripper],
                                     "images": imgs, "task": text})
        a = res["action"]
        u = node.abs_to_delta(a)
        u = layer.filter(node, u)
        node.send_delta(u, a[3] > 0.5)
        if step % 60 == 59:            # 진단 — 10초마다 필터 상태
            print(f"[SC]   step{step + 1}: 수정 {layer.filtered}회, "
                  f"h_min={layer.h_min:.2f}", flush=True)

    run_episodes(node, args.task, args.episodes, act,
                 on_episode_start=on_start, on_event=on_event,
                 timeout=args.timeout,
                 out_jsonl=f"{args.out}/SC_{args.task}.jsonl", tag=f"SC/{args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
