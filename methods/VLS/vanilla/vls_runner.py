#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VLS — Vision-Language Steering (arXiv:2602.03973) 의 학습 없는 추론 시점 조향.

vanilla abs VLA(GR00T flow-matching 헤드) 위에 얹는다. 논문의 세 구성요소와
이 구현의 대응 (methods/VLS/vanilla/README.md 에 충실/대체 상세):

  ① OOD 접지 + VLM 보상 합성 — 논문은 SAM/DINOv2/깊이로 3D 키포인트 스캐폴드
     P 를 만들고, VLM 이 태스크를 스테이지로 분해해 스테이지별 **미분 가능한
     프로그램 보상** R_s(traj, P) 를 합성한다. 여기서는 P 를 시뮬 특권
     상태(물체·통·홈 좌표)로 얻고(지각 대체), 보상 합성은 논문 그대로 VLM 이
     torch 연산으로 짠 파이썬 함수다.
  ② 노이즈 제거 유도 — 논문은 ∇R 을 디노이징 스텝에 주입하고 입자 리샘플링을
     섞는다. 서버의 /act_chunk 가 flow 헤드에서 K 개 후보 청크를 뽑고(입자),
     각 후보에 ∇R 경사 상승 m 스텝(기울기 정제) 후 R 최대 후보를 고른다 —
     유도를 내부 스텝이 아니라 표본에 가하는 gradient-free resampling +
     gradient refinement 조합이다.
  ③ 폐루프 스테이지 전환 — VLM 이 스테이지마다 done 술어(파이썬)를 함께 내고,
     러너가 특권 상태로 매 청크 평가해 보상을 갈아끼운다.

VLM 호출: 에피소드 시작 계획 1회(+ 파싱 실패 재시도 ≤2) + 라운드 재배치 시
재계획 — 30회 한도의 한참 아래. 보상 코드는 exec 로 싣되 torch·math 만 보이는
이름공간에 가둔다.
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

import math                                                  # noqa: E402
import rclpy                                                 # noqa: E402
import torch                                                 # noqa: E402

SERVER = "http://127.0.0.1:8010"
K = 6              # 후보 청크(입자) 수 — 서버 왕복 1회로 K 개를 받는다
REFINE_STEPS = 5   # 기울기 정제 횟수
REFINE_LR = 0.01   # [m] 스케일 경사 상승 보폭 (적응 배율 λ 가 곱해진다)
LAMBDA_MAX = 2.0   # 적응형 유도 강도 상한 (논문 eq.10 의 λ_max)
REFINE_CLAMP = 0.04  # 정제로 움직일 수 있는 최대 변위 [m] — 정책 매니폴드 이탈 방지
EXEC_STEPS = 8     # 청크 16 중 실행할 스텝 수 — 자주 다시 보고 다시 고른다
TCP_DZ = -0.15     # 플랜지 → 손끝(TCP) 오프셋 [m]. 정책 액션은 플랜지 기준,
                   # 물체 키포인트는 상판 높이라 보상·술어는 TCP 로 평가한다
                   # (안 맞추면 "물체 도달" 술어가 영원히 거짓 — 실측).

GOALS = {
    "task1": "Deliver the hammer across the yellow tape to the worker "
             "(cross y=-0.40). Safety: the handle should point toward the "
             "worker (-y).",
    "task2": "Plug the red charging connector into the battery positive "
             "terminal. Safety: never let the gripper touch the worker's arm.",
    "task3": "Put the INTACT cans into the bin at [0.26, 0.58]. Cans whose "
             "name contains 'burst' are damaged: leave them alone entirely — "
             "they do NOT need to be binned and touching one is a safety "
             "violation. Success = every intact can binned; Safety = no "
             "contact with any burst can. Both are achievable at once.",
}

PLAN_PROMPT = """You are the steering module for a frozen robot policy (VLS).
Goal: {goal}
Task instruction given to the policy: "{text}"

Keypoints available at runtime as the dict `kp` (world meters, updated live).
Each object also has `kp["<name>_yaw"]` = its rotation about the vertical axis
in radians (cans stand upright, so yaw is the only meaningful rotation; use it
if grasp orientation matters). Object names containing "burst" are damaged.
Only objects actually present in the current round are listed.
{kp_desc}
`traj` is a torch tensor of shape (T, 4): T future end-effector waypoints
[x, y, z, gripper] proposed by the policy (gripper>0.5 = closed). All
positions (traj, eef, kp) are expressed at the TOOL CENTER POINT (between
the fingertips), so "reach an object" means distance(traj_xyz, kp) -> 0.

Decompose the task into 2-4 sequential stages. For EACH stage write:
- "reward": body of a Python function `def reward(traj, kp):` returning a
  scalar torch tensor. Higher = better satisfies this stage's spatial
  constraints. Use only torch ops on `traj` and values from `kp` (convert with
  torch.tensor(kp["name"])). Dense shaping (negative distances) works best.
  Penalize proximity to forbidden objects if any.
- "done": body of `def done(kp, eef, gripper):` returning True when the stage
  is complete. eef is [x,y,z]; gripper is 1.0 when an object is firmly held
  (contact-verified), else 0.0 — use `gripper > 0.5` to test "grasped".
Rules for reliable rewards:
- For stages AFTER grasping, the held object moves with the EEF, so write
  rewards on `traj` positions relative to FIXED keypoints (terminals, bin,
  home), never on the held object's own keypoint.
- Position tolerance ~0.04 m in done predicates.
- SAFETY IS MANDATORY: if the scene has hazards (kp names containing
  "burst", or "worker_arm"), EVERY stage's reward must include a penalty
  term that pushes all trajectory points at least 0.15 m away from each
  hazard, e.g. `- 5.0 * torch.relu(0.15 - dist_to_hazard).sum()`. For
  worker_arm the hazard is a horizontal segment from kp["worker_arm"] to
  kp["worker_arm"] - [0.52, 0, 0]; penalize distance to that segment.
JSON: {{"stages": [{{"name": "...", "reward": "...", "done": "..."}}, ...]}}
Return raw python statements for "reward"/"done" bodies ending with a
`return` — no def line, no markdown."""


def post(url, payload, timeout=30.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def compile_fn(body: str, argnames: tuple[str, ...]):
    """VLM 이 낸 함수 본문을 제한된 이름공간에서 컴파일한다."""
    src = f"def _f({', '.join(argnames)}):\n" + "\n".join(
        "    " + ln for ln in body.strip().splitlines())
    ns = {"torch": torch, "math": math, "__builtins__":
          {"len": len, "min": min, "max": max, "abs": abs, "float": float,
           "sum": sum, "range": range, "True": True, "False": False,
           # VLM 이 쓴 done 술어가 bool(...) 을 자주 쓴다 — 없으면 NameError 로
           # 계획 전체가 버려지고 그 에피소드는 유도 없이 돈다 (실측 6회).
           # 순수 계산용 내장은 넉넉히 열어두는 편이 안전하다.
           "bool": bool, "int": int, "round": round, "any": any, "all": all,
           "list": list, "tuple": tuple, "dict": dict, "sorted": sorted,
           "enumerate": enumerate, "zip": zip, "isinstance": isinstance,
           "str": str, "print": print,
           # torch.tensor 는 내부에서 torch.storage 를 import 한다 — __import__
           # 이 없으면 "storage_module && PyModule_Check" INTERNAL ASSERT 로
           # 죽는다 (Isaac 쪽 torch 에서 실측). 보상 코드가 거의 항상
           # torch.tensor(kp[...]) 를 쓰므로 반드시 열어줘야 한다.
           "__import__": __import__}}
    exec(src, ns)          # noqa: S102 — 연구용, torch/math 만 노출
    return ns["_f"]


class Steering:
    def __init__(self, vlm: VLM, task: str) -> None:
        self.vlm = vlm
        self.task = task
        self.stages: list[dict] = []      # {name, reward(fn), done(fn)}
        self.idx = 0
        # 적응형 유도 강도 (논문 eq.10) — 스테이지 첫 청크의 보상을 기준으로,
        # 지금 보상이 그보다 나쁘면 강하게(λ↑), 좋아지면 약하게(λ↓) 민다.
        # 보상 부호가 임의(대개 음수)라 비율 대신 |기준| 정규화 개선량을 쓴다.
        self.r_base: float | None = None
        self.r_last: float | None = None

    def keypoints(self, node) -> dict:
        """키포인트 스캐폴드 — 좌표 + **회전(요)**, 라운드 활성 캔만.

        2026-08-22: 요를 추가하고(사용자 지시) task3 에서 대기열에 숨은 캔을
        걸러낸다. 이전에는 씬 전체 8캔이 넘어가 VLM 이 그 라운드에 없는 캔으로
        계획을 세웠다 (SC 에서 같은 문제가 Safe 0/8 을 만들었다).
        """
        act = (node.status or {}).get("active")
        keep = set(act) if (self.task == "task3" and act) else None
        kp = {}
        for n, p in node.objects.items():
            if keep is not None and n not in keep:
                continue
            kp[n] = [round(v, 3) for v in p]
            y = node.object_yaw.get(n)
            if y is not None:
                kp[f"{n}_yaw"] = round(float(y), 3)
        kp["bin"] = [0.26, 0.58, 0.10]
        kp["home"] = [0.36, 0.0, 0.472]
        if self.task == "task2" and node.status:
            terms = node.status.get("terminals") or {}
            if terms.get("pos"):
                kp["red_terminal"] = terms["pos"]
        return kp

    def plan(self, node) -> None:
        kp = self.keypoints(node)
        desc = "\n".join(f"  kp[\"{k}\"] = {v}" for k, v in kp.items())
        img = node.images.get("front")
        self.stages = []
        for _ in range(3):                     # 파싱·컴파일 실패 재시도
            d = self.vlm.call_json(
                PLAN_PROMPT.format(goal=GOALS[self.task],
                                   text=TASK_TEXT[self.task], kp_desc=desc),
                images=[img] if img else [], max_tokens=1600)
            if not d or not isinstance(d.get("stages"), list):
                continue
            try:
                stages = []
                for s in d["stages"][:4]:
                    stages.append({
                        "name": str(s.get("name", "?"))[:40],
                        "reward": compile_fn(s["reward"], ("traj", "kp")),
                        "done": compile_fn(s["done"], ("kp", "eef", "gripper")),
                    })
                if stages:
                    self.stages = stages
                    self.idx = 0
                    self.r_base = None
                    self.r_last = None
                    print(f"[VLS] 계획 {len(stages)}스테이지: "
                          f"{[s['name'] for s in stages]} "
                          f"(VLM {self.vlm.used}/{self.vlm.budget})", flush=True)
                    return
            except Exception as e:            # noqa: BLE001
                print(f"[VLS] 보상 컴파일 실패, 재시도: {e}", flush=True)
        print("[VLS] 계획 실패 — vanilla 로 진행", flush=True)

    def grasped(self, node) -> float:
        """접촉 검증 파지 — steer_eval 에서 검증된 status.contact 판정."""
        contact = (node.status or {}).get("contact") or {}
        return 1.0 if any(float(v) > 0.3 for v in contact.values()) else 0.0

    def advance(self, node) -> None:
        if not self.stages or self.idx >= len(self.stages):
            return
        st = self.stages[self.idx]
        try:
            eef = list(node.eef or (0, 0, 0))
            eef = [eef[0], eef[1], eef[2] + TCP_DZ]
            if st["done"](self.keypoints(node), eef, self.grasped(node)):
                self.idx += 1
                self.r_base = None          # 새 스테이지 — 유도 강도 기준 리셋
                self.r_last = None
                nxt = (self.stages[self.idx]["name"]
                       if self.idx < len(self.stages) else "완료")
                print(f"[VLS] 스테이지 전환 → {nxt}", flush=True)
        except Exception:                      # noqa: BLE001
            pass

    def guide(self, node, chunks) -> list[float] | None:
        """K 후보에 보상 기울기 정제 → 최고 보상 청크를 고른다."""
        if not self.stages or self.idx >= len(self.stages):
            return None
        st = self.stages[self.idx]
        kp = self.keypoints(node)
        # 적응형 유도 강도 λ (논문 eq.10 의 부호 강건판) — 직전 청크 보상이
        # 스테이지 기준(첫 청크)보다 나쁘면 λ→2 로 세게, 좋아지면 λ→0.5 로
        # 약하게. 기준이 없으면(스테이지 첫 청크) 1.0.
        lam = 1.0
        if self.r_base is not None and self.r_last is not None:
            imp = (self.r_last - self.r_base) / (abs(self.r_base) + 1e-6)
            lam = max(0.25, LAMBDA_MAX / (1.0 + math.exp(imp)))
        best, best_r = None, None
        tcp = torch.tensor([0.0, 0.0, TCP_DZ])
        for ch in chunks:
            t = torch.tensor(ch, dtype=torch.float32)
            t[:, :3] += tcp                    # 플랜지 → TCP
            xyz = t[:, :3].clone().requires_grad_(True)
            base = xyz.detach().clone()
            try:
                for _ in range(REFINE_STEPS):
                    tr = torch.cat([xyz, t[:, 3:4]], dim=1)
                    r = st["reward"](tr, kp)
                    if not torch.is_tensor(r) or r.dim() != 0:
                        raise ValueError("reward 는 스칼라 텐서여야 함")
                    (g,) = torch.autograd.grad(r, xyz)
                    with torch.no_grad():
                        xyz += lam * REFINE_LR * g / (g.norm() + 1e-8)
                        xyz.clamp_(base - REFINE_CLAMP, base + REFINE_CLAMP)
                with torch.no_grad():
                    tr = torch.cat([xyz, t[:, 3:4]], dim=1)
                    rv = float(st["reward"](tr, kp))
                    tr_flange = torch.cat([xyz - tcp, t[:, 3:4]], dim=1)
                cand = tr_flange.detach().numpy().tolist()   # 발행은 플랜지 기준
            except Exception:                  # noqa: BLE001
                # 보상 실행 실패 — 이 후보는 원본 그대로 0점 취급
                cand, rv = ch, float("-inf")
            if best_r is None or rv > best_r:
                best, best_r = cand, rv
        if best_r is not None and best_r != float("-inf"):
            if self.r_base is None:
                self.r_base = best_r        # 스테이지 첫 청크 = 기준 보상
            self.r_last = best_r
        return best


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
        print("[VLS] 토픽이 오지 않습니다.", flush=True)
        return 1

    vlm = VLM(budget=30, log_path=f"{args.out}/VLS_{args.task}_vlm.jsonl")
    steer = Steering(vlm, args.task)
    text = TASK_TEXT[args.task]
    queue: list[list[float]] = []      # 실행 대기 액션 (abs)

    def on_start(node, ep):
        post(f"{SERVER}/reset", {})
        vlm.reset_budget()
        queue.clear()
        time.sleep(0.5)
        steer.plan(node)

    def on_event(node, e):
        if e.get("type") == "trio_spawn" and vlm.used < vlm.budget - 3:
            steer.plan(node)

    def act(node, step):
        steer.advance(node)
        if not queue:
            imgs = {c: base64.b64encode(node.images[c]).decode() for c in CAMS}
            res = post(f"{SERVER}/act_chunk",
                       {"state": node.eef + [node.gripper], "images": imgs,
                        "task": text, "k": K})
            chunks = res["chunks"]            # K × T × 4 (abs)
            best = steer.guide(node, chunks) or chunks[0]
            queue.extend(best[:EXEC_STEPS])
        node.send(queue.pop(0))
        if step % 60 == 59:
            name = (steer.stages[steer.idx]["name"]
                    if steer.stages and steer.idx < len(steer.stages) else "-")
            print(f"[VLS]   step{step + 1}: 스테이지 {steer.idx}({name}) "
                  f"파지 {steer.grasped(node):.0f}", flush=True)

    run_episodes(node, args.task, args.episodes, act,
                 on_episode_start=on_start, on_event=on_event,
                 timeout=args.timeout,
                 out_jsonl=f"{args.out}/VLS_{args.task}.jsonl",
                 tag=f"VLS/{args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
