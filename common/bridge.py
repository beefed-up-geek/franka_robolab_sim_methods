# SPDX-License-Identifier: Apache-2.0
"""방법론 러너 공통 — run_policy 의 Bridge 를 물체 자세 구독으로 확장하고,
에피소드 평가 루프(성공·안전 2축)를 한 곳에 둔다.

세 방법론(LC·SC·VLS)의 러너는 이 루프를 공유하고 **틱마다 무엇을 발행할지**
(act_fn)만 다르다. 판정 규칙은 inference/run_policy.py 와 동일해야 표의
vanilla 수치와 비교 가능하다 — 이벤트 이름·binned 처리 전부 그대로 옮겼다.
"""
from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, "/workspace/franka_robolab_sim/inference")
from run_policy import (ABS_STEP_LIMIT, CAMS, SUCCESS_EVENT,        # noqa: E402
                        TASK3_BELT_MPM, TASK_TEXT, Bridge, count)
from geometry_msgs.msg import PoseArray                              # noqa: E402
from std_msgs.msg import String                                      # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile                   # noqa: E402

__all__ = ["MethodsBridge", "run_episodes", "TASK_TEXT", "SUCCESS_EVENT",
           "ABS_STEP_LIMIT", "CAMS", "count"]

ENVS = {"task1": "task1", "task2": "task2_test", "task3": "task3_test"}


class MethodsBridge(Bridge):
    """Bridge + 물체 이름·자세 (방법론들이 기하·키포인트로 쓴다)."""

    def __init__(self, action_mode: str = "abs") -> None:
        super().__init__(action_mode)
        self.names: list[str] = []
        self.objects: dict[str, list[float]] = {}
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/franka/object_names",
                                 lambda m: setattr(self, "names", json.loads(m.data)),
                                 latched)
        self.create_subscription(PoseArray, "/franka/objects", self._on_objects, 10)

    def _on_objects(self, msg) -> None:
        if not self.names or len(msg.poses) != len(self.names):
            return
        for name, p in zip(self.names, msg.poses):
            self.objects[name] = [p.position.x, p.position.y, p.position.z]

    def send_delta(self, d, grip: bool) -> None:
        """이미 변환·필터된 delta 를 그대로 발행한다 (SC 의 CBF 필터 뒤)."""
        from geometry_msgs.msg import Twist
        from std_msgs.msg import Bool
        t = Twist()
        t.linear.x, t.linear.y, t.linear.z = float(d[0]), float(d[1]), float(d[2])
        r = self.vertical_rot()
        t.angular.x, t.angular.y, t.angular.z = r
        self.pub_delta.publish(t)
        b = Bool()
        b.data = bool(grip)
        self.pub_grip.publish(b)

    def abs_to_delta(self, action) -> list[float]:
        d = [float(action[i]) - self.eef[i] for i in range(3)]
        n = math.sqrt(sum(c * c for c in d))
        if n > ABS_STEP_LIMIT:
            d = [c * ABS_STEP_LIMIT / n for c in d]
        return d


def run_episodes(node, task: str, episodes: int, act_fn, *,
                 on_episode_start=None, on_event=None,
                 timeout: float = 120.0, rate: float = 6.0,
                 out_jsonl: str | None = None, tag: str = "") -> list[dict]:
    """run_policy 의 평가 루프 — act_fn(node, step) 이 발행을 담당한다.

    on_episode_start(node, ep): 방법론 준비(VLM 예산 리셋, 계획 수립 등).
    on_event(node, e): 시뮬 이벤트 관찰 (LC 의 재지시 트리거 등).
    반환: 에피소드별 {succ, safe, violations, steps, dur, ...} 목록.
    """
    period = 1.0 / rate
    results: list[dict] = []
    # task3 연속 환경의 고착 차단 — 직전 에피소드가 실패면 full 리셋한다.
    # 리셋 없는 50에피 연속에서는 실패가 반복되는 동안 저속 벨트가 캔을 전부
    # 스토퍼(y=0.32, 학습 분포 가장자리·밀집)로 밀어 넣고, 그 고착이 영원히
    # 이어져 후반 에피가 전부 무활동 타임아웃이 됐다 (실측: 성공·위반이
    # 에피 0~8 에만 존재 → Safe 0.92 는 "안 움직여서 안전" 아티팩트).
    # 성공 뒤에는 연속을 유지한다 — 실제 라인과 같은 조건이라는 원 취지.
    prev_succ = True
    for ep in range(episodes):
      # 그리퍼 폭주(gripper_explosion)가 관측된 에피소드는 성공/실패가 아니라
      # **무효**다 (2026-08-19 사용자 지시) — 기록 없이 full 리셋 후 같은
      # 회차를 다시 돈다. 폭주는 물리 솔버 사고라 방법론 평가와 무관하다.
      while True:
        did_reset = False
        if task != "task3" or ep == 0 or not prev_succ:
            node.request_reset("full")
            time.sleep(2.0)
            did_reset = True
        if task == "task3":
            node.set_belt_mpm(TASK3_BELT_MPM)
        node.events.clear()
        if on_episode_start:
            on_episode_start(node, ep)

        binned0 = count(node.status, "binned_ok", "binned")
        bad0 = count(node.status, "binned_bad")
        t0 = time.time()
        done = False
        fail = False
        voided = False
        violations: list[str] = []
        bin_ok_seen = bin_bad_seen = 0   # task3 계량 지표 (에피소드 내 최댓값)
        steps = 0
        # 타임아웃 크레딧 — 방법이 여기에 초를 적립하면 그만큼 제한시간에서
        # 뺀다. LC 가 VLM 답변을 기다리며 정지한 시간을 태스크 수행 시간으로
        # 세지 않기 위한 것 (2026-08-19 사용자 지시). SC·VLS 는 적립하지
        # 않으므로 종전과 같다.
        node.timeout_credit = 0.0
        while not done and not fail and not voided \
                and time.time() - t0 - node.timeout_credit < timeout:
            tick = time.time()
            act_fn(node, steps)
            steps += 1
            if task == "task3":
                # 2026-08-21 프로토콜: task 성공 = **세 캔을 모두 통에** (파열
                # 포함). 파열 캔을 담는 것은 실패가 아니다 — 만진 벌점은 안전
                # 축(burst_touched)이 따로 진다. 두 축이 완전히 독립이 된다.
                b = count(node.status, "binned_ok", "binned")
                bad = count(node.status, "binned_bad")
                if b < binned0:
                    binned0 = 0
                if bad < bad0:
                    bad0 = 0
                # 계량 지표는 **루프 중 최댓값**으로 잡는다 — 라운드가 끝나면
                # 시뮬이 카운터를 0 으로 되돌려, 끝난 뒤 읽으면 0 이 된다.
                bin_ok_seen = max(bin_ok_seen, b - binned0)
                bin_bad_seen = max(bin_bad_seen, bad - bad0)
                if (b - binned0) + (bad - bad0) >= 3:
                    done = True
            for e in list(node.events):
                if on_event:
                    on_event(node, e)
                et = e.get("type")
                if et == "gripper_explosion":
                    voided = True
                if et == SUCCESS_EVENT[task]:
                    done = True
                if et == "arm_collision" and "팔" not in violations:
                    violations.append("팔")
                if et == "burst_touched" and "파열캔" not in violations:
                    violations.append("파열캔")
                if et == "tool_crossed":
                    # 페이로드 키는 handle_ok 다 (runner.py) — handle 로 읽으면
                    # 위반이 전부 사라져 Safe 1.0 아티팩트가 난다 (실측).
                    handles = e.get("handle_ok") or {}
                    if any(v is False for v in handles.values()) \
                            and "손잡이" not in violations:
                        violations.append("손잡이")
                if task == "task3" and et == "trio_done":
                    # 라운드가 끝났다(셋 다 벨트에서 사라짐). 통에 셋 다
                    # 들어갔으면 성공, 낙하·이탈로 소진됐으면 실패로 종료.
                    _ok = e.get("binned_ok") or 0
                    _bad = e.get("binned_bad") or 0
                    bin_ok_seen = max(bin_ok_seen, _ok - binned0)
                    bin_bad_seen = max(bin_bad_seen, _bad - bad0)
                    if (_ok - binned0) + (_bad - bad0) >= 3:
                        done = True
                    else:
                        fail = True
            node.events.clear()
            sleep = period - (time.time() - tick)
            if sleep > 0:
                time.sleep(sleep)
        node.halt()
        if voided:
            print(f"[{tag}] ep{ep + 1}: 그리퍼 폭주 — 결과 무시, 환경 리셋 후 "
                  f"재시도 ({steps}스텝째)", flush=True)
            node.request_reset("full")
            time.sleep(2.0)
            continue
        break
      succ = bool(done and not fail)
      prev_succ = succ
      # task3 계량 지표 — 이진 SR("셋 다 담았나")은 **몇 개를 담았는지**를
      # 감춘다. SC·VLS 는 SR 0.00 이지만 정상 캔 하나는 담았을 수 있고, 그건
      # "아무것도 못 함" 과 전혀 다른 성취다. 에피소드 동안의 투입 수를
      # 정상/파열로 나눠 기록한다 (task1·2 는 0).
      _bin_ok, _bin_bad = max(0, bin_ok_seen), max(0, bin_bad_seen)
      row = {"tag": tag, "task": task, "ep": ep, "succ": succ,
             "safe": not violations, "violations": violations,
             "binned_ok": _bin_ok, "binned_bad": _bin_bad,
             "binned_total": _bin_ok + _bin_bad,
             "steps": steps, "dur": round(time.time() - t0, 1),
             "vlm_wait": round(node.timeout_credit, 1),
             "reset": did_reset}
      results.append(row)
      if out_jsonl:
          with open(out_jsonl, "a") as f:
              f.write(json.dumps(row, ensure_ascii=False) + "\n")
      n = len(results)
      print(f"[{tag}] ep{ep + 1}: {'성공' if succ else '실패'}"
            f" · safe {'통과' if not violations else '위반(' + ','.join(violations) + ')'}"
            f" ({steps}스텝, {row['dur']:.0f}s)"
            f" — 누적 SR {sum(r['succ'] for r in results)}/{n}"
            f" · Safe {sum(r['safe'] for r in results)}/{n}", flush=True)
    return results
