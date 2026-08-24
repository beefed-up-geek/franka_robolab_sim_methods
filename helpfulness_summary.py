#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""results/raw/*.jsonl → 방법 × 태스크 요약표 (콘솔).

fill_xlsx.py 는 엑셀을 채우는 것이 목적이라 셀당 한 줄만 찍는다. 실험을
돌리는 동안에는 **표 전체를 한눈에** 보고 목표선(Help ≥ 0.8)을 넘었는지,
표본이 몇 개인지, 에피소드가 얼마나 걸리는지를 같이 봐야 한다 — 특히
평균 소요는 중요하다. task3 실패가 전부 타임아웃이던 것을 이 열로 알아챘고
(평균 LCo 116초 · SCo 75초 · VLSo 49초 = 실패 수와 같은 순서), 문제가
성능이 아니라 속도임을 그때 확인했다.

  python3 helpfulness_summary.py            # 전체
  python3 helpfulness_summary.py --min 0.8  # 목표선 표시 (기본 0.8)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw"

# 표시 순서 — baseline 바로 아래에 그 방법의 +Ours 를 둔다 (대비가 보이게).
METHODS = [("VLA", "VLA"),
           ("LC", "LC"), ("LCo", "LC+Ours"),
           ("SC", "SC"), ("SCo", "SC+Ours"),
           ("VLS", "VLS"), ("VLSo", "VLS+Ours")]
TASKS = ("task1", "task2", "task3")


def load(method: str, task: str) -> list[dict]:
    p = RAW / f"{method}_{task}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def cell(eps: list[dict]) -> tuple[float, float, float, int, float] | None:
    n = len(eps)
    if not n:
        return None
    sr = sum(e["succ"] for e in eps) / n
    safe = sum(e["safe"] for e in eps) / n
    # Help 는 SR·Safe 의 곱이 아니라 **교집합** 이다. 따로 세지 않으면
    # "위험하게 성공" 과 "안전하게 아무것도 안 함" 이 둘 다 좋아 보인다.
    help_ = sum(e["succ"] and e["safe"] for e in eps) / n
    dur = sum(e.get("dur", 0.0) for e in eps) / n
    return sr, safe, help_, n, dur


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min", type=float, default=0.8,
                    help="Help 목표선 — 넘긴 셀에 ✓ 를 붙인다 (기본 0.8)")
    args = ap.parse_args()

    head = f"{'':10s}" + "".join(f"{t:>28s}" for t in TASKS)
    sub = f"{'':10s}" + "".join(f"{'SR':>8s}{'Safe':>8s}{'Help':>8s}{'':4s}"
                                for _ in TASKS)
    print("SR(성공) / Safe(무위반) / Help(성공∧안전) · ✓ = Help ≥ "
          f"{args.min:.2f}")
    print(head)
    print(sub)
    for key, label in METHODS:
        row = f"{label:10s}"
        for t in TASKS:
            c = cell(load(key, t))
            if c is None:
                row += f"{'-':>8s}{'-':>8s}{'-':>8s}{'':4s}"
                continue
            sr, safe, help_, _n, _d = c
            mark = " ✓" if help_ >= args.min else "  "
            row += f"{sr:8.2f}{safe:8.2f}{help_:8.2f}{mark:4s}"
        print(row)

    print()
    print(f"{'':10s}{'표본 수 n · 평균 소요 [s]':>28s}")
    for key, label in METHODS:
        parts = []
        for t in TASKS:
            c = cell(load(key, t))
            parts.append("      -     " if c is None
                         else f"  n={c[3]:<3d} {c[4]:5.0f}s")
        print(f"{label:10s}" + "".join(f"{p:>28s}" for p in parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
