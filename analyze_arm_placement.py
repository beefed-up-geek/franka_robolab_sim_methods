#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""task2 — 작업자 팔 배치와 안전 위반의 상관을 본다 (러너 로그 파싱).

task2 의 Safe 는 에피소드마다 크게 흔들리는데, 그 흔들림이 방법의 성질이
아니라 **팔이 어디에 놓였는지** 에서 온다. 팔의 y 는 매 에피소드
RAND_Y=(-0.30, 0.10) 에서 뽑히고, 그 값이 로봇의 홈 자세(y=0)와 겹치는
구간(|y| < ARM_HALF=0.075, 약 37%)에서는 팔이 **정지한 로봇 쪽으로 쓸고
들어와** 첫 제어 스텝에 접촉이 기록된다. 방법이 손쓸 수 없는 구간이다.

이 도구가 그 사실을 처음 드러냈다. 팔이 단자 쪽(y < -0.12)이면 위반 0/5,
홈 쪽(y ≥ -0.12)이면 3/5 — 같은 코드에서 표본만 다르면 Safe 가 0.9 와 0.4
사이를 오간다. 그래서 task2 결과는 반드시 배치 분포와 함께 읽어야 한다.
(그 뒤 시작 전 비켜서기 + 부호 있는 상자 SDF 로 이 구간도 막았다.)

사용:
  python3 analyze_arm_placement.py /tmp/chain.log [--method SCo] [--split -0.12]

러너가 스텝마다 찍는 진단 줄에서 팔 좌표를 읽으므로, 그 진단이 켜진
로그여야 한다 (예: "팔[0.71,-0.26,0.30]").
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ARM_RE = re.compile(r"\[0\.\d+,\s*(-?\d+\.\d+),")      # 팔[x, y, z]
EP_RE = re.compile(r"\[(\w+)/task2\] ep\d+: .*?(성공|실패) · safe (통과|위반)")


def episodes(path: Path, method: str | None):
    """(팔 y, 성공, 안전) 목록 — 에피소드 순서대로."""
    arm_y, out = None, []
    for line in path.read_text(errors="ignore").splitlines():
        m = ARM_RE.search(line)
        if m:
            arm_y = float(m.group(1))
        e = EP_RE.search(line)
        if e and (method is None or e.group(1) == method):
            out.append((arm_y, e.group(2) == "성공", e.group(3) == "통과"))
            arm_y = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="러너 로그 (예: /tmp/chain.log)")
    ap.add_argument("--method", default=None, help="한 방법만 (SCo/LCo/VLSo…)")
    ap.add_argument("--split", type=float, default=-0.12,
                    help="팔 y 를 어려움/쉬움으로 가르는 경계 (기본 -0.12)")
    args = ap.parse_args()

    eps = episodes(args.log, args.method)
    if not eps:
        print("에피소드를 찾지 못했다 — 진단이 켜진 로그인지 확인할 것")
        return 1

    print(f"{'팔 y':>8s}  {'성공':>4s}  {'안전':>4s}")
    for y, succ, safe in eps:
        ys = "  ?  " if y is None else f"{y:+.2f}"
        print(f"{ys:>8s}  {'O' if succ else 'X':>4s}  "
              f"{'O' if safe else '위반':>4s}")

    known = [(y, s, sf) for y, s, sf in eps if y is not None]
    near = [sf for y, _s, sf in known if y >= args.split]   # 홈·커넥터 쪽
    far = [sf for y, _s, sf in known if y < args.split]     # 단자 쪽
    print()
    print(f"팔이 홈·커넥터 쪽 (y ≥ {args.split:+.2f}): "
          f"위반 {sum(1 for x in near if not x)}/{len(near)}")
    print(f"팔이 단자 쪽      (y <  {args.split:+.2f}): "
          f"위반 {sum(1 for x in far if not x)}/{len(far)}")
    print("\n두 줄의 차이가 크면, 그 측정의 Safe 는 방법이 아니라 "
          "배치 분포를 보고 있는 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
