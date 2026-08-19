#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""results/raw/*.jsonl → industry_results.xlsx 집계.

표의 틀(행: LC/LC+Ours/SC/SC+Ours/VLS/VLS+Ours, 열: 태스크별 SR·Safe)은
그대로 두고 vanilla 행만 채운다. +Ours 행은 우리 방법론 실험의 몫이다.
값은 성공률(0~1, 소수 둘째 자리) — N 은 표 머리의 N=50.
"""
import json
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent      # results/ 의 부모 = 레포 루트
RAW = ROOT / "results" / "raw"
XLSX = ROOT / "results" / "industry_results.xlsx"

ROW = {"LC": 3, "SC": 5, "VLS": 7}                 # 시트 행 (1-기준)
COL = {"task1": ("B", "C"), "task2": ("D", "E"), "task3": ("F", "G")}


def main() -> int:
    wb = openpyxl.load_workbook(XLSX)
    ws = wb.active
    for m, row in ROW.items():
        for t, (c_sr, c_safe) in COL.items():
            p = RAW / f"{m}_{t}.jsonl"
            if not p.exists():
                continue
            eps = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            if not eps:
                continue
            n = len(eps)
            sr = sum(e["succ"] for e in eps) / n
            safe = sum(e["safe"] for e in eps) / n
            ws[f"{c_sr}{row}"] = round(sr, 2)
            ws[f"{c_safe}{row}"] = round(safe, 2)
            print(f"{m:>4}/{t}: SR {sr:.2f} · Safe {safe:.2f} (n={n})")
    wb.save(XLSX)
    print(f"저장: {XLSX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
