import json, os
from collections import Counter
G = "/home/gty/franka_robolab_sim_methods/results/graded"
print(f"{'방법':6s} {'n':>2s} {'SR':>5s} {'Safe':>5s} | {'담은캔/3':>8s} {'정상':>5s} {'파열':>5s} | 담은수 분포")
for m in ("VLA", "LC", "SC", "VLS"):
    p = f"{G}/{m}_task3.jsonl"
    if not os.path.exists(p): continue
    e = [json.loads(l) for l in open(p)]
    n = len(e)
    tot = sum(x["binned_total"] for x in e) / n
    ok = sum(x["binned_ok"] for x in e) / n
    bad = sum(x["binned_bad"] for x in e) / n
    dist = Counter(x["binned_total"] for x in e)
    print(f"{m:6s} {n:2d} {sum(x['succ'] for x in e)/n:5.2f} {sum(x['safe'] for x in e)/n:5.2f} | "
          f"{tot:8.2f} {ok:5.2f} {bad:5.2f} | {dict(sorted(dist.items()))}")
print()
print("정상 캔(1개) 회수율 — '위험은 피하고 멀쩡한 건 챙겼나':")
for m in ("VLA", "LC", "SC", "VLS"):
    p = f"{G}/{m}_task3.jsonl"
    if not os.path.exists(p): continue
    e = [json.loads(l) for l in open(p)]
    got = sum(1 for x in e if x["binned_ok"] >= 1)
    print(f"  {m:5s} {got}/{len(e)} ({got/len(e):.0%})")
