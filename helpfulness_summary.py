import json, os
R = "/home/gty/franka_robolab_sim_methods/results/raw"
M = ("VLA", "LC", "SC", "VLS"); T = ("task1", "task2", "task3")
LOAD = lambda p: [json.loads(l) for l in open(p)] if os.path.exists(p) else []

print("N=50 · SR(성공) / Safe(무위반) / Help(성공∧안전)")
print(f"{'':6s}" + "".join(f"{t:>27s}" for t in T))
print(f"{'':6s}" + "".join(f"{'SR':>9s}{'Safe':>9s}{'Help':>9s}" for _ in T))
for m in M:
    row = f"{m:6s}"
    for t in T:
        e = LOAD(f"{R}/{m}_{t}.jsonl"); n = len(e)
        if not n:
            row += f"{'-':>9s}" * 3; continue
        sr = sum(x["succ"] for x in e) / n
        sf = sum(x["safe"] for x in e) / n
        hp = sum(x["succ"] and x["safe"] for x in e) / n
        row += f"{sr:9.2f}{sf:9.2f}{hp:9.2f}"
    print(row)

print("\n에피소드 4분면 분해 (N=50, 개수) — 성공×안전")
print(f"{'':6s}{'':8s}" + "".join(f"{t:>26s}" for t in T))
print(f"{'':6s}{'':8s}" + "".join(f"{'✓✓':>6s}{'✓✗':>7s}{'✗✓':>7s}{'✗✗':>6s}" for _ in T))
for m in M:
    row = f"{m:6s}{'':8s}"
    for t in T:
        e = LOAD(f"{R}/{m}_{t}.jsonl")
        q = [sum(1 for x in e if x["succ"] == a and x["safe"] == b)
             for a, b in ((1,1),(1,0),(0,1),(0,0))]
        row += "".join(f"{v:>6d}" if i==0 else f"{v:>7d}" for i, v in enumerate(q[:3])) + f"{q[3]:>6d}"
    print(row)
print("  ✓✓ 성공·안전(Help) | ✓✗ 성공했지만 위반 | ✗✓ 안전했지만 실패 | ✗✗ 둘 다 실패")
