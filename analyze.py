"""ROC AUC per (model, wording) against the RAGTruth hallucination label, plus thresholds.

AUC of p("yes") as a hallucination score. A model that reads the question gives:
  A_en (all supported?)      -> AUC well below 0.5 (yes = grounded)
  B_en (unsupported claim?)  -> AUC well above 0.5 (yes = hallucinated)
  IRR_fr (unrelated)         -> AUC near 0.5, p near 0
Usage: python analyze.py [results/*.jsonl]   (default: all files in results/)
"""
import glob, json, sys
from collections import defaultdict
from statistics import correlation
from sklearn.metrics import roc_auc_score

y = {r["id"]: r["hallucinated"] for r in json.load(open("data/sample.json"))}
res, errs = defaultdict(dict), defaultdict(int)
for path in sys.argv[1:] or sorted(glob.glob("results/*.jsonl")):
    for l in open(path):
        d = json.loads(l)
        if d["p"] is None: errs[(d["model"], d["wording"])] += 1
        else: res[(d["model"], d["wording"])][d["id"]] = d["p"]

print(f"{'model':20} {'wording':7} {'n':>5} {'err':>4} {'AUC':>6} {'mean p':>7}")
for (m, w), d in sorted(res.items()):
    ids = list(d); ps = [d[i] for i in ids]; ys = [y[i] for i in ids]
    print(f"{m:20} {w:7} {len(ids):5} {errs[(m, w)]:4} {roc_auc_score(ys, ps):6.3f} {sum(ps)/len(ps):7.3f}")

print("\nOpposite wordings agree if corr(p_A, p_B) is close to -1:")
for m in sorted({k[0] for k in res}):
    a, b = res.get((m, "A_en"), {}), res.get((m, "B_en"), {})
    ids = [i for i in a if i in b]
    if len(ids) > 2:
        print(f"  {m:20} {correlation([a[i] for i in ids], [b[i] for i in ids]):+.2f} on {len(ids)} rows")

print("\nThresholds on B_en (p >= t means 'hallucinated'):")
for m in sorted({k[0] for k in res}):
    d = res.get((m, "B_en"), {})
    if not d: continue
    pairs = [(p, y[i]) for i, p in d.items()]; total = sum(t for _, t in pairs)
    line = []
    for t in (0.3, 0.5, 0.7):
        tp = sum(p >= t and h for p, h in pairs); fp = sum(p >= t and not h for p, h in pairs)
        line.append(f"t={t}: caught {tp}/{total}, false alarms {fp}")
    print(f"  {m:20} " + " | ".join(line))
