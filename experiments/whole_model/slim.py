# Copies results/*.json to slim/ without bulky raw timing arrays and exclusivity observations.
# 2026-11-22: a pair of per-step arrays (reference_ms, candidate_ms) is reduced to the median and the
# quartiles of the per-step ratio reference/candidate before the arrays are dropped ("paired_ratio";
# receipts written before that date do not contain it).
import json, glob, os, statistics
os.makedirs("slim", exist_ok=True)
DROP = ("observations", "arm_order", "a", "b", "samples_us", "reference_ms", "candidate_ms", "samples_ms")
def slim(o):
    if isinstance(o, dict):
        out = {k: ("[%d items]" % len(v) if k in DROP and isinstance(v, list) else slim(v)) for k, v in o.items()}
        a, b = o.get("reference_ms"), o.get("candidate_ms")
        if isinstance(a, list) and isinstance(b, list) and a and len(a) == len(b) and "paired_ratio" not in o:
            pr = sorted(x / y for x, y in zip(a, b))
            out["paired_ratio"] = {"median": statistics.median(pr), "q25": pr[len(pr) // 4], "q75": pr[(3 * len(pr)) // 4],
                                   "n": len(pr), "computed_by": "slim.py"}
        return out
    if isinstance(o, list):
        return [slim(x) for x in o]
    return o
for f in sorted(glob.glob("results/*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    json.dump(slim(d), open(os.path.join("slim", os.path.basename(f)), "w"))
