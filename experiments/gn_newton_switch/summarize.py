"""Tables from results.jsonl (compare.py): per case and method, the cost of
each variant, and the switch's savings (positive: cheaper) against pure
Gauss-Newton and, for the better of the two switch runs, against the better
of the two full-Hessian runs.

Cost: diffusion, PDE solves (all of them); rosenbrock and mc, Hessian-vector
products, unweighted (gn + full) and, for mc, with a full product weighted
1.6 (its measured time against a Gauss-Newton one). A run that did not
converge is marked with >.

Run:  python summarize.py [results.jsonl]
"""

import json
import os
import sys
from collections import defaultdict
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ("gn", "switch", "switch-p0.5", "full", "full-p1")
MC_WEIGHT = 1.6


def cost(r: dict[str, Any], weight: float = 1.0) -> float:
    if r["family"] == "diffusion":
        return float(r["pde_solves"])
    return r["gn_hv"] + weight * r["full_hv"]


def cell(r: dict[str, Any], weight: float = 1.0) -> str:
    mark = "" if r["reason"] == "converged" else ">"
    return f"{mark}{cost(r, weight):.0f} ({r['iterations']})"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.jsonl")
    runs: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    order: list[tuple[str, str, str]] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["family"], r["case"], r["method"])
            if key not in runs:
                order.append(key)
            runs[key][r["variant"]] = r
    for family in ("rosenbrock", "diffusion", "mc"):
        keys = [k for k in order if k[0] == family and len(runs[k]) == len(VARIANTS)]
        if not keys:
            continue
        unit = "PDE solves" if family == "diffusion" else "Hessian products (gn + full)"
        print(f"\n### {family}: {unit} (iterations)\n")
        extra = family == "mc"
        head = ("| case | method | gn | switch | at | switch p=0.5 | full | full p=1 | "
                "switch vs gn | switch p=0.5 vs gn | better switch vs better full |")
        if extra:
            head += f" switch vs gn, full x{MC_WEIGHT} |"
        print(head)
        print("|" + "---|" * (head.count("|") - 1))
        tally = {"switch < gn": 0, "switch p=0.5 < gn": 0, "better switch < better full": 0}
        for k in keys:
            v = runs[k]
            g, s, s5 = v["gn"], v["switch"], v["switch-p0.5"]
            best_full = min(cost(v["full"]), cost(v["full-p1"]))
            best_switch = min(cost(s), cost(s5))
            tally["switch < gn"] += cost(s) < cost(g)
            tally["switch p=0.5 < gn"] += cost(s5) < cost(g)
            tally["better switch < better full"] += best_switch < best_full
            row = (f"| {k[1]} | {k[2]} | {cell(g)} | {cell(s)} | {s['switch_iteration']} | "
                   f"{cell(s5)} | {cell(v['full'])} | {cell(v['full-p1'])} | "
                   f"{1.0 - cost(s) / cost(g):+.0%} | {1.0 - cost(s5) / cost(g):+.0%} | "
                   f"{1.0 - best_switch / best_full:+.0%} |")
            if extra:
                row += f" {1.0 - cost(s, MC_WEIGHT) / cost(g, MC_WEIGHT):+.0%} |"
            print(row)
        print("\n" + "; ".join(f"{name}: {n} of {len(keys)}" for name, n in tally.items()))


if __name__ == "__main__":
    main()
