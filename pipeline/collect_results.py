"""Aggregate outputs/*/*/results/*.json into a Table-1-style GRA table (markdown + CSV).

    python main.py stage=collect_results 
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

METHODS = ["human", "ws_po", "cw_po", "ms_po"]


def run(cfg) -> None:
    outputs = cfg.paths.outputs_dir
    out = os.path.join(outputs, "results.md")
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(outputs, "*", "*", "results", "*.json")))]
    if not rows:
        print("No results found.")
        return

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out.replace(".md", ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # (weak -> strong, loss) -> dataset -> method(+suffix) -> GRA
    table = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        table[(f"{r['weak']} -> {r['strong']}", r["loss"])][r["dataset"]][r["method"] + r.get("run_suffix", "")] = r["gra"]

    lines = []
    for (pair, loss), by_ds in sorted(table.items()):
        cols = METHODS + sorted({m for d in by_ds.values() for m in d} - set(METHODS))
        lines += [f"### {pair} | {loss.upper()}", "", "| Dataset | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        sums = defaultdict(list)
        for ds, vals in sorted(by_ds.items()):
            cells = []
            for m in cols:
                v = vals.get(m)
                cells.append(f"{v:.1f}" if v is not None else "-")
                if v is not None:
                    sums[m].append(v)
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")
        lines.append("| Avg. | " + " | ".join(f"{sum(sums[m]) / len(sums[m]):.1f}" if sums[m] else "-" for m in cols) + " |")
        lines.append("")
    open(out, "w").write("\n".join(lines))
    print("\n".join(lines))

