"""Aggregate outputs/*/*/results/*.json into Table-1-style GRA tables.

    python main.py stage=collect_results

Writes outputs/results.md and outputs/results.csv and, with W&B enabled, logs a run
`results-summary` (group `results`) containing:
  * results/all                         : every evaluated run (one row per result JSON)
  * results/<weak>-to-<strong>/<loss>   : dataset x method GRA table with an Avg. row (as in Table 1)
  * results/gra/<dataset>/<pair>/<loss>/<method> : scalar summaries, for W&B charts and reports
  * an artifact `results-summary` with the CSV and Markdown files.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

from utils.logging import finish_wandb, log_artifact, log_table, setup_wandb, wandb_log

METHODS = ["human", "ws_po", "cw_po", "ms_po"]


def _tables(rows: list[dict]) -> dict:
    """{(pair, loss): (columns, [[dataset, gra...], ..., ["Avg.", ...]])}"""
    grouped = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        grouped[(f"{r['weak']} -> {r['strong']}", r["loss"])][r["dataset"]][r["method"] + r.get("run_suffix", "")] = r["gra"]

    tables = {}
    for key, by_ds in sorted(grouped.items()):
        cols = METHODS + sorted({m for d in by_ds.values() for m in d} - set(METHODS))
        body, sums = [], defaultdict(list)
        for ds, vals in sorted(by_ds.items()):
            body.append([ds] + [vals.get(m) for m in cols])
            for m in cols:
                if vals.get(m) is not None:
                    sums[m].append(vals[m])
        body.append(["Avg."] + [sum(sums[m]) / len(sums[m]) if sums[m] else None for m in cols])
        tables[key] = (["dataset"] + cols, body)
    return tables


def _markdown(tables: dict) -> str:
    lines = []
    for (pair, loss), (cols, body) in tables.items():
        lines += [f"### {pair} | {loss.upper()}", "", "| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
        for row in body:
            lines.append("| " + " | ".join([row[0]] + [f"{v:.1f}" if v is not None else "-" for v in row[1:]]) + " |")
        lines.append("")
    return "\n".join(lines)


def run(cfg) -> None:
    outputs = cfg.paths.outputs_dir
    md_file, csv_file = os.path.join(outputs, "results.md"), os.path.join(outputs, "results.csv")
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(outputs, "*", "*", "results", "*.json")))]
    if not rows:
        print("No results found.")
        return

    os.makedirs(outputs, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tables = _tables(rows)
    md = _markdown(tables)
    with open(md_file, "w") as f:
        f.write(md)
    print(md)

    if setup_wandb(cfg, "results-summary", "results", group="results") is None:
        return
    log_table("results/all", fields, [[r.get(k) for k in fields] for r in rows])
    for (pair, loss), (cols, body) in tables.items():
        log_table(f"results/{pair.replace(' -> ', '-to-')}/{loss}", cols, body)
    wandb_log({
        f"results/gra/{r['dataset']}/{r['weak']}-to-{r['strong']}/{r['loss']}/{r['method']}{r.get('run_suffix', '')}": r["gra"]
        for r in rows
    })
    log_artifact("results-summary", "results", [csv_file, md_file], metadata={"n_results": len(rows)})
    finish_wandb()
