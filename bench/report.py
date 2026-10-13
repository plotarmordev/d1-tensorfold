#!/usr/bin/env python3
"""Historical pre-audit tables from a results directory: each pair at the same weight width.

    python3 bench/report.py results

Latency rows: p50 (p99) in ms over every sample; packed: states/s (p50 of the per-batch rates). Parity rows
compare each engine's answers with results/reference.json (the model's own transformers code, bf16).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity import compare  # noqa: E402
from systemone_bench import stats

PAIRS = [("llamacpp-BF16", "tensorfold-bf16", "BF16 checkpoints"),
         ("llamacpp-Q8_0", "tensorfold-fp8", "8-bit weights"),
         ("llamacpp-Q4_K_M", "tensorfold-int4", "4-bit weights")]
ROWS = [("one", "one question, 39 tokens"), ("three", "three questions on one state"),
        ("long", "one question, 3446 prompt tokens"), ("image", "one question about a 384x288 picture")]


def load(d: Path, label: str):
    bench, parity = d / f"{label}.json", d / f"parity-{label}.json"
    if not bench.exists() or not parity.exists():
        return None
    result = json.loads(bench.read_text())
    for workload, cell in result["workloads"].items():
        rounds = cell.get("round_samples")
        if not rounds or len(rounds) != result["method"]["rounds"]:
            raise ValueError(f"{label}/{workload}: missing raw measurement rounds")
        flat = [sample for samples in rounds for sample in samples]
        if stats(flat) != cell["all"]:
            raise ValueError(f"{label}/{workload}: summary does not match retained samples")
    return result, json.loads(parity.read_text())


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    ref = json.loads((d / "reference.json").read_text())
    print("# Historical pre-audit implementation\n")
    print("These results do not describe the current native-BF16 engine. FP8/INT4 are no longer supported.\n")
    for base, ours, title in PAIRS:
        a, b = load(d, base), load(d, ours)
        if a is None or b is None:
            print(f"### {title}\n\n(missing results for {base if a is None else ours})\n")
            continue
        print(f"### {title}: {base} vs {ours}\n")
        print(f"| workload | {base} | {ours} | speedup |\n|---|---|---|---|")
        for key, what in ROWS:
            x, y = a[0]["workloads"][key]["all"], b[0]["workloads"][key]["all"]
            print(f"| {key}: {what} | {x['p50']:.2f} ({x['p99']:.2f}), {100*x['cv']:.2f}% "
                  f"| {y['p50']:.2f} ({y['p99']:.2f}), {100*y['cv']:.2f}% | {x['p50'] / y['p50']:.2f}x |")
        x, y = a[0]["workloads"]["packed"]["all"], b[0]["workloads"]["packed"]["all"]
        print(f"| packed: 64 states at once, states/s | {x['p50']:.1f}, CV {100*x['cv']:.2f}% "
              f"| {y['p50']:.1f}, CV {100*y['cv']:.2f}% | {y['p50'] / x['p50']:.2f}x |")
        print()
        print(f"| parity vs the model's own code | {base} | {ours} |\n|---|---|---|")
        pa, pb = compare(ref["parity"], a[1]), compare(ref["parity"], b[1])
        qa = compare(ref["workloads"]["packed"], a[0]["answers"]["packed"])
        qb = compare(ref["workloads"]["packed"], b[0]["answers"]["packed"])
        print(f"| top answer agrees ({pa['questions']} questions) | {pa['top_answer_agreement']:.4f} "
              f"| {pb['top_answer_agreement']:.4f} |")
        for k, name in (("p50_abs_prob_diff", "median"), ("p95_abs_prob_diff", "p95"), ("max_abs_prob_diff", "max")):
            print(f"| {name} abs probability difference | {pa[k]:.4f} | {pb[k]:.4f} |")
        print(f"| packed set: top answer agrees ({qa['questions']}) | {qa['top_answer_agreement']:.4f} "
              f"| {qb['top_answer_agreement']:.4f} |")
        print()


if __name__ == "__main__":
    main()
