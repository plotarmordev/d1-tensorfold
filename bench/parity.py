#!/usr/bin/env python3
"""Answers of one engine against a reference, on the parity set (and any workload answers both dumps hold).

    python3 bench/parity.py --url http://127.0.0.1:8131 --dump results/parity-llamacpp.json
    python3 bench/parity.py --compare results/parity-reference.json results/parity-llamacpp.json

Per question: the largest absolute probability difference over its options (P(yes) for a noul), and
whether the top answer agrees.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workloads as W  # noqa: E402
from systemone_bench import post  # noqa: E402


def probs(ans: dict) -> tuple[dict[str, float], str]:
    t = ans["type"]
    if t == "noul":
        p = float(ans["noul"])
        pr = {"yes": p, "no": 1 - p}
        top = "yes" if p >= 0.5 else "no"
    else:
        pr = {k: float(v) for k, v in ans["probabilities"].items()}
        top = ans["choice"] if t == "choice" else max(pr, key=pr.get)
    if not pr or any(not math.isfinite(v) or not 0 <= v <= 1 for v in pr.values()):
        raise ValueError("answer probabilities must be finite and within [0, 1]")
    if not math.isclose(sum(pr.values()), 1.0, abs_tol=1e-4) or str(top) not in pr:
        raise ValueError("answer probabilities or selected option are inconsistent")
    return pr, str(top)


def compare(ref: list[dict], got: list[dict]) -> dict:
    if not ref or len(ref) != len(got):
        raise ValueError("reference and candidate must contain the same nonzero request count")
    diffs, agree, n = [], 0, 0
    for r, g in zip(ref, got):
        if not r or r.keys() != g.keys():
            raise ValueError("reference and candidate question IDs must match")
        for qid, ra in r.items():
            if ra["type"] != g[qid]["type"]:
                raise ValueError("answer types must match")
            pr, tr = probs(ra)
            pg, tg = probs(g[qid])
            if pr.keys() != pg.keys():
                raise ValueError("answer option IDs must match")
            diffs.append(max(abs(pr[key] - pg[key]) for key in pr))
            agree += tr == tg
            n += 1
    diffs.sort()
    return {"questions": n, "top_answer_agreement": round(agree / n, 4), "max_abs_prob_diff": round(diffs[-1], 4),
            "p50_abs_prob_diff": round(diffs[len(diffs) // 2], 5), "p95_abs_prob_diff": round(diffs[int(0.95 * len(diffs))], 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--dump")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.url:
        answers = [post(a.url, body)["answers"] for body in W.parity_set()]
        Path(a.dump).write_text(json.dumps(answers, indent=1))
        print(f"wrote {a.dump} ({len(answers)} requests)")
    if a.compare:
        ref, got = (json.loads(Path(p).read_text()) for p in a.compare)
        ref = ref["parity"] if isinstance(ref, dict) else ref        # results/reference.json holds more than the set
        print(json.dumps(compare(ref, got), indent=1))


if __name__ == "__main__":
    main()
