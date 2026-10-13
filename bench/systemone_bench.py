#!/usr/bin/env python3
"""Latency and throughput of a `/v1/systemone` server on the five d1 workloads, plus an answer dump.

    python3 bench/systemone_bench.py --url http://127.0.0.1:8080 --out results/tensorfold-bf16.json \
        [--tokenizer path/to/tokenizer.json] [--n 50] [--warmup 5] [--rounds 3] [--workloads one,three,...]

Method (one request at a time, warm): each workload is sent `--warmup` times unmeasured, then `--n` times
per round for `--rounds` rounds; every sample is kept. `packed` sends its 64 requests concurrently and
reports states/s per burst. Stdlib only, so the same client measures every engine.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workloads as W  # noqa: E402


def post(url: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def stats(xs: list[float]) -> dict:
    m = statistics.mean(xs)
    return {"n": len(xs), "p50": round(statistics.median(xs), 2), "p90": round(pct(xs, 0.90), 2),
            "p99": round(pct(xs, 0.99), 2), "mean": round(m, 2),
            "cv": round(statistics.pstdev(xs) / m, 4) if m else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json, to size the long state exactly")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workloads", default="one,three,long,image,packed")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    count = None
    if a.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731
    wl = W.build(count)
    res = {"label": a.label, "url": a.url, "method": {"n": a.n, "warmup": a.warmup, "rounds": a.rounds},
           "workloads": {}, "answers": {}}
    root = Path(__file__).resolve().parent
    res["provenance"] = {
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workloads_sha256": hashlib.sha256((root / "workloads.py").read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256((root / "assets/cats-384.jpg").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest() if a.tokenizer else None,
    }
    for name in a.workloads.split(","):
        spec = wl[name]
        reqs = spec["requests"]
        if spec["concurrent"]:
            with cf.ThreadPoolExecutor(len(reqs)) as ex:
                for _ in range(a.warmup):
                    list(ex.map(lambda b: post(a.url, b), reqs))
                rounds = []
                for _ in range(a.rounds):
                    rates = []
                    for _ in range(max(3, a.n // 10)):
                        t0 = time.perf_counter()
                        outs = list(ex.map(lambda b: post(a.url, b), reqs))
                        rates.append(len(reqs) / (time.perf_counter() - t0))
                    rounds.append(rates)
            flat = [x for r in rounds for x in r]
            res["workloads"][name] = {"unit": "states/s", "all": stats(flat), "round_samples": rounds,
                                      "round_medians": [round(statistics.median(r), 1) for r in rounds]}
            res["answers"][name] = [o.get("answers") for o in outs]
        else:
            body = reqs[0]
            for _ in range(a.warmup):
                post(a.url, body)
            rounds = []
            for _ in range(a.rounds):
                ts = []
                for _ in range(a.n):
                    t0 = time.perf_counter()
                    out = post(a.url, body)
                    ts.append((time.perf_counter() - t0) * 1000)
                rounds.append(ts)
            flat = [x for r in rounds for x in r]
            res["workloads"][name] = {"unit": "ms", "input_tokens": out.get("usage", {}).get("input_tokens"),
                                      "all": stats(flat), "round_samples": rounds,
                                      "round_medians": [round(statistics.median(r), 2) for r in rounds]}
            res["answers"][name] = out.get("answers")
        summary = {k: v for k, v in res["workloads"][name].items() if k != "round_samples"}
        print(f"{name:7s} {json.dumps(summary)}", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
