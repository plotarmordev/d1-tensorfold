"""Qualify a served d1 checkpoint: the same decision alone, again, and beside other requests must agree bit for
bit, and every answer must generate no tokens.

Runs the family's own engine and server in this process, so it needs the checkpoint and one CUDA GPU. The request
set is fixed and small; every row (and every failure) is kept in the output JSON.

    python3 bench/qualify_recipe.py --model /path/to/d1-3B --output d1-qualification.json --label bf16
"""

import argparse
import base64
import hashlib
import http.client
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from tensorfold.cuda.server import Server, make_handler
from tensorfold.families.lfm2 import cuda_engine
from tensorfold.families.lfm2.cuda.app import DecisionApp

# a noul, a score and a choice question over a text state, two questions over one state, and one over a picture
REQUESTS = [
    {"name": "refund", "pictures": 0,
     "state": "A customer writes: my order never arrived and I want my money back.",
     "questions": {"refund": {"type": "noul", "instructions": "Is this a refund request?"}}},
    {"name": "urgency", "pictures": 0,
     "state": {"ticket": 4211, "body": "The export button does nothing since the upgrade."},
     "questions": {"urgency": {"type": "score", "instructions": "How urgent is it?",
                               "criteria": ["no rush", "soon", "today"]}}},
    {"name": "team", "pictures": 0,
     "state": "The export button does nothing since the upgrade.",
     "questions": {"team": {"type": "choice", "instructions": "Which team should take this?",
                            "criteria": {"billing": "money", "platform": "the web app", "other": "none"}}}},
    {"name": "two questions", "pictures": 0,
     "state": "Please stop calling me at work, or I will cancel everything.",
     "questions": {"sentiment": {"type": "choice", "instructions": "What is the tone?",
                                 "criteria": {"angry": "upset", "calm": "settled"}},
                   "escalate": {"type": "noul", "instructions": "Does a supervisor need to see it?",
                                "criteria": {"true": "the customer is upset", "false": "nothing is wrong"}}}},
    {"name": "picture", "pictures": 1,
     "state": None,
     "questions": {"colour": {"type": "choice", "instructions": "What colour is the picture?",
                              "criteria": {"red": "mostly red", "blue": "mostly blue"}}}},
]


def picture_url() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 30, 200)).save(buffer, format="PNG")     # blue, for the picture row
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def body(request: dict) -> dict:
    """The ``/v1/systemone`` request for a row: a picture row carries a data URL and no state."""

    out = {"state": None if request["pictures"] else request["state"], "questions": request["questions"]}
    if request["pictures"]:
        out["images"] = [picture_url()]
    return out


def post(port: int, payload: dict, timeout: float = 600) -> tuple[dict, float]:
    """One ``/v1/systemone`` request, its answer and how long it took."""

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    start = time.perf_counter()
    try:
        connection.request("POST", "/v1/systemone", json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        elapsed = time.perf_counter() - start
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {raw[:400]!r}")
        out = json.loads(raw)
        if "answers" not in out or "usage" not in out:
            raise RuntimeError(f"unexpected reply: {out}")
        return out, elapsed
    finally:
        connection.close()


def digest(out: dict) -> str:
    """A row's answers, so two runs can be compared and reported in one line."""

    return hashlib.sha256(json.dumps(out["answers"], sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=3, help="alone runs of a row, compared with its first")
    args = parser.parse_args()

    engine = cuda_engine(args.model, context=args.context, context_explicit=args.context is not None)
    has_vision = getattr(engine, "vision", None) is not None
    rows, payloads = [], []
    for request in REQUESTS:
        row = {"name": request["name"], "alone": [], "together": None}
        if request["pictures"] and not has_vision:
            row["skipped"] = "this checkpoint has no vision tower"
        else:
            payloads.append((row, body(request)))
        rows.append(row)

    app = DecisionApp(engine, args.model, "d1-qualification", max_tokens=1, context_window=engine.context_window)
    graphs = getattr(engine, "graphs", None)

    def document() -> dict:
        """The run so far, with the graph cache's own counts, so a replay can be told from an eager forward."""

        return {"label": args.label, "rows": rows, "failures": failures,
                "graphs": graphs.stats() if graphs is not None else None}
    server = Server(("127.0.0.1", 0), make_handler(app))
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    port, failures = server.server_port, []
    try:
        for row, payload in payloads:
            try:
                torch.cuda.reset_peak_memory_stats()
                first, elapsed = post(port, payload)
                row["alone"].append({"sha256": digest(first), "usage": first["usage"], "s": elapsed,
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
                for _ in range(max(0, args.repeat - 1)):
                    again, elapsed = post(port, payload)
                    row["alone"].append({"sha256": digest(again), "usage": again["usage"], "s": elapsed})
                    if again != first:
                        failures.append(f"{row['name']}: a repeated request read different bits")
                if first["usage"]["output_tokens"] != 0:
                    failures.append(f"{row['name']}: a decision generated {first['usage']['output_tokens']} tokens")
            except Exception as exc:            # noqa: BLE001 - a failed qualification row is kept as a failure
                row["error"] = f"{type(exc).__name__}: {exc}"
                failures.append(f"{row['name']}: {row['error']}")
            args.output.write_text(json.dumps(document(), indent=2))
        if len(payloads) > 1:
            with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
                answers = list(pool.map(lambda item: post(port, item[1]), payloads))
            for (row, _), (answer, elapsed) in zip(payloads, answers):
                row["together"] = {"sha256": digest(answer), "usage": answer["usage"], "s": elapsed}
                if digest(answer) != row["alone"][0]["sha256"]:
                    failures.append(f"{row['name']}: beside other requests it read different bits")
        for row in rows:
            if row["alone"]:
                timings = sorted(item["s"] for item in row["alone"])
                print(json.dumps({"row": row["name"], "median_s": timings[len(timings) // 2],
                                  "sha256": row["alone"][0]["sha256"],
                                  "input_tokens": row["alone"][0]["usage"]["input_tokens"],
                                  "output_tokens": row["alone"][0]["usage"]["output_tokens"],
                                  "peak_allocated_bytes": row["alone"][0]["peak_allocated_bytes"]}), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)
    args.output.write_text(json.dumps(document(), indent=2))
    if failures:
        raise SystemExit("; ".join(failures))
    print(f"[tensorfold] {len(payloads)} decisions agree with themselves, alone and together ({args.label}); "
          f"graph cache {document()['graphs']}", flush=True)


if __name__ == "__main__":
    main()
