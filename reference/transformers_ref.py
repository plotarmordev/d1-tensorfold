#!/usr/bin/env python3
"""Reference answers from the model's own transformers code (`system_one`, bf16), in the bench's dump format.

    python3 reference/transformers_ref.py --model /path/to/d1-3B --out results/reference.json [--tokenizer ...]

Writes {"parity": [answers...], "workloads": {name: answers}} for bench/parity.py.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import workloads as W  # noqa: E402


def images_of(body: dict):
    from PIL import Image
    out = []
    for url in body.get("images") or []:
        out.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoModel

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    run = lambda b: model.system_one(b["state"], b["questions"], images=images_of(b))["answers"]  # noqa: E731

    count = None
    if a.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731
    out = {"parity": [run(b) for b in W.parity_set()], "workloads": {}}
    for name, spec in W.build(count).items():
        answers = [run(b) for b in spec["requests"]]
        out["workloads"][name] = answers if spec["concurrent"] else answers[0]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
