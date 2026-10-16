#!/usr/bin/env python3
"""Calibration of a `/v1/systemone` server on the Fast Decisions dev split: accuracy and top-label ECE.

    python3 bench/calibration.py fetch                         # dataset -> data/cache (git-ignored)
    python3 bench/calibration.py run [--url URL] [--container NAME | --provenance FILE]
                                                               # -> results/calibration/answers.jsonl, manifest.json
    python3 bench/calibration.py score                         # -> results/calibration/summary.json
    python3 bench/calibration_chart.py                         # -> results/calibration/*.svg

`fetch` downloads fastino/fast-decisions at REVISION and checks every file against checksums/fast-decisions.sha256;
`run` checks them again, then sends one request per row to URL (default http://127.0.0.1:8080), one at a time, in
file then row order. Every single-label head becomes a `choice` question over the head's labels, verbatim and in
order; multi-label heads are skipped, since a choice has one answer. A run never overwrites; use a fresh `--out`.

Before any question, `run` records what serves URL. By default `docker inspect` of the recipe's container
(`--container`, default d1-tensorfold) gives the image id, the command with its `--context`, and the
TENSORFOLD_* / TORCH_* environment (precision, token budget, CUDA graphs, TF32 override). That only holds when URL
is a loopback address (127.0.0.0/8, ::1, localhost), the Docker daemon is local (a unix socket, not a remote
DOCKER_HOST or context), and the running container publishes URL's host and port, with 0.0.0.0 and :: matching
any loopback host. The checkpoint mounted at /model is hashed against checksums/d1-3B.sha256, reading all of it,
and its revision is read from the metadata `hf download` leaves there. The container must be the same one when the
run ends. Any other server needs `--provenance FILE`: a JSON object with the same `image`, `model` and `settings`
fields, recorded as supplied by the operator. Missing evidence stops the run before any question is sent.

`score` reads the top option's probability as the confidence. ECE is sum_b (n_b / N) |accuracy_b - confidence_b|
over 10 equal-width bins (lo, hi], the first also holding 0. Each figure has a 95% percentile interval from 1000
bootstrap resamples of rows, not questions, because a row's heads share one input. The seed is fixed, so the same
answers give the same summary. The dev split is not the dataset's benchmark: it describes calibration only.
"""
from __future__ import annotations

import argparse
import bisect
import datetime
import hashlib
import ipaddress
import json
import math
import random
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from systemone_bench import post

DATASET = "fastino/fast-decisions"
REVISION = "35afb79a0f611a1db44380e855ac2a821b0a4da1"
ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "checksums/fast-decisions.sha256"
MODEL_PINS = ROOT / "checksums/d1-3B.sha256"
CACHE = ROOT / "data/cache"
OUT = ROOT / "results/calibration"
CONTAINER = "d1-tensorfold"
# the environment start.sh gives the container; with --context, the settings that change what is served
SETTINGS = {"precision": "TENSORFOLD_LFM2_PRECISION", "token_budget": "TENSORFOLD_LFM2_TOKEN_BUDGET",
            "cuda_graphs": "TENSORFOLD_LFM2_GRAPHS", "tf32_override": "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"}
REQUIRED = (("image", "id"), ("model", "revision"), ("model", "files_sha256"),
            *(("settings", key) for key in (*SETTINGS, "context")))
INSTRUCTIONS = "Choose the {task} label."
BINS = 10
SAMPLES = 1000
SEED = 0


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pins(path: Path = PINS) -> dict[str, str]:
    return {name: digest for digest, name in (line.split() for line in path.read_text().splitlines() if line)}


def verify(directory: Path, pinned: dict[str, str]) -> None:
    for name, digest in pinned.items():
        got = sha256(directory / name)
        if got != digest:
            raise ValueError(f"{directory / name}: sha256 {got}, expected {digest}")


def fetch(directory: Path, pinned: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in pinned.items():
        path = directory / name
        if path.exists():
            continue
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
        raw = urllib.request.urlopen(url, timeout=60).read()
        got = hashlib.sha256(raw).hexdigest()
        if got != digest:
            raise SystemExit(f"{url}: sha256 {got}, expected {digest}")
        part = path.with_name(name + ".part")
        part.write_bytes(raw)
        part.replace(path)
    verify(directory, pinned)


def rows(directory: Path, names: list[str]) -> list[dict]:
    out = []
    for name in names:
        for i, line in enumerate((directory / name).read_text(encoding="utf-8").splitlines()):
            record = json.loads(line)
            heads = [h for h in record["output"]["classifications"] if not h["multi_label"]]
            for h in heads:
                if len(h["true_label"]) != 1 or h["true_label"][0] not in h["labels"]:
                    raise ValueError(f"{name}:{i} {h['task']}: a single-label head needs one gold label of its own")
            if len({h["task"] for h in heads}) != len(heads):
                raise ValueError(f"{name}:{i}: head names repeat")
            if heads:
                out.append({"row": f"{name}:{i}", "input": record["input"], "heads": heads})
    return out


def request(row: dict) -> dict:
    return {"state": row["input"], "questions": {
        h["task"]: {"type": "choice", "instructions": INSTRUCTIONS.format(task=h["task"].replace("_", " ")),
                    "criteria": {label: None for label in h["labels"]}} for h in row["heads"]}}


def get(url: str, path: str) -> object:
    with urllib.request.urlopen(url + path, timeout=30) as r:
        return json.loads(r.read())


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def docker(*args: str) -> dict:
    return json.loads(subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout)[0]


def loopback(host: str | None) -> set:
    """The addresses a URL's host names on this machine; empty unless it is a loopback address."""
    if host == "localhost":
        return {ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")}
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return set()
    return {address} if address.is_loopback else set()


def publishes(binding: dict, addresses: set, port: str) -> bool:
    # an unspecified HostIp ("", 0.0.0.0, ::) publishes on every address, loopback included
    host = binding["HostIp"]
    return binding["HostPort"] == port and (not host or ipaddress.ip_address(host).is_unspecified
                                            or ipaddress.ip_address(host) in addresses)


def container(name: str, url: str) -> tuple[dict, Path]:
    """What a running recipe container serves, and the checkpoint directory it mounts. Only a container on this
    machine's Docker daemon, published on URL's loopback host and port, can vouch for URL."""
    parts = urllib.parse.urlsplit(url)
    addresses = loopback(parts.hostname)
    if not addresses:
        raise SystemExit(f"{url} is not a loopback address, so no local container can vouch for it; "
                         "pass --provenance FILE")
    try:
        daemon = docker("context", "inspect")["Endpoints"]["docker"]["Host"]
        if not daemon.startswith("unix://"):
            raise SystemExit(f"the Docker daemon at {daemon} is not local; pass --provenance FILE")
        c = docker("container", "inspect", name)
        image = docker("image", "inspect", c["Image"])
    except (OSError, subprocess.CalledProcessError) as e:
        why = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else e
        raise SystemExit(f"cannot inspect container {name} ({why}); pass --provenance FILE for a server without one")
    if not c["State"]["Running"]:
        raise SystemExit(f"container {name} is not running")
    bindings = (c["HostConfig"]["PortBindings"] or {}).get("8080/tcp") or []
    if not any(publishes(b, addresses, str(parts.port or 80)) for b in bindings):
        raise SystemExit(f"container {name} does not publish {url}")
    mount = next((m["Source"] for m in c["Mounts"] if m["Destination"] == "/model"), None)
    if mount is None:
        raise SystemExit(f"container {name} mounts no checkpoint at /model")
    env = dict(e.split("=", 1) for e in c["Config"]["Env"])
    args = c["Args"]
    settings = {key: env.get(var) for key, var in SETTINGS.items()}
    settings["context"] = args[args.index("--context") + 1] if "--context" in args else None
    return {"source": "docker inspect",
            "container": {"name": name, "id": c["Id"], "started": c["State"]["StartedAt"], "daemon": daemon},
            "image": {"id": image["Id"], "tag": c["Config"]["Image"], "repo_digests": image["RepoDigests"],
                      "recipe_hash": (image["Config"]["Labels"] or {}).get("d1.image-hash")},
            "command": [c["Path"], *args],
            "env": {k: v for k, v in env.items() if k.startswith(("TENSORFOLD_", "TORCH_"))},
            "settings": settings}, Path(mount)


def checkpoint(directory: Path, pinned: dict[str, str]) -> dict:
    verify(directory, pinned)
    revisions = set()
    for name in pinned:
        meta = directory / ".cache/huggingface/download" / f"{name}.metadata"
        revisions.add(meta.read_text().split("\n", 1)[0] if meta.exists() else None)
    if len(revisions) != 1 or None in revisions:
        raise SystemExit(f"{directory}: its files do not record one `hf download` revision")
    return {"dir": str(directory), "revision": revisions.pop(), "files_sha256": pinned}


def serving(url: str, name: str, provenance: Path | None) -> dict:
    if provenance:
        record = {"source": "file", "file": str(provenance), "file_sha256": sha256(provenance),
                  **json.loads(provenance.read_text())}
    else:
        record, mount = container(name, url)
        record["model"] = checkpoint(mount, pins(MODEL_PINS))
    missing = [".".join(k) for k in REQUIRED if (record.get(k[0]) or {}).get(k[1]) is None]
    if missing:
        raise SystemExit(f"serving provenance lacks {', '.join(missing)}")
    return record


def run(url: str, directory: Path, pinned: dict[str, str], out: Path, name: str = CONTAINER,
        provenance: Path | None = None) -> None:
    answers, manifest = out / "answers.jsonl", out / "manifest.json"
    if answers.exists() or manifest.exists():
        raise SystemExit(f"{out} already holds a run; use a fresh --out")
    verify(directory, pinned)
    data = rows(directory, list(pinned))
    started = utc()
    served = serving(url, name, provenance)
    reported = {"url": url, "health": get(url, "/health"), "models": get(url, "/v1/models")}
    out.mkdir(parents=True, exist_ok=True)
    questions = 0
    with answers.open("x", encoding="utf-8") as f:
        for row in data:
            got = post(url, request(row))["answers"]
            for h in row["heads"]:
                a = got[h["task"]]
                f.write(json.dumps({"row": row["row"], "task": h["task"], "labels": h["labels"],
                                    "gold": h["true_label"][0], "choice": a["choice"],
                                    "confidence": a["confidence"], "probabilities": a["probabilities"]},
                                   ensure_ascii=False) + "\n")
                questions += 1
    if not provenance and container(name, url)[0]["container"] != served["container"]:
        raise SystemExit(f"container {name} was replaced or restarted during the run; {answers} is not one run")
    manifest.write_text(json.dumps({
        "dataset": {"repo": DATASET, "revision": REVISION, "split": "dev", "files_sha256": pinned},
        "mapping": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": "the head's labels in order, "
                    "without descriptions", "skipped": "multi-label heads"},
        "serving": served,
        "server_reported": reported,
        "recipe": {"commit": git("rev-parse", "HEAD"),
                   "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
        "collector_sha256": sha256(Path(__file__)),
        "requests": len(data), "questions": questions, "answers_sha256": sha256(answers),
        "started_utc": started, "finished_utc": utc(),
    }, indent=1, ensure_ascii=False))
    print(f"wrote {answers} ({len(data)} requests, {questions} questions) and {manifest}")


# ---------------------------------------------------------------------------------------------------- scoring

def edges(bins: int = BINS) -> list[float]:
    return [i / bins for i in range(bins + 1)]


def bin_of(confidence: float, bins: int = BINS) -> int:
    return max(bisect.bisect_left(edges(bins), confidence) - 1, 0)


def tally(items: list[tuple[float, bool]], bins: int = BINS) -> list[list[float]]:
    """Per bin: [count, confidence sum, correct count]."""
    t = [[0, 0.0, 0] for _ in range(bins)]
    for confidence, correct in items:
        b = t[bin_of(confidence, bins)]
        b[0] += 1
        b[1] += confidence
        b[2] += correct
    return t


def accuracy(t: list[list[float]]) -> float:
    return sum(b[2] for b in t) / sum(b[0] for b in t)


def ece(t: list[list[float]]) -> float:
    # (n_b / N) * |correct_b / n_b - confidence_sum_b / n_b| = |correct_b - confidence_sum_b| / N
    return sum(abs(b[2] - b[1]) for b in t) / sum(b[0] for b in t)


def resample(groups: list[list[tuple[float, bool]]], samples: int = SAMPLES, seed: int = SEED,
             bins: int = BINS) -> list[list[list[float]]]:
    """Bootstrap tallies, drawing whole groups with replacement."""
    sparse = [[(i, *b) for i, b in enumerate(tally(g, bins)) if b[0]] for g in groups]
    rng = random.Random(seed)
    out = []
    for _ in range(samples):
        t = [[0, 0.0, 0] for _ in range(bins)]
        for g in rng.choices(sparse, k=len(sparse)):
            for i, n, s, k in g:
                b = t[i]
                b[0] += n
                b[1] += s
                b[2] += k
        out.append(t)
    return out


def interval(xs: list[float], level: float = 0.95) -> list[float]:
    xs = sorted(xs)
    tail = (1 - level) / 2
    return [xs[round(tail * (len(xs) - 1))], xs[round((1 - tail) * (len(xs) - 1))]]


def estimate(stat, t: list[list[float]], boot: list[list[list[float]]]) -> dict:
    return {"value": round(stat(t), 6), "ci95": [round(x, 6) for x in interval([stat(b) for b in boot])]}


def validate(a: dict) -> None:
    p = a["probabilities"]
    if set(p) != set(a["labels"]) or a["gold"] not in p or a["choice"] not in p:
        raise ValueError(f"{a['row']} {a['task']}: answer options differ from the head's labels")
    values = list(p.values())
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or not math.isclose(sum(values), 1, abs_tol=1e-4):
        raise ValueError(f"{a['row']} {a['task']}: probabilities must be finite, within [0, 1] and sum to 1")
    if p[a["choice"]] != max(p.values()) or a["confidence"] != p[a["choice"]]:
        raise ValueError(f"{a['row']} {a['task']}: the choice is not the most probable option")


def score(out: Path) -> dict:
    manifest = json.loads((out / "manifest.json").read_text())
    answers = out / "answers.jsonl"
    if sha256(answers) != manifest["answers_sha256"]:
        raise ValueError(f"{answers} differs from the one its manifest recorded")
    groups: dict[str, list[tuple[float, bool]]] = {}
    for line in answers.read_text(encoding="utf-8").splitlines():
        a = json.loads(line)
        validate(a)
        groups.setdefault(a["row"], []).append((a["confidence"], a["choice"] == a["gold"]))
    items = [x for g in groups.values() for x in g]
    if len(items) != manifest["questions"] or len(groups) != manifest["requests"]:
        raise ValueError("answers do not cover every question of the run")
    t = tally(items)
    boot = resample(list(groups.values()))
    lo = edges()
    return {
        "dataset": {k: manifest["dataset"][k] for k in ("repo", "revision", "split")},
        "model": [m["id"] for m in manifest["server_reported"]["models"]["data"]],
        "questions": len(items), "rows": len(groups),
        "accuracy": estimate(accuracy, t, boot),
        "ece": estimate(ece, t, boot),
        "bins": [{"lo": lo[i], "hi": lo[i + 1], "n": n,
                  "confidence": round(s / n, 6) if n else None, "accuracy": round(k / n, 6) if n else None,
                  "accuracy_minus_confidence": round((k - s) / n, 6) if n else None}
                 for i, (n, s, k) in enumerate(t)],
        "method": {"confidence": "probability of the chosen option", "bins": f"{BINS} equal-width, (lo, hi], the "
                   "first also holding 0", "ece": "sum over bins of n_b/N * |accuracy_b - confidence_b|",
                   "bootstrap": {"samples": SAMPLES, "seed": SEED, "unit": "row", "interval": "percentile, 95%"}},
        "run": manifest,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fetch", "run", "score"))
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--container", default=CONTAINER)
    ap.add_argument("--provenance", type=Path, help="serving evidence for a server without a local container")
    a = ap.parse_args()
    directory = a.cache / "fast-decisions" / REVISION
    pinned = pins()
    if a.step == "fetch":
        fetch(directory, pinned)
        print(f"{directory}: {len(pinned)} files match {PINS.relative_to(ROOT)}")
    elif a.step == "run":
        run(a.url, directory, pinned, a.out, a.container, a.provenance)
    else:
        summary = score(a.out)
        (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        print(json.dumps({k: summary[k] for k in ("questions", "rows", "accuracy", "ece")}))
        print(f"wrote {a.out / 'summary.json'}")


if __name__ == "__main__":
    main()
