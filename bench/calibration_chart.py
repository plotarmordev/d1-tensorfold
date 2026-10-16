#!/usr/bin/env python3
"""A reliability diagram and the accuracy-minus-confidence bars of one calibration run, as SVG.

    python3 bench/calibration_chart.py [--summary results/calibration/summary.json]

Writes reliability.svg and accuracy-minus-confidence.svg next to the summary, and reads nothing else, so every
figure on a chart can be checked against that JSON. Standard library only: the image has no matplotlib.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1f2328", "#59636e", "#d1d9e0"
BLUE, ORANGE = "#0969da", "#bc4c00"
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def text(x: float, y: float, s: str, size: int = 12, anchor: str = "start", fill: str = INK,
         weight: str = "normal", vertical: bool = False) -> str:
    turn = f' transform="rotate(-90 {x:.1f} {y:.1f})"' if vertical else ""
    # the halo keeps labels legible where they cross grid lines, bars or the diagonal
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
            f'font-weight="{weight}" stroke="#ffffff" stroke-width="3" paint-order="stroke"{turn}>{escape(s)}</text>')


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = INK, width: float = 1, dash: str = "") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{width}"{d}/>')


def rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"/>'


def document(width: int, height: int, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="{FONT}">\n{rect(0, 0, width, height, "#ffffff")}\n'
            + "\n".join(body) + "\n</svg>\n")


def percent(figure: dict) -> str:
    lo, hi = figure["ci95"]
    return f"{figure['value'] * 100:.1f}% (95% CI {lo * 100:.1f}–{hi * 100:.1f})"


def frame(s: dict, title: str, height: int) -> list[str]:
    d, run = s["dataset"], s["run"]
    served, settings = run["serving"], run["serving"]["settings"]
    dirty = ", uncommitted changes" if run["recipe"]["dirty"] else ""
    return [
        text(24, 32, f"{title}: {', '.join(s['model'])} on this recipe", 18, weight="600"),
        text(24, 54, f"{d['repo']}@{d['revision'][:8]}, {d['split']} split (calibration only, not the benchmark); "
             f"n = {s['questions']:,} questions in {s['rows']:,} rows", 12, fill=MUTED),
        text(24, 72, f"ECE {percent(s['ece'])} · accuracy {percent(s['accuracy'])}", 12, fill=MUTED),
        text(24, height - 44, f"Our run only: {len(s['bins'])} equal-width bins, ECE weighted by bin count, 95% CIs "
             f"from {s['method']['bootstrap']['samples']} bootstrap resamples of rows.", 10, fill=MUTED),
        text(24, height - 30, f"Image {served['image']['id'].split(':')[-1][:12]}, model revision "
             f"{served['model']['revision'][:8]}, {settings['precision']}, context {settings['context']}, token "
             f"budget {settings['token_budget']}; evidence from {served['source']}.", 10, fill=MUTED),
        text(24, height - 16, f"Recipe {run['recipe']['commit'][:7]}{dirty}; run {run['started_utc']} to "
             f"{run['finished_utc']}.", 10, fill=MUTED),
    ]


def reliability(s: dict) -> str:
    width, height, left, top, side = 720, 640, 140, 100, 440

    def x(v: float) -> float:
        return left + v * side

    def y(v: float) -> float:
        return top + side - v * side

    body = frame(s, "Reliability", height)
    for i in range(11):
        v = i / 10
        body += [line(x(v), y(0), x(v), y(1), GRID), line(x(0), y(v), x(1), y(v), GRID),
                 text(x(v), y(0) + 18, f"{i * 10}", 11, "middle", MUTED),
                 text(x(0) - 8, y(v) + 4, f"{i * 10}", 11, "end", MUTED)]
    counts = []
    for b in s["bins"]:
        if not b["n"]:
            continue
        lo, hi = x(b["lo"]) + 2, x(b["hi"]) - 2
        body += [rect(lo, y(b["accuracy"]), hi - lo, b["accuracy"] * side, BLUE),
                 line(lo, y(b["confidence"]), hi, y(b["confidence"]), INK, 2)]
        counts.append(text((lo + hi) / 2, y(max(b["accuracy"], b["confidence"])) - 9, f"n={b['n']:,}", 10,
                           "middle", MUTED))
    key = x(0.03)
    body += [line(x(0), y(0), x(1), y(1), MUTED, 1.5, "6 4"), *counts,
             line(x(0), y(0), x(1), y(0)), line(x(0), y(0), x(0), y(1)),
             text(x(0.5), y(0) + 40, "Confidence of the chosen option (%)", 13, "middle"),
             text(left - 44, y(0.5), "Accuracy (%)", 13, "middle", vertical=True),
             rect(key, y(0.97) - 10, 18, 10, BLUE), text(key + 26, y(0.97), "accuracy in bin", 11),
             line(key, y(0.92) - 4, key + 18, y(0.92) - 4, INK, 2),
             text(key + 26, y(0.92), "mean confidence in bin", 11),
             line(key, y(0.87) - 4, key + 18, y(0.87) - 4, MUTED, 1.5, "6 4"),
             text(key + 26, y(0.87), "perfect calibration", 11)]
    return document(width, height, body)


def gaps(s: dict) -> str:
    width, height, left, top, plot_w, plot_h = 720, 540, 80, 100, 600, 320
    bins = s["bins"]
    widest = max(abs(b["accuracy_minus_confidence"]) for b in bins if b["n"]) * 100
    step = 5 if widest <= 25 else 10
    limit = max(step, math.ceil(widest / step) * step)

    def y(pts: float) -> float:
        return top + plot_h / 2 - pts / limit * plot_h / 2

    slot = plot_w / len(bins)
    body = frame(s, "Accuracy minus confidence", height)
    for pts in range(-limit, limit + 1, step):
        body += [line(left, y(pts), left + plot_w, y(pts), GRID),
                 text(left - 8, y(pts) + 4, f"{pts:+d}" if pts else "0", 11, "end", MUTED)]
    for i, b in enumerate(bins):
        mid = left + (i + 0.5) * slot
        body += [text(mid, y(-limit) + 18, f"{b['lo'] * 100:.0f}–{b['hi'] * 100:.0f}", 11, "middle", MUTED),
                 text(mid, y(-limit) + 32, f"n={b['n']:,}", 10, "middle", MUTED)]
        if not b["n"]:
            continue
        pts = b["accuracy_minus_confidence"] * 100
        colour = BLUE if pts >= 0 else ORANGE
        body += [rect(mid - slot * 0.35, y(max(pts, 0)), slot * 0.7, abs(y(pts) - y(0)), colour),
                 text(mid, y(pts) - 6 if pts >= 0 else y(pts) + 14, f"{pts:+.1f}", 11, "middle")]
    body += [line(left, y(0), left + plot_w, y(0)),
             text(left + plot_w / 2, y(-limit) + 54, "Confidence band of the chosen option (%)", 13, "middle"),
             text(left - 44, y(0), "Accuracy − confidence (points)", 13, "middle", vertical=True),
             text(left + plot_w - 4, y(limit) + 14, "above 0: under-confident", 11, "end", BLUE),
             text(left + plot_w - 4, y(limit) + 30, "below 0: over-confident", 11, "end", ORANGE)]
    return document(width, height, body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=ROOT / "results/calibration/summary.json")
    a = ap.parse_args()
    s = json.loads(a.summary.read_text())
    for name, render in (("reliability.svg", reliability), ("accuracy-minus-confidence.svg", gaps)):
        path = a.summary.parent / name
        path.write_text(render(s))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
