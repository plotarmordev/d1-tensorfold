#!/usr/bin/env python3
"""The bench picture: COCO val2017 image 39769 (two cats on a couch), fetched and checked against its sha256, then
resized to 384x288. It is not committed here: COCO's images keep their Flickr owners' licenses.

    python3 bench/assets.py        # writes bench/assets/cats-384.jpg if it is missing
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
SHA256 = "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e"
RESIZED_SHA256 = "09b416b56972a9a46651e559cde81ff77fe64248dd9f85c789a798604a28982e"
DIR = Path(__file__).resolve().parent / "assets"
PICTURE = DIR / "cats-384.jpg"


def ensure() -> Path:
    if PICTURE.exists():
        if hashlib.sha256(PICTURE.read_bytes()).hexdigest() != RESIZED_SHA256:
            raise ValueError("cached benchmark image differs from the pinned corpus")
        return PICTURE
    from PIL import Image

    raw = urllib.request.urlopen(URL, timeout=60).read()
    got = hashlib.sha256(raw).hexdigest()
    if got != SHA256:
        raise SystemExit(f"{URL}: sha256 {got}, expected {SHA256}")
    DIR.mkdir(exist_ok=True)
    Image.open(io.BytesIO(raw)).convert("RGB").resize((384, 288), Image.Resampling.LANCZOS).save(PICTURE, quality=92)
    if hashlib.sha256(PICTURE.read_bytes()).hexdigest() != RESIZED_SHA256:
        raise ValueError("resized benchmark image differs; use the recipe's pinned Pillow environment")
    return PICTURE


if __name__ == "__main__":
    print(ensure())
