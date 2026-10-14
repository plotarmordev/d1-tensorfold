"""LFM2-VL picture features from tensorfold's Python port on the CPU, in d1's Docker image: vision.py MODEL_DIR OUT."""

import base64
import json
import sys
from pathlib import Path

import torch

from tensorfold.families.lfm2.cuda.vision import Vision, _cap
from tensorfold.vision.images import ImageSource, load_images

NAMES = ("turned", "banner", "photo")  # one tile, a wide tile, six tiles and a thumbnail


def features(vision, pic):
    """vision.py's trunk for one picture: the tower over its tiles, then the projector a tile at a time."""
    out = vision.processor(text=["<image>"], images=[[pic]], return_tensors="pt", add_special_tokens=False)
    pv, mask, shapes = out["pixel_values"], out["pixel_attention_mask"], out["spatial_shapes"]
    n = int(mask.sum(1).max())
    hidden = vision.tower(pixel_values=pv[:, :n].to(torch.bfloat16), spatial_shapes=shapes,
                          pixel_attention_mask=mask[:, :n], return_dict=True).last_hidden_state
    lengths = mask.sum(dim=1)
    feats = []
    for i in range(hidden.shape[0]):
        h, w = (int(v) for v in shapes[i])
        f = hidden[i, : int(lengths[i])].reshape(1, h, w, -1)
        feats.append(vision.projector(f).reshape(-1, vision.projector.linear_2.out_features))
    return torch.cat(feats).to(torch.bfloat16)


def picked(rows):
    """The rows kept whole: the first, the quarters and the last."""
    return sorted({0, rows // 4, rows // 2, 3 * rows // 4, rows - 1})


@torch.inference_mode()
def main(model_dir, out_path):
    vision = Vision.load(Path(model_dir), device="cpu")
    pictures = {c["name"]: c["url"] for c in json.loads((Path(__file__).parent / "images.json").read_text())["cases"]}
    cases = []
    for name in NAMES:
        (img,) = load_images([ImageSource(pictures[name])])
        f = features(vision, _cap(img.to_pil()))
        wide = f.float()
        rows = picked(f.shape[0])
        cases.append({"name": name, "rows": f.shape[0], "norms": wide.norm(dim=1).tolist(), "sums": wide.sum(dim=1).tolist(),
                      "picked": rows, "values": base64.b64encode(f[rows].view(torch.int16).numpy().tobytes()).decode()})
    with open(out_path, "w") as out:
        json.dump({"cases": cases}, out)
        out.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
