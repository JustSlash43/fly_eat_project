"""
Builds a real retinotopic projection for the R1-R6 photoreceptors: where in
visual space each one anatomically "points", inferred from the connectome
itself rather than assumed.

This replicates the approach used by the public doomfly project (Janelia
MaleCNS + ViZDoom): each R1-R6 photoreceptor's column identity is read off
the optic-lobe hex-lattice coordinates (`assignedOlHex1`/`assignedOlHex2`)
annotated on the L1/L2/L3 lamina neurons it synapses onto — the modal
(highest synaptic-weight) column among those contacts is taken as that
photoreceptor's anatomical position. Hex axial coordinates are converted to
Cartesian and normalized per eye (rootSide) into a single panoramic [0,1]^2
UV viewport, left eye on the left half, right eye on the right half, with a
small overlap in the middle akin to fly binocular overlap.

Result is cached to data/vision_map.npz since building it requires scanning
the full edge table once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / "vision_map.npz"


def build_r1r6_projection(data_dir: Path | str = DATA_DIR, minconf: bool = True) -> dict:
    data_dir = Path(data_dir)
    suffix = "-minconf-0.5" if minconf else ""
    ann_path = data_dir / f"body-annotations-male-cns-v1.0{suffix}.feather"
    weights_path = data_dir / f"connectome-weights-male-cns-v1.0{suffix}.feather"

    ann = pd.read_feather(
        ann_path, columns=["bodyId", "type", "rootSide", "assignedOlHex1", "assignedOlHex2"]
    )
    types = ann["type"].fillna("")

    r16 = ann.loc[types == "R1-R6", ["bodyId", "rootSide"]].set_index("bodyId")
    if r16.empty:
        raise ValueError("No R1-R6 photoreceptors found in annotation table")

    lamina_mask = types.isin(["L1", "L2", "L3"]) & ann["assignedOlHex1"].notna() & ann["assignedOlHex2"].notna()
    lamina = ann.loc[lamina_mask, ["bodyId", "assignedOlHex1", "assignedOlHex2"]].set_index("bodyId")

    print(f"R1-R6 photoreceptors: {len(r16)}; hex-annotated L1/L2/L3 anchors: {len(lamina)}")

    edges = pd.read_feather(weights_path, columns=["body_pre", "body_post", "weight"])
    e = edges[edges["body_pre"].isin(r16.index) & edges["body_post"].isin(lamina.index)]
    print(f"R1-R6 -> L1/L2/L3 contacts found: {len(e)}")

    e = e.join(lamina, on="body_post")
    e["hex1"] = e["assignedOlHex1"]
    e["hex2"] = e["assignedOlHex2"]

    votes = e.groupby(["body_pre", "hex1", "hex2"], as_index=False)["weight"].sum()
    totals = votes.groupby("body_pre")["weight"].sum()
    winner_idx = votes.groupby("body_pre")["weight"].idxmax()
    winners = votes.loc[winner_idx].set_index("body_pre")
    winners["confidence"] = winners["weight"] / totals

    body_ids = winners.index.to_numpy(dtype=np.int64)
    hexes = winners[["hex1", "hex2"]].to_numpy(dtype=np.float64)
    confidence = winners["confidence"].to_numpy(dtype=np.float32)
    root_side = r16.loc[body_ids, "rootSide"].to_numpy().astype("<U1")

    # axial hex -> cartesian
    xy = np.column_stack([hexes[:, 0] - 0.5 * hexes[:, 1], (np.sqrt(3) / 2) * hexes[:, 1]])

    uv = np.empty_like(xy, dtype=np.float32)
    for side in ("L", "R"):
        sel = root_side == side
        if not sel.any():
            continue
        lo = xy[sel].min(axis=0)
        span = np.ptp(xy[sel], axis=0)
        span[span == 0] = 1.0
        z = (xy[sel] - lo) / span
        uv[sel, 0] = 0.6 * z[:, 0] if side == "L" else 0.4 + 0.6 * (1 - z[:, 0])
        uv[sel, 1] = 1 - z[:, 1]

    uv = np.clip(uv, 0, 1).astype(np.float32)
    unmapped = len(r16) - len(body_ids)
    print(
        f"Mapped {len(body_ids)}/{len(r16)} R1-R6 cells to retinal UV positions "
        f"({unmapped} had no annotated lamina contact); median confidence {np.median(confidence):.2f}"
    )

    return {"body_ids": body_ids, "uv": uv, "confidence": confidence, "root_side": root_side}


def load_or_build_projection(data_dir: Path | str = DATA_DIR, cache_path: Path = CACHE_PATH) -> dict:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as f:
            return {
                "body_ids": f["body_ids"],
                "uv": f["uv"],
                "confidence": f["confidence"],
                "root_side": f["root_side"],
            }
    projection = build_r1r6_projection(data_dir)
    np.savez(cache_path, **projection)
    return projection


if __name__ == "__main__":
    p = load_or_build_projection()
    print("body_ids:", p["body_ids"][:10])
    print("uv sample:", p["uv"][:10])
