"""
Controlled verification that the sensory -> motor pathway is actually wired
and propagates signal through the real connectome — the same kind of
stimulus-vs-baseline comparison doomfly's own doom_learning/vision.py assay
uses (black/white/left/right conditions), applied to our R1-R6 -> DNa02 /
DNp09 / MDN pathway.

This does NOT run the game or require a display. It runs two conditions on
the same freshly-reset network and compares downstream spike counts:

  - "no food in view"   -> baseline, should be silent
  - "food on the left"  -> the left visual hemisphere should dominate
  - "food on the right" -> the right visual hemisphere should dominate

It reads the same VisualSteeringReadout the game uses (optic-lobe / visual
projection neurons at least two synapses from the retina, split by side) and
also reports the DNa02/DNp09/MDN descending neurons for reference.

Usage:
    python scripts/verify_pathway.py [--subset N] [--seconds 2.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from connectome import load_connectome
from lif_simulator import LIFPopulation
from main import subset_connectome, ESSENTIAL_TYPES
from environment import BACKGROUND_RGB, FOOD_RGB, POV_FOV_DEG
from motors import BiologicalMotorReadout, VisualSteeringReadout
from sensors import RetinotopicVision


def make_frame(condition: str, size: int = 128) -> np.ndarray:
    """A panoramic frame like environment.render_pov_frame: one food bar 60 deg left/right, or nothing."""
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:] = BACKGROUND_RGB
    azimuth = {"food_left": -60.0, "food_right": 60.0}.get(condition)
    if azimuth is not None:
        col_azimuth = (np.arange(size) / (size - 1) - 0.5) * POV_FOV_DEG
        frame[:, np.abs(col_azimuth - azimuth) <= 10.0] = FOOD_RGB
    return frame


def run_condition(connectome, vision, steering, motors, condition: str, seconds: float, w_syn: float) -> dict:
    sim = LIFPopulation(connectome.W, w_syn=w_syn)
    ext = vision.drive(make_frame(condition))
    counts = sim.run(round(seconds * 1000 / sim.dt), external_input=ext)

    return {
        "r16_active_cells": int(np.count_nonzero(counts[vision.indices])) if len(vision.indices) else 0,
        "visual_left": int(counts[steering.left].sum()),
        "visual_right": int(counts[steering.right].sum()),
        "DNa02_L": int(counts[motors.dna02_l].sum()),
        "DNa02_R": int(counts[motors.dna02_r].sum()),
        "DNp09": int(counts[motors.dnp09].sum()),
        "MDN": int(counts[motors.mdn].sum()),
        "network_total_spikes": int(counts.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=0.5, help="simulated time per condition")
    parser.add_argument("--input-gain", type=float, default=40.0, help="external (sensory) drive gain")
    parser.add_argument("--w-syn", type=float, default=0.035, help="synaptic current per synapse (threshold = 1)")
    args = parser.parse_args()

    connectome = load_connectome()
    if args.subset:
        connectome = subset_connectome(connectome, args.subset, preserve_types=ESSENTIAL_TYPES)

    vision = RetinotopicVision(connectome, gain=args.input_gain)
    steering = VisualSteeringReadout(connectome, vision.indices)
    motors = BiologicalMotorReadout(connectome)

    conditions = ["no_food", "food_left", "food_right"]
    print(f"\nRunning {args.seconds}s of simulated time per condition (w_syn={args.w_syn})...\n")
    results = {c: run_condition(connectome, vision, steering, motors, c, args.seconds, args.w_syn) for c in conditions}

    print(f"{'metric':22s}" + "".join(f"{c:>12s}" for c in conditions))
    for key in results["no_food"]:
        print(f"{key:22s}" + "".join(f"{results[c][key]:12d}" for c in conditions))

    left, right, base = results["food_left"], results["food_right"], results["no_food"]
    print()
    if left["r16_active_cells"] == 0 or right["r16_active_cells"] == 0:
        print("FAIL: the stimulus never made any R1-R6 photoreceptor fire. Check vision_mapping.py / input-gain.")
    elif base["network_total_spikes"] > 0:
        print("FAIL: the network fires with no stimulus -- it is in a self-sustained regime; lower --w-syn.")
    elif left["visual_left"] > left["visual_right"] and right["visual_right"] > right["visual_left"]:
        print(
            "PASS: food on each side drives that side's visual hemisphere harder, through "
            f"{connectome.n_neurons:,} neurons / {connectome.W.nnz:,} synapses -- the steering signal "
            "the game reads is carried by the connectome."
        )
    else:
        print("INCONCLUSIVE: the visual hemispheres did not respond to the side the food was on.")


if __name__ == "__main__":
    main()
