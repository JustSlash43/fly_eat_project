"""
Closed loop: MaleCNS connectome (LIF simulation) <-> 2D fly-seeks-food world.

Vision and motor pathways replicate the public doomfly project's approach:
real R1-R6 retinotopic photoreceptor positions (vision_mapping.py) sampling
a locally-rendered frame, and specific identified descending neurons
(DNa02 steering, DNp09/MDN forward-backward) for motor readout, rather than
a generic "any visual/descending neuron" pool.

Usage:
    python src/main.py                  # full connectome (needs data/ downloaded)
    python src/main.py --subset 20000   # random subset of neurons, faster/lighter
    python src/main.py --fake           # synthetic random graph, no data needed (smoke test)
    python src/main.py --headless --frames 3000   # no window; prints the score (benchmark)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from connectome import Connectome, load_connectome
from environment import FlyEnvironment
from lif_simulator import LIFPopulation
from motors import BiologicalMotorReadout, VisualSteeringReadout
from sensors import RetinotopicVision

ESSENTIAL_TYPES = ["R1-R6", "DNa02", "DNp09", "MDN"]


def make_fake_connectome(n: int = 5000, avg_degree: int = 30, seed: int = 0) -> Connectome:
    """Synthetic random graph with the same cell-type labels the real pipeline
    looks for, so --fake exercises the exact same vision/motor code paths."""
    from scipy import sparse

    rng = np.random.default_rng(seed)
    n_edges = n * avg_degree
    rows = rng.integers(0, n, n_edges)
    cols = rng.integers(0, n, n_edges)
    signs = rng.choice([1.0, -1.0], n_edges, p=[0.8, 0.2])
    weights = rng.integers(1, 10, n_edges).astype(np.float32) * signs
    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))

    body_ids = np.arange(n, dtype=np.int64)
    types = np.full(n, "other", dtype=object)
    # sprinkle in a handful of each essential type, deterministically
    types[0:120] = "R1-R6"
    types[120:122] = "DNa02"
    types[122:124] = "DNp09"
    types[124:128] = "MDN"

    soma_side = np.array(["L" if i % 2 == 0 else "R" for i in range(n)], dtype=object)
    root_side = soma_side.copy()
    superclass = np.full(n, "ol_intrinsic", dtype=object)
    superclass[120:128] = "descending_neuron"

    return Connectome(
        body_ids=body_ids,
        id_to_index={b: i for i, b in enumerate(body_ids)},
        types=types,
        soma_side=soma_side,
        root_side=root_side,
        W=W,
        superclass=superclass,
    )


def make_fake_projection(connectome: Connectome, seed: int = 0) -> dict:
    """A synthetic R1-R6 -> UV mapping for --fake mode, laid out on a small grid
    so left/right visual field split is still meaningful."""
    rng = np.random.default_rng(seed)
    r16 = connectome.indices_for_exact_type("R1-R6")
    body_ids = connectome.body_ids[r16]
    uv = rng.uniform(0, 1, size=(len(r16), 2)).astype(np.float32)
    confidence = np.ones(len(r16), dtype=np.float32)
    root_side = connectome.root_side[r16]
    return {"body_ids": body_ids, "uv": uv, "confidence": confidence, "root_side": root_side}


def subset_connectome(c: Connectome, n: int, seed: int = 0, preserve_types: list[str] | None = None) -> Connectome:
    """Random subset of n neurons, always keeping any neuron whose type is in
    preserve_types (the sensory/motor cells the demo's readouts depend on)."""
    rng = np.random.default_rng(seed)
    if n >= c.n_neurons:
        return c

    preserve_types = preserve_types or []
    preserve_mask = np.isin(c.types, preserve_types)
    preserve_idx = np.nonzero(preserve_mask)[0]

    remaining_n = max(0, n - len(preserve_idx))
    pool = np.nonzero(~preserve_mask)[0]
    sampled = rng.choice(pool, size=min(remaining_n, len(pool)), replace=False)

    keep = np.sort(np.concatenate([preserve_idx, sampled]))
    W_sub = c.W[keep][:, keep]
    return Connectome(
        body_ids=c.body_ids[keep],
        id_to_index={b: i for i, b in enumerate(c.body_ids[keep])},
        types=c.types[keep],
        soma_side=c.soma_side[keep],
        root_side=c.root_side[keep],
        W=W_sub,
        superclass=c.superclass[keep] if c.superclass is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="use a synthetic random graph instead of real data")
    parser.add_argument("--subset", type=int, default=None, help="use a random subset of N neurons for speed")
    parser.add_argument("--sim-substeps", type=int, default=20, help="LIF steps (ms) per rendered frame")
    parser.add_argument("--input-gain", type=float, default=40.0, help="external current gain for R1-R6 luminance drive")
    parser.add_argument("--w-syn", type=float, default=0.035, help="synaptic current per synapse (threshold = 1)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None, help="world seed (food placement, start heading)")
    parser.add_argument("--headless", action="store_true", help="no window; run --frames frames and print the score")
    parser.add_argument("--frames", type=int, default=3000, help="frames to run in --headless mode")
    args = parser.parse_args()

    if args.fake:
        connectome = make_fake_connectome()
        projection = make_fake_projection(connectome)
    else:
        connectome = load_connectome()
        if args.subset:
            connectome = subset_connectome(connectome, args.subset, preserve_types=ESSENTIAL_TYPES)
        projection = None  # RetinotopicVision loads/builds the real one from data/

    sim = LIFPopulation(connectome.W, w_syn=args.w_syn)
    vision = RetinotopicVision(connectome, gain=args.input_gain, projection=projection)
    steering = VisualSteeringReadout(connectome, vision.indices)
    descending = BiologicalMotorReadout(connectome)  # diagnostic only, see motors.VisualSteeringReadout
    env = FlyEnvironment(seed=args.seed, headless=args.headless)

    frame_no = 0
    started = time.perf_counter()
    running = True
    while running:
        frame = env.render_pov_frame()
        ext_input = vision.drive(frame)

        spike_counts = sim.run(args.sim_substeps, external_input=ext_input)
        turn, speed = steering.read(spike_counts)

        running = env.step(turn, speed)
        frame_no += 1

        if args.headless:
            if frame_no % 500 == 0:
                print(f"frame {frame_no}: score {env.score}  ({frame_no / (time.perf_counter() - started):.1f} frames/s)")
            if frame_no >= args.frames:
                break
            continue

        diagnostics = {
            "r16_active": int(np.count_nonzero(spike_counts[vision.indices])) if len(vision.indices) else 0,
            "r16_total": len(vision.indices),
            "active": int(np.count_nonzero(spike_counts)),
            "vis_l": steering.last_left,
            "vis_r": steering.last_right,
            "dna02_l": int(spike_counts[descending.dna02_l].sum()),
            "dna02_r": int(spike_counts[descending.dna02_r].sum()),
            "dnp09": int(spike_counts[descending.dnp09].sum()),
            "mdn": int(spike_counts[descending.mdn].sum()),
        }
        env.render(turn, speed, diagnostics, pov=frame)
        env.tick(args.fps)

    if args.headless:
        print(f"final score: {env.score} food in {frame_no} frames")
    env.close()


if __name__ == "__main__":
    main()
