# FLY PROJECT

Downloads the real **MaleCNS v1.0** connectome (Janelia FlyEM + Google
Research, ~211k annotated neurons at minconf 0.5, CC-BY 4.0) and uses it as
a leaky integrate-and-fire (LIF) spiking network to control a fly agent in a
small 2D Pygame world (walk toward food).

Data source: https://male-cns.janelia.org/ — public bucket
`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, no account or
API key required.

The vision and motor pathways replicate the approach used by the public
**doomfly** project (a MaleCNS-driven ViZDoom agent): real anatomical
photoreceptor positions for sensory input, and specific, functionally
identified descending neurons for motor output — not a generic pool of
"visual" or "motor" cells. See **How it works** below.

## Requirements

- Python 3.11+ (tested on 3.14)
- ~1.2 GB of free disk for the connectome data (not included in this repo)
- RAM: ~32 GB recommended for the full connectome; use `--subset` or
  `--fake` on smaller machines
- A display for the Pygame window (or use `--headless`)

## Project layout

```
src/                     simulator, world and closed loop (see "How it works")
scripts/verify_pathway.py  sensory -> motor pathway check on the real data
data/                    downloaded connectome files + vision_map.npz cache (git-ignored)
requirements.txt
```

## Setup

```bash
git clone <repo-url>
cd <repo-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Smoke test (no download needed)

Runs the full pipeline end-to-end on a small synthetic random graph, to
confirm pygame, scipy and the sim loop all work before pulling gigabytes of
real data:

```bash
python src/main.py --fake
```

## 2. Download the real connectome

```bash
python src/download_connectome.py
```

Downloads ~1.2 GB by default (body annotations, neurotransmitter
predictions, and the weighted connectivity graph). Add `--full` to also
pull the raw synapse-point files (~22 GB extra) — not needed for the
simulator.

## 3. Run the fly

```bash
# full ~211k-neuron connectome (needs a decent CPU/RAM, ~32GB recommended)
python src/main.py

# lighter: random subset of neurons (always keeps the R1-R6/DNa02/DNp09/MDN
# cells the readouts depend on, so it stays wired correctly)
python src/main.py --subset 20000
```

Controls: close the window or press ESC to quit.

The window shows the arena, a HUD with the brain's activity, and (top right)
the fly's 330-degree panoramic view that the photoreceptors sample.

```bash
# no window: run N frames and print the score (benchmark)
python src/main.py --headless --frames 5000 --seed 1

# check the retina -> optic lobe -> steering pathway on the real connectome
python scripts/verify_pathway.py
```

Measured on the full connectome, 5000 frames per run:

| seeds 1-4                      | food eaten       |
|--------------------------------|------------------|
| connectome-controlled fly      | 51, 58, 57, 53   |
| random walk, same speed        | 2, 4, 6, 6       |
| blind fly (`--input-gain 0`)   | 0                |

## 4. Several flies in one arena

```bash
python src/main.py --flies 6                       # 6 flies, each with its own brain
python src/main.py --flies 6 --fly-brightness 0    # flies cannot see each other
python src/main.py --flies 6 --headless --frames 1500 --seed 1
```

Each fly runs its own copy of the connectome's LIF state (`LIFBatch`: the
weight matrix is stored once, every brain advances in the same step). Flies
compete for the same food, see each other in their panoramic views as grey
bars (`--fly-brightness`, default 0.35 vs. full-brightness food), and are
pushed apart when they overlap; each new contact counts as a collision.
In the window, TAB or a click picks which fly's view and brain the HUD shows.

Measured on the full connectome, 6 flies, 1500 frames, seed 1 (~7 frames/s):

| flies see each other           | food eaten | collisions |
|--------------------------------|------------|------------|
| yes (`--fly-brightness 0.35`)  | 47         | 55         |
| no (`--fly-brightness 0`)      | 81         | 19         |

The steering readout turns toward whatever is bright, so visible flies
attract each other just like food does: they bump into each other more and
eat less.

## How it works

- `src/download_connectome.py` — fetches the official feather files over HTTPS.
- `src/connectome.py` — loads them into a signed `scipy.sparse` weight
  matrix. Synapse sign (excitatory/inhibitory) is inferred from each
  presynaptic neuron's predicted neurotransmitter (ACh → excitatory,
  GABA/glutamate → inhibitory, aminergic → weak/modulatory).
- `src/lif_simulator.py` — event-driven LIF population following the
  whole-brain model of Shiu et al. 2024: raw signed synapse counts, a fixed
  current per synapse (`w_syn = 0.035`, threshold = 1), synaptic current
  decaying with τ = 5 ms, membrane τ = 20 ms. Only neurons that spiked are
  propagated each step, so the full graph runs at ~30-50 frames/s.
- `src/vision_mapping.py` — builds a **real retinotopic map** for the 3,335
  R1-R6 photoreceptors: each cell's anatomical eye position is read off the
  optic-lobe hex-lattice coordinates (`assignedOlHex1`/`assignedOlHex2`)
  annotated on the L1/L2/L3 lamina neurons it synapses onto (the
  highest-weight/"modal" column among those contacts), converted to
  Cartesian and normalized per eye into one panoramic UV viewport. Cached to
  `data/vision_map.npz` after the first run. This mirrors the projection
  built by the public **doomfly** project (Janelia MaleCNS + ViZDoom) —
  3,335 mapped cells matches their reported figure exactly.
- `src/sensors.py` — bilinearly samples the locally-rendered frame
  (`environment.render_pov_frame`) at each photoreceptor's real UV position
  (the same `retinal_samples` mechanism doomfly uses), then applies
  photoreceptor light adaptation: a saturating response per cell, a global
  cap on the summed drive, and equal total drive for both eyes (the
  connectome has 1107 mapped R1-R6 in the left eye vs 2228 in the right).
- `src/motors.py` — `VisualSteeringReadout` drives the fly like a
  Braitenberg vehicle: turn toward the visual hemisphere (optic-lobe and
  visual projection neurons, split by soma side) that is firing more; walk
  faster the more visual activity there is. Photoreceptors and every
  neuron that receives direct photoreceptor input are excluded, so the
  signal has to cross at least two synapses of the connectome. No learned
  weights. `BiologicalMotorReadout` still reads the identified descending
  neurons, replicating doomfly's `NeuralControls` biological-mode mapping,
  and is shown in the HUD as a diagnostic:
  - **DNa02** (right − left spike rate) → turning (Namiki et al. 2018;
    Rayshubskiy & Wilson 2023)
  - **DNp09 − MDN** (forward-walking command minus the "moonwalker"
    backward-walking command; Bidaye et al. 2020, 2014) → forward speed
- `src/environment.py` — the Pygame world, rendering, and
  `render_pov_frame()`, a 330-degree panoramic first-person view (columns =
  azimuth, straight ahead in the middle, 30-degree blind spot behind) where
  each food item is a vertical bar whose width is its angular size and whose
  brightness falls with distance. The fly bounces off the walls.
- `src/main.py` — the closed loop tying it together.

## Important caveats

- The world itself (2D food-seeking, food drawn as flat bars in a
  panoramic view) is a simplification. The *sampling mechanism* (real
  photoreceptor positions, bilinear luminance sampling) is a faithful
  replication of doomfly's approach; the *content of the frame*, the light
  adaptation and the bilateral steering readout are our own toy-game
  choices, outside the connectome. The connectome itself (neurons,
  synapses, counts, neurotransmitter-inferred signs) is used as downloaded.
- **Why the fly is not steered by DNa02/DNp09/MDN.** Those are single
  identified neurons (DNa02: 1 per side, DNp09: 2, MDN: 4). With a plain LIF
  model and neurotransmitter-inferred signs, visual activity is strongly
  lateralized through the optic lobes but almost never reaches them in the
  regime where the brain stays stimulus-driven (w_syn ≤ 0.035). Raising the
  gain until they do fire (w_syn ≳ 0.06) tips the central brain into
  self-sustained activity (~45k spikes per 20 ms with no stimulus) that no
  longer depends on what the fly sees, and DNa02-left wins regardless of
  where the food is. The earlier version (row-normalized weights,
  `input_gain=3`) never let activity leave the retina at all, which is why
  the fly never moved. The game therefore steers from the lateralized
  visual hemispheres.
- The bilateral readout has a small rightward bias for food within ~20°
  of straight ahead (binocular overlap in the UV map), so the fly approaches
  food in a slight curve rather than a straight line.
- None of this is a validated model of real fly sensorimotor behavior or
  behavior prediction — it's a faithful *replication of the wiring/sampling
  architecture*, run on real connectome data, not a claim that this is how
  a fly actually navigates.

## Swapping in a different "game"

The connectome/simulator code is decoupled from `environment.py`. To drive
a different task, give your environment a `render_pov_frame()`-style method
returning an (H,W,3) uint8 frame and a `step(turn, speed)` method, and reuse
`sensors.RetinotopicVision` / `motors.VisualSteeringReadout` as-is.

## License and credits

- The code in this repository is released under the **MIT License**
  (see [LICENSE](LICENSE)).
- The connectome data is **MaleCNS v1.0** by Janelia FlyEM and Google
  Research, released under **CC-BY 4.0**
  (https://male-cns.janelia.org/). It is downloaded at runtime and is not
  redistributed in this repository; if you publish results based on it,
  cite the MaleCNS release as requested on that site.
- The LIF model follows Shiu et al. 2024 (whole-brain *Drosophila* LIF
  model); the vision/motor mapping replicates the public **doomfly**
  project's approach.
