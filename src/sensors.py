"""
Real retinotopic visual sensor array for the R1-R6 photoreceptor population.

Bilinearly samples a locally-rendered frame of the fly's forward view at
each photoreceptor's true anatomical UV position (built in vision_mapping.py
from the connectome's own optic-lobe hex-column annotations) — the same
mechanism the public doomfly project uses to sample Doom game frames at
real R1-R6 positions (see doom/game.py:retinal_samples in that project).
"""
from __future__ import annotations

import numpy as np

from connectome import Connectome
from vision_mapping import load_or_build_projection

_LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def retinal_samples(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear *linear* luminance at each uv sample point. rgb: (H,W,3) uint8, uv in [0,1]^2."""
    h, w = rgb.shape[:2]
    x = uv[:, 0] * (w - 1)
    y = uv[:, 1] * (h - 1)
    x0 = x.astype(int)
    y0 = y.astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0

    def linear_luma(pixels: np.ndarray) -> np.ndarray:
        p = pixels.astype(np.float32) / 255
        p = np.where(p <= 0.04045, p / 12.92, ((p + 0.055) / 1.055) ** 2.4)
        return p @ _LUMA_WEIGHTS

    return (
        (1 - dx) * (1 - dy) * linear_luma(rgb[y0, x0])
        + dx * (1 - dy) * linear_luma(rgb[y0, x1])
        + (1 - dx) * dy * linear_luma(rgb[y1, x0])
        + dx * dy * linear_luma(rgb[y1, x1])
    ).astype(np.float32)


class RetinotopicVision:
    """
    Drives R1-R6 photoreceptors using their real anatomical retinal position.

    Two forms of photoreceptor light adaptation shape the drive:
      - each receptor's response saturates, r = L / (L + half_saturation)
        (Naka-Rushton), so dim, distant food still excites the retina while
        near food cannot drive it without bound;
      - a global gain control caps the summed drive at max_total_drive.
        Above ~20000 the full connectome tips into self-sustained central-
        brain activity that outlasts the stimulus (measured at w_syn=0.04),
        so the cap keeps the brain in its stimulus-driven regime.
    """

    def __init__(
        self,
        connectome: Connectome,
        gain: float = 40.0,
        projection: dict | None = None,
        half_saturation: float = 0.05,
        max_total_drive: float = 8000.0,
    ):
        self.gain = gain
        self.half_saturation = half_saturation
        self.max_total_drive = max_total_drive
        self.n = connectome.n_neurons

        if projection is None:
            projection = load_or_build_projection()
        body_ids, uv = projection["body_ids"], projection["uv"]

        present = np.array([b in connectome.id_to_index for b in body_ids])
        self.uv = uv[present]
        self.indices = np.array([connectome.id_to_index[b] for b in body_ids[present]], dtype=int)

        # The proofread connectome has ~2x more mapped R1-R6 cells in the
        # right eye than the left (2228 vs 1107), so the same stimulus drives
        # the right optic lobe much harder and biases every left/right
        # comparison. Scale each eye's drive so both eyes deliver the same
        # total current for the same scene -- a sensor calibration, the
        # connectome itself is untouched.
        eye = np.asarray(projection["root_side"])[present].astype(str)
        self.eye_gain = np.ones(len(self.indices), dtype=np.float32)
        counts = {s: int((eye == s).sum()) for s in ("L", "R")}
        if counts["L"] and counts["R"]:
            mean = (counts["L"] + counts["R"]) / 2
            for s in ("L", "R"):
                self.eye_gain[eye == s] = mean / counts[s]

        print(
            f"RetinotopicVision: {len(self.indices)}/{len(body_ids)} mapped R1-R6 "
            f"photoreceptors present in this graph (left eye {counts['L']}, right eye {counts['R']})"
        )

    def drive(self, frame_rgb: np.ndarray) -> np.ndarray:
        """frame_rgb: (H,W,3) uint8 array — the fly's locally rendered forward view."""
        ext = np.zeros(self.n, dtype=np.float32)
        if len(self.indices) == 0:
            return ext
        luminance = retinal_samples(frame_rgb, self.uv)
        response = luminance / (luminance + self.half_saturation)
        drive = response * self.gain * self.eye_gain
        total = float(drive.sum())
        if total > self.max_total_drive:
            drive *= self.max_total_drive / total
        ext[self.indices] = drive
        return ext
