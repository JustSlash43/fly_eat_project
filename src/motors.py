"""
Reads specific, functionally-identified descending neurons and turns their
spike rates into (turn, forward) actions — replicating the biological
control mapping used by the public doomfly project (doom/engine.py:
NeuralControls.decode, 'biological' mode) rather than pooling every
descending-neuron-type cell together:

  - DNa02 (right minus left): steering command neuron, well documented as
    driving turning (Rayshubskiy & Wilson 2023; Namiki et al. 2018).
  - DNp09 minus MDN: forward walking command (DNp09, Bidaye et al. 2020)
    versus backward/"moonwalker" command (MDN, Bidaye et al. 2014) — their
    difference drives net forward speed.

Gains below are joystick-style tuning constants for this toy game, not a
claim about the real neurons' behavioral output magnitude.
"""
from __future__ import annotations

import numpy as np

from connectome import Connectome


class BiologicalMotorReadout:
    def __init__(
        self,
        connectome: Connectome,
        smoothing: float = 0.3,
        turn_gain: float = 0.4,
        forward_gain: float = 0.4,
    ):
        self.smoothing = smoothing
        self.turn_gain = turn_gain
        self.forward_gain = forward_gain
        self.turn_ema = 0.0
        self.forward_ema = 0.0

        dna02 = connectome.indices_for_exact_type("DNa02")
        self.dna02_l = connectome.indices_for_side(dna02, "L")
        self.dna02_r = connectome.indices_for_side(dna02, "R")
        self.dnp09 = connectome.indices_for_exact_type("DNp09")
        self.mdn = connectome.indices_for_exact_type("MDN")

        print(
            f"BiologicalMotorReadout: DNa02 L={len(self.dna02_l)} R={len(self.dna02_r)}, "
            f"DNp09={len(self.dnp09)}, MDN={len(self.mdn)}"
        )

    def read(self, spike_counts: np.ndarray, seconds: float) -> tuple[float, float]:
        def rate(idx: np.ndarray) -> float:
            return float(spike_counts[idx].sum()) / seconds if len(idx) and seconds > 0 else 0.0

        turn_raw = (rate(self.dna02_r) - rate(self.dna02_l)) * self.turn_gain
        forward_raw = (rate(self.dnp09) - rate(self.mdn)) * self.forward_gain

        self.turn_ema = (1 - self.smoothing) * self.turn_ema + self.smoothing * turn_raw
        self.forward_ema = (1 - self.smoothing) * self.forward_ema + self.smoothing * forward_raw

        turn = float(np.clip(self.turn_ema, -1, 1))
        speed = float(np.clip(self.forward_ema, 0, 1))
        return turn, speed


# Visual neuron populations whose left/right activity is compared for steering.
STEERING_SUPERCLASSES = ("ol_intrinsic", "visual_projection", "visual_centrifugal")


class VisualSteeringReadout:
    """
    Turns the connectome's own lateralized visual activity into (turn, speed),
    like a Braitenberg vehicle: steer toward the hemisphere that is more active.

    Why not DNa02/DNp09/MDN: in the stable LIF regime (see lif_simulator.py)
    visual activity is strongly lateralized through the optic lobes but
    rarely reaches those few descending neurons, and pushing the gain until
    it does throws the central brain into self-sustained activity that no
    longer depends on what the fly sees. BiologicalMotorReadout is still
    reported in the HUD as a diagnostic.

    Pool: every optic-lobe / visual projection neuron, split by soma side,
    EXCLUDING the photoreceptors and every neuron that receives direct
    photoreceptor input. The readout is therefore at least two synapses
    deep: the signal has to be carried by the connectome's own wiring
    (medulla, lobula, lobula plate, visual projection neurons), not copied
    from the retina.

    No learned weights: turn = (R - L) / (R + L + softening),
    speed = exploratory baseline + total visual drive.
    """

    def __init__(
        self,
        connectome: Connectome,
        photoreceptor_idx: np.ndarray,
        turn_gain: float = 1.5,
        softening: float = 20.0,
        base_speed: float = 0.35,
        speed_scale: float = 150.0,
        smoothing: float = 0.5,
    ):
        self.turn_gain = turn_gain
        self.softening = softening
        self.base_speed = base_speed
        self.speed_scale = speed_scale
        self.smoothing = smoothing
        self.turn_ema = 0.0
        self.speed_ema = base_speed
        self.last_left = self.last_right = 0.0

        n = connectome.n_neurons
        first_order = np.zeros(n, dtype=bool)
        # W rows are postsynaptic: neurons with a nonzero entry in a photoreceptor column
        first_order[np.unique(connectome.W[:, photoreceptor_idx].nonzero()[0])] = True
        first_order[photoreceptor_idx] = True

        superclass = connectome.superclass if connectome.superclass is not None else np.full(n, "")
        pool = np.isin(superclass.astype(str), STEERING_SUPERCLASSES) & ~first_order
        self.left = np.nonzero(pool & (connectome.soma_side.astype(str) == "L"))[0]
        self.right = np.nonzero(pool & (connectome.soma_side.astype(str) == "R"))[0]

        print(
            f"VisualSteeringReadout: {len(self.left)} left / {len(self.right)} right visual neurons "
            f"(excluding {int(first_order.sum())} photoreceptors and their direct targets)"
        )

    def read(self, spike_counts: np.ndarray) -> tuple[float, float]:
        left = float(spike_counts[self.left].sum())
        right = float(spike_counts[self.right].sum())

        turn_raw = self.turn_gain * (right - left) / (right + left + self.softening)
        speed_raw = self.base_speed + (1 - self.base_speed) * np.tanh((left + right) / self.speed_scale)

        self.turn_ema = (1 - self.smoothing) * self.turn_ema + self.smoothing * turn_raw
        self.speed_ema = (1 - self.smoothing) * self.speed_ema + self.smoothing * speed_raw

        self.last_left, self.last_right = left, right
        return float(np.clip(self.turn_ema, -1, 1)), float(np.clip(self.speed_ema, 0, 1))
