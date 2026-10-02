"""
Event-driven leaky integrate-and-fire (LIF) simulation over a signed sparse
connectivity graph, following the whole-brain Drosophila LIF model of
Shiu et al. 2024 (Nature, "A Drosophila computational brain model reveals
sensorimotor processing"):

  - weights are raw signed synapse counts; every synapse adds a fixed
    amount of synaptic current (w_syn) when its presynaptic neuron spikes,
  - synaptic current decays with tau_syn (5 ms) and is integrated by the
    membrane (tau_m = 20 ms), so spikes arriving a few ms apart summate.

w_syn = 0.04 (threshold = 1) would match Shiu et al.'s 0.275 mV per synapse
with a 7 mV threshold gap. Measured on the full MaleCNS graph: above ~0.06
the central brain falls into a self-sustaining, input-independent state
almost immediately; at 0.04 it still ignites after ~1500 game frames of
continuous visual drive; at 0.035 (default) and 0.03 it stayed stimulus-
driven for 5000+ frames, with activity dying out when the stimulus goes away.

Only presynaptic neurons that spiked on a step are propagated (a row slice
of the transposed matrix), so a step costs O(synapses of active neurons),
not O(all 26M synapses) -- the full MaleCNS graph runs at game frame rates.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class LIFPopulation:
    def __init__(
        self,
        W: sparse.spmatrix,
        tau_m: float = 20.0,       # membrane time constant, ms
        tau_syn: float = 5.0,      # synaptic current decay, ms
        v_th: float = 1.0,         # spike threshold
        v_reset: float = 0.0,
        refractory_ms: float = 2.0,
        w_syn: float = 0.035,      # current per synapse (W holds raw signed synapse counts)
        dt: float = 1.0,           # ms per sub-step
    ):
        # row = presynaptic neuron, so the targets of the neurons that fired
        # this step are a cheap CSR row slice.
        self.W_pre = sparse.csr_matrix(W.T, dtype=np.float32)
        self.n = W.shape[0]
        self.tau_m = tau_m
        self.syn_decay = float(np.exp(-dt / tau_syn))
        self.v_th = v_th
        self.v_reset = v_reset
        self.refractory_steps = max(1, round(refractory_ms / dt))
        self.w_syn = w_syn
        self.dt = dt

        self.v = np.zeros(self.n, dtype=np.float32)
        self.i_syn = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.fired = np.zeros(0, dtype=np.int64)

    def reset(self) -> None:
        self.v.fill(0.0)
        self.i_syn.fill(0.0)
        self.refractory.fill(0)
        self.fired = np.zeros(0, dtype=np.int64)

    def step(self, external_input: np.ndarray | None = None) -> np.ndarray:
        """Advance one dt and return the indices of the neurons that spiked."""
        self.i_syn *= self.syn_decay
        if len(self.fired):
            out = self.W_pre[self.fired]
            self.i_syn += self.w_syn * np.bincount(out.indices, weights=out.data, minlength=self.n).astype(np.float32)

        drive = self.i_syn if external_input is None else self.i_syn + external_input
        active = self.refractory <= 0
        self.v[active] += (self.dt / self.tau_m) * (drive[active] - self.v[active])

        fired = active & (self.v >= self.v_th)
        self.v[fired] = self.v_reset
        self.refractory[~active] -= 1
        self.refractory[fired] = self.refractory_steps

        self.fired = np.nonzero(fired)[0]
        return self.fired

    def run(self, n_steps: int, external_input: np.ndarray | None = None) -> np.ndarray:
        """Run several steps, return spike counts per neuron over the window."""
        counts = np.zeros(self.n, dtype=np.float32)
        for _ in range(n_steps):
            counts[self.step(external_input)] += 1
        return counts


class LIFBatch:
    """
    B independent brains (one per fly) on the same connectome, with the same
    dynamics as LIFPopulation. The weight matrix is stored once and every
    state array has shape (B, n), so one step advances every fly together:
    the targets of all neurons that fired in any brain come from a single CSR
    row slice and are scattered back to their own brain with one bincount.
    """

    def __init__(
        self,
        W: sparse.spmatrix,
        batch: int,
        tau_m: float = 20.0,
        tau_syn: float = 5.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        refractory_ms: float = 2.0,
        w_syn: float = 0.035,
        dt: float = 1.0,
    ):
        self.W_pre = sparse.csr_matrix(W.T, dtype=np.float32)
        self.n = W.shape[0]
        self.batch = batch
        self.tau_m = tau_m
        self.syn_decay = float(np.exp(-dt / tau_syn))
        self.v_th = v_th
        self.v_reset = v_reset
        self.refractory_steps = max(1, round(refractory_ms / dt))
        self.w_syn = w_syn
        self.dt = dt

        self.v = np.zeros((batch, self.n), dtype=np.float32)
        self.i_syn = np.zeros((batch, self.n), dtype=np.float32)
        self.refractory = np.zeros((batch, self.n), dtype=np.int32)
        self.fired_brain = np.zeros(0, dtype=np.int64)
        self.fired_neuron = np.zeros(0, dtype=np.int64)

    def step(self, external_input: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Advance one dt. external_input: (B, n). Returns (brain, neuron) indices of the spikes."""
        self.i_syn *= self.syn_decay
        if len(self.fired_neuron):
            out = self.W_pre[self.fired_neuron]
            brain = np.repeat(self.fired_brain, np.diff(out.indptr))
            flat = np.bincount(brain * self.n + out.indices, weights=out.data, minlength=self.batch * self.n)
            self.i_syn += self.w_syn * flat.reshape(self.batch, self.n).astype(np.float32)

        drive = self.i_syn if external_input is None else self.i_syn + external_input
        active = self.refractory <= 0
        self.v[active] += (self.dt / self.tau_m) * (drive[active] - self.v[active])

        fired = active & (self.v >= self.v_th)
        self.v[fired] = self.v_reset
        self.refractory[~active] -= 1
        self.refractory[fired] = self.refractory_steps

        self.fired_brain, self.fired_neuron = np.nonzero(fired)
        return self.fired_brain, self.fired_neuron

    def run(self, n_steps: int, external_input: np.ndarray | None = None) -> np.ndarray:
        """Run several steps, return (B, n) spike counts per brain and neuron."""
        counts = np.zeros((self.batch, self.n), dtype=np.float32)
        for _ in range(n_steps):
            brain, neuron = self.step(external_input)
            counts[brain, neuron] += 1
        return counts
