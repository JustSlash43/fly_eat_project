"""
Loads the MaleCNS v1.0 feather files and builds a signed sparse weight matrix
(scipy.sparse.csr_matrix) suitable for a leaky integrate-and-fire simulation.

Column names in FlyEM exports vary slightly between dataset releases, so
loaders here inspect the actual columns present and fail with a readable
error (listing what was found) rather than silently assuming a schema.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Fly CNS neurotransmitter -> sign heuristic.
# ACh is the dominant excitatory transmitter; GABA and glutamate are
# predominantly inhibitory in the insect CNS (glutamate is inhibitory at
# many fly NMJs/central synapses, unlike in vertebrates). Aminergic/peptide
# transmitters are treated as weakly excitatory/modulatory by default.
NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "dopamine": 0.3,
    "serotonin": 0.3,
    "octopamine": 0.3,
    "tyramine": 0.3,
    "histamine": -0.5,
    "unknown": 0.5,
}


def _pick_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a column for '{purpose}'. Tried {candidates}. "
        f"Actual columns present: {list(df.columns)}"
    )


@dataclass
class Connectome:
    body_ids: np.ndarray          # (N,) int64, index position == neuron index
    id_to_index: dict             # bodyId -> row/col index into W
    types: np.ndarray             # (N,) str, cell type annotation (may be "")
    soma_side: np.ndarray         # (N,) str, one of 'L','R','M',''
    root_side: np.ndarray         # (N,) str, `rootSide` annotation (eye/neuropil side for sensory cells)
    W: sparse.csr_matrix          # (N, N) signed synapse-count weight matrix
    superclass: np.ndarray | None = None  # (N,) str, e.g. 'ol_intrinsic', 'visual_projection', 'descending_neuron'

    @property
    def n_neurons(self) -> int:
        return len(self.body_ids)

    def indices_for_type(self, substring: str) -> np.ndarray:
        """Row indices of neurons whose `type` annotation contains substring (case-insensitive)."""
        mask = np.char.find(np.char.lower(self.types.astype(str)), substring.lower()) >= 0
        return np.nonzero(mask)[0]

    def indices_for_exact_type(self, type_name: str) -> np.ndarray:
        """Row indices of neurons whose `type` annotation exactly equals type_name."""
        return np.nonzero(self.types.astype(str) == type_name)[0]

    def indices_for_side(self, idx: np.ndarray, side: str) -> np.ndarray:
        return idx[self.soma_side[idx] == side]

    def indices_for_root_side(self, idx: np.ndarray, side: str) -> np.ndarray:
        return idx[self.root_side[idx] == side]


def load_connectome(data_dir: Path | str = DATA_DIR, minconf: bool = True, normalize_rows: bool = False) -> Connectome:
    data_dir = Path(data_dir)
    suffix = "-minconf-0.5" if minconf else ""

    ann_path = data_dir / f"body-annotations-male-cns-v1.0{suffix}.feather"
    nt_path = data_dir / "body-neurotransmitters-male-cns-v1.0.feather"
    weights_path = data_dir / f"connectome-weights-male-cns-v1.0{suffix}.feather"

    for p in (ann_path, nt_path, weights_path):
        if not p.exists():
            print(
                f"[error] missing {p.name} in {data_dir}. "
                f"Run: python src/download_connectome.py",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Loading {ann_path.name} ...")
    ann = pd.read_feather(ann_path)
    id_col = _pick_column(ann, ["bodyId", "bodyid", "body_id", "bodyId_pre"], "body id")
    type_col_candidates = ["type", "cellType", "celltype", "consensusType"]
    side_col_candidates = ["somaSide", "soma_side", "hemisphere", "side"]

    ann = ann.drop_duplicates(subset=id_col).set_index(id_col)
    body_ids = ann.index.to_numpy(dtype=np.int64)
    id_to_index = {bid: i for i, bid in enumerate(body_ids)}

    try:
        type_col = _pick_column(ann.reset_index(), type_col_candidates, "cell type")
        types = ann[type_col].fillna("").astype(str).to_numpy()
    except ValueError:
        print("[warn] no cell-type column found; type-based lookups will return nothing.")
        types = np.full(len(body_ids), "", dtype=object)

    try:
        side_col = _pick_column(ann.reset_index(), side_col_candidates, "soma side")
        raw_side = ann[side_col].fillna("").astype(str)
        soma_side = raw_side.where(raw_side == "", raw_side.str[0]).to_numpy().astype(str)
    except ValueError:
        print("[warn] no soma-side column found; left/right lookups will return nothing.")
        soma_side = np.full(len(body_ids), "", dtype=object)

    try:
        root_col = _pick_column(ann.reset_index(), ["rootSide", "root_side"], "root side")
        root_side = ann[root_col].fillna("").astype(str).to_numpy()
    except ValueError:
        print("[warn] no rootSide column found; eye-side lookups will return nothing.")
        root_side = np.full(len(body_ids), "", dtype=object)

    if "superclass" in ann.columns:
        superclass = ann["superclass"].fillna("").astype(str).to_numpy()
    else:
        print("[warn] no superclass column found; superclass-based readouts will return nothing.")
        superclass = np.full(len(body_ids), "", dtype=object)

    print(f"Loading {nt_path.name} ...")
    nt = pd.read_feather(nt_path)
    nt_id_col = _pick_column(nt, ["bodyId", "bodyid", "body_id", "body"], "body id")
    nt_col = _pick_column(
        nt, ["consensus_nt", "predicted_nt", "predictedNt", "nt", "neurotransmitter", "top_nt"], "neurotransmitter"
    )
    nt = nt.drop_duplicates(subset=nt_id_col).set_index(nt_id_col)[nt_col].fillna("unknown").str.lower()
    sign_by_body = nt.map(lambda x: NT_SIGN.get(x, NT_SIGN["unknown"]))

    print(f"Loading {weights_path.name} (this is the big one) ...")
    edges = pd.read_feather(weights_path)
    pre_col = _pick_column(
        edges, ["bodyId_pre", "bodyid_pre", "body_pre", "pre", "pre_id"], "presynaptic body id"
    )
    post_col = _pick_column(
        edges, ["bodyId_post", "bodyid_post", "body_post", "post", "post_id"], "postsynaptic body id"
    )
    weight_col = _pick_column(edges, ["weight", "syn_count", "count", "weightHR"], "synapse weight")

    keep = edges[pre_col].isin(id_to_index) & edges[post_col].isin(id_to_index)
    dropped = (~keep).sum()
    if dropped:
        print(f"[info] dropping {dropped} edges referencing bodies outside the annotation table")
    edges = edges.loc[keep]

    pre_idx = edges[pre_col].map(id_to_index).to_numpy()
    post_idx = edges[post_col].map(id_to_index).to_numpy()
    w = edges[weight_col].to_numpy(dtype=np.float32)

    pre_bodies = edges[pre_col].to_numpy()
    signs = pd.Series(pre_bodies).map(sign_by_body).fillna(NT_SIGN["unknown"]).to_numpy(dtype=np.float32)
    signed_w = w * signs

    # R1-R6/R7/R8 photoreceptors are genuinely histaminergic/inhibitory onto
    # their lamina targets in vivo -- but that sign only makes sense together
    # with real graded, hyperpolarizing phototransduction (light REDUCES
    # photoreceptor release). Our simplified LIF proxy does the opposite:
    # "more light -> more photoreceptor spikes" (see sensors.py), so taking
    # the literal inhibitory sign here would cancel the signal rather than
    # encode it. We force photoreceptor output synapses positive (excitatory)
    # instead, matching the same correction the public doomfly project makes
    # for its R8 photoreceptor pathway (doom_learning_v6/visual.py) for the
    # same reason: a documented, deliberate proxy, not the literal biology.
    photoreceptor_types = {"R1-R6", "R7", "R8p", "R8y"}
    pre_types = types[pre_idx]
    is_photoreceptor_pre = np.isin(pre_types, list(photoreceptor_types))
    signed_w = np.where(is_photoreceptor_pre, np.abs(signed_w), signed_w)

    n = len(body_ids)
    W = sparse.csr_matrix((signed_w, (post_idx, pre_idx)), shape=(n, n))
    # row = postsynaptic neuron, col = presynaptic neuron, so W @ spike_vector
    # gives, per postsynaptic neuron, the sum of incoming signed synaptic input.

    # By default W keeps raw signed synapse counts, as in the whole-brain LIF
    # model of Shiu et al. 2024 (Nature): each synapse contributes a fixed
    # amount of current (lif_simulator.LIFPopulation.w_syn). Row-normalizing
    # (each neuron's incoming |weight| summing to 1) was tried and makes
    # every neuron need a third of all its inputs firing at once to spike, so
    # visual activity never left the retina -- kept only as an option.
    if normalize_rows:
        row_abs_sum = np.asarray(np.abs(W).sum(axis=1)).ravel()
        row_abs_sum[row_abs_sum == 0] = 1.0
        W = sparse.diags(1.0 / row_abs_sum) @ W
    W = W.astype(np.float32).tocsr()

    label = "row-normalized" if normalize_rows else "raw synapse counts"
    print(f"Connectome loaded: {n:,} neurons, {W.nnz:,} directed synaptic edges ({label})")
    return Connectome(
        body_ids=body_ids,
        id_to_index=id_to_index,
        types=types,
        soma_side=soma_side,
        root_side=root_side,
        W=W,
        superclass=superclass,
    )


if __name__ == "__main__":
    c = load_connectome()
    print("Example cell types found:", np.unique(c.types)[:20])
