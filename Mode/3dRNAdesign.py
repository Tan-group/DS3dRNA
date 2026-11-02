#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__VERSION__ = "v1.0"

TRIRNADE_BANNER = f"""
============================================================
                    DS3dRNA_v1.0 [2026]
                 De novo design of 3D RNAs 
               via higher-order interactions
------------------------------------------------------------
Author        : Tongwei Yuan [https://github.com/TOVI-YUEN]
Affiliation   : School of Physics. WHU. CN
Contact_1     : tongwei.tovi.yuan@gmail.com
Contact_2     : tovi_yuen@whu.edu.cn
License       : DS3dRNA academic/non-commercial license
Commercial use: Licensing available upon request
------------------------------------------------------------
Version       : {__VERSION__}
============================================================
"""
import os
import sys
import math
import copy
import random
import glob
import argparse
from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from Src.structure_fixed import RNA_Structure_Fixed, IDX_TO_BASE
from Src.potential import TriRNASP_Potential
from Src.scorer import TriRNASP_Scorer
from Src.ss_constraint import (
    build_ss_constraint as _build_ss_constraint_base,
    looks_like_dotbracket,
    sample_penalty_free_sequence as _sample_penalty_free_sequence_base,
)

# DS3dRNA.py sets this value before loading the computational core. Direct
# imports default to RNA mode.
DS3DRNA_MOL_MODE = str(
    globals().get("DS3DRNA_MOL_MODE", os.environ.get("DS3DRNA_MOL_MODE", "RNA"))
).strip().upper()
if DS3DRNA_MOL_MODE not in {"RNA", "DNA"}:
    raise ValueError(f"Unsupported DS3DRNA_MOL_MODE={DS3DRNA_MOL_MODE!r}")
DNA_MODE = DS3DRNA_MOL_MODE == "DNA"

if DNA_MODE:
    from Src.MidThermoPrior_DNA import build_mid_thermo_prior
else:
    from Src.MidThermoPrior import build_mid_thermo_prior

_PAIR_OPTIONS = (
    (
        (0, 1),  # A-T
        (1, 0),  # T-A
        (2, 3),  # C-G
        (3, 2),  # G-C
    )
    if DNA_MODE
    else (
        (0, 1),  # A-U
        (1, 0),  # U-A
        (2, 3),  # C-G
        (3, 2),  # G-C
        (1, 3),  # U-G
        (3, 1),  # G-U
    )
)
_PAIR_DESCRIPTION = "AT/TA/CG/GC" if DNA_MODE else "AU/UA/CG/GC/UG/GU"


def build_ss_constraint(*args, **kwargs):
    """Build the shared SS constraint, restricting DNA to Watson-Crick pairs."""
    constraint = _build_ss_constraint_base(*args, **kwargs)
    if DNA_MODE and constraint is not None:
        constraint.allowed[1 * 4 + 3] = False  # T-G (U-G in tensor encoding)
        constraint.allowed[3 * 4 + 1] = False  # G-T (G-U in tensor encoding)
    return constraint


def sample_penalty_free_sequence(
    ss_constraint,
    generator=None,
    device=None,
    pair_weights=None,
    batch_size: int = 320,
    max_rounds: int = 8,
):
    """Use the shared sampler, assigning zero probability to GT/TG in DNA mode."""
    if DNA_MODE:
        if pair_weights is None:
            pair_weights = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        else:
            weights = torch.as_tensor(pair_weights, dtype=torch.float32).flatten()
            if int(weights.numel()) == 4:
                weights = torch.cat([
                    weights,
                    torch.zeros(2, dtype=weights.dtype, device=weights.device),
                ])
            elif int(weights.numel()) != 6:
                raise ValueError(
                    "DNA pair_weights must contain four Watson-Crick weights "
                    "(or six RNA-layout weights with GT/TG entries)."
                )
            weights[4:] = 0.0
            pair_weights = weights
    return _sample_penalty_free_sequence_base(
        ss_constraint,
        generator=generator,
        device=device,
        pair_weights=pair_weights,
        batch_size=batch_size,
        max_rounds=max_rounds,
    )

def _tqdm_stream():
    # Prefer an explicitly configured terminal stream.
    tty_path = os.environ.get("TQDM_TTY", "").strip()
    if tty_path:
        try:
            return open(tty_path, "w")
        except Exception:
            pass

    # Otherwise use the controlling terminal when available.
    try:
        if os.path.exists("/dev/tty"):
            return open("/dev/tty", "w")
    except Exception:
        pass

    # Fall back to standard error.
    return sys.stderr

_tty = _tqdm_stream()


def set_seed(seed: int):
    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =========================
# Mid-thermo reranker config
# =========================
USE_MID_THERMO = True
MID_THERMO_TOP_K = 5
MID_THERMO_TEMP_C = 37.0
MID_THERMO_UNSUPPORTED_PAIR_PENALTY = 0.0
MID_THERMO_CACHE_LIMIT = 20000


# ======================================================================
# Multi-state acceleration helpers (merge triplets across states)
# ======================================================================

def _propose_batch_multi_independent(
    current_seq: torch.Tensor,
    batch_size: int,
    p_mut_vec: torch.Tensor,
    gens,
    device: torch.device,
    ss_constraint=None,
    ss_fast: Optional["SSProposalFast"] = None,
    frz_constraint: Optional["FrozenConstraint"] = None,
) -> torch.Tensor:
    K, N = current_seq.shape
    B = int(batch_size)

    cand_batch = current_seq[:, None, :].expand(K, B, N).clone()

    # original mode
    if ss_constraint is None or ss_constraint.n_pairs == 0:
        for k in range(K):
            pk = float(p_mut_vec[k].item())
            gk = gens[k]

            mut_mask = (torch.rand((B, N), device=device, generator=gk) < pk)
            if frz_constraint is not None and frz_constraint.n_fixed > 0:
                mut_mask[:, frz_constraint.fixed_pos.to(device)] = False

            no_mut = ~mut_mask.any(dim=1)
            if no_mut.any():
                rows = torch.nonzero(no_mut, as_tuple=False).squeeze(1)
                if frz_constraint is not None and frz_constraint.n_fixed > 0:
                    free_idx = frz_constraint.free_pos.to(device)
                else:
                    free_idx = torch.arange(N, device=device)
                if free_idx.numel() > 0:
                    cols_local = torch.randint(0, free_idx.numel(), (rows.numel(),), device=device, generator=gk)
                    cols = free_idx[cols_local]
                    mut_mask[rows, cols] = True

            old = cand_batch[k]
            rand3 = torch.randint(0, 3, (B, N), device=device, generator=gk, dtype=torch.long)
            new_base = rand3 + (rand3 >= old).long()
            cand_batch[k] = torch.where(mut_mask, new_base, old)
            enforce_frz_and_repair_ss_(
                cand_batch[k],
                frz_constraint=frz_constraint,
                ss_constraint=ss_constraint,
                generator=gk,
            )

        return cand_batch

    # Vectorized hard-SS path
    if ss_fast is None:
        raise ValueError("_propose_batch_multi_independent: ss_fast is required when ss_constraint is enabled")

    for k in range(K):
        cand_batch[k] = _propose_one_center_ss_fast(
            center_seq=current_seq[k],
            batch_size=B,
            p_mut=float(p_mut_vec[k].item()),
            device=device,
            ss_fast=ss_fast,
            generator=gens[k],
            force_one_mut=False,
        )
        enforce_frz_and_repair_ss_(
            cand_batch[k],
            frz_constraint=frz_constraint,
            ss_constraint=ss_constraint,
            generator=gens[k],
        )

    return cand_batch

def merge_multistate_structures(structs: List["RNA_Structure_Fixed"]):

    if not structs:
        raise ValueError("merge_multistate_structures: empty structs")

    s0 = structs[0]
    N_res = int(getattr(s0, "N_res"))

    # --- concat atom-level tensors ---
    coords_list = []
    atom_res_list = []
    atom_family_list = []
    atom_codes_list = []
    atom_offsets = [0]

    total_atoms = 0
    for s in structs:
        if int(getattr(s, "N_res")) != N_res:
            raise ValueError(f"multi-state length mismatch: {getattr(s,'N_res')} vs {N_res}")

        Na = int(getattr(s, "N_atoms"))
        total_atoms += Na
        atom_offsets.append(total_atoms)

        coords_list.append(getattr(s, "coords"))
        atom_res_list.append(getattr(s, "atom_res_indices"))
        atom_family_list.append(getattr(s, "atom_family"))
        atom_codes_list.append(getattr(s, "atom_codes"))

    coords = torch.cat(coords_list, dim=0)
    atom_res = torch.cat(atom_res_list, dim=0)
    atom_family = torch.cat(atom_family_list, dim=0)
    atom_codes = torch.cat(atom_codes_list, dim=0)

    # --- concat triplets with atom-index offsets ---
    trip_list = []
    for si, s in enumerate(structs):
        trip = getattr(s, "triplets")
        off = int(atom_offsets[si])
        if off != 0:
            trip = trip.clone()
            trip[:, 0] += off
            trip[:, 1] += off
            trip[:, 2] += off
        trip_list.append(trip)
    triplets = torch.cat(trip_list, dim=0)

    # --- build merged structure object ---
    merged = copy.copy(s0)  # keep any extra metadata from the first state
    merged.N_res = N_res
    merged.N_atoms = int(total_atoms)
    merged.coords = coords
    merged.atom_res_indices = atom_res
    merged.atom_family = atom_family
    merged.atom_codes = atom_codes
    merged.triplets = triplets

    merged._multistate_merged = True
    merged.n_states = int(len(structs))
    return merged, merged.n_states

def build_directional_proposals_from_rough(
    seq_batch: torch.Tensor,
    E_rough_used: torch.Tensor,
    n_dir: int = 22,
    beta_rough: float = 0.5, #1.0/T_min
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:

    if seq_batch.ndim != 2:
        raise ValueError(f"seq_batch must have shape (B, N), got {tuple(seq_batch.shape)}")
    if E_rough_used.ndim != 1:
        raise ValueError(f"E_rough_used must have shape (B,), got {tuple(E_rough_used.shape)}")
    if seq_batch.shape[0] != E_rough_used.shape[0]:
        raise ValueError(
            f"seq_batch and E_rough_used size mismatch: "
            f"{seq_batch.shape[0]} vs {E_rough_used.shape[0]}"
        )

    B, N = seq_batch.shape
    if B <= 0:
        raise ValueError("seq_batch is empty")
    if n_dir <= 0:
        return torch.empty((0, N), dtype=seq_batch.dtype, device=seq_batch.device)

    device = seq_batch.device
    dtype_prob = torch.float32

    # --------------------------------------------------
    # Boltzmann weights
    # --------------------------------------------------
    E_shift = E_rough_used - E_rough_used.min()
    w = torch.softmax(-float(beta_rough) * E_shift, dim=0).to(dtype_prob)  # (B,)

    # --------------------------------------------------
    # One-hot encode sequences
    # seq_batch: (B,N) -> (B,N,4)
    # --------------------------------------------------
    one_hot = F.one_hot(seq_batch.to(torch.long), num_classes=4).to(dtype_prob)

    # --------------------------------------------------
    # Weighted marginal probabilities
    # --------------------------------------------------
    # (B,1,1) * (B,N,4) -> (B,N,4)
    probs = (w.view(B,1,1) * one_hot).sum(dim=0)  # (N,4)

    # normalize
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------
    probs_expand = probs.unsqueeze(0).expand(n_dir, N, 4).reshape(-1,4)

    seq_dir = torch.multinomial(
        probs_expand,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).reshape(n_dir, N).to(seq_batch.dtype)

    return seq_dir

def build_directional_probs_grouped_from_rough(
    cand_batch: torch.Tensor,      # (K,B,N)
    E_rough_used: torch.Tensor,    # (K,B)
    beta_rough: float = 0.5,
) -> torch.Tensor:
    """
    Build Boltzmann-weighted per-position base probabilities for all chains at once.

    Returns:
        probs: (K, N, 4) float32
    """
    if cand_batch.ndim != 3:
        raise ValueError(f"cand_batch must have shape (K,B,N), got {tuple(cand_batch.shape)}")
    if E_rough_used.ndim != 2:
        raise ValueError(f"E_rough_used must have shape (K,B), got {tuple(E_rough_used.shape)}")
    if cand_batch.shape[:2] != E_rough_used.shape:
        raise ValueError(
            f"shape mismatch: cand_batch[:2]={cand_batch.shape[:2]} vs E_rough_used={E_rough_used.shape}"
        )

    K, B, N = cand_batch.shape
    dtype_prob = torch.float32

    # per-chain Boltzmann weights: (K,B)
    E_shift = E_rough_used - E_rough_used.min(dim=1, keepdim=True).values
    w = torch.softmax(-float(beta_rough) * E_shift, dim=1).to(dtype_prob)  # (K,B)

    # one-hot: (K,B,N,4)
    one_hot = F.one_hot(cand_batch.to(torch.long), num_classes=4).to(dtype_prob)

    # weighted sum over B -> (K,N,4)
    probs = (w[:, :, None, None] * one_hot).sum(dim=1)

    probs = probs / probs.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return probs
    
def sample_directional_from_grouped_probs(
    probs: torch.Tensor,   # (K,N,4)
    n_dir: int,
    generators: Optional[List[torch.Generator]] = None,
    dtype_out: torch.dtype = torch.long,
) -> torch.Tensor:
    """
    Sample directional proposals per chain from grouped probs.

    Returns:
        seq_dir: (K, n_dir, N)
    """
    if probs.ndim != 3 or probs.shape[-1] != 4:
        raise ValueError(f"probs must have shape (K,N,4), got {tuple(probs.shape)}")

    K, N, _ = probs.shape
    device = probs.device

    if n_dir <= 0:
        return torch.empty((K, 0, N), dtype=dtype_out, device=device)

    seq_dir_list = []
    for kk in range(K):
        gk = None if generators is None else generators[kk]
        probs_expand = probs[kk].unsqueeze(0).expand(n_dir, N, 4).reshape(-1, 4)
        seq_dir_k = torch.multinomial(
            probs_expand,
            num_samples=1,
            replacement=True,
            generator=gk,
        ).reshape(n_dir, N).to(dtype_out)
        seq_dir_list.append(seq_dir_k)

    return torch.stack(seq_dir_list, dim=0)  # (K,n_dir,N)

def _valid_res_count_from_native_idx(native_idx: torch.Tensor | None, N_res_fallback: int) -> int:
    """
    Effective residue count for a state.
    Rule: count positions whose native_base_idx is in {0,1,2,3}.
    If native_idx is None -> fallback to N_res_fallback.
    """
    if native_idx is None:
        return int(N_res_fallback)
    # native_idx could be on CPU; keep it simple
    v = native_idx.detach().cpu()
    return int(((v >= 0) & (v <= 3)).sum().item())


def compute_multistate_norm_factor(structs: List["RNA_Structure_Fixed"]) -> Tuple[float, int, List[int]]:
    """
    Compute normalization factor for multi-state energy:
        norm_factor = sum(valid_i) / native_max
    where:
        valid_i = effective residue count of state i (0<=native_idx<=3)
        native_max = max(valid_i)

    Returns:
        norm_factor (float, >=1 if at least one state has >=1 valid res)
        native_max (int)
        valid_list (List[int])
    """
    if not structs:
        raise ValueError("compute_multistate_norm_factor: empty structs")

    valid_list = []
    for s in structs:
        N_res = int(getattr(s, "N_res"))
        native_idx = getattr(s, "native_base_idx", None)
        valid_list.append(_valid_res_count_from_native_idx(native_idx, N_res_fallback=N_res))

    native_max = max(valid_list) if valid_list else 0
    if native_max <= 0:
        # pathological: all states have 0 valid residues
        # keep it safe to avoid division by zero; fall back to number of states
        native_max = 1

    norm_factor = float(sum(valid_list)) / float(native_max)
    # safety
    if norm_factor <= 0:
        norm_factor = float(len(structs))

    return norm_factor, native_max, valid_list



# ======================================================================
# Helper functions
# ======================================================================

def batch_recovery(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    pred: (B, N)
    true: (N,)
    return: (B,)
    Ignore positions where true < 0.
    """
    if pred.dim() != 2 or true.dim() != 1:
        raise ValueError(f"pred shape={tuple(pred.shape)}, true shape={tuple(true.shape)}")

    if true.device != pred.device:
        true = true.to(pred.device)

    mask = (true >= 0)  # (N,)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return torch.full((pred.shape[0],), float("nan"), dtype=torch.float32, device=pred.device)

    pred_v = pred[:, mask]     # (B, Nv)
    true_v = true[mask]        # (Nv,)
    return (pred_v == true_v.unsqueeze(0)).float().mean(dim=1)


def batch_macro_f1(
    pred: torch.Tensor,
    true: torch.Tensor,
    num_classes: int = 4,
    ignore_empty_classes: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    pred: (B, N)
    true: (N,)
    return: (B,)
    Ignore positions where true < 0.
    """
    if pred.dim() != 2 or true.dim() != 1:
        raise ValueError(f"pred shape={tuple(pred.shape)}, true shape={tuple(true.shape)}")

    B, N = pred.shape
    device = pred.device

    if true.device != device:
        true = true.to(device)

    mask = (true >= 0)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return torch.full((B,), float("nan"), dtype=torch.float32, device=device)

    pred = pred[:, mask]   # (B,Nv)
    true = true[mask]      # (Nv,)

    include = torch.ones((num_classes,), dtype=torch.bool, device=device)
    if ignore_empty_classes:
        for c in range(num_classes):
            include[c] = (true == c).any()

    idx = torch.nonzero(include, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return torch.full((B,), float("nan"), dtype=torch.float32, device=device)

    f1s = []
    for c in idx.tolist():
        pred_c = (pred == c)
        true_c = (true == c).unsqueeze(0)

        tp = (pred_c & true_c).sum(dim=1).float()
        fp = (pred_c & (~true_c)).sum(dim=1).float()
        fn = ((~pred_c) & true_c).sum(dim=1).float()

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2.0 * precision * recall / (precision + recall + eps)
        f1s.append(f1)

    return torch.stack(f1s, dim=0).mean(dim=0)

# ============================================================
# Decode-time global poly-run parameters
# ============================================================
# Base code convention:
#   A=0, U=1, C=2, G=3
#
# Semantics:
#   max_run[b] = maximum allowed consecutive run length for base b.
#   Example:
#     DECODE_POLY_G_MAX_RUN = 3 means GGG is allowed, GGGG is forbidden.
#     DECODE_POLY_A_MAX_RUN = 6 means AAAAAA is allowed, AAAAAAA is forbidden.
#
# Set a value to None or <=0 to disable the poly-run constraint for that base.
# ============================================================

DECODE_POLY_A_MAX_RUN = 5
DECODE_POLY_U_MAX_RUN = 5 
DECODE_POLY_C_MAX_RUN = 5
DECODE_POLY_G_MAX_RUN = 5

DECODE_BASE_NAMES = ("A", "U", "C", "G")

DECODE_POLY_MAX_RUN_BY_BASE = np.array(
    [
        10**9 if DECODE_POLY_A_MAX_RUN is None or int(DECODE_POLY_A_MAX_RUN) <= 0 else int(DECODE_POLY_A_MAX_RUN),
        10**9 if DECODE_POLY_U_MAX_RUN is None or int(DECODE_POLY_U_MAX_RUN) <= 0 else int(DECODE_POLY_U_MAX_RUN),
        10**9 if DECODE_POLY_C_MAX_RUN is None or int(DECODE_POLY_C_MAX_RUN) <= 0 else int(DECODE_POLY_C_MAX_RUN),
        10**9 if DECODE_POLY_G_MAX_RUN is None or int(DECODE_POLY_G_MAX_RUN) <= 0 else int(DECODE_POLY_G_MAX_RUN),
    ],
    dtype=np.int64,
)

DECODE_POLY_RMAX = int(np.max(DECODE_POLY_MAX_RUN_BY_BASE[DECODE_POLY_MAX_RUN_BY_BASE < 10**9])) \
    if np.any(DECODE_POLY_MAX_RUN_BY_BASE < 10**9) else 1


def _decode_poly_rule_summary() -> str:
    items = []
    for b, name in enumerate(DECODE_BASE_NAMES):
        m = int(DECODE_POLY_MAX_RUN_BY_BASE[b])
        if m >= 10**9:
            items.append(f"{name}: disabled")
        else:
            items.append(f"{name}: max {m} allowed, length {m + 1}+ forbidden")
    return "; ".join(items)


def _decode_prev_same_chain_mask(chain_constraint, N: int) -> np.ndarray:
    """
    Return prev_same_chain[pos] == True iff pos and pos-1 are in the same chain.

    Decode-time use ONLY:
      - '&' breaks chain
      - homopolymer run does NOT cross '&'

    It does NOT use pair information.
    It does NOT use terminal-pair information.
    """
    prev_same_chain = np.zeros((N,), dtype=np.bool_)
    if N <= 1:
        return prev_same_chain

    segs = getattr(chain_constraint, "chain_segments", None) if chain_constraint is not None else None
    if segs is not None:
        for seg in segs:
            start, end = int(seg[0]), int(seg[1])
            if end - start >= 2:
                prev_same_chain[start + 1 : end] = True
        return prev_same_chain

    raw_ss = getattr(chain_constraint, "raw_ss_string", None) if chain_constraint is not None else None
    if raw_ss is not None:
        cur = 0
        in_chain = False
        chain_start = 0

        for ch in raw_ss:
            if ch.isspace():
                continue

            if ch == "&":
                if in_chain:
                    if cur - chain_start >= 2:
                        prev_same_chain[chain_start + 1 : cur] = True
                    in_chain = False
                continue

            if not in_chain:
                chain_start = cur
                in_chain = True

            cur += 1

        if in_chain and cur - chain_start >= 2:
            prev_same_chain[chain_start + 1 : cur] = True

        return prev_same_chain

    # No chain info available -> compact sequence is treated as one chain.
    prev_same_chain[1:] = True
    return prev_same_chain


# Decode-only terminal-orientation search. It branches over terminal CG/GC
# orientations and then applies the homopolymer-run dynamic program.
DECODE_TERMINAL_ORIENT_BEAMS = (512, 4096, 65536)


def _decode_terminal_pairs(ss_constraint, N: int):
    """
    Extract stem terminal pairs from ss_constraint.pairs.

    Terminal rule:
      for each stem:
        - length == 1: use its only pair
        - length >= 2: use first and last pair

    This function only returns terminal pair positions.
    It does NOT enforce full pair legality.
    """
    if ss_constraint is None:
        return []

    pairs = getattr(ss_constraint, "pairs", None)
    if not pairs:
        return []

    pair_set = set()
    for i, j in pairs:
        i = int(i)
        j = int(j)
        if 0 <= i < N and 0 <= j < N and i != j:
            if i < j:
                pair_set.add((i, j))
            else:
                pair_set.add((j, i))

    if not pair_set:
        return []

    terminal_pairs = []

    for i, j in sorted(pair_set):
        # Only start from the outer/start pair of a stacked stem.
        if (i - 1, j + 1) in pair_set:
            continue

        stem = []
        a, b = i, j
        while (a, b) in pair_set:
            stem.append((a, b))
            a += 1
            b -= 1

        if len(stem) == 1:
            terminal_pairs.append(stem[0])
        else:
            terminal_pairs.append(stem[0])
            terminal_pairs.append(stem[-1])

    # De-duplicate defensively.
    out = []
    seen = set()
    for p in terminal_pairs:
        if p not in seen:
            out.append(p)
            seen.add(p)

    return out


def _fixed_base_has_forbidden_run(
    fixed_base: np.ndarray,
    prev_same_chain: np.ndarray,
    max_run: np.ndarray = DECODE_POLY_MAX_RUN_BY_BASE,
) -> bool:
    """
    Quick impossibility check on already-fixed bases.

    fixed_base[pos] = -1 means free, so it can break a run.
    Only fully fixed consecutive same-base runs can prove impossibility.

    max_run is controlled by global decode-time parameters:
      DECODE_POLY_A_MAX_RUN
      DECODE_POLY_U_MAX_RUN
      DECODE_POLY_C_MAX_RUN
      DECODE_POLY_G_MAX_RUN
    """
    last = -1
    run = 0

    N = int(fixed_base.shape[0])
    for pos in range(N):
        b = int(fixed_base[pos])

        if b < 0:
            last = -1
            run = 0
            continue

        if pos == 0 or not bool(prev_same_chain[pos]):
            last = b
            run = 1
        else:
            if b == last:
                run += 1
            else:
                last = b
                run = 1

        if run > int(max_run[b]):
            return True

    return False


def _make_terminal_cg_lock_candidates(
    ss_constraint,
    scores_np: np.ndarray,
    prev_same_chain: np.ndarray,
    beam_size: int,
    fixed_base_init: Optional[np.ndarray] = None,
):
    """
    Build candidate fixed_base arrays for decode-only terminal CG/GC clamp.

    fixed_base[pos] = -1 means free.
    fixed_base[pos] in {0,1,2,3} means forced base.

    For each terminal pair (i,j), force either:
      C-G: i=2, j=3
      G-C: i=3, j=2

    This is a light orientation search over terminal pairs only.
    It does NOT do full pair DP.
    It does NOT enforce AU/UA/GU/etc.
    It does NOT call score_batch().
    """
    N = int(scores_np.shape[1])
    max_run = DECODE_POLY_MAX_RUN_BY_BASE

    terminal_pairs = _decode_terminal_pairs(ss_constraint, N)

    if fixed_base_init is None:
        base_fixed0 = np.full((N,), -1, dtype=np.int8)
    else:
        base_fixed0 = np.asarray(fixed_base_init, dtype=np.int8).copy()
        if base_fixed0.shape[0] != N:
            raise ValueError(
                f"[frz] fixed_base_init length mismatch: {base_fixed0.shape[0]} vs N={N}"
            )

    if _fixed_base_has_forbidden_run(
        fixed_base=base_fixed0,
        prev_same_chain=prev_same_chain,
        max_run=DECODE_POLY_MAX_RUN_BY_BASE,
    ):
        return []

    # No terminal pairs -> keep only pre-existing fixed bases.
    if not terminal_pairs:
        return [(0.0, base_fixed0)]

    # Sort by left position for stable behavior.
    terminal_pairs = sorted(terminal_pairs)

    init_fixed = base_fixed0.copy()
    states = [(0.0, init_fixed)]

    for i, j in terminal_pairs:
        new_states = []

        for score0, fixed0 in states:
            # Option 1: C-G
            # Option 2: G-C
            local_added = False

            for bi, bj in ((2, 3), (3, 2)):
                # conflict with previous/frozen lock
                if fixed0[i] >= 0 and int(fixed0[i]) != bi:
                    continue
                if fixed0[j] >= 0 and int(fixed0[j]) != bj:
                    continue

                fixed = fixed0.copy()
                fixed[i] = bi
                fixed[j] = bj

                # Hard prune: terminal locks themselves must not already
                # force a forbidden poly-run under the global decode rules.
                if _fixed_base_has_forbidden_run(
                    fixed_base=fixed,
                    prev_same_chain=prev_same_chain,
                    max_run=max_run,
                ):
                    continue

                local_score = float(scores_np[bi, i]) + float(scores_np[bj, j])
                new_states.append((score0 + local_score, fixed))
                local_added = True

            # If frozen bases conflict with terminal CG/GC heuristic,
            # frozen wins and this terminal pair is not clamped.
            if not local_added:
                fixed = fixed0.copy()
                if not _fixed_base_has_forbidden_run(
                    fixed_base=fixed,
                    prev_same_chain=prev_same_chain,
                    max_run=max_run,
                ):
                    new_states.append((score0, fixed))

        if not new_states:
            return []

        # Keep only high-local-score terminal orientation candidates.
        # The final winner is still selected by full poly-run DP score.
        if len(new_states) > int(beam_size):
            new_states.sort(key=lambda x: x[0], reverse=True)
            new_states = new_states[: int(beam_size)]

        states = new_states

    states.sort(key=lambda x: x[0], reverse=True)
    return states


def _poly_run_dp_with_fixed_bases(
    scores_np: np.ndarray,
    fixed_base: np.ndarray,
    prev_same_chain: np.ndarray,
):
    """
    DP under two hard constraints:

      1) fixed_base[pos] if specified
      2) poly-run limits controlled by global decode parameters:
           DECODE_POLY_A_MAX_RUN
           DECODE_POLY_U_MAX_RUN
           DECODE_POLY_C_MAX_RUN
           DECODE_POLY_G_MAX_RUN

    Returns:
      (best_seq_idx, best_score)
      or None if no valid sequence exists.
    """
    s = scores_np
    N = int(s.shape[1])

    max_run = DECODE_POLY_MAX_RUN_BY_BASE
    RMAX = DECODE_POLY_RMAX
    NEG_INF = -1.0e100

    dp = np.full((4, RMAX + 1), NEG_INF, dtype=np.float64)
    parent_base = np.full((N, 4, RMAX + 1), -1, dtype=np.int8)
    parent_run = np.full((N, 4, RMAX + 1), -1, dtype=np.int8)

    # Initialize pos=0
    if int(fixed_base[0]) >= 0:
        allowed0 = [int(fixed_base[0])]
    else:
        allowed0 = [0, 1, 2, 3]

    for b in allowed0:
        # If a base is disabled by max_run <= 0 before normalization, it has
        # effectively infinite max_run after normalization, so run=1 is valid.
        if int(max_run[b]) >= 1:
            dp[b, 1] = s[b, 0]

    for pos in range(1, N):
        new_dp = np.full((4, RMAX + 1), NEG_INF, dtype=np.float64)

        if int(fixed_base[pos]) >= 0:
            allowed_bases = [int(fixed_base[pos])]
        else:
            allowed_bases = [0, 1, 2, 3]

        if not bool(prev_same_chain[pos]):
            flat_idx = int(np.argmax(dp))
            prev_b, prev_r = np.unravel_index(flat_idx, dp.shape)
            best_prev = float(dp[prev_b, prev_r])

            if best_prev <= NEG_INF / 2:
                return None

            for b in allowed_bases:
                if int(max_run[b]) >= 1:
                    new_dp[b, 1] = best_prev + s[b, pos]
                    parent_base[pos, b, 1] = prev_b
                    parent_run[pos, b, 1] = prev_r

            dp = new_dp
            continue

        for b in allowed_bases:
            mb = int(max_run[b])
            mb_dp = min(mb, RMAX)

            if mb_dp < 1:
                continue

            # Continue same base.
            for old_r in range(1, mb_dp):
                old_score = float(dp[b, old_r])
                if old_score <= NEG_INF / 2:
                    continue

                new_r = old_r + 1
                cand = old_score + s[b, pos]

                if cand > new_dp[b, new_r]:
                    new_dp[b, new_r] = cand
                    parent_base[pos, b, new_r] = b
                    parent_run[pos, b, new_r] = old_r

            # Switch from another base.
            best_prev = NEG_INF
            best_pb = -1
            best_pr = -1

            for pb in range(4):
                if pb == b:
                    continue

                pr = int(np.argmax(dp[pb]))
                val = float(dp[pb, pr])

                if val > best_prev:
                    best_prev = val
                    best_pb = pb
                    best_pr = pr

            if best_prev > NEG_INF / 2:
                cand = best_prev + s[b, pos]

                if cand > new_dp[b, 1]:
                    new_dp[b, 1] = cand
                    parent_base[pos, b, 1] = best_pb
                    parent_run[pos, b, 1] = best_pr

        dp = new_dp

    flat_idx = int(np.argmax(dp))
    cur_b, cur_r = np.unravel_index(flat_idx, dp.shape)
    best_score = float(dp[cur_b, cur_r])

    if best_score <= NEG_INF / 2:
        return None

    out = np.zeros((N,), dtype=np.int64)

    for pos in range(N - 1, -1, -1):
        out[pos] = int(cur_b)

        if pos == 0:
            break

        pb = int(parent_base[pos, cur_b, cur_r])
        pr = int(parent_run[pos, cur_b, cur_r])

        if pb < 0 or pr < 0:
            return None

        cur_b, cur_r = pb, pr

    return out, best_score


def decode_consensus_idx_poly_runs(
    scores,
    ss_constraint=None,
    frz_constraint: Optional["FrozenConstraint"] = None,
) -> np.ndarray:
    """
    Decode consensus with simultaneous hard constraints:

      1) '&' chain break from ss_constraint
      2) terminal stem pairs must be CG or GC
      3) poly-run limits controlled by global decode parameters:
           DECODE_POLY_A_MAX_RUN
           DECODE_POLY_U_MAX_RUN
           DECODE_POLY_C_MAX_RUN
           DECODE_POLY_G_MAX_RUN

    Important:
      - terminal CG/GC is decode-only
      - sampling/proposal is not changed
      - full pair legality is not restored
      - no terminal weak AU/GU logic
      - no pair-beam search
      - no final score_batch fallback
      - if no sequence satisfies all hard constraints, raise RuntimeError
    """
    if isinstance(scores, torch.Tensor):
        s = scores.detach().cpu().numpy()
    else:
        s = np.asarray(scores)

    if s.ndim != 2 or s.shape[0] != 4:
        raise ValueError(f"scores must have shape (4, N), got {tuple(s.shape)}")

    s = s.astype(np.float64, copy=False)

    N = int(s.shape[1])
    if N == 0:
        return np.zeros((0,), dtype=np.int64)

    if ss_constraint is not None and int(getattr(ss_constraint, "compact_len", N)) != N:
        raise ValueError(
            f"Consensus length mismatch: scores_N={N} "
            f"vs ss_compact_len={int(ss_constraint.compact_len)}"
        )

    prev_same_chain = _decode_prev_same_chain_mask(ss_constraint, N)

    fixed_base_init = None
    if frz_constraint is not None and frz_constraint.n_fixed > 0:
        fixed_base_init = frz_constraint.fixed_idx.detach().cpu().numpy().astype(np.int8, copy=True)

    best_seq = None
    best_score = -1.0e100

    # Try progressively larger terminal-orientation candidate sets.
    # This keeps normal cases fast while avoiding false failure in conflicts.
    for beam_size in DECODE_TERMINAL_ORIENT_BEAMS:
        candidates = _make_terminal_cg_lock_candidates(
            ss_constraint=ss_constraint,
            scores_np=s,
            prev_same_chain=prev_same_chain,
            beam_size=int(beam_size),
            fixed_base_init=fixed_base_init,
        )

        if not candidates:
            continue

        for _, fixed_base in candidates:
            ret = _poly_run_dp_with_fixed_bases(
                scores_np=s,
                fixed_base=fixed_base,
                prev_same_chain=prev_same_chain,
            )

            if ret is None:
                continue

            seq_idx, seq_score = ret

            if float(seq_score) > best_score:
                best_score = float(seq_score)
                best_seq = seq_idx

        if best_seq is not None:
            return best_seq.astype(np.int64, copy=False)

    raise RuntimeError(
        "decode_consensus_idx_poly_runs: no valid consensus satisfies "
        "terminal CG/GC + decode-time poly-run constraints. "
        f"Current poly rule: {_decode_poly_rule_summary()}. "
        "This means the terminal locks and homopolymer constraints are mutually "
        "infeasible under the searched terminal orientations."
    )


BASE_MAP_NP = np.array(["A", "U", "C", "G"], dtype="<U1")

def topk_unique_energy_and_consensus(
    tail_seq_u8_list,
    tail_E_list,
    tail_rec_list,
    tail_f1_list,
    k: int = 5,
    energy_decimals: int = 6,
    chain_constraint=None,
    frz_constraint: Optional["FrozenConstraint"] = None,
):
    """
    Build top-k unique-energy groups and decode consensus sequence.

    Consensus decode uses ONLY:
      - G-run < xx
      - A/U/C-run < xx
      - '&' chain break from chain_constraint

    Removed:
      - pair legality
      - terminal clamp
      - beam search
      - final score_batch check
    """
    n = len(tail_E_list)
    if n == 0:
        return [], ""

    k = max(1, int(k))
    scale = 10 ** int(energy_decimals)

    items = []
    for E, s_u8, r, f1 in zip(
        tail_E_list,
        tail_seq_u8_list,
        tail_rec_list,
        tail_f1_list,
    ):
        Ei = float(E)
        key_int = int(round(Ei * scale))
        b_cache = s_u8.tobytes()

        items.append((
            Ei,
            key_int,
            s_u8,
            float(r),
            float(f1),
            b_cache,
        ))

    # Avoid numpy-array tuple-comparison issues when energies tie.
    items.sort(key=lambda x: (x[0], x[1], x[5]))

    groups = []
    cur_key_int = None

    for Ei, key_int, s_u8, r, f1, b_cache in items:
        if cur_key_int is None or key_int != cur_key_int:
            groups.append({
                "E_int": key_int,
                "E": key_int / scale,
                "seqs": [],
                "recs": [],
                "f1s": [],
                "bytes": [],
            })
            cur_key_int = key_int

        groups[-1]["seqs"].append(s_u8)
        groups[-1]["recs"].append(r)
        groups[-1]["f1s"].append(f1)
        groups[-1]["bytes"].append(b_cache)

    def pick_rep(g):
        freq = {}
        meta = {}

        for s_u8, r, f1, b in zip(
            g["seqs"],
            g["recs"],
            g["f1s"],
            g["bytes"],
        ):
            freq[b] = freq.get(b, 0) + 1

            if (
                (b not in meta)
                or (r > meta[b][0])
                or (r == meta[b][0] and f1 > meta[b][1])
            ):
                meta[b] = (r, f1, s_u8)

        best_b = None
        best_tuple = None

        for b, c in freq.items():
            r, f1, _ = meta[b]
            tup = (c, r, f1)

            if best_tuple is None or tup > best_tuple:
                best_tuple = tup
                best_b = b

        rep_r, rep_f1, rep_s = meta[best_b]
        return rep_s, rep_r, rep_f1, int(freq[best_b]), best_b

    top_groups = []
    seen = set()

    for g in groups:
        rep_s, rep_r, rep_f1, rep_freq, best_b = pick_rep(g)

        if best_b in seen:
            continue

        seen.add(best_b)

        top_groups.append({
            "E": float(g["E"]),
            "n_samples": len(g["seqs"]),
            "rep_seq_u8": rep_s,
            "rep_rec": float(rep_r),
            "rep_f1": float(rep_f1),
            "rep_freq": int(rep_freq),
            "all_seqs_u8": g["seqs"],
        })

        if len(top_groups) >= k:
            break

    if not top_groups:
        return [], ""

    N_res = int(top_groups[0]["rep_seq_u8"].shape[0])
    counts = np.zeros((4, N_res), dtype=np.int64)
    idx_cols = np.arange(N_res, dtype=np.intp)

    for tg in top_groups:
        for s_u8 in tg["all_seqs_u8"]:
            np.add.at(counts, (s_u8, idx_cols), 1)

    consensus_idx = decode_consensus_idx_poly_runs(
        scores=counts,
        ss_constraint=chain_constraint,
        frz_constraint=frz_constraint,
    )

    consensus_seq = "".join(BASE_MAP_NP[consensus_idx].tolist())
    return top_groups, consensus_seq

def _write_single_outputs(
    out_prefix: str,
    pdb_name: str,
    consensus_seq_str: str,
    consensus_rec: float,
    macro_f1_consensus: float,
    ppl: float,
    diversity: float,
    E_rough: float,  # <<< NEW
    E_cons: float,
    best_rec: float,
    best_macro_f1: float,
    E_best_rec: float,
    E_best_f1: float,
    best_rec_seq_str: str,
    best_macro_seq_str: str,
    counts_cpu: torch.Tensor,
    native_idx: torch.Tensor,
    traj_step_list,
    traj_rec_list,
    traj_f1_list,
    traj_seq_u8_list,
    traj_E_list,
    tail_rec_list,
    tail_f1_list,
    tail_seq_u8_list,
    tail_E_list,
    decode_chain_constraint=None,
    frz_constraint: Optional["FrozenConstraint"] = None,
):
    """
    Write a full single-run output set:
      <prefix>.3dRNAdesign.result
      <prefix>.logo.csv
      <prefix>.tail_hist.csv
      <prefix>.design.csv
      <prefix>.traj.csv
      <prefix>.top_energy.csv (optional)
    """
    rec_percent = float(consensus_rec) * 100.0
    best_rec_percent = float(best_rec) * 100.0

    # (1) result
    result_path = f"{out_prefix}.3dRNAdesign.result"
    with open(result_path, "w") as f:
        f.write("# 3dRNAdesign single-file result (consensus over tail_steps)\n")
        f.write("# Two-stage Rough/Fine TriRNASP (fixed hyperparameters)\n\n")
        f.write(
            f"{pdb_name},{float(E_cons):.6f},{rec_percent:.2f},{float(macro_f1_consensus):.4f},"
            f"{float(ppl):.6f},{float(diversity):.6f},"
            f"{float(E_best_rec):.6f},{best_rec_percent:.2f},"
            f"{float(E_best_f1):.6f},{float(best_macro_f1):.4f}\n"
        )
        f.write("[Consensus]\n" + str(consensus_seq_str) + "\n")
        f.write("[BestRec]\n" + str(best_rec_seq_str) + "\n")
        f.write("[BestMacroF1]\n" + str(best_macro_seq_str) + "\n")

    # MIN-E append
    minE_pack = get_minE_from_tail(
        tail_seq_u8_list,
        tail_E_list,
        tail_rec_list,
        tail_f1_list,
        native_idx=native_idx,
    )
    if minE_pack is not None:
        minE_seq, minE_E, minE_rec, minE_f1 = minE_pack
        with open(result_path, "a") as rf:
            rf.write(f"[MIN-E] E={minE_E:.6f}, Recovery={minE_rec:.4f}, MacroF1={minE_f1:.4f}\n")
            rf.write(f"{minE_seq}\n\n")

    # (2) logo
    counts_np = counts_cpu.numpy()
    N_res = counts_np.shape[1]
    logo_csv = f"{out_prefix}.logo.csv"
    with open(logo_csv, "w") as f_csv:
        header = ["Base"] + [str(i + 1) for i in range(N_res)]
        f_csv.write(",".join(header) + "\n")
        label_order = ["A", "U", "C", "G"]
        for row_idx, basech in enumerate(label_order):
            row_counts = counts_np[row_idx].tolist()
            f_csv.write(",".join([basech] + [f"{float(c):.8f}" for c in row_counts]) + "\n")

    # (3) tail hist
    hist_csv = f"{out_prefix}.tail_hist.csv"
    write_tail_metric_hist_csv(hist_csv, rec_list=tail_rec_list, f1_list=tail_f1_list, n_bins=50)

    # (4) tail design
    design_csv = f"{out_prefix}.design.csv"
    write_design_csv(
        design_csv,
        seq_u8_list=tail_seq_u8_list,
        energy_list=tail_E_list,
        rec_list=tail_rec_list,
        f1_list=tail_f1_list,
        native_idx=native_idx,
    )

    # (5) traj
    traj_csv = f"{out_prefix}.traj.csv"
    write_traj_csv(
        traj_csv,
        step_list=traj_step_list,
        seq_u8_list=traj_seq_u8_list,
        energy_list=traj_E_list,
        rec_list=traj_rec_list,
        f1_list=traj_f1_list,
        native_idx=native_idx,
    )

    # (6) TOP-energy (tail)
    top_groups, top10_cons = topk_unique_energy_and_consensus(
        tail_seq_u8_list=tail_seq_u8_list,
        tail_E_list=tail_E_list,
        tail_rec_list=tail_rec_list,
        tail_f1_list=tail_f1_list,
        k=10,
        energy_decimals=6,
        chain_constraint=decode_chain_constraint,
        frz_constraint=frz_constraint,
    )
    if top_groups:
        top_csv = f"{out_prefix}.top_energy.csv"
        write_top10_energy_csv(top_csv, top_groups, top10_cons, native_idx=native_idx)
        with open(result_path, "a") as rf:
            rf.write("[Top_Energy]\n")
            for i, g in enumerate(top_groups, start=1):
                rep_seq = "".join(np.array(["A","U","C","G"], dtype="<U1")[g["rep_seq_u8"]].tolist())
                rf.write(
                    f"{i:02d} E={float(g['E']):.6f} n={int(g['n_samples'])} repFreq={int(g['rep_freq'])} "
                    f"repRec={float(g['rep_rec']):.4f} repF1={float(g['rep_f1']):.4f}\n"
                )
                rf.write(rep_seq + "\n")
            rf.write("[Top_Consensus]\n")
            rf.write(top10_cons + "\n\n")

    return result_path

def write_top10_energy_csv(csv_path: str, top_groups: List[dict], top10_consensus: str, native_idx: torch.Tensor):
    """
    Write TOP10 unique-energy groups to CSV + the TOP10 consensus sequence.

    Columns:
      rank,energy,n_samples,rep_freq,rep_recovery,rep_macroF1,rep_sequence

    Plus footer line:
      #TOP10_CONSENSUS,<seq>
    """
    base_map = np.array(["A", "U", "C", "G"], dtype="<U1")

    with open(csv_path, "w") as f:
        f.write("rank,energy(kBT),n_samples,rep_freq,rep_recovery,rep_macroF1,rep_sequence\n")
        for i, g in enumerate(top_groups, start=1):
            seq_t = torch.as_tensor(g["rep_seq_u8"], dtype=torch.long)
            seq_str = seq_idx_to_masked_str(seq_t, native_idx)
            f.write(
                f"{i},{g['E']:.6f},{g['n_samples']},{g['rep_freq']},"
                f"{g['rep_rec']:.6f},{g['rep_f1']:.6f},{seq_str}\n"
            )
        f.write(f"#CONSENSUS,{top10_consensus}\n")

BASES = "ATCG" if DNA_MODE else "AUCG"

# ============================================================
# Frozen sequence constraint
# ============================================================
# --frz semantics:
#   A/U/C/G/T : fixed base, T is converted to U
#   -         : free position
#   . or _    : also treated as free for convenience
#   &         : chain separator, ignored in compact residue indexing
# ============================================================

@dataclass
class FrozenConstraint:
    fixed_idx: torch.Tensor      # (N,), long, -1 for free, 0/1/2/3 for A/U/C/G
    fixed_mask: torch.Tensor     # (N,), bool
    fixed_pos: torch.Tensor      # (F,), long fixed positions
    fixed_vals: torch.Tensor     # (F,), long fixed bases
    free_pos: torch.Tensor       # (N-F,), long designable positions
    n_fixed_int: int
    raw: str
    compact: str
    _ss_repair_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_fixed(self) -> int:
        return int(self.n_fixed_int)


_FRZ_BASE_TO_IDX = {
    "A": 0,
    "U": 1,
    "T": 1,
    "C": 2,
    "G": 3,
    "a": 0,
    "u": 1,
    "t": 1,
    "c": 2,
    "g": 3,
}


def _read_frz_text(frz_arg: str) -> str:
    """
    Read a frozen pattern from either a direct CLI string or a file path.

    File format:
      - FASTA header lines starting with '>' are ignored
      - comment lines starting with '#' are ignored
      - remaining non-empty lines are concatenated
    """
    frz_arg = str(frz_arg).strip()

    if os.path.isfile(frz_arg):
        parts = []
        with open(frz_arg, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">") or line.startswith("#"):
                    continue
                parts.append(line)
        return "".join(parts)

    return frz_arg


def build_frz_constraint(
    frz_arg: Optional[str],
    native_len: int,
    device: torch.device,
    label: str = "",
) -> Optional[FrozenConstraint]:
    """Build a frozen sequence constraint in compact residue indexing."""
    if frz_arg is None:
        return None

    raw = _read_frz_text(frz_arg)
    raw_no_space = "".join(ch for ch in raw if not ch.isspace())

    compact_chars = []
    fixed = []

    for ch in raw_no_space:
        if ch == "&":
            continue

        if ch in ("-", ".", "_"):
            compact_chars.append("-")
            fixed.append(-1)
            continue

        if ch in _FRZ_BASE_TO_IDX:
            b = int(_FRZ_BASE_TO_IDX[ch])
            compact_chars.append(BASES[b])
            fixed.append(b)
            continue

        raise ValueError(
            f"[frz] Invalid character {repr(ch)} in --frz pattern. "
            "Allowed: A/U/C/G/T, '-', '.', '_', '&', whitespace."
        )

    if len(fixed) != int(native_len):
        raise ValueError(
            f"[frz] Length mismatch for {label}: "
            f"--frz compact_len={len(fixed)} vs structure N_res={int(native_len)}. "
            "Remember: '&' is ignored, '-' still counts as one residue."
        )

    fixed_np = np.asarray(fixed, dtype=np.int64)
    fixed_idx = torch.as_tensor(fixed_np, dtype=torch.long, device=device)
    fixed_mask = fixed_idx >= 0
    fixed_pos = torch.nonzero(fixed_mask, as_tuple=False).squeeze(1)
    fixed_vals = fixed_idx[fixed_pos]
    free_pos = torch.nonzero(~fixed_mask, as_tuple=False).squeeze(1)
    n_fixed_int = int(fixed_pos.numel())

    fc = FrozenConstraint(
        fixed_idx=fixed_idx,
        fixed_mask=fixed_mask,
        fixed_pos=fixed_pos,
        fixed_vals=fixed_vals,
        free_pos=free_pos,
        n_fixed_int=n_fixed_int,
        raw=raw,
        compact="".join(compact_chars),
    )

    if fc.n_fixed > 0:
        print(
            f"[INFO][frz] enabled for {label}: "
            f"n_fixed={fc.n_fixed}/{int(native_len)}"
        )
        print(f"[INFO][frz] compact mask: {fc.compact}")

    return fc


def apply_frz_constraint_(
    seqs: torch.Tensor,
    frz_constraint: Optional[FrozenConstraint],
) -> torch.Tensor:
    """
    In-place hard projection:
      seqs[..., fixed_positions] = frozen_base
    """
    if frz_constraint is None or frz_constraint.n_fixed <= 0:
        return seqs

    fixed_idx = frz_constraint.fixed_idx
    fixed_pos = frz_constraint.fixed_pos.to(seqs.device)
    fixed_vals = frz_constraint.fixed_vals.to(seqs.device)

    if seqs.shape[-1] != fixed_idx.numel():
        raise ValueError(
            f"[frz] seq length mismatch: seqs_N={seqs.shape[-1]} "
            f"vs frz_N={fixed_idx.numel()}"
        )

    seqs[..., fixed_pos] = fixed_vals
    return seqs


_ALLOWED_PAIR_SET = set(_PAIR_OPTIONS)

_COMPAT_RIGHT = {
    base: [right for left, right in _PAIR_OPTIONS if left == base]
    for base in range(4)
}

_COMPAT_LEFT = {
    base: [left for left, right in _PAIR_OPTIONS if right == base]
    for base in range(4)
}


@dataclass
class _FRZSSRepairPlan:
    left_targets: torch.Tensor
    left_allowed0: torch.Tensor
    left_allowed1: torch.Tensor
    left_has_alt: torch.Tensor
    left_any_alt: bool
    right_targets: torch.Tensor
    right_allowed0: torch.Tensor
    right_allowed1: torch.Tensor
    right_has_alt: torch.Tensor
    right_any_alt: bool


def _make_empty_frz_ss_repair_plan(device: torch.device) -> _FRZSSRepairPlan:
    empty_long = torch.empty((0,), dtype=torch.long, device=device)
    empty_bool = torch.empty((0,), dtype=torch.bool, device=device)
    return _FRZSSRepairPlan(
        left_targets=empty_long,
        left_allowed0=empty_long,
        left_allowed1=empty_long,
        left_has_alt=empty_bool,
        left_any_alt=False,
        right_targets=empty_long,
        right_allowed0=empty_long,
        right_allowed1=empty_long,
        right_has_alt=empty_bool,
        right_any_alt=False,
    )


def _get_ss_pairs_list(ss_constraint):
    if ss_constraint is None:
        return []

    pairs = getattr(ss_constraint, "pairs", None)
    if pairs:
        return [(int(i), int(j)) for i, j in pairs]

    pi = getattr(ss_constraint, "pairs_i", None)
    pj = getattr(ss_constraint, "pairs_j", None)
    if pi is not None and pj is not None:
        pi_cpu = pi.detach().cpu().tolist()
        pj_cpu = pj.detach().cpu().tolist()
        return [(int(i), int(j)) for i, j in zip(pi_cpu, pj_cpu)]

    return []


def _get_frz_ss_repair_plan(
    frz_constraint: FrozenConstraint,
    ss_constraint,
    device: torch.device,
) -> _FRZSSRepairPlan:
    """
    Build/cache a GPU-friendly repair plan for --frz + --ss.

    The previous implementation checked every pair with .item() on GPU each
    proposal, which serialized the hot loop. This plan moves pair inspection
    to one cached CPU pass, then repairs all affected positions with batched
    tensor ops on the proposal device.
    """
    if ss_constraint is None:
        return _make_empty_frz_ss_repair_plan(device)

    dev_key = (device.type, None if device.index is None else int(device.index))
    cache_key = (id(ss_constraint), dev_key)
    cached = frz_constraint._ss_repair_cache.get(cache_key)
    if cached is not None:
        return cached

    pairs = _get_ss_pairs_list(ss_constraint)
    if not pairs:
        plan = _make_empty_frz_ss_repair_plan(device)
        frz_constraint._ss_repair_cache[cache_key] = plan
        return plan

    N = int(frz_constraint.fixed_idx.numel())
    fixed_idx_cpu = frz_constraint.fixed_idx.detach().cpu().numpy()
    fixed_mask_cpu = frz_constraint.fixed_mask.detach().cpu().numpy()

    left_targets = []
    left_allowed0 = []
    left_allowed1 = []
    left_has_alt = []

    right_targets = []
    right_allowed0 = []
    right_allowed1 = []
    right_has_alt = []

    for i, j in pairs:
        if not (0 <= i < N and 0 <= j < N):
            continue

        fi = bool(fixed_mask_cpu[i])
        fj = bool(fixed_mask_cpu[j])

        if not fi and not fj:
            continue

        bi = int(fixed_idx_cpu[i]) if fi else -1
        bj = int(fixed_idx_cpu[j]) if fj else -1

        if fi and fj:
            if (bi, bj) not in _ALLOWED_PAIR_SET:
                raise ValueError(
                    f"[frz] Frozen bases conflict with --ss pair ({i+1},{j+1}): "
                    f"{BASES[bi]}-{BASES[bj]} is not in {_PAIR_DESCRIPTION}."
                )
            continue

        if fi and not fj:
            allowed = _COMPAT_RIGHT[bi]
            left_targets.append(int(j))
            left_allowed0.append(int(allowed[0]))
            left_allowed1.append(int(allowed[-1]))
            left_has_alt.append(len(allowed) > 1)
            continue

        if fj and not fi:
            allowed = _COMPAT_LEFT[bj]
            right_targets.append(int(i))
            right_allowed0.append(int(allowed[0]))
            right_allowed1.append(int(allowed[-1]))
            right_has_alt.append(len(allowed) > 1)

    def _long(xs):
        return torch.as_tensor(xs, dtype=torch.long, device=device)

    def _bool(xs):
        return torch.as_tensor(xs, dtype=torch.bool, device=device)

    plan = _FRZSSRepairPlan(
        left_targets=_long(left_targets),
        left_allowed0=_long(left_allowed0),
        left_allowed1=_long(left_allowed1),
        left_has_alt=_bool(left_has_alt),
        left_any_alt=any(left_has_alt),
        right_targets=_long(right_targets),
        right_allowed0=_long(right_allowed0),
        right_allowed1=_long(right_allowed1),
        right_has_alt=_bool(right_has_alt),
        right_any_alt=any(right_has_alt),
    )
    frz_constraint._ss_repair_cache[cache_key] = plan
    return plan


def _repair_frz_partner_block_(
    flat: torch.Tensor,
    targets: torch.Tensor,
    allowed0: torch.Tensor,
    allowed1: torch.Tensor,
    has_alt: torch.Tensor,
    any_alt: bool,
    generator: Optional[torch.Generator] = None,
):
    if targets.numel() <= 0:
        return

    cur = flat[:, targets]
    ok = (cur == allowed0.unsqueeze(0))
    ok = ok | (has_alt.unsqueeze(0) & (cur == allowed1.unsqueeze(0)))

    if any_alt:
        pick_alt = torch.randint(
            0,
            2,
            size=cur.shape,
            device=flat.device,
            generator=generator,
            dtype=torch.long,
        ).bool()
        replacement = torch.where(
            has_alt.unsqueeze(0) & pick_alt,
            allowed1.unsqueeze(0),
            allowed0.unsqueeze(0),
        )
    else:
        replacement = allowed0.unsqueeze(0)
    flat[:, targets] = torch.where(ok, cur, replacement)


def enforce_frz_and_repair_ss_(
    seqs: torch.Tensor,
    frz_constraint: Optional[FrozenConstraint],
    ss_constraint=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Hard-enforce frozen bases, then repair the non-frozen partner
    when a frozen base participates in an SS pair.
    """
    if frz_constraint is None or frz_constraint.n_fixed <= 0:
        return seqs

    apply_frz_constraint_(seqs, frz_constraint)

    if ss_constraint is None:
        return seqs

    n_pairs = getattr(ss_constraint, "n_pairs", None)
    if n_pairs is not None and int(n_pairs) <= 0:
        return seqs

    N = int(seqs.shape[-1])
    flat = seqs.reshape(-1, N)

    plan = _get_frz_ss_repair_plan(
        frz_constraint=frz_constraint,
        ss_constraint=ss_constraint,
        device=seqs.device,
    )
    _repair_frz_partner_block_(
        flat,
        plan.left_targets,
        plan.left_allowed0,
        plan.left_allowed1,
        plan.left_has_alt,
        plan.left_any_alt,
        generator=generator,
    )
    _repair_frz_partner_block_(
        flat,
        plan.right_targets,
        plan.right_allowed0,
        plan.right_allowed1,
        plan.right_has_alt,
        plan.right_any_alt,
        generator=generator,
    )

    apply_frz_constraint_(seqs, frz_constraint)
    return seqs


def get_valid_eval_mask(native_idx: torch.Tensor) -> torch.Tensor:
    """
    native_idx: (N,), valid bases in {0,1,2,3}, unknown = -1
    return: bool mask (N,)
    """
    if not isinstance(native_idx, torch.Tensor):
        native_idx = torch.as_tensor(native_idx, dtype=torch.long)
    if native_idx.dim() != 1:
        raise ValueError(f"native_idx must be 1D, got shape={tuple(native_idx.shape)}")
    return (native_idx >= 0)


def count_valid_eval_sites(native_idx: torch.Tensor) -> int:
    if not isinstance(native_idx, torch.Tensor):
        native_idx = torch.as_tensor(native_idx, dtype=torch.long)
    return int((native_idx >= 0).sum().item())


def format_seq_with_native_mask(
    seq_idx: torch.Tensor,
    native_idx: torch.Tensor,
    unknown_char: str = "-",
) -> str:
    """
    Output formatter:
      if native_idx[i] < 0 -> show '-'
      else show designed base seq_idx[i] as A/U/C/G
    """
    if not isinstance(seq_idx, torch.Tensor):
        seq_idx = torch.as_tensor(seq_idx, dtype=torch.long)
    if not isinstance(native_idx, torch.Tensor):
        native_idx = torch.as_tensor(native_idx, dtype=torch.long)

    if seq_idx.dim() == 2:
        if seq_idx.size(0) != 1:
            raise ValueError(f"seq_idx must be (N,) or (1,N), got {tuple(seq_idx.shape)}")
        seq_idx = seq_idx[0]

    if native_idx.device != seq_idx.device:
        native_idx = native_idx.to(seq_idx.device)

    if seq_idx.shape != native_idx.shape:
        raise ValueError(f"shape mismatch: seq_idx={tuple(seq_idx.shape)} native_idx={tuple(native_idx.shape)}")

    seq = seq_idx.detach().cpu().tolist()
    nat = native_idx.detach().cpu().tolist()

    out = []
    for s, n in zip(seq, nat):
        if int(n) < 0:
            out.append(unknown_char)
        else:
            s = int(s)
            if 0 <= s < 4:
                out.append(BASES[s])
            else:
                out.append("N")
    return "".join(out)


def format_native_with_mask(native_idx: torch.Tensor, unknown_char: str = "-") -> str:
    """
    Native display formatter:
      valid {0,1,2,3} -> A/U/C/G
      unknown (<0)    -> '-'
    """
    if not isinstance(native_idx, torch.Tensor):
        native_idx = torch.as_tensor(native_idx, dtype=torch.long)

    nat = native_idx.detach().cpu().tolist()
    out = []
    for n in nat:
        n = int(n)
        if n < 0:
            out.append(unknown_char)
        elif 0 <= n < 4:
            out.append(BASES[n])
        else:
            out.append("N")
    return "".join(out)


def seq_idx_to_masked_str(seq_idx: torch.Tensor, native_idx: torch.Tensor) -> str:
    return format_seq_with_native_mask(seq_idx, native_idx, unknown_char="-")

def seq_idx_to_str(seq_idx: torch.Tensor) -> str:
    """
    Convert sequence indices (0,1,2,3) to an A/U/C/G string.
    """
    if seq_idx.dim() == 2:
        assert seq_idx.size(0) == 1
        seq_idx = seq_idx[0]
    seq_list = seq_idx.detach().cpu().tolist()
    return "".join(IDX_TO_BASE[i] for i in seq_list)

def fine_energy_of_seq(
    seq_idx: torch.Tensor,
    scorer: "TriRNASP_Scorer",
    norm_factor: float = 1.0,
) -> float:
    """
    Return Fine energy for one sequence.
    For multi-state merged scorer, pass norm_factor to normalize.
    """
    if seq_idx.dim() == 1:
        x = seq_idx.unsqueeze(0)
    else:
        x = seq_idx
    with torch.inference_mode():
        E = scorer.score_batch(x, stage="fine", with_kl=False)[0].item()
    return float(E) / float(norm_factor)

def seq_recovery(seq_idx: torch.Tensor, native_idx: torch.Tensor) -> float:
    """
    Recovery on valid target-native positions only.
    """
    if not isinstance(seq_idx, torch.Tensor):
        seq_idx = torch.as_tensor(seq_idx, dtype=torch.long)
    if not isinstance(native_idx, torch.Tensor):
        native_idx = torch.as_tensor(native_idx, dtype=torch.long)

    if seq_idx.dim() == 2:
        if seq_idx.size(0) != 1:
            raise ValueError(f"seq_idx must be (N,) or (1,N), got {tuple(seq_idx.shape)}")
        seq_idx = seq_idx[0]

    if native_idx.device != seq_idx.device:
        native_idx = native_idx.to(seq_idx.device)

    if seq_idx.shape != native_idx.shape:
        raise ValueError(f"shape mismatch: seq_idx={tuple(seq_idx.shape)} native_idx={tuple(native_idx.shape)}")

    mask = (native_idx >= 0)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return float("nan")

    matches = (seq_idx[mask] == native_idx[mask]).sum().item()
    return matches / float(n_valid)


def macro_f1_from_idx(
    pred_idx: torch.Tensor,
    true_idx: torch.Tensor,
    num_classes: int = 4,
    ignore_empty_classes: bool = True,
) -> float:
    """
    Macro-F1 over valid positions only (true_idx >= 0).
    """
    if not isinstance(pred_idx, torch.Tensor):
        pred_idx = torch.as_tensor(pred_idx, dtype=torch.long)
    if not isinstance(true_idx, torch.Tensor):
        true_idx = torch.as_tensor(true_idx, dtype=torch.long)

    if pred_idx.dim() == 2:
        if pred_idx.size(0) != 1:
            raise ValueError(f"pred_idx must be (N,) or (1,N), got {tuple(pred_idx.shape)}")
        pred_idx = pred_idx[0]

    if true_idx.device != pred_idx.device:
        true_idx = true_idx.to(pred_idx.device)

    if pred_idx.shape != true_idx.shape:
        raise ValueError(f"shape mismatch: pred_idx={tuple(pred_idx.shape)} true_idx={tuple(true_idx.shape)}")

    mask = (true_idx >= 0)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return float("nan")

    pred = pred_idx[mask].detach().cpu().numpy().astype(int)
    true = true_idx[mask].detach().cpu().numpy().astype(int)

    f1_list = []
    for c in range(num_classes):
        support = (true == c).sum()
        if ignore_empty_classes and support == 0:
            continue

        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / float(tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        f1_list.append(f1)

    if not f1_list:
        return float("nan")
    return float(sum(f1_list) / len(f1_list))


def perplexity_from_counts(counts: torch.Tensor, native_idx: torch.Tensor) -> float:
    """
    counts: (4, N)
    native_idx: (N,)

    PPL does NOT use native base identity as labels,
    but it DOES mask out positions where native_idx < 0.
    """
    if not isinstance(counts, torch.Tensor):
        counts = torch.as_tensor(counts, dtype=torch.float32)
    if not isinstance(native_idx, torch.Tensor):
        native_idx = torch.as_tensor(native_idx, dtype=torch.long)

    if counts.dim() != 2 or counts.shape[0] != 4:
        raise ValueError(f"counts must be (4,N), got {tuple(counts.shape)}")

    if native_idx.device != counts.device:
        native_idx = native_idx.to(counts.device)

    if counts.shape[1] != native_idx.shape[0]:
        raise ValueError(f"length mismatch: counts N={counts.shape[1]} native N={native_idx.shape[0]}")

    mask = (native_idx >= 0)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return float("nan")

    counts_f = counts.float()[:, mask]
    col_sum = counts_f.sum(dim=0, keepdim=True)
    eps = 1e-9

    probs = (counts_f + eps) / (col_sum + 4 * eps)

    # native-free definition, but only on valid target positions
    consensus_probs = probs.max(dim=0).values
    ppl = torch.exp(-torch.log(consensus_probs + eps).mean()).item()
    return float(ppl)

def logo4xN_from_tail_boltzmann(
    tail_seq_u8_list: List[np.ndarray],
    tail_E_list: List[float],
    N_res: int,
    clip_deltaE: float = 100.0,
    Beta_en: float = 0.0,
    dedup: bool = True,  # False : F = E - TS ; True : only F = E (adjust Beta for best); Beta = 0 , only entropy F = -TS
    ) -> torch.Tensor:
    """
    Build tail partition function Z_tail and return Boltzmann-normalized logo probs.
    Output: torch.float32 (4, N_res), each column sums to ~1.
    """
    if tail_seq_u8_list is None or len(tail_seq_u8_list) == 0:
        return torch.zeros((4, N_res), dtype=torch.float32)

    X = np.stack(tail_seq_u8_list, axis=0).astype(np.uint8, copy=False)  # (M,N)
    if X.size == 0:
        return torch.zeros((4, N_res), dtype=torch.float32)

    E = np.asarray(tail_E_list, dtype=np.float64)
    if E.size != X.shape[0]:
        raise ValueError(f"tail_E_list length {E.size} != tail_seq size {X.shape[0]}")

    # filter invalid rows if any
    valid = (X <= 3).all(axis=1) & np.isfinite(E)
    X = X[valid]
    E = E[valid]
    if X.shape[0] == 0:
        return torch.zeros((4, N_res), dtype=torch.float32)

    # optional exact dedup: merge same sequence by summing weights
    if dedup:
        beta = Beta_en

        # 1) per unique sequence: keep ONE representative energy (min)
        Erep = {}  # bytes -> minE
        for row, Ei in zip(X, E):
            b = row.tobytes()
            prev = Erep.get(b, None)
            if prev is None or float(Ei) < prev:
                Erep[b] = float(Ei)

        # 2) global reference Emin over unique states (for numerical stability)
        Emin_u = min(Erep.values())
        # 3) one weight per unique sequence (no multiplicity)
        keys = list(Erep.keys())
        X_uniq = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), N_res)

        dE = np.clip(np.array([Erep[b] - Emin_u for b in keys], dtype=np.float64), 0.0, float(clip_deltaE))
        w = np.exp(-beta * dE)
    else:
        Emin = float(E.min())
        beta = Beta_en
        dE = np.clip(E - Emin, 0.0, float(clip_deltaE))
        w = np.exp(-beta * dE)  # unnormalized weights
        X_uniq = X

    Z = float(w.sum())
    if Z <= 0 or (not np.isfinite(Z)):
        # fallback to uniform weights
        w = np.ones_like(w, dtype=np.float64)
        Z = float(w.sum())

    p = w / Z  # normalized over tail

    probs = np.zeros((4, N_res), dtype=np.float64)
    idx_cols = np.arange(N_res, dtype=np.int64)
    for s_u8, ps in zip(X_uniq, p):
        s = s_u8.astype(np.int64, copy=False)
        np.add.at(probs, (s, idx_cols), float(ps))

    return torch.from_numpy(probs.astype(np.float32, copy=False)) 

def diversity_from_tail_dedup_sequences(
    tail_seq_u8_list: List[np.ndarray],
    native_idx: torch.Tensor,
) -> float:
    if tail_seq_u8_list is None or len(tail_seq_u8_list) == 0:
        return 0.0

    if not isinstance(native_idx, torch.Tensor):
        native_idx = torch.as_tensor(native_idx, dtype=torch.long)

    mask = (native_idx.detach().cpu().numpy() >= 0)

    X = np.stack(tail_seq_u8_list, axis=0).astype(np.uint8, copy=False)
    if X.size == 0:
        return 0.0

    valid_rows = (X <= 3).all(axis=1)
    X = X[valid_rows]
    if X.shape[0] == 0:
        return 0.0

    X = X[:, mask]
    if X.shape[1] == 0:
        return float("nan")

    X_uniq = np.unique(X, axis=0)
    M, N = X_uniq.shape
    if M <= 1:
        return 0.0

    counts = np.zeros((4, N), dtype=np.float64)
    for b in range(4):
        counts[b, :] = (X_uniq == b).sum(axis=0)

    p = counts / float(M)
    div_i = 1.0 - np.sum(p * p, axis=0)
    return float(np.mean(div_i))
    
_BANNER_PRINTED = False

def print_trirnade_banner():
    global _BANNER_PRINTED
    if _BANNER_PRINTED:
        return
    _BANNER_PRINTED = True
    print(TRIRNADE_BANNER)

def _hist_0_1(values: List[float], n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histogram on [0,1], n_bins bins.
    Returns:
        centers: (n_bins,)
        counts : (n_bins,)
    """
    if not values:
        centers = (np.arange(n_bins) + 0.5) / n_bins
        return centers, np.zeros(n_bins, dtype=np.int64)

    v = np.asarray(values, dtype=float)
    v = np.clip(v, 0.0, 1.0)

    # bins edges: 0..1 with n_bins
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
    cnt, _ = np.histogram(v, bins=edges)
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers, cnt


def write_tail_metric_hist_csv(
    csv_path: str,
    rec_list: List[float],
    f1_list: List[float],
    n_bins: int = 50,
):
    """
    Write histograms of tail-window Recovery and MacroF1 into one CSV.

    Columns:
      bin_center, rec_count, rec_prob, f1_count, f1_prob
    """
    centers_r, cnt_r = _hist_0_1(rec_list, n_bins=n_bins)
    centers_f, cnt_f = _hist_0_1(f1_list, n_bins=n_bins)
    assert np.allclose(centers_r, centers_f)

    total_r = max(int(cnt_r.sum()), 1)
    total_f = max(int(cnt_f.sum()), 1)
    prob_r = cnt_r.astype(float) / total_r
    prob_f = cnt_f.astype(float) / total_f

    with open(csv_path, "w") as f:
        f.write("bin_center,rec_count,rec_prob,f1_count,f1_prob\n")
        for i in range(n_bins):
            f.write(
                f"{centers_r[i]:.6f},"
                f"{int(cnt_r[i])},{prob_r[i]:.8f},"
                f"{int(cnt_f[i])},{prob_f[i]:.8f}\n"
            )

def write_design_csv(
    csv_path: str,
    seq_u8_list: List[np.ndarray],
    energy_list: List[float],
    rec_list: List[float],
    f1_list: List[float],
    native_idx: torch.Tensor,
):
    """
    Write tail-window per-step records to one CSV, written ONCE at the end.

    Columns:
      sequence, energy, recovery, macroF1
    """
    assert len(seq_u8_list) == len(energy_list) == len(rec_list) == len(f1_list)

    # Fast idx->base map
    base_map = np.array(["A", "U", "C", "G"], dtype="<U1")

    with open(csv_path, "w") as f:
        f.write("sequence,energy,recovery,macroF1\n")
        for s_u8, E, r, f1 in zip(seq_u8_list, energy_list, rec_list, f1_list):
            
            seq_t = torch.as_tensor(s_u8, dtype=torch.long)
            seq_str = seq_idx_to_masked_str(seq_t, native_idx)
            f.write(f"{seq_str},{E:.6f},{r:.6f},{f1:.6f}\n")

def write_traj_csv(
    csv_path: str,
    step_list: List[int],
    seq_u8_list: List[np.ndarray],
    energy_list: List[float],
    rec_list: List[float],
    f1_list: List[float],
    native_idx: torch.Tensor,
):
    """
    Write whole-trajectory records to CSV.

    Columns:
      step, sequence, energy, recovery, macroF1
    """
    assert len(step_list) == len(seq_u8_list) == len(energy_list) == len(rec_list) == len(f1_list)

    base_map = np.array(["A", "U", "C", "G"], dtype="<U1")

    with open(csv_path, "w") as f:
        f.write("step,sequence,energy(kBT),recovery,macroF1\n")
        for st, s_u8, E, r, f1 in zip(step_list, seq_u8_list, energy_list, rec_list, f1_list):
            seq_t = torch.as_tensor(s_u8, dtype=torch.long)
            seq_str = seq_idx_to_masked_str(seq_t, native_idx)
            f.write(f"{int(st)},{seq_str},{float(E):.6f},{float(r):.6f},{float(f1):.6f}\n")

def get_minE_from_tail(
    tail_seq_u8_list: list,   # List[np.ndarray]
    tail_E_list: list,        # List[float]
    tail_rec_list: list,      # List[float]
    tail_f1_list: list,       # List[float]
    native_idx: torch.Tensor,
):
    """
    Return the minimum-energy sample from tail window:
      (seq_str, minE, rec, macroF1)
    """
    n = len(tail_E_list)
    if n == 0:
        return None

    min_i = 0
    minE = tail_E_list[0]
    for i in range(1, n):
        Ei = tail_E_list[i]
        if Ei < minE:
            minE = Ei
            min_i = i

    base_map = np.array(["A", "U", "C", "G"], dtype="<U1")
    seq_t = torch.as_tensor(tail_seq_u8_list[min_i], dtype=torch.long)
    seq_str = seq_idx_to_masked_str(seq_t, native_idx)
    rec = float(tail_rec_list[min_i])
    f1  = float(tail_f1_list[min_i])
    return seq_str, float(minE), rec, f1


def propose_batch(
    center_seq: torch.Tensor,
    batch_size: int,
    p_mut: float,
    device: torch.device,
    ss_constraint=None,
    ss_fast: Optional["SSProposalFast"] = None,
    generator: Optional[torch.Generator] = None,
    frz_constraint: Optional["FrozenConstraint"] = None,
) -> torch.Tensor:
    B = int(batch_size)
    N = int(center_seq.numel())

    seqs = center_seq.unsqueeze(0).expand(B, N).clone()
    enforce_frz_and_repair_ss_(
        seqs,
        frz_constraint=frz_constraint,
        ss_constraint=ss_constraint,
        generator=generator,
    )

    if p_mut <= 0.0:
        return seqs

    # original mode
    if ss_constraint is None or ss_constraint.n_pairs == 0:
        u = torch.rand((B, N), device=device, generator=generator)
        mut_mask = (u < p_mut)
        if frz_constraint is not None and frz_constraint.n_fixed > 0:
            mut_mask[:, frz_constraint.fixed_pos.to(device)] = False

        if not mut_mask.any():
            enforce_frz_and_repair_ss_(
                seqs,
                frz_constraint=frz_constraint,
                ss_constraint=ss_constraint,
                generator=generator,
            )
            return seqs

        old = seqs
        rand3 = torch.randint(0, 3, size=seqs.shape, device=device, generator=generator, dtype=torch.long)
        new_val = rand3 + (rand3 >= old).long()
        seqs[mut_mask] = new_val[mut_mask]
        enforce_frz_and_repair_ss_(
            seqs,
            frz_constraint=frz_constraint,
            ss_constraint=ss_constraint,
            generator=generator,
        )
        return seqs

    # Vectorized hard-SS path
    if ss_fast is None:
        raise ValueError("propose_batch: ss_fast is required when ss_constraint is enabled")

    seqs = _propose_one_center_ss_fast(
        center_seq=center_seq,
        batch_size=batch_size,
        p_mut=p_mut,
        device=device,
        ss_fast=ss_fast,
        generator=generator,
        force_one_mut=False,
    )
    enforce_frz_and_repair_ss_(
        seqs,
        frz_constraint=frz_constraint,
        ss_constraint=ss_constraint,
        generator=generator,
    )
    return seqs
    
def propose_batch_multi(
    center_seqs: torch.Tensor,   # (K, N)
    batch_size: int,
    p_mut: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Vectorized propose for K chains at once.

    Returns:
        cand: (K, B, N)
    """
    assert center_seqs.dim() == 2
    K, N = center_seqs.shape
    B = int(batch_size)

    cand = center_seqs.unsqueeze(1).expand(K, B, N).clone()  # (K,B,N)

    if p_mut <= 0.0:
        return cand

    u = torch.rand((K, B, N), device=device)
    mut_mask = (u < float(p_mut))
    if not mut_mask.any():
        return cand

    rand2 = torch.randint(0, 3, size=(K, B, N), device=device)
    old = cand
    new_val = rand2 + (rand2 >= old).long()  # avoid original base
    cand[mut_mask] = new_val[mut_mask]
    return cand
    
def _sample_allowed_pairs_excluding_current(
    old_left: torch.Tensor,
    old_right: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample allowed pair states for given old pair states.
    Prefer changing to a *different* allowed pair when old pair is already legal.

    old_left / old_right:
        same shape, dtype long, values in {0,1,2,3}

    Returns:
        new_pair: (..., 2) long
    """
    device = old_left.device

    allowed_pairs = torch.tensor(
        _PAIR_OPTIONS,
        dtype=torch.long,
        device=device,
    )
    n_allowed = int(allowed_pairs.shape[0])

    pair_to_idx = torch.full((16,), -1, dtype=torch.long, device=device)
    for pair_idx, (left, right) in enumerate(_PAIR_OPTIONS):
        pair_to_idx[left * 4 + right] = pair_idx

    old_code = old_left * 4 + old_right
    old_idx = pair_to_idx[old_code]   # same shape, -1 if old pair not in allowed set

    new_idx = torch.empty_like(old_idx)

    valid_old = (old_idx >= 0)
    if valid_old.any():
        # Choose among all legal pairs except the current one.
        alternatives = n_allowed - 1
        random_alt = torch.randint(
            0, alternatives,
            size=old_idx[valid_old].shape,
            device=device,
            generator=generator,
        )
        new_idx_valid = random_alt + (random_alt >= old_idx[valid_old]).long()
        new_idx[valid_old] = new_idx_valid

    if (~valid_old).any():
        # old pair illegal -> choose any allowed pair
        random_pair = torch.randint(
            0, n_allowed,
            size=old_idx[~valid_old].shape,
            device=device,
            generator=generator,
        )
        new_idx[~valid_old] = random_pair

    return allowed_pairs[new_idx]  # (...,2)
# ======================================================================
# Two-stage scoring: single state
# ======================================================================

def score_two_stage_single(
    seq_batch: torch.Tensor,
    scorer: TriRNASP_Scorer,
    top_k: int,
    kl_lambda: float,
    n_dir: int = 22,
    beta_rough: float = 0.5,
    mid_thermo=None,
    mid_top_k: int = MID_THERMO_TOP_K,
    frz_constraint: Optional["FrozenConstraint"] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two-stage scoring for a batch of sequences on a single structure.

    New scheme:
      1) Rough(+KL) on all proposals
      2) Keep rough top-k anchors
      3) Build directional proposals from Boltzmann-weighted marginals
      4) Fine rerank anchors + directional proposals together

    Returns:
      fine_sorted_vals : (k+n_dir,)
      rough_sorted_vals: (k+n_dir,)
      seq_sorted       : (k+n_dir, N)
      idx_sorted       : original seq_batch idx for anchors; -1 for directional
    """
    B, N = seq_batch.shape
    k = min(int(top_k), int(B))

    # 1) Rough(+KL)
    E_rough_used = scorer.score_batch(
        seq_batch,
        stage="rough",
        with_kl=False,
        kl_lambda=float(kl_lambda),
    )  # (B,)

    # anchors
    rough_top_vals, rough_top_idx = torch.topk(
        E_rough_used, k=k, largest=False
    )  # (k,)
    seq_anchor = seq_batch[rough_top_idx]  # (k, N)

    # directional proposals
    seq_dir = build_directional_proposals_from_rough(
        seq_batch=seq_batch,
        E_rough_used=E_rough_used,
        n_dir=int(n_dir),
        beta_rough=float(beta_rough),
    )  # (n_dir, N)
    apply_frz_constraint_(seq_dir, frz_constraint)

    if seq_dir.shape[0] > 0:
        E_dir_rough_used = scorer.score_batch(
            seq_dir,
            stage="rough",
            with_kl=False,
            kl_lambda=float(kl_lambda),
        )  # (n_dir,)
    else:
        E_dir_rough_used = torch.empty(
            (0,),
            dtype=E_rough_used.dtype,
            device=E_rough_used.device,
        )

    # fine on anchors + directional
    seq_fine_in = torch.cat([seq_anchor, seq_dir], dim=0)   # (k+n_dir, N)
    apply_frz_constraint_(seq_fine_in, frz_constraint)
    rough_fine_in = torch.cat([rough_top_vals, E_dir_rough_used], dim=0)

    idx_fine_in = torch.cat([
        rough_top_idx,
        torch.full(
            (seq_dir.shape[0],),
            -1,
            dtype=rough_top_idx.dtype,
            device=rough_top_idx.device,
        )
    ], dim=0)

    # seq-prior gate ONLY inside this candidate pool
    if USE_MID_THERMO and (mid_thermo is not None) and (mid_top_k is not None):
        M = int(seq_fine_in.shape[0])
        k_mid = min(int(mid_top_k), M)
        if k_mid < M:
            E_mid = mid_thermo.score_batch(seq_fine_in)   # (M,)
            _, keep_idx = torch.topk(E_mid, k=k_mid, largest=False)
            seq_fine_in = seq_fine_in[keep_idx]
            rough_fine_in = rough_fine_in[keep_idx]
            idx_fine_in = idx_fine_in[keep_idx]

    E_fine_used = scorer.score_batch(
        seq_fine_in,
        stage="fine",
        with_kl=False,
    )  # (M_keep,)
    
    fine_sorted_vals, order = torch.sort(E_fine_used, descending=False)
    rough_sorted_vals = rough_fine_in[order]
    seq_sorted = seq_fine_in[order]
    idx_sorted = idx_fine_in[order]

    return fine_sorted_vals, rough_sorted_vals, seq_sorted, idx_sorted

def score_two_stage_single_grouped(
    cand_batch: torch.Tensor,
    scorer: TriRNASP_Scorer,
    top_k: int,
    kl_lambda: float,
    n_dir: int = 22,
    beta_rough: float = 0.5,
    generators: Optional[List[torch.Generator]] = None,
    mid_thermo=None,
    mid_top_k: int = MID_THERMO_TOP_K,
    frz_constraint: Optional["FrozenConstraint"] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two-stage scoring per chain (grouped by K), with:
      top-k anchors + directional proposals -> Fine competition.

    Returns:
        best_E:   (K,)  best fine energy per chain
        best_seq: (K,N) best sequence per chain
    """
    assert cand_batch.dim() == 3
    K, B, N = cand_batch.shape
    k = min(int(top_k), int(B))

    device = cand_batch.device
    row_idx = torch.arange(K, device=device)

    # ---- Rough for all original candidates in one big batch (K*B) ----
    flat = cand_batch.reshape(K * B, N)  # (K*B, N)
    E_rough_flat = scorer.score_batch(
        flat,
        stage="rough",
        with_kl=False,
        kl_lambda=float(kl_lambda),
    )  # (K*B,)
    E_rough = E_rough_flat.reshape(K, B)  # (K,B)

    # ---- Top-k anchors per chain ----
    rough_top_vals, rough_top_idx = torch.topk(E_rough, k=k, largest=False, dim=1)  # (K,k)
    idx_exp = rough_top_idx.unsqueeze(-1).expand(K, k, N)
    seq_anchor = torch.gather(cand_batch, dim=1, index=idx_exp)  # (K,k,N)

    # ---- Directional proposals per chain ----
    probs_dir = build_directional_probs_grouped_from_rough(
        cand_batch=cand_batch,      # (K,B,N)
        E_rough_used=E_rough,       # (K,B)
        beta_rough=float(beta_rough),
    )  # (K,N,4)

    seq_dir = sample_directional_from_grouped_probs(
        probs=probs_dir,
        n_dir=int(n_dir),
        generators=generators,
        dtype_out=cand_batch.dtype,
    )  # (K,n_dir,N)
    apply_frz_constraint_(seq_dir, frz_constraint)

    # ---- Rough score directional proposals ----
    if seq_dir.shape[1] > 0:
        dir_flat = seq_dir.reshape(K * int(n_dir), N)
        E_dir_rough_flat = scorer.score_batch(
            dir_flat,
            stage="rough",
            with_kl=False,
            kl_lambda=float(kl_lambda),
        )  # (K*n_dir,)
        E_dir_rough = E_dir_rough_flat.reshape(K, int(n_dir))  # (K,n_dir)
    else:
        E_dir_rough = torch.empty((K, 0), dtype=E_rough.dtype, device=device)

    # ---- Fine competition on anchors + directional ----
    seq_fine_in = torch.cat([seq_anchor, seq_dir], dim=1)  # (K, M, N)
    apply_frz_constraint_(seq_fine_in, frz_constraint)
    rough_fine_in = torch.cat([rough_top_vals, E_dir_rough], dim=1)  # (K, M)
    M = int(seq_fine_in.shape[1])

    # seq-prior gate PER CHAIN, not across K
    if USE_MID_THERMO and (mid_thermo is not None) and (mid_top_k is not None):
        k_mid = min(int(mid_top_k), M)
        if k_mid < M:
            E_mid_flat = mid_thermo.score_batch(seq_fine_in.reshape(K * M, N))  # (K*M,)
            E_mid = E_mid_flat.reshape(K, M)

            keep_idx = torch.topk(E_mid, k=k_mid, largest=False, dim=1).indices
            keep_idx_exp = keep_idx.unsqueeze(-1).expand(K, k_mid, N)

            seq_fine_in = torch.gather(seq_fine_in, dim=1, index=keep_idx_exp)
            rough_fine_in = torch.gather(rough_fine_in, dim=1, index=keep_idx)
            M = k_mid

    fine_flat = seq_fine_in.reshape(K * M, N)

    E_fine_flat = scorer.score_batch(
        fine_flat,
        stage="fine",
        with_kl=False,
    )  # (K*M,)
    E_fine = E_fine_flat.reshape(K, M)  # (K,M)

    best_pos = torch.argmin(E_fine, dim=1)  # (K,)
    best_E = E_fine[row_idx, best_pos]      # (K,)
    best_E_rough = rough_fine_in[row_idx, best_pos]   # <<< NEW
    best_seq = seq_fine_in[row_idx, best_pos, :]      # (K,N)

    return best_E, best_E_rough, best_seq

def make_chain_generators(seeds: List[int], device: torch.device) -> List[torch.Generator]:
    gens = []
    dev_type = device.type
    for s in seeds:
        g = torch.Generator(device=dev_type)
        g.manual_seed(int(s))
        gens.append(g)
    return gens

@dataclass
class SSProposalFast:
    pair_i: torch.Tensor         # (Kp,)
    pair_j: torch.Tensor         # (Kp,)
    unpaired_idx: torch.Tensor   # (Nu,)
    pair_alt_pairs: torch.Tensor # (16, n_allowed, 2)
    pair_alt_count: torch.Tensor # (16,)

def build_ss_proposal_fast(
    ss_constraint,
    N_res: int,
    device: torch.device,
) -> Optional[SSProposalFast]:
    """
    Build once, use many times.
    Only for hard-SS proposal acceleration.
    """
    if ss_constraint is None or ss_constraint.n_pairs == 0:
        return None

    pair_i = ss_constraint.pairs_i.to(device)
    pair_j = ss_constraint.pairs_j.to(device)

    paired_mask_1d = torch.zeros((N_res,), dtype=torch.bool, device=device)
    paired_mask_1d[pair_i] = True
    paired_mask_1d[pair_j] = True
    unpaired_idx = torch.nonzero(~paired_mask_1d, as_tuple=False).squeeze(1)

    allowed_pairs = torch.tensor(
        _PAIR_OPTIONS,
        dtype=torch.long,
        device=device,
    )
    n_allowed = int(allowed_pairs.shape[0])

    pair_to_idx = torch.full((16,), -1, dtype=torch.long, device=device)
    for pair_idx, (left, right) in enumerate(_PAIR_OPTIONS):
        pair_to_idx[left * 4 + right] = pair_idx

    # pair_alt_pairs[old_code] -> candidate legal pairs
    # Legal old pairs exclude themselves; illegal old pairs can use every pair.
    pair_alt_pairs = torch.empty((16, n_allowed, 2), dtype=torch.long, device=device)
    pair_alt_count = torch.empty((16,), dtype=torch.long, device=device)

    for old_code in range(16):
        old_idx = int(pair_to_idx[old_code].item())
        if old_idx >= 0:
            keep = [i for i in range(n_allowed) if i != old_idx]
            n_keep = len(keep)
            pair_alt_pairs[old_code, :n_keep] = allowed_pairs[torch.tensor(keep, device=device)]
            pair_alt_pairs[old_code, n_keep:] = pair_alt_pairs[old_code, n_keep - 1]
            pair_alt_count[old_code] = n_keep
        else:
            pair_alt_pairs[old_code] = allowed_pairs
            pair_alt_count[old_code] = n_allowed

    return SSProposalFast(
        pair_i=pair_i,
        pair_j=pair_j,
        unpaired_idx=unpaired_idx,
        pair_alt_pairs=pair_alt_pairs,
        pair_alt_count=pair_alt_count,
    )

def _propose_one_center_ss_fast(
    center_seq: torch.Tensor,
    batch_size: int,
    p_mut: float,
    device: torch.device,
    ss_fast: "SSProposalFast",
    generator: Optional[torch.Generator] = None,
    force_one_mut: bool = False,
) -> torch.Tensor:
    """
    Fast hard-SS proposer for ONE center sequence.
    Strategy:
      - unpaired: single-site mutation
      - paired: joint pair mutation via LUT
    """
    B = int(batch_size)
    N = int(center_seq.numel())

    seqs = center_seq.unsqueeze(0).expand(B, N).clone()  # (B,N)

    if p_mut <= 0.0:
        return seqs

    pair_i = ss_fast.pair_i
    pair_j = ss_fast.pair_j
    unpaired_idx = ss_fast.unpaired_idx
    pair_alt_pairs = ss_fast.pair_alt_pairs
    pair_alt_count = ss_fast.pair_alt_count

    Kp = int(pair_i.numel())
    Nu = int(unpaired_idx.numel())

    any_mut = torch.zeros((B,), dtype=torch.bool, device=device)

    # --------------------------------------------------
    # 1) unpaired: original fast single-site mutation
    # --------------------------------------------------
    if Nu > 0:
        u_un = torch.rand((B, Nu), device=device, generator=generator)
        mut_un = (u_un < p_mut)
        any_mut |= mut_un.any(dim=1)

        old_un = seqs[:, unpaired_idx]  # (B,Nu)
        rand3 = torch.randint(0, 3, size=(B, Nu), device=device, generator=generator, dtype=torch.long)
        new_un = rand3 + (rand3 >= old_un).long()

        seqs[:, unpaired_idx] = torch.where(mut_un, new_un, old_un)

    # --------------------------------------------------
    # 2) paired: joint mutation via prebuilt LUT
    # --------------------------------------------------
    if Kp > 0:
        u_pair = torch.rand((B, Kp), device=device, generator=generator)
        mut_pair = (u_pair < p_mut)
        any_mut |= mut_pair.any(dim=1)

        old_left = seqs[:, pair_i]    # (B,Kp)
        old_right = seqs[:, pair_j]   # (B,Kp)
        old_code = old_left * 4 + old_right  # (B,Kp)

        # candidate legal pairs for each old_code
        cand_pairs = pair_alt_pairs[old_code]   # (B,Kp,6,2)
        counts = pair_alt_count[old_code]       # (B,Kp), each entry 5 or 6

        # sample integer in [0, count)
        u_pick = torch.rand((B, Kp), device=device, generator=generator)
        pick = torch.floor(u_pick * counts.to(torch.float32)).long()  # (B,Kp)

        picked = torch.gather(
            cand_pairs,
            2,
            pick.unsqueeze(-1).unsqueeze(-1).expand(B, Kp, 1, 2)
        ).squeeze(2)  # (B,Kp,2)

        new_left = picked[..., 0]
        new_right = picked[..., 1]

        seqs[:, pair_i] = torch.where(mut_pair, new_left, old_left)
        seqs[:, pair_j] = torch.where(mut_pair, new_right, old_right)

    # --------------------------------------------------
    # 3) optional: force at least one mutation per candidate
    #    For speed, default False is recommended.
    # --------------------------------------------------
    if force_one_mut:
        no_mut = ~any_mut
        if no_mut.any():
            rows = torch.nonzero(no_mut, as_tuple=False).squeeze(1)
            total_events = Nu + Kp

            if total_events > 0:
                choice = torch.randint(
                    0, total_events,
                    size=(rows.numel(),),
                    device=device,
                    generator=generator,
                )
                choose_unpaired = (choice < Nu)

                if choose_unpaired.any():
                    rr = rows[choose_unpaired]
                    cc_local = choice[choose_unpaired]
                    cols = unpaired_idx[cc_local]

                    old = seqs[rr, cols]
                    rand3 = torch.randint(0, 3, size=(rr.numel(),), device=device, generator=generator, dtype=torch.long)
                    new = rand3 + (rand3 >= old).long()
                    seqs[rr, cols] = new

                if (~choose_unpaired).any():
                    rr = rows[~choose_unpaired]
                    kk = choice[~choose_unpaired] - Nu

                    ii = pair_i[kk]
                    jj = pair_j[kk]

                    old_left = seqs[rr, ii]
                    old_right = seqs[rr, jj]
                    old_code = old_left * 4 + old_right  # (R,)

                    cand_pairs = pair_alt_pairs[old_code]  # (R,6,2)
                    counts = pair_alt_count[old_code]      # (R,)

                    u_pick = torch.rand((rr.numel(),), device=device, generator=generator)
                    pick = torch.floor(u_pick * counts.to(torch.float32)).long()  # (R,)

                    new_pair = torch.gather(
                        cand_pairs,
                        1,
                        pick.unsqueeze(-1).unsqueeze(-1).expand(rr.numel(), 1, 2)
                    ).squeeze(1)  # (R,2)

                    seqs[rr, ii] = new_pair[:, 0]
                    seqs[rr, jj] = new_pair[:, 1]

    return seqs
# ======================================================================
# Single-state design
# ======================================================================
def design_one_pdb_single(
    pdb_path: str,
    pot: TriRNASP_Potential,
    device: torch.device,
    steps: int,
    batch_size: int,
    tail_steps: int,
    kl_lambda: float,
    top_k: int = 10,
    T_min=2.0,
    thermo_ss_arg: Optional[str] = None,
    hard_ss_arg: Optional[str] = None,
    ss_penalty: float = 10000.0,
    frz_arg: Optional[str] = None,
):
    """
    Single-chain SA, SPEED-first for K=1:
      - accept control flow uses python (1 sync/step)
      - recovery/macroF1 computed ONLY when accept (like old fast behavior)
      - trajectory/tail samples stored in GPU buffers during loop
      - one-time transfer to CPU at end
    Return schema aligned with your current 22-field version.
    """
    pdb_name = os.path.basename(pdb_path)

    # 1) Fixed structure
    struct = RNA_Structure_Fixed(pdb_path, R0=9.5000, bin_width=0.52778)
    native_idx = struct.native_base_idx.to(device)
    N_res = int(struct.N_res)

    frz_constraint = build_frz_constraint(
        frz_arg=frz_arg,
        native_len=N_res,
        device=device,
        label=pdb_name,
    )

    ss_constraint = build_ss_constraint(
        ss_arg=hard_ss_arg,
        native_len=N_res,
        device=device,
        penalty=ss_penalty,
    )

    if ss_constraint is not None:
        print(f"[INFO][ss] enabled for {pdb_name}: n_pairs={len(ss_constraint.pairs)}")

    # 2) Scorer
    scorer = TriRNASP_Scorer(
        potential=pot,
        structure=struct,
        device=device,
        R0_orig=9.5000,
        bw_rough=1.35714,
        bw_fine=0.52778,
        kl_coeff=kl_lambda,
        ss_constraint=ss_constraint,   # NEW
    )
    mid_thermo = None
    if USE_MID_THERMO:
        mid_thermo = build_mid_thermo_prior(
            ss_arg=thermo_ss_arg,
            pdb_path=pdb_path,
            temperature_c=MID_THERMO_TEMP_C,
            top_k=MID_THERMO_TOP_K,
            unsupported_pair_penalty=MID_THERMO_UNSUPPORTED_PAIR_PENALTY,
            cache_limit=MID_THERMO_CACHE_LIMIT,
            verbose=False,
        )
        mid_thermo.set_fast_tensor(True, verify_samples=0)
    """Debug
    if mid_thermo is not None:
        print(f"[INFO][thermo] MidThermoPrior enabled: top_k={MID_THERMO_TOP_K}, par={mid_thermo.par_path}")
    """
    ss_fast = None
    if ss_constraint is not None:
        ss_fast = build_ss_proposal_fast(
            ss_constraint=ss_constraint,
            N_res=N_res,
            device=device,
        )

    decode_chain_constraint = build_ss_constraint(
        ss_arg=thermo_ss_arg,
        native_len=N_res,
        device=device,
        penalty=ss_penalty,
    )

    # 3) Init seq
    if ss_constraint is not None:
        current_seq = sample_penalty_free_sequence(
            ss_constraint,
            device=device,
        )
    else:
        current_seq = torch.randint(
            0, 4,
            size=(N_res,),
            dtype=torch.long,
            device=device,
        )
    enforce_frz_and_repair_ss_(
        current_seq,
        frz_constraint=frz_constraint,
        ss_constraint=ss_constraint,
    )
        
    with torch.inference_mode():
        current_E = float(
            scorer.score_batch(current_seq.unsqueeze(0), stage="fine", with_kl=False)[0].item()
        )
    with torch.inference_mode():
        current_E_rough = float(
            scorer.score_batch(current_seq.unsqueeze(0), stage="rough", with_kl=False)[0].item()
        )

    # metrics (ONLY updated on accept)
    current_rec = seq_recovery(current_seq, native_idx)
    current_f1  = macro_f1_from_idx(current_seq, native_idx, ignore_empty_classes=True)

    best_rec = float(current_rec)
    best_rec_seq = current_seq.clone()

    best_f1 = float(current_f1)
    best_f1_seq = current_seq.clone()

    # 4) Anneal
    T0 = 100.000
    cooling = 0.999
    T = float(T0)
    if T_min is None:
        T_min = 2.0
    else:
        T_min = float(T_min)

    p_mut = 0.3
    min_p_mut = 0.001
    max_p_mut = 0.9
    adapt_interval = 10
    target_accept_low = 0.1
    target_accept_high = 0.5
    n_prop = 0
    n_acc = 0

    steps = int(steps)
    tail_steps = max(1, min(int(tail_steps), steps))
    tail_len = int(tail_steps)

    # 5) Logo counts (GPU)
    # Logo probs (computed from tail by Boltzmann at the end)
    counts = None  # placeholder; will be built from tail

    # -------------------------
    # GPU buffers (NO K dim)
    # -------------------------
    record_traj = True
    traj_seq_u8_gpu = torch.empty((steps, N_res), dtype=torch.uint8, device=device) if record_traj else None
    traj_E_gpu      = torch.empty((steps,), dtype=torch.float32, device=device) if record_traj else None
    traj_rec_gpu    = torch.empty((steps,), dtype=torch.float16, device=device) if record_traj else None
    traj_f1_gpu     = torch.empty((steps,), dtype=torch.float16, device=device) if record_traj else None

    tail_seq_u8_gpu = torch.empty((tail_len, N_res), dtype=torch.uint8, device=device)
    tail_E_gpu      = torch.empty((tail_len,), dtype=torch.float32, device=device)
    tail_rec_gpu    = torch.empty((tail_len,), dtype=torch.float16, device=device)
    tail_f1_gpu     = torch.empty((tail_len,), dtype=torch.float16, device=device)
    tail_ptr = 0

    # Limit progress-bar updates to reduce display overhead.
    POSTFIX_EVERY = 200

    pbar = tqdm(range(steps), desc=pdb_name, dynamic_ncols=True, file=_tty, mininterval=0.2)
    for step in pbar:
        # propose (B,N)
        cand_batch = propose_batch(
            current_seq,
            batch_size=batch_size,
            p_mut=p_mut,
            device=device,
            ss_constraint=ss_constraint,
            ss_fast=ss_fast,
            frz_constraint=frz_constraint,
        )
        # two-stage
        with torch.inference_mode():
            fine_sorted, rough_sorted, seq_sorted, idx_sorted = score_two_stage_single(
                cand_batch,
                scorer=scorer,
                top_k=top_k,
                kl_lambda=kl_lambda,
                beta_rough=float(1.0/T_min),
                mid_thermo=mid_thermo,
                mid_top_k=MID_THERMO_TOP_K,
                frz_constraint=frz_constraint,
            )
            cand_best_E = float(fine_sorted[0].item())
            cand_best_E_rough = float(rough_sorted[0].item())   # <<< NEW
            cand_best_seq = seq_sorted[0].clone()
        # Metropolis acceptance in Python control flow.
        dE = cand_best_E - current_E
        if dE <= 0.0:
            accept = True
        else:
            acc_prob = math.exp(-dE / max(T, 1e-8))
            accept = (torch.rand(1, device=device).item() < acc_prob)  # sync anyway; fine

        n_prop += 1
        if accept:
            n_acc += 1
            current_seq = cand_best_seq
            current_E = cand_best_E
            current_E_rough = cand_best_E_rough

            # metrics ONLY when accept (key for speed)
            current_rec = seq_recovery(current_seq, native_idx)
            current_f1  = macro_f1_from_idx(current_seq, native_idx, ignore_empty_classes=True)

            if current_rec > best_rec:
                best_rec = float(current_rec)
                best_rec_seq = current_seq.clone()
            if current_f1 > best_f1:
                best_f1 = float(current_f1)
                best_f1_seq = current_seq.clone()

        # record traj (GPU)
        if record_traj:
            traj_seq_u8_gpu[step].copy_(current_seq.to(torch.uint8))
            traj_E_gpu[step].fill_(float(current_E))
            traj_rec_gpu[step].fill_(float(current_rec))
            traj_f1_gpu[step].fill_(float(current_f1))

        # anneal
        T *= cooling
        if T < T_min:
            T = T_min

        # adapt p_mut
        if n_prop >= adapt_interval:
            acc_rate = float(n_acc) / float(max(n_prop, 1))
            if acc_rate < target_accept_low:
                p_mut *= 0.5
            elif acc_rate > target_accept_high:
                p_mut *= 1.2
            p_mut = max(min_p_mut, min(max_p_mut, p_mut))
            n_prop = 0
            n_acc = 0

        # tail window
        if step >= steps - tail_steps:
            tail_seq_u8_gpu[tail_ptr].copy_(current_seq.to(torch.uint8))
            tail_E_gpu[tail_ptr].fill_(float(current_E))
            tail_rec_gpu[tail_ptr].fill_(float(current_rec))
            tail_f1_gpu[tail_ptr].fill_(float(current_f1))
            tail_ptr += 1

        # Update progress details at a lower frequency.
        if (step % POSTFIX_EVERY) == 0 or (step == steps - 1):
            pbar.set_postfix({
                "E": f"{current_E:7.1f}",
                "Rec%": f"{best_rec*100:5.1f}",
                "F1": f"{best_f1:6.3f}",
                "p_mut": f"{p_mut:5.3f}",
            })

    pbar.close()

    # -------------------------
    # One-time transfer
    # -------------------------
    traj_step_list = list(range(steps))

    if record_traj:
        traj_seq_cpu = traj_seq_u8_gpu.cpu().numpy()
        traj_E_cpu   = traj_E_gpu.cpu().numpy().astype(np.float32)
        traj_rec_cpu = traj_rec_gpu.cpu().numpy().astype(np.float32)
        traj_f1_cpu  = traj_f1_gpu.cpu().numpy().astype(np.float32)

        traj_seq_u8_list = [traj_seq_cpu[i].copy() for i in range(steps)]
        traj_E_list      = [float(traj_E_cpu[i])   for i in range(steps)]
        traj_rec_list    = [float(traj_rec_cpu[i]) for i in range(steps)]
        traj_f1_list     = [float(traj_f1_cpu[i])  for i in range(steps)]
    else:
        traj_seq_u8_list, traj_E_list, traj_rec_list, traj_f1_list = [], [], [], []

    eff_tail = min(int(tail_ptr), tail_len)
    tail_seq_cpu = tail_seq_u8_gpu[:eff_tail].cpu().numpy()
    tail_E_cpu   = tail_E_gpu[:eff_tail].cpu().numpy().astype(np.float32)
    tail_rec_cpu = tail_rec_gpu[:eff_tail].cpu().numpy().astype(np.float32)
    tail_f1_cpu  = tail_f1_gpu[:eff_tail].cpu().numpy().astype(np.float32)

    tail_seq_u8_list = [tail_seq_cpu[i].copy() for i in range(eff_tail)]
    tail_E_list      = [float(tail_E_cpu[i])   for i in range(eff_tail)]
    tail_rec_list    = [float(tail_rec_cpu[i]) for i in range(eff_tail)]
    tail_f1_list     = [float(tail_f1_cpu[i])  for i in range(eff_tail)]

    # -------------------------
    # finalize outputs (Boltzmann-weighted logo from tail)
    # -------------------------
    # best_beta for Boltzmann
    counts = logo4xN_from_tail_boltzmann(
        tail_seq_u8_list=tail_seq_u8_list,
        tail_E_list=tail_E_list,
        N_res=N_res,
        clip_deltaE=100.0,
        Beta_en=0.0,
        dedup=True,
    )  # torch.float32 (4,N)

    consensus_idx_np = decode_consensus_idx_poly_runs(
        counts,
        ss_constraint=decode_chain_constraint,
        frz_constraint=frz_constraint,
    )
    consensus_idx = torch.as_tensor(
        consensus_idx_np,
        dtype=torch.long,
        device=counts.device,
    )

    consensus_seq_str = seq_idx_to_masked_str(consensus_idx, native_idx)
    consensus_rec = seq_recovery(consensus_idx, native_idx)
    macro_f1_cons = macro_f1_from_idx(consensus_idx, native_idx, ignore_empty_classes=True)

    E_cons     = fine_energy_of_seq(consensus_idx.to(device), scorer, norm_factor=1.0)
    E_best_rec = fine_energy_of_seq(best_rec_seq.to(device), scorer, norm_factor=1.0)
    E_best_f1  = fine_energy_of_seq(best_f1_seq.to(device), scorer, norm_factor=1.0)
    
    ppl = perplexity_from_counts(counts, native_idx)
    diversity = diversity_from_tail_dedup_sequences(tail_seq_u8_list, native_idx)

    best_rec_seq_str = seq_idx_to_masked_str(best_rec_seq, native_idx)
    best_f1_seq_str  = seq_idx_to_masked_str(best_f1_seq, native_idx)

    native_seq_str = format_native_with_mask(native_idx)
    n_valid_sites  = count_valid_eval_sites(native_idx)

    counts_cpu = counts.cpu()  # (4,N) float

    return (
        consensus_seq_str,
        float(consensus_rec),
        float(macro_f1_cons),
        float(ppl),
        float(diversity),
        float(current_E_rough),   # <<< NEW
        float(E_cons),
        float(best_rec),
        float(best_f1),
        float(E_best_rec),
        float(E_best_f1),
        best_rec_seq_str,
        best_f1_seq_str,
        counts_cpu,
        native_idx.detach().cpu().clone(),
        native_seq_str,
        int(n_valid_sites),
        # traj
        traj_step_list,
        traj_rec_list,
        traj_f1_list,
        traj_seq_u8_list,
        traj_E_list,
        # tail
        tail_rec_list,
        tail_f1_list,
        tail_seq_u8_list,
        tail_E_list,
    )

def design_one_pdb_single_multi(
    pdb_path: str,
    pot: TriRNASP_Potential,
    device: torch.device,
    steps: int,
    batch_size: int,
    tail_steps: int,
    kl_lambda: float,
    top_k: int = 10,
    T_min=2.0,
    seeds: Optional[List[int]] = None,
    thermo_ss_arg: Optional[str] = None,
    hard_ss_arg: Optional[str] = None,
    ss_penalty: float = 10.0,
    frz_arg: Optional[str] = None,
):
    """
    SIMD-style parallel single-state design with per-chain independent dynamics.

    Optimized but replay-safe version:
      1) per-chain RNG streams
      2) per-chain adaptive p_mut
      3) no global acceptance-rate coupling
      4) main loop wrapped in torch.inference_mode()
      5) preserves current replay semantics / return schema
    """
    if seeds is None or len(seeds) == 0:
        raise ValueError("design_one_pdb_single_multi requires a non-empty seeds list")

    K = int(len(seeds))
    pdb_name = os.path.basename(pdb_path)

    # -------------------------
    # 1) Fixed structure + scorer (shared read-only)
    # -------------------------
    struct = RNA_Structure_Fixed(pdb_path, R0=9.5000, bin_width=0.52778)
    native_idx = struct.native_base_idx.to(device)
    N_res = int(struct.N_res)

    frz_constraint = build_frz_constraint(
        frz_arg=frz_arg,
        native_len=N_res,
        device=device,
        label=pdb_name,
    )

    ss_constraint = build_ss_constraint(
        ss_arg=hard_ss_arg,
        native_len=N_res,
        device=device,
        penalty=ss_penalty,
    )
    if ss_constraint is not None:
        print(f"[INFO][ss] enabled for {pdb_name}: n_pairs={len(ss_constraint.pairs)}")

    scorer = TriRNASP_Scorer(
        potential=pot,
        structure=struct,
        device=device,
        R0_orig=9.5000,
        bw_rough=1.35714,
        bw_fine=0.52778,
        kl_coeff=kl_lambda,
        ss_constraint=ss_constraint,
    )
    mid_thermo = None
    if USE_MID_THERMO:
        mid_thermo = build_mid_thermo_prior(
            ss_arg=thermo_ss_arg,
            pdb_path=pdb_path,
            temperature_c=MID_THERMO_TEMP_C,
            top_k=MID_THERMO_TOP_K,
            unsupported_pair_penalty=MID_THERMO_UNSUPPORTED_PAIR_PENALTY,
            cache_limit=MID_THERMO_CACHE_LIMIT,
            verbose=False,
        )
        mid_thermo.set_fast_tensor(True, verify_samples=0)
    """Debug
    if mid_thermo is not None:
        print(f"[INFO][thermo] MidThermoPrior enabled: top_k={MID_THERMO_TOP_K}, par={mid_thermo.par_path}")
    """
    # -------------------------
    # 2) Per-chain RNG
    # -------------------------
    gens = make_chain_generators(seeds, device=device)

    ss_fast = None
    if ss_constraint is not None:
        ss_fast = build_ss_proposal_fast(
            ss_constraint=ss_constraint,
            N_res=N_res,
            device=device,
        )

    decode_chain_constraint = build_ss_constraint(
        ss_arg=thermo_ss_arg,
        native_len=N_res,
        device=device,
        penalty=ss_penalty,
    )

    # -------------------------
    # 3) Init K sequences independently
    # -------------------------
    with torch.inference_mode():
        if ss_constraint is not None:
            current_seq = torch.stack(
                [
                    sample_penalty_free_sequence(
                        ss_constraint,
                        generator=gens[k],
                        device=device,
                    )
                    for k in range(K)
                ],
                dim=0
            )  # (K, N)
        else:
            current_seq = torch.stack(
                [
                    torch.randint(
                        0, 4,
                        size=(N_res,),
                        dtype=torch.long,
                        device=device,
                        generator=gens[k],
                    )
                    for k in range(K)
                ],
                dim=0
            )  # (K, N)
        enforce_frz_and_repair_ss_(
            current_seq,
            frz_constraint=frz_constraint,
            ss_constraint=ss_constraint,
        )
        E0 = scorer.score_batch(current_seq, stage="fine", with_kl=False)  # (K,)
        current_E = E0.detach().to(torch.float32).clone()

        E0_rough = scorer.score_batch(current_seq, stage="rough", with_kl=False)  # (K,)
        current_E_rough = E0_rough.detach().to(torch.float32).clone()

        current_rec = batch_recovery(current_seq, native_idx).detach().to(torch.float32)  # (K,)
        current_f1  = batch_macro_f1(current_seq, native_idx, ignore_empty_classes=True).detach().to(torch.float32)  # (K,)

        best_rec = current_rec.clone()
        best_rec_seq = current_seq.clone()

        best_f1 = current_f1.clone()
        best_f1_seq = current_seq.clone()

    # -------------------------
    # 4) Anneal params
    # -------------------------
    T0 = 100.000
    cooling = 0.999
    T = float(T0)
    T_min = 2.0 if T_min is None else float(T_min)

    # per-chain adaptive mutation probabilities
    p_mut = torch.full((K,), 0.3, dtype=torch.float32, device=device)

    min_p_mut = 0.001
    max_p_mut = 0.9
    adapt_interval = 10
    target_accept_low = 0.1
    target_accept_high = 0.5

    # per-chain acceptance statistics
    n_prop = torch.zeros((K,), dtype=torch.int32, device=device)
    n_acc  = torch.zeros((K,), dtype=torch.int32, device=device)

    steps = int(steps)
    tail_steps = max(1, min(int(tail_steps), steps))
    tail_len = int(tail_steps)

    # -------------------------
    # 5) GPU buffers
    # -------------------------
    tail_rec_list = [[] for _ in range(K)]
    tail_f1_list = [[] for _ in range(K)]
    tail_seq_u8_list = [[] for _ in range(K)]
    tail_E_list = [[] for _ in range(K)]

    traj_step_list = [[] for _ in range(K)]
    traj_rec_list = [[] for _ in range(K)]
    traj_f1_list = [[] for _ in range(K)]
    traj_seq_u8_list = [[] for _ in range(K)]
    traj_E_list = [[] for _ in range(K)]

    record_traj = True
    traj_seq_u8_gpu = torch.empty((steps, K, N_res), dtype=torch.uint8, device=device) if record_traj else None
    traj_E_gpu      = torch.empty((steps, K), dtype=torch.float32, device=device) if record_traj else None
    traj_rec_gpu    = torch.empty((steps, K), dtype=torch.float16, device=device) if record_traj else None
    traj_f1_gpu     = torch.empty((steps, K), dtype=torch.float16, device=device) if record_traj else None

    tail_seq_u8_gpu = torch.empty((tail_len, K, N_res), dtype=torch.uint8, device=device)
    tail_E_gpu      = torch.empty((tail_len, K), dtype=torch.float32, device=device)
    tail_rec_gpu    = torch.empty((tail_len, K), dtype=torch.float16, device=device)
    tail_f1_gpu     = torch.empty((tail_len, K), dtype=torch.float16, device=device)
    tail_ptr = 0

    POSTFIX_EVERY = 100
    pbar = tqdm(
        range(steps),
        desc=f"{pdb_name}-K{K}",
        dynamic_ncols=True,
        file=_tty,
        mininterval=0.2,
        smoothing=0.05,
    )

    shared_step_list = list(range(steps))

    # -------------------------
    # 6) Main loop
    # -------------------------
    with torch.inference_mode():
        for step in pbar:
            # (K,B,N), independent proposal per chain
            cand_batch = _propose_batch_multi_independent(
                current_seq=current_seq,
                batch_size=batch_size,
                p_mut_vec=p_mut,
                gens=gens,
                device=device,
                ss_constraint=ss_constraint,
                ss_fast=ss_fast,
                frz_constraint=frz_constraint,
            )
            # grouped two-stage scoring
            cand_best_E, cand_best_E_rough, cand_best_seq = score_two_stage_single_grouped(
                cand_batch,
                scorer=scorer,
                top_k=top_k,
                kl_lambda=kl_lambda,
                beta_rough=float(1.0/T_min),
                generators=gens,
                mid_thermo=mid_thermo,
                mid_top_k=MID_THERMO_TOP_K,
                frz_constraint=frz_constraint,
            )

            cand_best_E = cand_best_E.to(torch.float32)
            dE = cand_best_E - current_E  # (K,)

            accept = (dE <= 0.0)

            uphill = (dE > 0.0)
            if uphill.any():
                prob = torch.exp(-dE / max(T, 1e-8))
                u = torch.stack(
                    [torch.rand((), device=device, generator=gens[k]) for k in range(K)],
                    dim=0
                )
                accept = accept | (uphill & (u < prob))

            # per-chain proposal/accept counters
            n_prop += 1
            n_acc += accept.to(torch.int32)

            # update only accepted chains
            if accept.any():
                idx = torch.nonzero(accept, as_tuple=False).squeeze(1)  # (A,)

                current_seq[idx] = cand_best_seq[idx]
                current_E[idx]   = cand_best_E[idx]
                current_E_rough[idx] = cand_best_E_rough[idx]

                new_rec = batch_recovery(current_seq[idx], native_idx).to(torch.float32)
                new_f1  = batch_macro_f1(current_seq[idx], native_idx, ignore_empty_classes=True).to(torch.float32)

                current_rec[idx] = new_rec
                current_f1[idx]  = new_f1

                better_rec = (new_rec > best_rec[idx])
                if better_rec.any():
                    idx2 = idx[better_rec]
                    best_rec[idx2] = new_rec[better_rec]
                    best_rec_seq[idx2] = current_seq[idx2]

                better_f1 = (new_f1 > best_f1[idx])
                if better_f1.any():
                    idx3 = idx[better_f1]
                    best_f1[idx3] = new_f1[better_f1]
                    best_f1_seq[idx3] = current_seq[idx3]

            # record trajectory
            if record_traj:
                traj_seq_u8_gpu[step].copy_(current_seq.to(torch.uint8))
                traj_E_gpu[step].copy_(current_E)
                traj_rec_gpu[step].copy_(current_rec.to(torch.float16))
                traj_f1_gpu[step].copy_(current_f1.to(torch.float16))

            # shared temperature schedule
            T *= cooling
            if T < T_min:
                T = T_min

            # per-chain adaptive p_mut
            adapt_mask = (n_prop >= adapt_interval)
            if adapt_mask.any():
                acc_rate = n_acc.to(torch.float32) / torch.clamp(n_prop.to(torch.float32), min=1.0)

                lower_mask = adapt_mask & (acc_rate < target_accept_low)
                upper_mask = adapt_mask & (acc_rate > target_accept_high)

                if lower_mask.any():
                    p_mut[lower_mask] *= 0.5
                if upper_mask.any():
                    p_mut[upper_mask] *= 1.2

                p_mut.clamp_(min=min_p_mut, max=max_p_mut)

                n_prop[adapt_mask] = 0
                n_acc[adapt_mask] = 0

            # tail window
            if step >= steps - tail_steps:
                tail_seq_u8_gpu[tail_ptr].copy_(current_seq.to(torch.uint8))
                tail_E_gpu[tail_ptr].copy_(current_E)
                tail_rec_gpu[tail_ptr].copy_(current_rec.to(torch.float16))
                tail_f1_gpu[tail_ptr].copy_(current_f1.to(torch.float16))
                tail_ptr += 1

            if (step % POSTFIX_EVERY) == 0 or (step == steps - 1):
                pbar.set_postfix({
                    "E": f"{float(current_E.mean().item()):7.1f}",
                    "Rec%": f"{float(best_rec.max().item() * 100):5.1f}",
                    "F1": f"{float(best_f1.max().item()):6.3f}",
                    "p_mut": f"{float(p_mut.mean().item()):5.3f}",
                })

    pbar.close()

    # -------------------------
    # 7) Transfer to CPU
    # -------------------------
    if record_traj:
        traj_seq_cpu = traj_seq_u8_gpu.cpu().numpy()  # (steps,K,N)
        traj_E_cpu   = traj_E_gpu.cpu().numpy()
        traj_rec_cpu = traj_rec_gpu.cpu().numpy().astype(np.float32)
        traj_f1_cpu  = traj_f1_gpu.cpu().numpy().astype(np.float32)

        for k in range(K):
            traj_step_list[k] = shared_step_list.copy()
            traj_seq_u8_list[k] = [traj_seq_cpu[st, k].copy() for st in range(steps)]
            traj_E_list[k]      = [float(traj_E_cpu[st, k]) for st in range(steps)]
            traj_rec_list[k]    = [float(traj_rec_cpu[st, k]) for st in range(steps)]
            traj_f1_list[k]     = [float(traj_f1_cpu[st, k]) for st in range(steps)]

    eff_tail = min(int(tail_ptr), tail_len)
    tail_seq_cpu = tail_seq_u8_gpu[:eff_tail].cpu().numpy()
    tail_E_cpu   = tail_E_gpu[:eff_tail].cpu().numpy()
    tail_rec_cpu = tail_rec_gpu[:eff_tail].cpu().numpy().astype(np.float32)
    tail_f1_cpu  = tail_f1_gpu[:eff_tail].cpu().numpy().astype(np.float32)

    for k in range(K):
        tail_seq_u8_list[k] = [tail_seq_cpu[i, k].copy() for i in range(eff_tail)]
        tail_E_list[k]      = [float(tail_E_cpu[i, k]) for i in range(eff_tail)]
        tail_rec_list[k]    = [float(tail_rec_cpu[i, k]) for i in range(eff_tail)]
        tail_f1_list[k]     = [float(tail_f1_cpu[i, k]) for i in range(eff_tail)]

    # -------------------------
    # 8) Finalize per chain
    # -------------------------
    rets = []
    for k in range(K):
        c = logo4xN_from_tail_boltzmann(
            tail_seq_u8_list=tail_seq_u8_list[k],
            tail_E_list=tail_E_list[k],
            N_res=N_res,
            clip_deltaE=100.0,
            Beta_en=0.0,
            dedup=True,
        )  # (4,N)

        consensus_idx_np = decode_consensus_idx_poly_runs(
            c,
            ss_constraint=decode_chain_constraint,
            frz_constraint=frz_constraint,
        )
        consensus_idx = torch.as_tensor(
            consensus_idx_np,
            dtype=torch.long,
            device=c.device,
        )
        consensus_seq_str = seq_idx_to_masked_str(consensus_idx, native_idx)
        consensus_rec = seq_recovery(consensus_idx, native_idx)
        macro_f1_cons = macro_f1_from_idx(consensus_idx, native_idx, ignore_empty_classes=True)

        E_cons     = fine_energy_of_seq(consensus_idx.to(device), scorer, norm_factor=1.0)
        E_best_rec = fine_energy_of_seq(best_rec_seq[k].to(device), scorer, norm_factor=1.0)
        E_best_f1  = fine_energy_of_seq(best_f1_seq[k].to(device), scorer, norm_factor=1.0)

        ppl = perplexity_from_counts(c, native_idx)
        diversity = diversity_from_tail_dedup_sequences(tail_seq_u8_list[k], native_idx)

        best_rec_seq_str = seq_idx_to_masked_str(best_rec_seq[k], native_idx)
        best_f1_seq_str  = seq_idx_to_masked_str(best_f1_seq[k], native_idx)

        native_seq_str = format_native_with_mask(native_idx)
        n_valid_sites  = count_valid_eval_sites(native_idx)

        rets.append((
            consensus_seq_str,
            float(consensus_rec),
            float(macro_f1_cons),
            float(ppl),
            float(diversity),
            float(current_E_rough[k].item()),   # <<< NEW
            float(E_cons),
            float(best_rec[k].item()),
            float(best_f1[k].item()),
            float(E_best_rec),
            float(E_best_f1),
            best_rec_seq_str,
            best_f1_seq_str,
            c,
            native_idx.detach().cpu().clone(),
            native_seq_str,
            int(n_valid_sites),
            # traj
            traj_step_list[k],
            traj_rec_list[k],
            traj_f1_list[k],
            traj_seq_u8_list[k],
            traj_E_list[k],
            # tail
            tail_rec_list[k],
            tail_f1_list[k],
            tail_seq_u8_list[k],
            tail_E_list[k],
        ))

    return rets

# ======================================================================
# Multi-state design
# ======================================================================
def propose_batch_multi_perchain(
    center_seqs,
    batch_size,
    p_mut,
    generators,
    ss_constraint=None,
    ss_fast: Optional["SSProposalFast"] = None,
    frz_constraint: Optional["FrozenConstraint"] = None,
):
    """
    Per-chain independent proposal for multi-state design.
    """
    K, N = center_seqs.shape
    B = int(batch_size)
    device = center_seqs.device

    cand = center_seqs.unsqueeze(1).expand(K, B, N).clone()

    # original mode
    if ss_constraint is None or ss_constraint.n_pairs == 0:
        for k in range(K):
            pm = float(p_mut[k].item())
            gk = generators[k]

            pm = max(0.0, min(1.0, pm))

            mut_mask = (torch.rand((B, N), device=device, generator=gk) < pm)
            if frz_constraint is not None and frz_constraint.n_fixed > 0:
                mut_mask[:, frz_constraint.fixed_pos.to(device)] = False

            no_mut = ~mut_mask.any(dim=1)
            if no_mut.any():
                rows = torch.nonzero(no_mut, as_tuple=False).squeeze(1)
                if frz_constraint is not None and frz_constraint.n_fixed > 0:
                    free_idx = frz_constraint.free_pos.to(device)
                else:
                    free_idx = torch.arange(N, device=device)
                if free_idx.numel() > 0:
                    cols_local = torch.randint(0, free_idx.numel(), (rows.numel(),), device=device, generator=gk)
                    cols = free_idx[cols_local]
                    mut_mask[rows, cols] = True

            old = cand[k]
            rand3 = torch.randint(0, 3, size=(B, N), device=device, generator=gk, dtype=torch.long)
            new_val = rand3 + (rand3 >= old).long()
            old[mut_mask] = new_val[mut_mask]
            enforce_frz_and_repair_ss_(
                cand[k],
                frz_constraint=frz_constraint,
                ss_constraint=ss_constraint,
                generator=gk,
            )

        return cand

    # Vectorized hard-SS path
    if ss_fast is None:
        raise ValueError("propose_batch_multi_perchain: ss_fast is required when ss_constraint is enabled")

    for k in range(K):
        cand[k] = _propose_one_center_ss_fast(
            center_seq=center_seqs[k],
            batch_size=B,
            p_mut=float(p_mut[k].item()),
            device=device,
            ss_fast=ss_fast,
            generator=generators[k],
            force_one_mut=False,
        )
        enforce_frz_and_repair_ss_(
            cand[k],
            frz_constraint=frz_constraint,
            ss_constraint=ss_constraint,
            generator=generators[k],
        )

    return cand


def score_two_stage_multi_merged_grouped(
    cand_batch: torch.Tensor,
    merged_scorer: "TriRNASP_Scorer",
    top_k: int,
    kl_lambda: float,
    norm_factor: float,
    n_dir: int = 22,
    beta_rough: float = 0.5,
    generators: Optional[List[torch.Generator]] = None,
    mid_thermo=None,
    mid_top_k: int = MID_THERMO_TOP_K,
    frz_constraint: Optional["FrozenConstraint"] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two-stage scoring on merged multi-state structure for grouped candidates.

    New scheme:
      top-k anchors + directional proposals -> Fine competition.

    Returns:
      best_E:   (K,)  best Fine energy (normalized by norm_factor) per chain
      best_seq: (K,N) best sequence per chain
    """
    assert cand_batch.dim() == 3
    K, B, N = cand_batch.shape
    k = min(int(top_k), int(B))

    nf = float(norm_factor)
    if nf <= 0.0:
        nf = 1.0

    device = cand_batch.device
    row_idx = torch.arange(K, device=device)

    # ---- Rough on all original candidates in one shot ----
    flat = cand_batch.reshape(K * B, N)  # (K*B,N)

    E_rough_sum = merged_scorer.score_batch(
        flat,
        stage="rough",
        with_kl=False,
        kl_lambda=float(kl_lambda) * nf,
    )  # (K*B,)

    E_rough = (E_rough_sum / nf).reshape(K, B)  # (K,B)

    # ---- Top-k anchors per chain ----
    rough_top_vals, rough_top_idx = torch.topk(E_rough, k=k, largest=False, dim=1)  # (K,k)
    idx_exp = rough_top_idx.unsqueeze(-1).expand(K, k, N)
    seq_anchor = torch.gather(cand_batch, dim=1, index=idx_exp)  # (K,k,N)

    # ---- Directional proposals per chain ----
    probs_dir = build_directional_probs_grouped_from_rough(
        cand_batch=cand_batch,
        E_rough_used=E_rough,
        beta_rough=float(beta_rough),
    )  # (K,N,4)

    seq_dir = sample_directional_from_grouped_probs(
        probs=probs_dir,
        n_dir=int(n_dir),
        generators=generators,
        dtype_out=cand_batch.dtype,
    )  # (K,n_dir,N)
    apply_frz_constraint_(seq_dir, frz_constraint)

    # ---- Rough energy for directional proposals ----
    if int(n_dir) > 0:
        dir_flat = seq_dir.reshape(K * int(n_dir), N)
        E_dir_rough_sum = merged_scorer.score_batch(
            dir_flat,
            stage="rough",
            with_kl=False,
        )
        nf = float(norm_factor)
        E_dir_rough = (E_dir_rough_sum / nf).reshape(K, int(n_dir))  # (K,n_dir)
    else:
        E_dir_rough = rough_top_vals.new_empty((K, 0))

    # ---- Fine competition on anchors + directional ----
    seq_fine_in = torch.cat([seq_anchor, seq_dir], dim=1)  # (K,M,N)
    apply_frz_constraint_(seq_fine_in, frz_constraint)
    rough_fine_in = torch.cat([rough_top_vals, E_dir_rough], dim=1)  # (K,M)
    M = int(seq_fine_in.shape[1])

    # seq-prior gate PER CHAIN
    if USE_MID_THERMO and (mid_thermo is not None) and (mid_top_k is not None):
        k_mid = min(int(mid_top_k), M)
        if k_mid < M:
            E_mid_flat = mid_thermo.score_batch(seq_fine_in.reshape(K * M, N))
            E_mid = E_mid_flat.reshape(K, M)

            keep_idx = torch.topk(E_mid, k=k_mid, largest=False, dim=1).indices
            keep_idx_exp = keep_idx.unsqueeze(-1).expand(K, k_mid, N)

            seq_fine_in = torch.gather(seq_fine_in, dim=1, index=keep_idx_exp)
            rough_fine_in = torch.gather(rough_fine_in, dim=1, index=keep_idx)
            M = k_mid

    fine_flat = seq_fine_in.reshape(K * M, N)
    E_fine_sum = merged_scorer.score_batch(
        fine_flat,
        stage="fine",
        with_kl=False,
    )  # (K*M,)

    E_fine = (E_fine_sum / nf).reshape(K, M)  # (K,M)

    # ---- Best fine per chain ----
    best_pos = torch.argmin(E_fine, dim=1)  # (K,)
    best_E = E_fine[row_idx, best_pos]      # (K,)
    best_E_rough = rough_fine_in[row_idx, best_pos]   # <<< NEW
    best_seq = seq_fine_in[row_idx, best_pos]         # (K,N)

    return best_E, best_E_rough, best_seq

def design_multi_state_multi(
    pdb_paths: List[str],
    seeds: List[int],
    pot: TriRNASP_Potential,
    device: torch.device,
    steps: int = 10000,
    batch_size: int = 256,
    tail_steps: int = 4000,
    kl_lambda: float = 0.0,
    top_k: int = 10,
    T_min: float = 2.0,
    T0: float = 100.000,
    cooling: float = 0.999,
    # adaptive mutation
    p_mut_init: float = 0.3,
    adapt_every: int = 10,
    acc_lo: float = 0.10,
    acc_hi: float = 0.50,
    thermo_ss_arg: Optional[str] = None,
    hard_ss_arg: Optional[str] = None,
    ss_penalty: float = 10000.0,
    frz_arg: Optional[str] = None,
):

    assert isinstance(pdb_paths, (list, tuple)) and len(pdb_paths) >= 2, \
        "design_multi_state_multi expects >=2 pdb_paths"
    assert isinstance(seeds, (list, tuple)) and len(seeds) >= 1, \
        "design_multi_state_multi expects non-empty seeds"

    steps = int(steps)
    tail_steps = max(1, min(int(tail_steps), steps))
    tail_len = int(tail_steps)

    # -------------------------
    # Load & merge structures
    # -------------------------
    structs = [RNA_Structure_Fixed(p, R0=9.5000, bin_width=0.52778) for p in pdb_paths]
    merged_struct, n_states = merge_multistate_structures(structs)

    norm_factor, native_max, valid_list = compute_multistate_norm_factor(structs)

    native_idx = getattr(structs[0], "native_base_idx", None)
    if native_idx is None:
        raise RuntimeError("[multi] structs[0].native_base_idx not found")
    native_idx = native_idx.to(device)
    N = int(native_idx.shape[0])

    frz_constraint = build_frz_constraint(
        frz_arg=frz_arg,
        native_len=N,
        device=device,
        label="multi-state",
    )

    ss_constraint = build_ss_constraint(
        ss_arg=hard_ss_arg,
        native_len=N,
        device=device,
        penalty=ss_penalty,
    )
    if ss_constraint is not None:
        print(f"[INFO][ss] enabled for multi-state: n_pairs={len(ss_constraint.pairs)}")

    merged_scorer = TriRNASP_Scorer(
        potential=pot,
        structure=merged_struct,
        device=device,
        R0_orig=9.5000,
        bw_rough=1.35714,
        bw_fine=0.52778,
        kl_coeff=kl_lambda,
        ss_constraint=ss_constraint,
    )
    mid_thermo = None
    if USE_MID_THERMO:
        mid_thermo = build_mid_thermo_prior(
            ss_arg=thermo_ss_arg,
            pdb_path=pdb_paths[0],
            temperature_c=MID_THERMO_TEMP_C,
            top_k=MID_THERMO_TOP_K,
            unsupported_pair_penalty=MID_THERMO_UNSUPPORTED_PAIR_PENALTY,
            cache_limit=MID_THERMO_CACHE_LIMIT,
            verbose=False,
        )
        mid_thermo.set_fast_tensor(True, verify_samples=0)
    """Debug
    if mid_thermo is not None:
        print(f"[INFO][thermo] MidThermoPrior enabled: top_k={MID_THERMO_TOP_K}, par={mid_thermo.par_path}")
    """
    # -------------------------
    # Multi-state INFO report
    # -------------------------
    pdb_stems = [os.path.splitext(os.path.basename(p))[0] for p in pdb_paths]
    valid_list_int = [int(x) for x in valid_list]
    native_max_int = int(native_max)
    norm_factor_f = float(norm_factor)

    print(f"[INFO][multi] n_states = {int(n_states)}")
    print(f"[INFO][multi] states = {', '.join(pdb_stems)}")
    print(f"[INFO][multi] valid_res_per_state = {valid_list_int} (native_max={native_max_int})")
    print(f"[INFO][multi] norm_factor = sum(valid)/native_max = {norm_factor_f:.6f}")
    print(f"[INFO][multi] energy_normalize_divisor(norm_factor) = {norm_factor_f:.6f}")

    ss_fast = None
    if ss_constraint is not None:
        ss_fast = build_ss_proposal_fast(
            ss_constraint=ss_constraint,
            N_res=N,
            device=device,
        )

    decode_chain_constraint = build_ss_constraint(
        ss_arg=thermo_ss_arg,
        native_len=N,
        device=device,
        penalty=ss_penalty,
    )

    # -------------------------
    # Init K chains (per-chain RNG)
    # -------------------------
    K = int(len(seeds))
    gens = []
    for s in seeds:
        g = torch.Generator(device=device.type)  # "cuda" or "cpu"
        g.manual_seed(int(s))
        gens.append(g)

    with torch.inference_mode():
        if ss_constraint is not None:
            current_seq = torch.stack(
                [
                    sample_penalty_free_sequence(
                        ss_constraint,
                        generator=gens[k],
                        device=device,
                    )
                    for k in range(K)
                ],
                dim=0
            )  # (K, N)
        else:
            current_seq = torch.stack(
                [
                    torch.randint(
                        0, 4,
                        size=(N,),
                        dtype=torch.long,
                        device=device,
                        generator=gens[k],
                    )
                    for k in range(K)
                ],
                dim=0
            )  # (K, N)
        enforce_frz_and_repair_ss_(
            current_seq,
            frz_constraint=frz_constraint,
            ss_constraint=ss_constraint,
        )
        # initial energies (Fine normalized)
        E_sum = merged_scorer.score_batch(current_seq, stage="fine", with_kl=False)  # (K,)
        E = (E_sum / float(norm_factor)).to(torch.float32)  # (K,)

        E_rough_sum = merged_scorer.score_batch(current_seq, stage="rough", with_kl=False)  # (K,)
        E_rough = (E_rough_sum / float(norm_factor)).to(torch.float32)  # (K,)

        # metrics
        rec = batch_recovery(current_seq, native_idx).to(torch.float32)  # (K,)
        f1  = batch_macro_f1(current_seq, native_idx, ignore_empty_classes=True).to(torch.float32)  # (K,)

        # best trackers per chain
        best_rec = rec.clone()
        best_rec_seq = current_seq.clone()

        best_f1 = f1.clone()
        best_f1_seq = current_seq.clone()

    # -------------------------
    # Temperature + p_mut per chain
    # -------------------------
    T = torch.full((K,), float(T0), device=device, dtype=torch.float32)
    T_min_val = float(T_min) if T_min is not None else 2.0
    T_min_tensor = torch.full((K,), T_min_val, device=device, dtype=torch.float32)
    p_mut = torch.full((K,), float(p_mut_init), device=device, dtype=torch.float32)

    # acceptance window stats per chain
    acc_cnt = torch.zeros((K,), device=device, dtype=torch.float32)
    prop_cnt = torch.zeros((K,), device=device, dtype=torch.float32)

    # -------------------------
    # CPU lists (kept for return compatibility; filled AFTER loop)
    # -------------------------
    tail_rec_list = [[] for _ in range(K)]
    tail_f1_list  = [[] for _ in range(K)]
    tail_seq_u8_list = [[] for _ in range(K)]
    tail_E_list   = [[] for _ in range(K)]

    traj_step_list = [[] for _ in range(K)]
    traj_rec_list  = [[] for _ in range(K)]
    traj_f1_list   = [[] for _ in range(K)]
    traj_seq_u8_list = [[] for _ in range(K)]
    traj_E_list    = [[] for _ in range(K)]

    # -------------------------
    # GPU trajectory buffers
    # -------------------------
    record_traj = True
    traj_seq_u8_gpu = torch.empty((steps, K, N), dtype=torch.uint8, device=device) if record_traj else None
    traj_E_gpu      = torch.empty((steps, K), dtype=torch.float32, device=device) if record_traj else None
    traj_rec_gpu    = torch.empty((steps, K), dtype=torch.float16, device=device) if record_traj else None
    traj_f1_gpu     = torch.empty((steps, K), dtype=torch.float16, device=device) if record_traj else None

    # -------------------------
    # GPU tail buffers
    # -------------------------
    tail_seq_u8_gpu = torch.empty((tail_len, K, N), dtype=torch.uint8, device=device)
    tail_E_gpu      = torch.empty((tail_len, K), dtype=torch.float32, device=device)
    tail_rec_gpu    = torch.empty((tail_len, K), dtype=torch.float16, device=device)
    tail_f1_gpu     = torch.empty((tail_len, K), dtype=torch.float16, device=device)
    tail_ptr = 0

    # -------------------------
    # SA loop
    # -------------------------
    first_base = os.path.splitext(os.path.basename(pdb_paths[0]))[0]
    POSTFIX_EVERY = 100
    pbar = tqdm(
        range(steps),
        desc=f"{first_base}-Multi-K{K}",
        dynamic_ncols=True,
        file=_tty,
        mininterval=0.2,
        smoothing=0.05,
    )

    shared_step_list = list(range(steps))

    with torch.inference_mode():
        for step in pbar:
            # 1) propose (K,B,N), MUST be per-chain RNG inside helper
            cand_batch = propose_batch_multi_perchain(
                center_seqs=current_seq,
                batch_size=batch_size,
                p_mut=p_mut,
                generators=gens,
                ss_constraint=ss_constraint,
                ss_fast=ss_fast,
                frz_constraint=frz_constraint,
            )
            # 2) two-stage grouped scoring -> best per chain
            cand_best_E, cand_best_E_rough, cand_best_seq = score_two_stage_multi_merged_grouped(
                cand_batch=cand_batch,
                merged_scorer=merged_scorer,
                top_k=top_k,
                kl_lambda=kl_lambda,
                norm_factor=float(norm_factor),
                beta_rough=float(1.0/T_min),
                generators=gens,
                mid_thermo=mid_thermo,
                mid_top_k=MID_THERMO_TOP_K,
                frz_constraint=frz_constraint,
            )

            cand_best_E = cand_best_E.to(torch.float32)
            dE = cand_best_E - E  # (K,)

            accept = (dE <= 0.0)

            # 3) Metropolis accept with PER-CHAIN RNG
            uphill = (~accept)
            if uphill.any():
                prob = torch.exp(torch.clamp(-dE / torch.clamp(T, min=1e-6), min=-50.0, max=50.0))
                u = torch.stack(
                    [torch.rand((), device=device, generator=gens[k]) for k in range(K)],
                    dim=0
                )
                accept = accept | (uphill & (u < prob))

            prop_cnt += 1.0
            acc_cnt += accept.to(torch.float32)

            # 4) update accepted chains only
            if accept.any():
                acc_idx = torch.nonzero(accept, as_tuple=False).squeeze(1)

                current_seq[acc_idx] = cand_best_seq[acc_idx]
                E[acc_idx] = cand_best_E[acc_idx]
                E_rough[acc_idx] = cand_best_E_rough[acc_idx]
                seq_acc = current_seq[acc_idx]
                rec_acc = batch_recovery(seq_acc, native_idx).to(torch.float32)
                f1_acc  = batch_macro_f1(seq_acc, native_idx, ignore_empty_classes=True).to(torch.float32)

                rec[acc_idx] = rec_acc
                f1[acc_idx]  = f1_acc

                better_rec = (rec_acc > best_rec[acc_idx])
                if better_rec.any():
                    idx2 = acc_idx[better_rec]
                    best_rec[idx2] = rec_acc[better_rec]
                    best_rec_seq[idx2] = current_seq[idx2]

                better_f1 = (f1_acc > best_f1[acc_idx])
                if better_f1.any():
                    idx3 = acc_idx[better_f1]
                    best_f1[idx3] = f1_acc[better_f1]
                    best_f1_seq[idx3] = current_seq[idx3]

            # 5) cooling
            T = torch.maximum(T * float(cooling), T_min_tensor)

            # 6) adaptive p_mut (per chain)
            if adapt_every > 0 and ((step + 1) % int(adapt_every) == 0):
                acc_rate = acc_cnt / torch.clamp(prop_cnt, min=1.0)
                p_mut = torch.where(acc_rate < float(acc_lo), torch.clamp(p_mut * 0.5, 0.001, 0.9), p_mut)
                p_mut = torch.where(acc_rate > float(acc_hi), torch.clamp(p_mut * 1.2, 0.001, 0.9), p_mut)
                acc_cnt.zero_()
                prop_cnt.zero_()

            # 7) record trajectory
            if record_traj:
                traj_seq_u8_gpu[step].copy_(current_seq.to(torch.uint8))
                traj_E_gpu[step].copy_(E)
                traj_rec_gpu[step].copy_(rec.to(torch.float16))
                traj_f1_gpu[step].copy_(f1.to(torch.float16))

            # 8) tail window
            if step >= steps - tail_steps:
                tail_seq_u8_gpu[tail_ptr].copy_(current_seq.to(torch.uint8))
                tail_E_gpu[tail_ptr].copy_(E)
                tail_rec_gpu[tail_ptr].copy_(rec.to(torch.float16))
                tail_f1_gpu[tail_ptr].copy_(f1.to(torch.float16))
                tail_ptr += 1

            if (step % POSTFIX_EVERY) == 0 or (step == steps - 1):
                pbar.set_postfix({
                    "E": f"{float(E.mean().item()):7.1f}",
                    "Rec%": f"{float(best_rec.max().item()*100):5.1f}",
                    "F1": f"{float(best_f1.max().item()):6.3f}",
                    "p_mut": f"{float(p_mut.mean().item()):5.3f}",
                })

    pbar.close()

    # -------------------------
    # One-time transfer to CPU
    # -------------------------
    if record_traj:
        traj_seq_cpu = traj_seq_u8_gpu.cpu().numpy()  # (steps,K,N) uint8
        traj_E_cpu   = traj_E_gpu.cpu().numpy()       # (steps,K) float32
        traj_rec_cpu = traj_rec_gpu.cpu().numpy().astype(np.float32)
        traj_f1_cpu  = traj_f1_gpu.cpu().numpy().astype(np.float32)

        for k in range(K):
            traj_step_list[k] = shared_step_list.copy()
            traj_seq_u8_list[k] = [traj_seq_cpu[st, k].copy() for st in range(steps)]
            traj_E_list[k]      = [float(traj_E_cpu[st, k])   for st in range(steps)]
            traj_rec_list[k]    = [float(traj_rec_cpu[st, k]) for st in range(steps)]
            traj_f1_list[k]     = [float(traj_f1_cpu[st, k])  for st in range(steps)]

    eff_tail = min(int(tail_ptr), tail_len)
    tail_seq_cpu = tail_seq_u8_gpu[:eff_tail].cpu().numpy()  # (tail,K,N) uint8
    tail_E_cpu   = tail_E_gpu[:eff_tail].cpu().numpy()       # (tail,K) float32
    tail_rec_cpu = tail_rec_gpu[:eff_tail].cpu().numpy().astype(np.float32)
    tail_f1_cpu  = tail_f1_gpu[:eff_tail].cpu().numpy().astype(np.float32)

    for k in range(K):
        tail_seq_u8_list[k] = [tail_seq_cpu[i, k].copy() for i in range(eff_tail)]
        tail_E_list[k]      = [float(tail_E_cpu[i, k])   for i in range(eff_tail)]
        tail_rec_list[k]    = [float(tail_rec_cpu[i, k]) for i in range(eff_tail)]
        tail_f1_list[k]     = [float(tail_f1_cpu[i, k])  for i in range(eff_tail)]

    # -------------------------
    # Finalize per chain
    # -------------------------
    runs = []
    for k in range(K):
        c = logo4xN_from_tail_boltzmann(
            tail_seq_u8_list=tail_seq_u8_list[k],
            tail_E_list=tail_E_list[k],
            N_res=N,
            clip_deltaE=100.0,
            Beta_en=0.0,
            dedup=True,
        )  # (4,N) float32

        consensus_idx_np = decode_consensus_idx_poly_runs(
            c,
            ss_constraint=decode_chain_constraint,
            frz_constraint=frz_constraint,
        )
        consensus_idx = torch.as_tensor(
            consensus_idx_np,
            dtype=torch.long,
            device=c.device,
        )
        consensus_seq_str = seq_idx_to_masked_str(consensus_idx, native_idx)
        consensus_rec = seq_recovery(consensus_idx, native_idx)
        macro_f1_cons = macro_f1_from_idx(consensus_idx, native_idx, ignore_empty_classes=True)
        # energies for reported sequences (Fine only; normalized by norm_factor)
        E_cons     = fine_energy_of_seq(consensus_idx.to(device), merged_scorer, norm_factor=float(norm_factor))
        E_best_rec = fine_energy_of_seq(best_rec_seq[k].to(device), merged_scorer, norm_factor=float(norm_factor))
        E_best_f1  = fine_energy_of_seq(best_f1_seq[k].to(device), merged_scorer, norm_factor=float(norm_factor))

        ppl = perplexity_from_counts(c, native_idx)
        diversity = diversity_from_tail_dedup_sequences(tail_seq_u8_list[k], native_idx)

        best_rec_seq_str = seq_idx_to_masked_str(best_rec_seq[k], native_idx)
        best_f1_seq_str  = seq_idx_to_masked_str(best_f1_seq[k], native_idx)

        native_seq_str = format_native_with_mask(native_idx)
        n_valid_sites  = count_valid_eval_sites(native_idx)

        runs.append((
            consensus_seq_str,
            float(consensus_rec),
            float(macro_f1_cons),
            float(ppl),
            float(diversity),
            float(E_rough[k].item()),   # <<< NEW
            float(E_cons),

            float(best_rec[k].item()),
            float(best_f1[k].item()),
            float(E_best_rec),
            float(E_best_f1),

            best_rec_seq_str,
            best_f1_seq_str,
            c,  # (4,N) cpu logo counts
            native_idx.detach().cpu().clone(),
            native_seq_str,
            int(n_valid_sites),

            # traj
            traj_step_list[k],
            traj_rec_list[k],
            traj_f1_list[k],
            traj_seq_u8_list[k],
            traj_E_list[k],

            # tail
            tail_rec_list[k],
            tail_f1_list[k],
            tail_seq_u8_list[k],
            tail_E_list[k],
        ))

    return runs

# ======================================================================
# Rank mode: score sequences from FASTA on a fixed backbone
# ======================================================================

def read_fasta_sequences(fa_path: str) -> List[Tuple[str, str]]:
    """
    Minimal FASTA reader.
    Returns list of (name, seq). If no header, name is generated.
    """
    seqs: List[Tuple[str, str]] = []
    name = None
    buf = []
    with open(fa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # flush previous
                if name is not None:
                    seqs.append((name, "".join(buf)))
                name = line[1:].strip() or f"seq{len(seqs)+1}"
                buf = []
            else:
                buf.append(line.replace(" ", "").replace("\t", ""))
    if name is not None:
        seqs.append((name, "".join(buf)))
    # handle headerless fasta
    if not seqs and buf:
        seqs.append((f"seq1", "".join(buf)))
    return seqs

"""
def seq_str_to_idx(seq: str, device: torch.device) -> torch.Tensor:

    seq = seq.strip().upper().replace("T", "U")
    mp = {"A": 0, "U": 1, "C": 2, "G": 3}
    idx_list = []
    for ch in seq:
        if ch not in mp:
            raise ValueError(f"Invalid base '{ch}' in sequence: {seq[:60]}...")
        idx_list.append(mp[ch])
    return torch.tensor(idx_list, dtype=torch.long, device=device)
"""

# ASCII lookup table: 255 marks invalid input; A/U/C/G map to 0/1/2/3,
# and T maps to U.
_ASCII_TO_IDX = np.full(256, 255, dtype=np.uint8)
_ASCII_TO_IDX[ord("A")] = 0
_ASCII_TO_IDX[ord("U")] = 1
_ASCII_TO_IDX[ord("C")] = 2
_ASCII_TO_IDX[ord("G")] = 3
_ASCII_TO_IDX[ord("T")] = 1
_ASCII_TO_IDX[ord("a")] = 0
_ASCII_TO_IDX[ord("u")] = 1
_ASCII_TO_IDX[ord("c")] = 2
_ASCII_TO_IDX[ord("g")] = 3
_ASCII_TO_IDX[ord("t")] = 1

def fasta_to_u8_matrix(fa_path: str, N_res: int):
    """
    Return:
      names: list[str]
      seq_out: list[str] (already T->U, upper)
      X: np.ndarray (M,N) uint8 in {0,1,2,3}
    Invalid seq (bad chars / wrong length) will be skipped.
    """
    names = []
    seq_out = []
    X_rows = []

    cur_name = None
    buf = []

    def flush_one(name, s):
        s = s.strip().replace(" ", "").replace("\t", "")
        if not s:
            return
        s_up = s.upper()
        # length check
        if len(s_up) != N_res:
            return
        # ascii map
        b = np.frombuffer(s_up.encode("ascii", "ignore"), dtype=np.uint8)
        idx = _ASCII_TO_IDX[b]
        if (idx == 255).any():
            return
        names.append(name)
        # output sequence with T->U
        seq_out.append(s_up.replace("T", "U"))
        X_rows.append(idx.astype(np.uint8, copy=False))

    with open(fa_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_name is not None:
                    flush_one(cur_name, "".join(buf))
                cur_name = line[1:].strip() or f"seq{len(names)+1}"
                buf = []
            else:
                buf.append(line)

    if cur_name is not None:
        flush_one(cur_name, "".join(buf))

    if not X_rows:
        return [], [], np.zeros((0, N_res), dtype=np.uint8)

    X = np.stack(X_rows, axis=0)  # (M,N)
    return names, seq_out, X

def has_valid_native(native_idx: torch.Tensor) -> bool:
    """
    Native sequence may be absent or contain invalid indices.
    We treat it as valid only if all indices are in {0,1,2,3}.
    """
    if native_idx is None:
        return False
    vmin = int(native_idx.min().item())
    vmax = int(native_idx.max().item())
    return (vmin >= 0) and (vmax <= 3)


def run_rank_mode(
    pdb_path: str,
    fasta_path: str,
    pot: TriRNASP_Potential,
    device: torch.device,
    out_csv: str,
    rank_bs: int = 10240,
    ss_arg: Optional[str] = None,
    ss_penalty: float = 10000.0,
):
    """
    Rank sequences in FASTA on a given structure using Fine.energy only.

    Optimized:
      - FASTA -> two-pass parse: (1) count valid seqs + length check, (2) stream encode in batches
      - Vectorized ASCII->idx encoding on CPU (np.uint8), then one H2D copy per batch
      - Avoid per-seq torch.tensor allocations on GPU
      - Write CSV streaming (avoid big rows list)

    Output CSV columns:
        sequence,energy(kBT),recovery,macroF1
    If native sequence is unavailable -> recovery/macroF1 = nan.
    """

    pdb_path = os.path.abspath(os.path.expanduser(pdb_path))
    fasta_path = os.path.abspath(os.path.expanduser(fasta_path))

    struct = RNA_Structure_Fixed(pdb_path, R0=9.5000, bin_width=0.52778)
    N_res = int(struct.N_res)

    native_idx = struct.native_base_idx.to(device) if hasattr(struct, "native_base_idx") else None
    native_ok = has_valid_native(native_idx) if native_idx is not None else False

    ss_constraint = build_ss_constraint(
        ss_arg=ss_arg,
        native_len=N_res,
        device=device,
        penalty=ss_penalty,
    )
    if ss_constraint is not None:
        print(
            f"[INFO][ss] enabled for rank(single): "
            f"n_pairs={len(ss_constraint.pairs)}"
        )
        
    scorer = TriRNASP_Scorer(
        potential=pot,
        structure=struct,
        device=device,
        R0_orig=9.5000,
        bw_rough=1.35714,
        bw_fine=0.52778,
        kl_coeff=0.0,
        ss_constraint=ss_constraint,
    )

    # ----------------------------
    # FASTA streaming reader
    # ----------------------------
    def _iter_fasta(path: str):
        name = None
        buf = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if name is not None:
                        yield name, "".join(buf)
                    name = line[1:].strip() or "seq"
                    buf = []
                else:
                    buf.append(line.replace(" ", "").replace("\t", ""))
        if name is not None:
            yield name, "".join(buf)

    # ----------------------------
    # ASCII map: A/U/C/G and T->U
    # ----------------------------
    # default 255 = invalid
    lut = np.full(256, 255, dtype=np.uint8)
    lut[ord("A")] = 0
    lut[ord("U")] = 1
    lut[ord("C")] = 2
    lut[ord("G")] = 3
    lut[ord("T")] = 1  # treat T as U
    # allow lowercase by mapping via upper() (fast path uses .upper() anyway)

    def _encode_seq_to_u8(seq: str) -> np.ndarray:
        # seq should be stripped, uppercase, no spaces already
        b = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        x = lut[b]
        return x

    # ----------------------------
    # Pass 1: count usable sequences
    # ----------------------------
    total = 0
    skipped = 0
    for name, seq in _iter_fasta(fasta_path):
        s = seq.strip().upper().replace(" ", "").replace("\t", "")
        if len(s) != N_res:
            skipped += 1
            continue
        x = _encode_seq_to_u8(s)
        if x.shape[0] != N_res or (x == 255).any():
            skipped += 1
            continue
        total += 1

    if total == 0:
        raise RuntimeError(f"No valid sequences found in FASTA (N_res={N_res}).")

    print(f"[INFO] Loaded sequences (valid for scoring): {total:,}  (skipped={skipped:,})")
    print("[INFO] Rank mode: single-state (Fine energy only)")
    print(f"[INFO] rank_bs={int(rank_bs)}  device={device}")

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)

    # ----------------------------
    # Pass 2: stream -> batch -> score
    # ----------------------------
    B = max(1, int(rank_bs))
    # CPU staging buffers
    buf_u8 = np.empty((B, N_res), dtype=np.uint8)
    buf_seq_out: list[str] = []
    buf_n = 0

    pbar = tqdm(total=total, desc="Ranking sequences", unit="seq", dynamic_ncols=True, file=_tty)

    def _flush_batch_to_csv(fh):
        nonlocal buf_n, buf_seq_out

        if buf_n <= 0:
            return

        # u8 -> long idx on device (one H2D copy per batch)
        # (buf_n,N) uint8 -> int64
        seq_batch = torch.from_numpy(buf_u8[:buf_n].astype(np.int64, copy=False)).to(device, non_blocking=True)

        with torch.inference_mode():
            E_batch = scorer.score_batch(seq_batch, stage="fine", with_kl=False)  # (buf_n,)

            if native_ok:
                rec_batch = batch_recovery(seq_batch, native_idx)  # (buf_n,)
                f1_batch = batch_macro_f1(seq_batch, native_idx, ignore_empty_classes=True)  # (buf_n,)
            else:
                rec_batch = None
                f1_batch = None

        E_cpu = E_batch.detach().cpu().numpy().astype(np.float64, copy=False)

        if native_ok:
            rec_cpu = rec_batch.detach().cpu().numpy().astype(np.float64, copy=False)
            f1_cpu = f1_batch.detach().cpu().numpy().astype(np.float64, copy=False)
            for i in range(buf_n):
                fh.write(f"{buf_seq_out[i]},{E_cpu[i]:.6f},{rec_cpu[i]:.6f},{f1_cpu[i]:.6f}\n")
        else:
            for i in range(buf_n):
                fh.write(f"{buf_seq_out[i]},{E_cpu[i]:.6f},nan,nan\n")

        # reset
        buf_n = 0
        buf_seq_out.clear()

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("sequence,energy(kBT),recovery,macroF1\n")

        for name, seq in _iter_fasta(fasta_path):
            s = seq.strip().upper().replace(" ", "").replace("\t", "")
            if len(s) != N_res:
                continue
            x = _encode_seq_to_u8(s)
            if x.shape[0] != N_res or (x == 255).any():
                continue

            # store
            buf_u8[buf_n, :] = x
            buf_seq_out.append(s.replace("T", "U"))  # keep output as AUCG with U
            buf_n += 1
            pbar.update(1)

            if buf_n >= B:
                _flush_batch_to_csv(f)

        # last partial batch
        _flush_batch_to_csv(f)

    pbar.close()
    print(f"[INFO] Rank CSV written to: {out_csv}")

def _default_rank_out_for_pdb(pdb_path: str, fasta_path: str, rank_out: str | None) -> str:
    """Resolve output CSV path for single-PDB ranking."""
    pdb_base = os.path.splitext(os.path.basename(pdb_path))[0]
    if rank_out:
        rank_out = os.path.abspath(os.path.expanduser(rank_out))
        # If rank_out is a directory (or ends with /), write <pdb>.rank.csv inside it
        if rank_out.endswith(os.sep) or os.path.isdir(rank_out) or (not rank_out.lower().endswith(".csv")):
            out_dir = rank_out
            os.makedirs(out_dir, exist_ok=True)
            return os.path.join(out_dir, f"{pdb_base}.rank.csv")
        # rank_out points to a csv path -> use exactly
        return rank_out
    # default: next to FASTA
    out_dir = os.path.dirname(os.path.abspath(fasta_path)) or "."
    return os.path.join(out_dir, f"{pdb_base}.rank.csv")


def _default_rank_out_for_multi(first_pdb_path: str, fasta_path: str, rank_out: str | None) -> str:
    first_base = os.path.splitext(os.path.basename(first_pdb_path))[0]
    out_name = f"{first_base}-Multi.rank.csv"
    if rank_out:
        rank_out = os.path.abspath(os.path.expanduser(rank_out))
        if rank_out.endswith(os.sep) or os.path.isdir(rank_out) or (not rank_out.lower().endswith(".csv")):
            out_dir = rank_out
            os.makedirs(out_dir, exist_ok=True)
            return os.path.join(out_dir, out_name)
        return rank_out
    out_dir = os.path.dirname(os.path.abspath(fasta_path)) or "."
    return os.path.join(out_dir, out_name)


def run_rank_batch(
    pdb_dir: str,
    fasta_path: str,
    pot: TriRNASP_Potential,
    device: torch.device,
    rank_out: str | None = None,
    rank_bs: int = 10240,
    ss_arg: str | None = None,
    ss_penalty: float = 10.0,
):
    """Batch rank: one FASTA scored on each *.pdb under pdb_dir (one CSV per PDB)."""
    pdb_dir = os.path.abspath(os.path.expanduser(pdb_dir))
    fasta_path = os.path.abspath(os.path.expanduser(fasta_path))

    if not os.path.isdir(pdb_dir):
        raise FileNotFoundError(f"--batch_rank expects --str to be a directory, got: {pdb_dir}")

    pdb_list = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdb_list:
        raise FileNotFoundError(f"No *.pdb found under: {pdb_dir}")

    print(f"[INFO] Batch-rank: {len(pdb_list)} PDB(s) found under {pdb_dir}")
    print(f"[INFO] Rank batch size: {rank_bs}")

    for pdb_path in pdb_list:
        out_csv = _default_rank_out_for_pdb(pdb_path, fasta_path, rank_out)
        ss_arg_this = _resolve_ss_for_batch(ss_arg, pdb_path)

        print(f"[INFO] Ranking on {os.path.basename(pdb_path)} -> {out_csv}")
        run_rank_mode(
            pdb_path=pdb_path,
            fasta_path=fasta_path,
            pot=pot,
            device=device,
            out_csv=out_csv,
            rank_bs=rank_bs,
            ss_arg=ss_arg_this,
            ss_penalty=ss_penalty,
        )



def run_rank_multi(
    multi_dir: str,
    fasta_path: str,
    pot: TriRNASP_Potential,
    device: torch.device,
    rank_out: str | None = None,
    rank_bs: int = 10240,
    ss_arg: Optional[str] = None,
    ss_penalty: float = 10.0,
):
    """
    Multi-state rank: mean Fine energy over multiple PDB states (single CSV).

    Optimized (same ideas as run_rank_mode):
      - FASTA two-pass: count valid for N_res, then stream encode in batches
      - Vectorized ASCII->idx encoding on CPU (np.uint8), one H2D copy per batch
      - Streaming CSV write (no large rows list)

    Note:
      - Uses merged_struct + merged_scorer, and divides Fine energy by norm_factor (mean-like).
    """
    multi_dir = os.path.abspath(os.path.expanduser(multi_dir))
    fasta_path = os.path.abspath(os.path.expanduser(fasta_path))

    if not os.path.isdir(multi_dir):
        raise FileNotFoundError(f"--multi_rank expects --str to be a directory, got: {multi_dir}")

    pdb_list = sorted(glob.glob(os.path.join(multi_dir, "*.pdb")))
    if not pdb_list:
        raise FileNotFoundError(f"No *.pdb found under: {multi_dir}")

    # ---- output csv ----
    if rank_out is not None:
        out_csv = os.path.abspath(os.path.expanduser(rank_out))
        # if rank_out is a directory (or endswith /), write default name inside it
        if out_csv.endswith(os.sep) or os.path.isdir(out_csv) or (not out_csv.lower().endswith(".csv")):
            os.makedirs(out_csv, exist_ok=True)
            base = os.path.basename(os.path.abspath(multi_dir.rstrip("/")))
            out_csv = os.path.join(out_csv, f"{base}-Multi.rank.csv")
    else:
        base = os.path.basename(os.path.abspath(multi_dir.rstrip("/")))
        out_csv = os.path.join(multi_dir, f"{base}-Multi.rank.csv")

    # ---- load states ----
    structs: list[RNA_Structure_Fixed] = []
    N_res: int | None = None

    for pdb_path in pdb_list:
        s = RNA_Structure_Fixed(pdb_path, R0=9.5000, bin_width=0.52778)
        if N_res is None:
            N_res = int(s.N_res)
        elif int(s.N_res) != int(N_res):
            print(f"[WARN] Skip state {pdb_path}: N_res {int(s.N_res)} != {int(N_res)}", file=sys.stderr)
            continue
        structs.append(s)

    if not structs or N_res is None:
        raise RuntimeError("No valid multi-state PDBs were loaded (check N_res consistency).")

    merged_struct, n_states = merge_multistate_structures(structs)
    norm_factor, native_max, valid_list = compute_multistate_norm_factor(structs)
    print(f"[INFO] Multi-state normalization: valid={valid_list}, native_max={native_max}, norm_factor={norm_factor:.6f}")
    ss_constraint = build_ss_constraint(
        ss_arg=ss_arg,
        native_len=int(N_res),
        device=device,
        penalty=ss_penalty,
    )
    if ss_constraint is not None:
        print(
            f"[INFO][ss] enabled for rank(multi): "
            f"n_pairs={len(ss_constraint.pairs)}"
        )
    merged_scorer = TriRNASP_Scorer(
        potential=pot,
        structure=merged_struct,
        device=device,
        R0_orig=9.5000,
        bw_rough=1.35714,
        bw_fine=0.52778,
        kl_coeff=0.0,
        ss_constraint=ss_constraint,
    )

    native_idx = structs[0].native_base_idx.to(device) if hasattr(structs[0], "native_base_idx") else None
    native_ok = has_valid_native(native_idx) if native_idx is not None else False

    # ----------------------------
    # FASTA streaming reader
    # ----------------------------
    def _iter_fasta(path: str):
        name = None
        buf = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if name is not None:
                        yield name, "".join(buf)
                    name = line[1:].strip() or "seq"
                    buf = []
                else:
                    buf.append(line.replace(" ", "").replace("\t", ""))
        if name is not None:
            yield name, "".join(buf)

    # ----------------------------
    # ASCII map: A/U/C/G and T->U
    # ----------------------------
    lut = np.full(256, 255, dtype=np.uint8)
    lut[ord("A")] = 0
    lut[ord("U")] = 1
    lut[ord("C")] = 2
    lut[ord("G")] = 3
    lut[ord("T")] = 1

    def _encode_seq_to_u8(seq: str) -> np.ndarray:
        b = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        return lut[b]

    # ----------------------------
    # Pass 1: count usable sequences
    # ----------------------------
    total = 0
    skipped = 0
    for name, seq in _iter_fasta(fasta_path):
        s = seq.strip().upper().replace(" ", "").replace("\t", "")
        if len(s) != int(N_res):
            skipped += 1
            continue
        x = _encode_seq_to_u8(s)
        if x.shape[0] != int(N_res) or (x == 255).any():
            skipped += 1
            continue
        total += 1

    if total == 0:
        raise RuntimeError(f"No valid sequences found in FASTA (N_res={N_res}).")

    print(f"[INFO] Loaded sequences (valid for scoring): {total:,}  (skipped={skipped:,})")
    print(f"[INFO] Rank mode: multi-state (states={int(n_states)}, norm_factor={float(norm_factor):.6f})")
    print(f"[INFO] rank_bs={int(rank_bs)}  device={device}")

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)

    # ----------------------------
    # Pass 2: stream -> batch -> score -> write
    # ----------------------------
    B = max(1, int(rank_bs))
    buf_u8 = np.empty((B, int(N_res)), dtype=np.uint8)
    buf_seq_out: list[str] = []
    buf_n = 0

    pbar = tqdm(total=total, desc="Ranking sequences (multi-state)", unit="seq", dynamic_ncols=True, file=_tty)

    def _flush_batch_to_csv(fh):
        nonlocal buf_n, buf_seq_out
        if buf_n <= 0:
            return

        seq_batch = torch.from_numpy(buf_u8[:buf_n].astype(np.int64, copy=False)).to(device, non_blocking=True)

        with torch.inference_mode():
            E_mean = merged_scorer.score_batch(seq_batch, stage="fine", with_kl=False) / float(norm_factor)

            if native_ok:
                rec = batch_recovery(seq_batch, native_idx)
                f1 = batch_macro_f1(seq_batch, native_idx, ignore_empty_classes=True)
            else:
                rec = None
                f1 = None

        E_cpu = E_mean.detach().cpu().numpy().astype(np.float64, copy=False)

        if native_ok:
            rec_cpu = rec.detach().cpu().numpy().astype(np.float64, copy=False)
            f1_cpu = f1.detach().cpu().numpy().astype(np.float64, copy=False)
            for i in range(buf_n):
                fh.write(f"{buf_seq_out[i]},{E_cpu[i]:.6f},{rec_cpu[i]:.6f},{f1_cpu[i]:.6f}\n")
        else:
            for i in range(buf_n):
                fh.write(f"{buf_seq_out[i]},{E_cpu[i]:.6f},nan,nan\n")

        buf_n = 0
        buf_seq_out.clear()

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("sequence,energy(kBT),recovery,macroF1\n")

        for name, seq in _iter_fasta(fasta_path):
            s = seq.strip().upper().replace(" ", "").replace("\t", "")
            if len(s) != int(N_res):
                continue
            x = _encode_seq_to_u8(s)
            if x.shape[0] != int(N_res) or (x == 255).any():
                continue

            buf_u8[buf_n, :] = x
            buf_seq_out.append(s.replace("T", "U"))
            buf_n += 1
            pbar.update(1)

            if buf_n >= B:
                _flush_batch_to_csv(f)

        _flush_batch_to_csv(f)

    pbar.close()
    print(f"[INFO] Rank CSV written to: {out_csv}")


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "3dRNAdesign: TriRNASP-based RNA sequence design.\n\n"
            "Modes:\n"
            "  Batch (directory):\n"
            "    python3 3dRNAdesign.py CASP16\n\n"
            "  Single file:\n"
            "    python3 3dRNAdesign.py 2yie.pdb\n\n"
            "  Multi-state (folder with multiple PDBs of same RNA):\n"
            "    python3 3dRNAdesign.py -m ./MultiSets/2yie_set\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        help="Batch directory OR single pdb file (default behavior).",
    )

    parser.add_argument(
        "-m", "--multi",
        type=str,
        default=None,
        help="Multi-state mode: a folder containing multiple PDBs (same RNA, different conformations).",
    )

    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tail_steps", type=int, default=4000)
    parser.add_argument("--kl_lambda", type=float, default=0.0)

    # Rank mode (Fine.energy only):
    #   python3 3dRNAdesign.py -rank --str ./x.pdb --fa ./seqs.fasta
    parser.add_argument(
        "-rank", "--rank",
        action="store_true",
        help="Rank sequences from FASTA on a structure (Fine.energy only).",
    )
    parser.add_argument(
        "--str",
        type=str,
        default=None,
        help="Structure PDB path for -rank mode.",
    )
    parser.add_argument(
        "--fa",
        type=str,
        default=None,
        help="FASTA file path for -rank mode.",
    )
    parser.add_argument(
        "--rank_out",
        type=str,
        default=None,
        help="Output CSV path for -rank mode (default: <pdb_basename>.rank.csv next to FASTA).",
    )

    
    parser.add_argument(
        "--batch_rank",
        action="store_true",
        help="Rank mode: treat --str as a directory and rank FASTA on each *.pdb independently (one CSV per PDB).",
    )

    parser.add_argument(
        "--multi_rank",
        action="store_true",
        help="Rank mode: treat --str as a directory of multi-state PDBs and rank by mean Fine energy over states (single CSV).",
    )
    
    parser.add_argument(
    "--rank_bs",
    type=int,
    default=10240,
    help="Batch size for rank modes (default: 10240)",
    )
    
    parser.add_argument(
    "--temp",
    type=float,
    default=2.0,
    help="Minimum temperature T_min for simulated annealing (override internal default)"
)
    parser.add_argument("--n_runs", type=int, default=1, help="Single-state: number of parallel SA chains (scheme-B).")
    parser.add_argument("--seed_base", type=int, default=123, help="Base seed to generate per-run seeds: seed_base + i.")

    parser.add_argument(
        "--ss",
        type=str,
        default=None,
        help=(
            "Secondary structure constraint. "
            "Can be: (1) a dot-bracket string; "
            "(2) a .dbn file; "
            "(3) for batch directory mode, a directory containing matched .dbn files."
        ),
    )

    parser.add_argument(
        "--frz",
        type=str,
        default=None,
        help=(
            "Frozen sequence mask. "
            "A/U/C/G/T positions are locked; '-' positions are free. "
            "'&' is allowed as a chain separator and ignored in compact indexing. "
            "Can be a direct mask string, a file, or for batch mode a directory "
            "containing matched .frz/.frozen/.txt files."
        ),
    )

    parser.add_argument(
        "--ss_penalty",
        type=float,
        default=10.0,
        help="Penalty added per violated base pair in the secondary structure constraint."
    )



    return parser.parse_args()

def _resolve_ss_for_single(ss_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if ss_arg is None:
        return None
    # Single-file mode accepts a direct string or one DBN file.
    return ss_arg


def _resolve_ss_for_batch(ss_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if ss_arg is None:
        return None

    # Accept a direct dot-bracket string.
    if looks_like_dotbracket(ss_arg):
        return ss_arg

    # A single file is shared by all targets; incompatible lengths raise an error.
    if os.path.isfile(ss_arg):
        return ss_arg

    # Match files in a directory by target basename.
    if os.path.isdir(ss_arg):
        base = os.path.splitext(os.path.basename(pdb_path))[0]
        cands = [
            os.path.join(ss_arg, base + ".dbn"),
            os.path.join(ss_arg, base + ".ss"),
            os.path.join(ss_arg, base + ".txt"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched SS file for {pdb_path} under directory: {ss_arg}")

    raise FileNotFoundError(f"Invalid --ss for batch mode: {ss_arg}")


def _resolve_ss_for_multi(ss_arg: Optional[str], multi_dir: str) -> Optional[str]:
    if ss_arg is None:
        return None

    # Multi-state mode accepts a direct string or one DBN file.
    if looks_like_dotbracket(ss_arg):
        return ss_arg

    if os.path.isfile(ss_arg):
        return ss_arg

    # For a directory, match the multi-state folder name.
    if os.path.isdir(ss_arg):
        folder = os.path.basename(os.path.normpath(multi_dir))
        cands = [
            os.path.join(ss_arg, folder + ".dbn"),
            os.path.join(ss_arg, folder + ".ss"),
            os.path.join(ss_arg, folder + ".txt"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched multi-state SS file for folder {multi_dir} under: {ss_arg}")

    raise FileNotFoundError(f"Invalid --ss for multi-state mode: {ss_arg}")


def _looks_like_frz_mask(s: str) -> bool:
    if s is None:
        return False
    allowed = set("AUCGTaucgt-._& \t\r\n")
    return all(ch in allowed for ch in str(s))


def _resolve_frz_for_single(frz_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if frz_arg is None:
        return None
    return frz_arg


def _resolve_frz_for_batch(frz_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if frz_arg is None:
        return None

    if _looks_like_frz_mask(frz_arg):
        return frz_arg

    if os.path.isfile(frz_arg):
        return frz_arg

    if os.path.isdir(frz_arg):
        base = os.path.splitext(os.path.basename(pdb_path))[0]
        cands = [
            os.path.join(frz_arg, base + ".frz"),
            os.path.join(frz_arg, base + ".frozen"),
            os.path.join(frz_arg, base + ".txt"),
            os.path.join(frz_arg, base + ".fa"),
            os.path.join(frz_arg, base + ".fasta"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched FRZ file for {pdb_path} under directory: {frz_arg}")

    raise FileNotFoundError(f"Invalid --frz for batch mode: {frz_arg}")


def _resolve_frz_for_multi(frz_arg: Optional[str], multi_dir: str) -> Optional[str]:
    if frz_arg is None:
        return None

    if _looks_like_frz_mask(frz_arg):
        return frz_arg

    if os.path.isfile(frz_arg):
        return frz_arg

    if os.path.isdir(frz_arg):
        folder = os.path.basename(os.path.normpath(multi_dir))
        cands = [
            os.path.join(frz_arg, folder + ".frz"),
            os.path.join(frz_arg, folder + ".frozen"),
            os.path.join(frz_arg, folder + ".txt"),
            os.path.join(frz_arg, folder + ".fa"),
            os.path.join(frz_arg, folder + ".fasta"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched multi-state FRZ file for folder {multi_dir} under: {frz_arg}")

    raise FileNotFoundError(f"Invalid --frz for multi-state mode: {frz_arg}")


def main():
    
    args = parse_args()
    print_trirnade_banner()
    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # Load potential
    # -------------------------
    pot = TriRNASP_Potential(
        rough_path="Energy/Rough.energy",
        fine_path="Energy/Fine.energy",
        D=12,
        R0=9.5000,
        bw_rough=1.35714,
        bw_fine=0.52778,
        device=device,
    )

    steps = args.steps
    batch_size = args.batch_size
    tail_steps = args.tail_steps
    kl_lambda = args.kl_lambda
    rank_bs = args.rank_bs
    ss_arg_cli = args.ss
    frz_arg_cli = args.frz
    ss_penalty = float(args.ss_penalty)
    T_min = float(args.temp) if args.temp is not None else 2.0
    n_runs = int(getattr(args, "n_runs", 1))
    seed_base = int(getattr(args, "seed_base", 123))


    
    # ==========================================================
    # (Rank) Score sequences in FASTA on a given structure (Fine only)
    # ==========================================================
    if getattr(args, "rank", False):
        if args.str is None or args.fa is None:
            print("[ERROR] -rank requires --str <pdb_or_dir> and --fa <fasta>", file=sys.stderr)
            sys.exit(1)

        if getattr(args, "batch_rank", False) and getattr(args, "multi_rank", False):
            print("[ERROR] --batch_rank and --multi_rank are mutually exclusive.", file=sys.stderr)
            sys.exit(1)

        str_path = os.path.abspath(os.path.expanduser(args.str.strip()))
        fa_path = os.path.abspath(os.path.expanduser(args.fa.strip()))

        if not os.path.isfile(fa_path):
            print(f"[ERROR] --fa not found: {fa_path}", file=sys.stderr)
            sys.exit(1)

        # Dispatch
        if getattr(args, "batch_rank", False):
            if not os.path.isdir(str_path):
                print(f"[ERROR] --batch_rank expects --str to be a directory, got: {str_path}", file=sys.stderr)
                sys.exit(1)
            run_rank_batch(
                pdb_dir=str_path,
                fasta_path=fa_path,
                pot=pot,
                device=device,
                rank_out=args.rank_out,
                rank_bs=rank_bs,
                ss_arg=ss_arg_cli,
                ss_penalty=ss_penalty,
            )
            return

        if getattr(args, "multi_rank", False):
            if not os.path.isdir(str_path):
                print(f"[ERROR] --multi_rank expects --str to be a directory, got: {str_path}", file=sys.stderr)
                sys.exit(1)
            ss_arg_this = _resolve_ss_for_multi(ss_arg_cli, str_path)

            run_rank_multi(
                multi_dir=str_path,
                fasta_path=fa_path,
                pot=pot,
                device=device,
                rank_out=args.rank_out,
                rank_bs=rank_bs,
                thermo_ss_arg=ss_arg_this,
                hard_ss_arg=ss_arg_this,
                ss_penalty=ss_penalty,
            )
            return

        # Default rank: single PDB
        if not os.path.isfile(str_path):
            print(f"[ERROR] --str not found: {str_path}", file=sys.stderr)
            sys.exit(1)

        ss_arg_this = _resolve_ss_for_single(ss_arg_cli, str_path)

        out_csv = _default_rank_out_for_pdb(str_path, fa_path, args.rank_out)
        run_rank_mode(
            pdb_path=str_path,
            fasta_path=fa_path,
            pot=pot,
            device=device,
            out_csv=out_csv,
            rank_bs=rank_bs,
            ss_arg=ss_arg_this,
            ss_penalty=ss_penalty,
        )
        return


    # ==========================================================
    # (A) Manual multi-state mode: -m / --multi <folder>
    # ==========================================================
    if getattr(args, "multi", None) is not None and args.multi is not None:
        multi_dir = os.path.abspath(os.path.expanduser(args.multi.strip()))
        print(f"[INFO] Multi-state folder: {multi_dir}")

        if not os.path.isdir(multi_dir):
            print(f"[ERROR] -m/--multi is not a directory: {multi_dir}", file=sys.stderr)
            sys.exit(1)

        pdb_paths = sorted(glob.glob(os.path.join(multi_dir, "*.pdb")))
        if len(pdb_paths) < 2:
            print(f"[ERROR] Multi-state folder must contain >=2 .pdb files: {multi_dir}", file=sys.stderr)
            print(f"[ERROR] Found {len(pdb_paths)} pdb(s).", file=sys.stderr)
            sys.exit(1)

        # -----------------------------
        # NEW: always use multi-parallel core (design_multi_state_multi)
        # - n_runs controls number of parallel chains (seeds)
        # - if n_runs==1, just one seed = seed_base
        # - if n_runs>1, seeds = seed_base + i
        # -----------------------------
        if n_runs <= 1:
            seeds = [int(seed_base)]
        else:
            seeds = [int(seed_base) + i for i in range(int(n_runs))]

        ss_arg_this = _resolve_ss_for_multi(ss_arg_cli, multi_dir)
        frz_arg_this = _resolve_frz_for_multi(frz_arg_cli, multi_dir)
        
        runs = design_multi_state_multi(
            pdb_paths=pdb_paths,
            seeds=seeds,
            pot=pot,
            device=device,
            steps=steps,
            batch_size=batch_size,
            tail_steps=tail_steps,
            kl_lambda=kl_lambda,
            top_k=10,
            T_min=(float(T_min) if T_min is not None else 2.0),
            thermo_ss_arg=ss_arg_this,
            hard_ss_arg=ss_arg_this,
            ss_penalty=ss_penalty,
            frz_arg=frz_arg_this,
        )

        # Use folder name as output prefix
        folder_name = os.path.basename(os.path.normpath(multi_dir))
        out_dir = multi_dir
        output_N = int(runs[0][14].shape[0])
        decode_chain_constraint = build_ss_constraint(
            ss_arg=ss_arg_this,
            native_len=output_N,
            device=device,
            penalty=ss_penalty,
        )
        frz_constraint_out = build_frz_constraint(
            frz_arg=frz_arg_this,
            native_len=output_N,
            device=torch.device("cpu"),
            label=f"{folder_name}-Multi-output",
        )

        # -----------------------------
        # If multiple runs: write per-seed outputs into <folder>-Multi.runs/
        # Then pick BEST run to write legacy <folder>-Multi.* outputs
        # -----------------------------
        if int(len(seeds)) > 1:
            runs_dir = os.path.join(out_dir, f"{folder_name}-Multi.runs")
            os.makedirs(runs_dir, exist_ok=True)

            for sd, ret in zip(seeds, runs):
                (
                    consensus_seq_str,
                    consensus_rec,
                    macro_f1_consensus,
                    ppl,
                    diversity,
                    E_rough,   # <<< NEW
                    E_cons,
                    best_rec,
                    best_macro_f1,
                    E_best_rec,
                    E_best_f1,
                    best_rec_seq_str,
                    best_macro_seq_str,
                    logo_counts_cpu,
                    native_idx_cpu,
                    native_seq_str,
                    n_valid_sites,
                    # traj
                    traj_step_list,
                    traj_rec_list,
                    traj_f1_list,
                    traj_seq_u8_list,
                    traj_E_list,
                    # tail
                    tail_rec_list,
                    tail_f1_list,
                    tail_seq_u8_list,
                    tail_E_list,
                ) = ret

                tag = f"{folder_name}-Multi.seed{int(sd)}"
                out_prefix = os.path.join(runs_dir, tag)

                # result + logo + tail_hist + design + traj + top_energy (+ MIN-E appended inside)
                _write_single_outputs(
                    out_prefix=out_prefix,
                    pdb_name=f"{folder_name}-Multi",
                    consensus_seq_str=consensus_seq_str,
                    consensus_rec=consensus_rec,
                    macro_f1_consensus=macro_f1_consensus,
                    ppl=ppl,
                    diversity=diversity,
                    E_rough=E_rough,
                    E_cons=E_cons,
                    best_rec=best_rec,
                    best_macro_f1=best_macro_f1,
                    E_best_rec=E_best_rec,
                    E_best_f1=E_best_f1,
                    best_rec_seq_str=best_rec_seq_str,
                    best_macro_seq_str=best_macro_seq_str,
                    counts_cpu=logo_counts_cpu,
                    native_idx=native_idx_cpu,
                    traj_step_list=traj_step_list,
                    traj_rec_list=traj_rec_list,
                    traj_f1_list=traj_f1_list,
                    traj_seq_u8_list=traj_seq_u8_list,
                    traj_E_list=traj_E_list,
                    tail_rec_list=tail_rec_list,
                    tail_f1_list=tail_f1_list,
                    tail_seq_u8_list=tail_seq_u8_list,
                    tail_E_list=tail_E_list,
                    decode_chain_constraint=decode_chain_constraint,
                    frz_constraint=frz_constraint_out,
                )

            print(f"[INFO] Per-seed multi-state outputs written under: {runs_dir}")

            # Select the best run for the standard output files.
            #   1) min consensus fine energy (E_cons)
            #   2) max consensus recovery
            #   3) max consensus macroF1
            def _key_multi(ret_tuple):
                E = float(ret_tuple[6])
                rec = float(ret_tuple[1])
                f1 = float(ret_tuple[2])
                return (E, -rec, -f1)

            best_i = min(range(len(runs)), key=lambda i: _key_multi(runs[i]))
            best_seed = int(seeds[best_i])
            ret_best = runs[best_i]
        else:
            best_seed = int(seeds[0])
            ret_best = runs[0]

        (
            consensus_seq_str,
            consensus_rec,
            macro_f1_consensus,
            ppl,
            diversity,
            E_rough,   # <<< NEW
            E_cons,
            best_rec,
            best_macro_f1,
            E_best_rec,
            E_best_f1,
            best_rec_seq_str,
            best_macro_seq_str,
            logo_counts_cpu,
            native_idx_cpu,
            native_seq_str,
            n_valid_sites,
            # traj
            traj_step_list,
            traj_rec_list,
            traj_f1_list,
            traj_seq_u8_list,
            traj_E_list,
            # tail
            tail_rec_list,
            tail_f1_list,
            tail_seq_u8_list,
            tail_E_list,
        ) = ret_best

        rec_percent = float(consensus_rec) * 100.0
        best_rec_percent = float(best_rec) * 100.0

        result_path = os.path.join(out_dir, f"{folder_name}-Multi.result")
        with open(result_path, "w") as f:
            f.write("# 3dRNAdesign Multi-state result (consensus over tail_steps)\n")
            f.write("# Energies are mean-like normalized over conformations via norm_factor in core.\n")
            f.write("# Two-stage Rough/Fine.\n\n")
            f.write(f"# Scheme-B multi: n_runs={int(n_runs)} seed_base={int(seed_base)} best_seed={int(best_seed)}\n\n")
            f.write(
                "# Consensus_Efine,Consensus_Recovery(%),ConsensusMacroF1,Perplexity,Diversity,"
                "BestRec_Efine,BestRecovery(%),BestF1_Efine,BestMacroF1\n\n"
            )
            f.write(
                f"{float(E_cons):.6f},{rec_percent:.2f},{float(macro_f1_consensus):.4f},"
                f"{float(ppl):.6f},{float(diversity):.6f},"
                f"{float(E_best_rec):.6f},{best_rec_percent:.2f},"
                f"{float(E_best_f1):.6f},{float(best_macro_f1):.4f}\n"
            )
            f.write("[Consensus]\n")
            f.write(consensus_seq_str + "\n")
            f.write("[BestRec]\n")
            f.write(best_rec_seq_str + "\n")
            f.write("[BestMacroF1]\n")
            f.write(best_macro_seq_str + "\n")

        print("\n=== Multi-state design finished ===")
        print(f"Consensus Recovery = {rec_percent:.2f}%")
        print(f"Consensus MacroF1  = {float(macro_f1_consensus):.4f}")
        print(f"BestRec            = {best_rec_percent:.2f}%")
        print(f"BestMacroF1        = {float(best_macro_f1):.4f}")

        # MIN-E from tail (best run)
        minE_pack = get_minE_from_tail(tail_seq_u8_list, tail_E_list, tail_rec_list, tail_f1_list, native_idx=native_idx_cpu)
        if minE_pack is not None:
            minE_seq, minE_E, minE_rec, minE_f1 = minE_pack
            print(f"[MIN-E] E={minE_E:.6f}, Recovery={minE_rec:.4f}, MacroF1={minE_f1:.4f}")
            print(f"[MIN-E] {minE_seq}")
            with open(result_path, "a") as rf:
                rf.write(f"[MIN-E] E={minE_E:.6f}, Recovery={minE_rec:.4f}, MacroF1={minE_f1:.4f}\n")
                rf.write(f"{minE_seq}\n\n")

        print(f"[INFO] Multi-state result written to: {result_path}")

        # Multi-state logo CSV (best run)
        counts_np = logo_counts_cpu.detach().cpu().numpy()
        N_res = counts_np.shape[1]
        csv_path = os.path.join(out_dir, f"{folder_name}-Multi.logo.csv")
        with open(csv_path, "w") as f_csv:
            header = ["Base"] + [str(i + 1) for i in range(N_res)]
            f_csv.write(",".join(header) + "\n")
            label_order = ["A", "U", "C", "G"]
            for row_idx, basech in enumerate(label_order):
                row_counts = counts_np[row_idx].tolist()
                row_str = [basech] + [f"{float(c):.8f}" for c in row_counts]
                f_csv.write(",".join(row_str) + "\n")
        print(f"[INFO] Multi-state logo CSV written to: {csv_path}")

        # Multi-state tail hist CSV (best run)
        hist_path = os.path.join(out_dir, f"{folder_name}-Multi.tail_hist.csv")
        write_tail_metric_hist_csv(
            hist_path,
            rec_list=tail_rec_list,
            f1_list=tail_f1_list,
            n_bins=50,
        )
        print(f"[INFO] Multi-state tail metric hist CSV written to: {hist_path}")

        # Multi-state tail design CSV (best run)
        design_csv_path = os.path.join(out_dir, f"{folder_name}-Multi.design.csv")
        write_design_csv(
            design_csv_path,
            seq_u8_list=tail_seq_u8_list,
            energy_list=tail_E_list,
            rec_list=tail_rec_list,
            f1_list=tail_f1_list,
            native_idx=native_idx_cpu,
        )
        print(f"[INFO] Wrote tail design CSV: {design_csv_path}")

        # Multi-state traj CSV (best run)
        traj_csv_path = os.path.join(out_dir, f"{folder_name}-Multi.traj.csv")
        write_traj_csv(
            traj_csv_path,
            traj_step_list,
            traj_seq_u8_list,
            traj_E_list,
            traj_rec_list,
            traj_f1_list,
            native_idx=native_idx_cpu,
        )
        print(f"[INFO] Wrote traj CSV: {traj_csv_path}")

        # TOP10 by unique energy (tail window) (best run)
        top_groups, top10_cons = topk_unique_energy_and_consensus(
            tail_seq_u8_list=tail_seq_u8_list,
            tail_E_list=tail_E_list,
            tail_rec_list=tail_rec_list,
            tail_f1_list=tail_f1_list,
            k=10,
            energy_decimals=6,
            chain_constraint=decode_chain_constraint,
            frz_constraint=frz_constraint_out,
        )

        if top_groups:
            top10_csv = os.path.join(out_dir, f"{folder_name}-Multi.top_energy.csv")
            write_top10_energy_csv(top10_csv, top_groups, top10_cons, native_idx=native_idx_cpu)
            print(f"[INFO] Wrote TOP-energy CSV: {top10_csv}")
            print(f"[TOP-Consensus] {top10_cons}")

            with open(result_path, "a") as rf:
                rf.write("[Top_Energy]\n")
                for i, g in enumerate(top_groups, start=1):
                    rep_seq = "".join(np.array(["A","U","C","G"], dtype="<U1")[g["rep_seq_u8"]].tolist())
                    rf.write(
                        f"{i:02d} E={g['E']:.6f} n={g['n_samples']} repFreq={g['rep_freq']} "
                        f"repRec={g['rep_rec']:.4f} repF1={g['rep_f1']:.4f}\n"
                    )
                    rf.write(rep_seq + "\n")
                rf.write("[Top_Consensus]\n")
                rf.write(top10_cons + "\n\n")

        return

    # ==========================================================
    # (B) Default modes (unchanged): batch dir OR single pdb file
    # ==========================================================
    if not args.inputs:
        print("[ERROR] No inputs provided.", file=sys.stderr)
        print("[HINT] Batch dir:   python3 3dRNAdesign.py <pdb_dir>", file=sys.stderr)
        print("[HINT] Single file: python3 3dRNAdesign.py <one.pdb>", file=sys.stderr)
        print("[HINT] Multi-state: python3 3dRNAdesign.py -m <multi_folder>", file=sys.stderr)
        sys.exit(1)

    raw_inputs = args.inputs
    inputs = [os.path.abspath(os.path.expanduser(x.strip())) for x in raw_inputs]

    # Debug (optional)
    print("[DEBUG] raw_inputs =", raw_inputs)
    print("[DEBUG] inputs     =", inputs)
    for p in inputs:
        print(f"[DEBUG] path: {repr(p)}  isdir={os.path.isdir(p)}  isfile={os.path.isfile(p)}")

    # Only accept ONE thing in default mode: a directory OR a single pdb file
    if len(inputs) != 1:
        print("[ERROR] Default mode accepts exactly ONE directory OR ONE pdb file.", file=sys.stderr)
        print("[HINT] Multi-state folder mode: python3 3dRNAdesign.py -m <multi_folder>", file=sys.stderr)
        sys.exit(1)

    one = inputs[0]

    # -------------------------
    # (B1) Batch directory
    # -------------------------
    if os.path.isdir(one):
        pdb_dir = one
        pdb_list = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
        if not pdb_list:
            print(f"[ERROR] No .pdb files found under {pdb_dir}", file=sys.stderr)
            sys.exit(1)

        result_path = os.path.join(pdb_dir, "3dRNAdesign.result")
        with open(result_path, "w") as f:
            f.write("# 3dRNAdesign batch design result (consensus over tail_steps)\n")
            f.write("# Two-stage Rough/Fine TriRNASP (fixed hyperparameters)\n")
            f.write("# format:\n")
            f.write(
                "# PDB_tag,seed,consensus_Efine,concensus_Recovery(%),MacroF1,Perplexity,Diversity,"
  "bestRec_Efine,BestRecovery(%),bestF1_Efine,BestMacroF1\n\n"
    )

        for pdb_path in pdb_list:
            pdb_name = os.path.basename(pdb_path)
            base = os.path.splitext(pdb_name)[0]
            print(f"[INFO] Designing: {pdb_name}")
            ss_arg_this = _resolve_ss_for_batch(ss_arg_cli, pdb_path)
            frz_arg_this = _resolve_frz_for_batch(frz_arg_cli, pdb_path)
            try:
                # -----------------------------
                # Scheme-B: single-state K runs
                # -----------------------------
                if n_runs <= 1:
                    seeds = [seed_base]
                    rets = [design_one_pdb_single(
                        pdb_path=pdb_path,
                        pot=pot,
                        device=device,
                        steps=steps,
                        batch_size=batch_size,
                        tail_steps=tail_steps,
                        kl_lambda=kl_lambda,
                        top_k=10,
                        T_min=T_min,
                        thermo_ss_arg=ss_arg_this,
                        hard_ss_arg=ss_arg_this,
                        ss_penalty=ss_penalty,
                        frz_arg=frz_arg_this,
                    )]
                else:
                    seeds = [seed_base + i for i in range(n_runs)]
                    rets = design_one_pdb_single_multi(
                        pdb_path=pdb_path,
                        pot=pot,
                        device=device,
                        steps=steps,
                        batch_size=batch_size,
                        tail_steps=tail_steps,
                        kl_lambda=kl_lambda,
                        top_k=10,
                        T_min=T_min,
                        seeds=seeds,
                        thermo_ss_arg=ss_arg_this,
                        hard_ss_arg=ss_arg_this,
                        ss_penalty=ss_penalty,
                        frz_arg=frz_arg_this,
                    )

            except Exception as e:
                print(f"{pdb_name}\tERROR: {e}", file=sys.stderr)
                continue

            output_N = int(rets[0][14].shape[0])
            decode_chain_constraint = build_ss_constraint(
                ss_arg=ss_arg_this,
                native_len=output_N,
                device=device,
                penalty=ss_penalty,
            )
            frz_constraint_out = build_frz_constraint(
                frz_arg=frz_arg_this,
                native_len=output_N,
                device=torch.device("cpu"),
                label=f"{base}-output",
            )

            # -----------------------------------------
            # Write outputs for EACH chain (seed)
            # -----------------------------------------
            for seed, ret in zip(seeds, rets):
                (
                    consensus_seq_str,
                    consensus_rec,
                    macro_f1_consensus,
                    ppl,
                    diversity,
                    E_rough,   # <<< NEW
                    E_cons,
                    best_rec,
                    best_macro_f1,
                    E_best_rec,
                    E_best_f1,
                    best_rec_seq_str,
                    best_macro_seq_str,
                    counts_cpu,
                    native_idx_cpu,
                    native_seq_str,
                    n_valid_sites,
                    # traj
                    traj_step_list,
                    traj_rec_list,
                    traj_f1_list,
                    traj_seq_u8_list,
                    traj_E_list,
                    # tail
                    tail_rec_list,
                    tail_f1_list,
                    tail_seq_u8_list,
                    tail_E_list,
                ) = ret

                # tag to avoid overwriting per-seed files
                tag = f"{base}.seed{seed}"

                rec_percent = float(consensus_rec) * 100.0
                best_rec_percent = float(best_rec) * 100.0

                print(
                    f"{tag}\tConsensus_Recovery={rec_percent:.2f}%, "
                    f"MacroF1={float(macro_f1_consensus):.4f}, "
                    f"PPL={float(ppl):.3f}, "
                    f"Diversity={float(diversity):.4f}, "
                    f"BestRec={best_rec_percent:.2f}%, "
                    f"BestMacroF1={float(best_macro_f1):.4f}"
                )
                print("[Consensus]")
                print(consensus_seq_str)
                print("[BestRec]")
                print(best_rec_seq_str)
                print("[BestMacroF1]")
                print(best_macro_seq_str)

                # ----------------
                # Append to batch result file (one line per seed)
                # ----------------
                with open(result_path, "a") as f:
                    f.write(
                        f"{tag},{seed},"
                        f"{float(E_cons):.6f},{rec_percent:.2f},{float(macro_f1_consensus):.4f},"
                        f"{float(ppl):.6f},{float(diversity):.6f},"
                        f"{float(E_best_rec):.6f},{best_rec_percent:.2f},"
                        f"{float(E_best_f1):.6f},{float(best_macro_f1):.4f}\n"
                    )
                    f.write("[Consensus]\n")
                    f.write(consensus_seq_str + "\n")
                    f.write("[BestRec]\n")
                    f.write(best_rec_seq_str + "\n")
                    f.write("[BestMacroF1]\n")
                    f.write(best_macro_seq_str + "\n\n")
                    f.write(f"[Consensus] E_fine={float(E_cons):.6f}\n")
                    f.write(consensus_seq_str + "\n")

                # ----------------
                # MIN-E from tail
                # ----------------
                minE_pack = get_minE_from_tail(tail_seq_u8_list, tail_E_list, tail_rec_list, tail_f1_list, native_idx=native_idx_cpu)
                if minE_pack is not None:
                    minE_seq, minE_E, minE_rec, minE_f1 = minE_pack
                    print(f"[MIN-E] {tag} E={minE_E:.6f}, Recovery={minE_rec:.4f}, MacroF1={minE_f1:.4f}")
                    with open(result_path, "a") as rf:
                        rf.write(f"[MIN-E] {tag} E={minE_E:.6f}, Recovery={minE_rec:.4f}, MacroF1={minE_f1:.4f}\n")
                        rf.write(f"{minE_seq}\n\n")

                # ----------------
                # logo CSV (per seed)
                # ----------------
                counts_np = counts_cpu.numpy()
                N_res = counts_np.shape[1]
                csv_path = os.path.join(pdb_dir, f"{tag}.logo.csv")
                with open(csv_path, "w") as f_csv:
                    header = ["Base"] + [str(i + 1) for i in range(N_res)]
                    f_csv.write(",".join(header) + "\n")
                    label_order = ["A", "U", "C", "G"]
                    for row_idx, basech in enumerate(label_order):
                        row_counts = counts_np[row_idx].tolist()
                        row_str = [basech] + [f"{float(c):.8f}" for c in row_counts]
                        f_csv.write(",".join(row_str) + "\n")
                print(f"[INFO] Wrote logo CSV: {csv_path}")

                # ----------------
                # tail hist CSV (per seed)
                # ----------------
                hist_path = os.path.join(pdb_dir, f"{tag}.tail_hist.csv")
                write_tail_metric_hist_csv(
                    hist_path,
                    rec_list=tail_rec_list,
                    f1_list=tail_f1_list,
                    n_bins=50,
                )
                print(f"[INFO] Wrote tail metric hist CSV: {hist_path}")

                # ----------------
                # tail design CSV (per seed)
                # ----------------
                design_csv_path = os.path.join(pdb_dir, f"{tag}.design.csv")
                write_design_csv(
                    design_csv_path,
                    seq_u8_list=tail_seq_u8_list,
                    energy_list=tail_E_list,
                    rec_list=tail_rec_list,
                    f1_list=tail_f1_list,
                    native_idx=native_idx_cpu,
                )
                print(f"[INFO] Wrote tail design CSV: {design_csv_path}")

                # ----------------
                # traj CSV (per seed)
                # ----------------
                traj_csv_path = os.path.join(pdb_dir, f"{tag}.traj.csv")
                write_traj_csv(
                    traj_csv_path,
                    step_list=traj_step_list,
                    seq_u8_list=traj_seq_u8_list,
                    energy_list=traj_E_list,
                    rec_list=traj_rec_list,
                    f1_list=traj_f1_list,
                    native_idx=native_idx_cpu,
                )
                print(f"[INFO] Wrote traj CSV: {traj_csv_path}")

                # ----------------
                # TOP10 unique-energy (tail window) CSV (per seed)
                # ----------------
                top_groups, top10_cons = topk_unique_energy_and_consensus(
                    tail_seq_u8_list=tail_seq_u8_list,
                    tail_E_list=tail_E_list,
                    tail_rec_list=tail_rec_list,
                    tail_f1_list=tail_f1_list,
                    k=10,
                    energy_decimals=6,
                    chain_constraint=decode_chain_constraint,
                    frz_constraint=frz_constraint_out,
                )

                if top_groups:
                    top10_csv = os.path.join(pdb_dir, f"{tag}.top_energy.csv")
                    write_top10_energy_csv(top10_csv, top_groups, top10_cons, native_idx=native_idx_cpu)
                    print(f"[INFO] Wrote TOP-energy CSV: {top10_csv}")
                    print(f"[TOP-Consensus] {tag} {top10_cons}")

                    with open(result_path, "a") as rf:
                        rf.write(f"[Top_Energy] {tag}\n")
                        for i, g in enumerate(top_groups, start=1):
                            rep_seq = "".join(np.array(["A","U","C","G"], dtype="<U1")[g["rep_seq_u8"]].tolist())
                            rf.write(
                                f"{i:02d} E={g['E']:.6f} n={g['n_samples']} repFreq={g['rep_freq']} "
                                f"repRec={g['rep_rec']:.4f} repF1={g['rep_f1']:.4f}\n"
                            )
                            rf.write(rep_seq + "\n")
                        rf.write("[Top_Consensus]\n")
                        rf.write(top10_cons + "\n\n")



        print(f"\n[INFO] Batch finished. Results: {result_path}")
        return

    # -------------------------
    # (B2) Single pdb file
    # -------------------------
    if os.path.isfile(one):
        pdb_path = one
        pdb_dir = os.path.dirname(pdb_path)
        pdb_name = os.path.basename(pdb_path)
        base = os.path.splitext(pdb_name)[0]
        ss_arg_this = _resolve_ss_for_single(ss_arg_cli, pdb_path)
        frz_arg_this = _resolve_frz_for_single(frz_arg_cli, pdb_path)
        print(f"[INFO] Designing (single-file): {pdb_name}")

        # For multiple runs, evaluate independent seeds and write the best
        # result to the standard single-target output paths.
        if n_runs <= 1:
            seeds = [int(seed_base)]
            (
                consensus_seq_str,
                consensus_rec,
                macro_f1_consensus,
                ppl,
                diversity,
                E_rough,
                E_cons,
                best_rec,
                best_macro_f1,
                E_best_rec,
                E_best_f1,
                best_rec_seq_str,
                best_macro_seq_str,
                counts_cpu,
                native_idx_cpu,
                native_seq_str,
                n_valid_sites,
                # traj
                traj_step_list,
                traj_rec_list,
                traj_f1_list,
                traj_seq_u8_list,
                traj_E_list,
                # tail
                tail_rec_list,
                tail_f1_list,
                tail_seq_u8_list,
                tail_E_list,
            ) = design_one_pdb_single(
                pdb_path=pdb_path,
                pot=pot,
                device=device,
                steps=steps,
                batch_size=batch_size,
                tail_steps=tail_steps,
                kl_lambda=kl_lambda,
                top_k=10,
                T_min=T_min,
                thermo_ss_arg=ss_arg_this,
                hard_ss_arg=ss_arg_this,
                ss_penalty=ss_penalty,
                frz_arg=frz_arg_this,
            )

            best_seed = seeds[0]

        else:
            seeds = [int(seed_base) + i for i in range(int(n_runs))]

            # Expect: rets is a list of design_one_pdb_single-like tuples (same schema)
            rets = design_one_pdb_single_multi(
                pdb_path=pdb_path,
                pot=pot,
                device=device,
                steps=steps,
                batch_size=batch_size,
                tail_steps=tail_steps,
                kl_lambda=kl_lambda,
                top_k=10,
                T_min=T_min,
                seeds=seeds,
                thermo_ss_arg=ss_arg_this,
                hard_ss_arg=ss_arg_this,
                ss_penalty=ss_penalty,
                frz_arg=frz_arg_this,
            )

            output_N = int(rets[0][14].shape[0])
            decode_chain_constraint = build_ss_constraint(
                ss_arg=ss_arg_this,
                native_len=output_N,
                device=device,
                penalty=ss_penalty,
            )
            frz_constraint_out = build_frz_constraint(
                frz_arg=frz_arg_this,
                native_len=output_N,
                device=torch.device("cpu"),
                label=f"{base}-output",
            )

            # ---- write per-seed outputs (recommended into a subfolder) ----
            runs_dir = os.path.join(pdb_dir, f"{base}.runs")
            os.makedirs(runs_dir, exist_ok=True)

            for i, ret in enumerate(rets):
                sd = int(seeds[i])
                out_prefix = os.path.join(runs_dir, f"{base}.seed{sd}")

                _write_single_outputs(
                    out_prefix=out_prefix,
                    pdb_name=pdb_name,
                    consensus_seq_str=ret[0],
                    consensus_rec=ret[1],
                    macro_f1_consensus=ret[2],
                    ppl=ret[3],
                    diversity=ret[4],
                    E_rough=ret[5],
                    E_cons=ret[6],
                    best_rec=ret[7],
                    best_macro_f1=ret[8],
                    E_best_rec=ret[9],
                    E_best_f1=ret[10],
                    best_rec_seq_str=ret[11],
                    best_macro_seq_str=ret[12],
                    counts_cpu=ret[13],
                    native_idx=ret[14],
                    traj_step_list=ret[17],
                    traj_rec_list=ret[18],
                    traj_f1_list=ret[19],
                    traj_seq_u8_list=ret[20],
                    traj_E_list=ret[21],
                    tail_rec_list=ret[22],
                    tail_f1_list=ret[23],
                    tail_seq_u8_list=ret[24],
                    tail_E_list=ret[25],
                    decode_chain_constraint=decode_chain_constraint,
                    frz_constraint=frz_constraint_out,
                )

            print(f"[INFO] Per-seed outputs written under: {runs_dir}")

            # -----------------------------
            # Select the best run for the standard output files.
            # Criterion (stable & simple):
            #   1) min consensus fine energy (E_cons)
            #   2) max consensus recovery
            #   3) max consensus macroF1
            # -----------------------------
            def _key(ret_tuple):
                # Indices follow design_one_pdb_single return order:
                # 0 consensus_seq_str
                # 1 consensus_rec
                # 2 macro_f1_consensus
                # 3 ppl
                # 4 diversity
                # 5 E_rough
                # 6 E_cons
                E = float(ret_tuple[6])
                rec = float(ret_tuple[1])
                f1 = float(ret_tuple[2])
                return (E, -rec, -f1)

            best_i = min(range(len(rets)), key=lambda i: _key(rets[i]))
            best_seed = int(seeds[best_i])

            (
                consensus_seq_str,
                consensus_rec,
                macro_f1_consensus,
                ppl,
                diversity,
                E_rough,
                E_cons,
                best_rec,
                best_macro_f1,
                E_best_rec,
                E_best_f1,
                best_rec_seq_str,
                best_macro_seq_str,
                counts_cpu,
                native_idx_cpu,
                native_seq_str,
                n_valid_sites,
                # traj
                traj_step_list,
                traj_rec_list,
                traj_f1_list,
                traj_seq_u8_list,
                traj_E_list,
                # tail
                tail_rec_list,
                tail_f1_list,
                tail_seq_u8_list,
                tail_E_list,
            ) = rets[best_i]

        output_N = int(native_idx_cpu.shape[0])
        decode_chain_constraint = build_ss_constraint(
            ss_arg=ss_arg_this,
            native_len=output_N,
            device=device,
            penalty=ss_penalty,
        )
        frz_constraint_out = build_frz_constraint(
            frz_arg=frz_arg_this,
            native_len=output_N,
            device=torch.device("cpu"),
            label=f"{base}-output",
        )

        rec_percent = consensus_rec * 100.0
        best_rec_percent = best_rec * 100.0

        result_path = os.path.join(pdb_dir, f"{base}.3dRNAdesign.result")
        with open(result_path, "w") as f:
            f.write("# 3dRNAdesign single-file result (consensus over tail_steps)\n")
            f.write("# Two-stage Rough/Fine TriRNASP (fixed hyperparameters)\n\n")
            f.write("#\n")
            f.write(f"# Scheme-B: n_runs={int(n_runs)} seed_base={int(seed_base)} best_seed={int(best_seed)}\n\n")
            f.write(
                f"{pdb_name},{E_rough:.6f},{E_cons:.6f},{rec_percent:.2f},{macro_f1_consensus:.4f},"
                f"{ppl:.6f},{diversity:.6f},"
                f"{E_best_rec:.6f},{best_rec_percent:.2f},"
                f"{E_best_f1:.6f},{best_macro_f1:.4f}\n"
            )
            f.write("[Consensus]\n" + consensus_seq_str + "\n")
            f.write("[BestRec]\n" + best_rec_seq_str + "\n")
            f.write("[BestMacroF1]\n" + best_macro_seq_str + "\n")
            
        
        minE_pack = get_minE_from_tail(tail_seq_u8_list, tail_E_list, tail_rec_list, tail_f1_list, native_idx=native_idx_cpu)
        if minE_pack is not None:
            minE_seq, minE_E, minE_rec, minE_f1 = minE_pack

            # 1) print to terminal
            print(f"[MIN-E] E={minE_E:.6f}, Recovery={minE_rec:.4f}, MacroF1={minE_f1:.4f}")
            print(f"[MIN-E] {minE_seq}")

            # 2) append to the SAME result file (result_path)
            with open(result_path, "a") as rf:
                rf.write(f"[MIN-E] E={minE_E:.6f}, Recovery={minE_rec:.4f}, MacroF1={minE_f1:.4f}\n")
                rf.write(f"{minE_seq}\n\n")


        print(f"[INFO] Single-file result written to: {result_path}")

        # logo
        counts_np = counts_cpu.numpy()
        N_res = counts_np.shape[1]
        csv_path = os.path.join(pdb_dir, f"{base}.logo.csv")
        with open(csv_path, "w") as f_csv:
            header = ["Base"] + [str(i + 1) for i in range(N_res)]
            f_csv.write(",".join(header) + "\n")
            label_order = ["A", "U", "C", "G"]
            for row_idx, basech in enumerate(label_order):
                row_counts = counts_np[row_idx].tolist()
                row_str = [basech] + [f"{float(c):.8f}" for c in row_counts]
                f_csv.write(",".join(row_str) + "\n")
        print(f"[INFO] Wrote logo CSV: {csv_path}")

        hist_path = os.path.join(pdb_dir, f"{base}.tail_hist.csv")
        write_tail_metric_hist_csv(
            hist_path,
            rec_list=tail_rec_list,
            f1_list=tail_f1_list,
            n_bins=50,
        )
        print(f"[INFO] Wrote tail metric hist CSV: {hist_path}")
        
        design_csv_path = os.path.join(pdb_dir, f"{base}.design.csv")
        write_design_csv(
            design_csv_path,
            seq_u8_list=tail_seq_u8_list,
            energy_list=tail_E_list,
            rec_list=tail_rec_list,
            f1_list=tail_f1_list,
            native_idx=native_idx_cpu,
        )
        print(f"[INFO] Wrote tail design CSV: {design_csv_path}")

        traj_csv_path = os.path.join(pdb_dir, f"{base}.traj.csv")
        write_traj_csv(traj_csv_path, traj_step_list, traj_seq_u8_list, traj_E_list, traj_rec_list, traj_f1_list, native_idx=native_idx_cpu)

        # TOP10 by unique energy (tail window)
        top_groups, top10_cons = topk_unique_energy_and_consensus(
            tail_seq_u8_list=tail_seq_u8_list,
            tail_E_list=tail_E_list,
            tail_rec_list=tail_rec_list,
            tail_f1_list=tail_f1_list,
            k=10,
            energy_decimals=6,
            chain_constraint=decode_chain_constraint,
            frz_constraint=frz_constraint_out,
        )

        if top_groups:
            top10_csv = os.path.join(pdb_dir, f"{base}.top_energy.csv")
            write_top10_energy_csv(top10_csv, top_groups, top10_cons, native_idx=native_idx_cpu)
            print(f"[INFO] Wrote TOP-energy CSV: {top10_csv}")
            print(f"[TOP-Consensus] {top10_cons}")

            # append to result file
            with open(result_path, "a") as rf:
                rf.write("[Top_Energy]\n")
                for i, g in enumerate(top_groups, start=1):
                    rep_seq = "".join(np.array(["A","U","C","G"], dtype="<U1")[g["rep_seq_u8"]].tolist())
                    rf.write(
                        f"{i:02d} E={g['E']:.6f} n={g['n_samples']} repFreq={g['rep_freq']} "
                        f"repRec={g['rep_rec']:.4f} repF1={g['rep_f1']:.4f}\n"
                    )
                    rf.write(rep_seq + "\n")
                rf.write("[Top_Consensus]\n")
                rf.write(top10_cons + "\n\n")


        return

    # If here, path is neither dir nor file
    print(f"[ERROR] Input is neither a directory nor a file: {one}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    set_seed(123)
    main()
