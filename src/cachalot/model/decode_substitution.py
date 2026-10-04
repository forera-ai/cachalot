"""Router-share substitution for DeepSeek decode (opt-in approximation, HANDOFF 18.62).

The decode miss budget (`set_decode_miss_budget`) drops a non-resident expert and rescales the remaining router
weights. This module is the other approximation of the same family, MiniMax's 0.39.0 rule ported to DeepSeek: a
non-resident expert whose share of the layer's routing weight is below `tau` is not read; the best *resident* expert
among the next `ranks` by selection score (the router's score plus its correction bias, the ranking top-k used) takes
its place and its weight. The layer's total routing mass is unchanged and no weight is rescaled. An expert that is
not substituted (share at or above `tau`, or no resident candidate) is read as usual, unless the decode miss budget
drops it.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def ranked_candidates(selection_scores: np.ndarray, expert_ids: Sequence[int], ranks: int) -> list[int]:
    """The `ranks` best experts by selection score that are not among `expert_ids`, best first."""
    chosen = set(int(e) for e in expert_ids)
    need = len(chosen) + max(0, int(ranks))
    order = np.argsort(-np.asarray(selection_scores, dtype=np.float64), kind="stable")[:need]
    return [int(e) for e in order if int(e) not in chosen][:ranks]


def plan_substitution(
    expert_ids: Sequence[int],
    weights: Sequence[float],
    candidates: Sequence[int],
    is_resident: Callable[[int], bool],
    tau: float,
) -> tuple[list[int], int]:
    """Replace each non-resident expert whose weight share is under `tau` by the best unused resident candidate.

    Returns the new expert ids (same positions, so the weights stay aligned) and how many were replaced."""
    ids = [int(e) for e in expert_ids]
    total = float(sum(weights))
    if total <= 0.0:
        return ids, 0
    spare = [int(c) for c in candidates if int(c) not in set(ids) and is_resident(int(c))]
    used = 0
    replaced = 0
    # weakest first: the cheapest substitutions take the best candidates
    for i in sorted(range(len(ids)), key=lambda j: weights[j]):
        if used >= len(spare):
            break
        if weights[i] >= tau * total or is_resident(ids[i]):
            continue
        ids[i] = spare[used]
        used += 1
        replaced += 1
    return ids, replaced
