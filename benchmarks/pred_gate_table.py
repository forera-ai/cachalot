"""
Predictor weight against precision, from a routing trace that carries predicted sets (0.61.3).

For each predicted (source layer -> next layer) set, the trace holds the predictor's own router
weights. A precision-gated prefetch would submit a predicted load only above a weight threshold.
This prints, per threshold, the share of predicted loads kept, their precision (used by the
target layer in the same decode token) and the recall of the target layer's experts. It is a
set-overlap screen: it does not know which predicted experts were already resident, so it is not
the "unused non-resident load" precision of /v1/stats (DS-PRED-PREC); price a gate with
cache_sim.py before building one.

Usage: PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/pred_gate_table.py TRACE.npz
"""
from __future__ import annotations

import sys

import numpy as np

from cachalot.metrics.routing_trace import load_trace, predicted_used_mask

THRESHOLDS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3)


def table(arrays) -> list[dict]:
    if "pred_experts" not in arrays:
        raise SystemExit("trace has no predicted sets (recorded since 0.61.3)")

    used = predicted_used_mask(arrays)
    weights = arrays["pred_weights"]
    total = used.size
    used_total = int(used.sum())
    rows = []

    for th in THRESHOLDS:
        kept = weights >= th
        n_kept = int(kept.sum())
        n_hit = int((used & kept).sum())
        rows.append(
            {
                "threshold": th,
                "kept_share": n_kept / total,
                "precision": n_hit / n_kept if n_kept else float("nan"),
                "recall": n_hit / used_total if used_total else float("nan"),
            }
        )

    return rows


def main() -> None:
    arrays, _ = load_trace(sys.argv[1])
    decode_tokens = int((arrays["layer"][arrays["phase"] == 1] == 0).sum())
    print(f"{decode_tokens} decode tokens, {arrays['pred_experts'].shape[0]} predicted sets, "
          f"top-{arrays['pred_experts'].shape[1]}")
    for r in table(arrays):
        print(f"weight >= {r['threshold']:.2f}: kept {r['kept_share']:.0%}, "
              f"precision {r['precision']:.2f}, recall {r['recall']:.2f}")


if __name__ == "__main__":
    main()
