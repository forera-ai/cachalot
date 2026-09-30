"""
A contiguous GLM-5.3-Flash routed-expert bank (HANDOFF 18.34).

The checkpoint keeps each expert as nine tensors (`{gate,up,down}_proj.{weight,scales,biases}`) that sit in
separate byte ranges of the safetensors shards, so one expert read is up to nine preads. The bank stores the same
bytes as one contiguous record per expert, in `ExpertFormat.tensor_names` order (the slot's own order), so a read
is one pread into the slot and outputs are bit-identical: nothing downstream changes.

Layout on disk: `bank.json` (version, tensor order, per-layer file and record size) and one `layer-NNN.bin` per MoE
layer, holding `n_experts` records of `record_bytes` each in expert order, padded to `ALIGN`. A bank may cover only
some layers; the others are read from the checkpoint as before. `CACHALOT_GLM_BANK` selects the bank directory;
`CACHALOT_GLM_BANK_ENABLED=0` ignores it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cachalot.storage.index import ExpertEntry, ExpertFormat, TensorRange

BANK_VERSION = 1
ALIGN = 16384
ENABLED = int(os.environ.get("CACHALOT_GLM_BANK_ENABLED", "1"))


def record_layout(fmt: ExpertFormat) -> tuple[dict[str, tuple[int, int]], int]:
    """({tensor name: (offset, size)} inside a record, record bytes padded to ALIGN)."""
    from cachalot.glm.experts import tensor_sizes

    sizes = tensor_sizes(fmt)
    layout, pos = {}, 0
    for name in fmt.tensor_names:
        layout[name] = (pos, sizes[name])
        pos += sizes[name]
    return layout, -(-pos // ALIGN) * ALIGN


def bank_dir() -> Path | None:
    raw = os.environ.get("CACHALOT_GLM_BANK", "")
    if not raw or not ENABLED:
        return None
    path = Path(raw)
    return path if (path / "bank.json").is_file() else None


def apply_bank(fmt: ExpertFormat, index: dict[tuple[int, int], ExpertEntry], bank: Path):
    """Replace the entries of every layer the bank holds by one range per expert inside its layer file."""
    meta = json.loads((bank / "bank.json").read_text())
    if meta["version"] != BANK_VERSION or tuple(meta["tensor_names"]) != fmt.tensor_names:
        raise ValueError(f"{bank}: bank format does not match the checkpoint's expert format")
    layout, record = record_layout(fmt)
    if record != meta["record_bytes"]:
        raise ValueError(f"{bank}: record size {meta['record_bytes']} differs from {record}")
    out = dict(index)
    for layer_s, info in meta["layers"].items():
        layer, path = int(layer_s), bank / info["file"]
        if not path.is_file() or path.stat().st_size < info["experts"] * record:
            raise ValueError(f"{path}: missing or short")
        for expert in range(info["experts"]):
            base = expert * record
            out[(layer, expert)] = ExpertEntry(
                layer=layer,
                expert=expert,
                tensors=tuple(
                    TensorRange(
                        name=f"layers.{layer}.experts.{expert}.{name}",
                        shard=path,
                        start=base + off,
                        end=base + off + size,
                    )
                    for name, (off, size) in layout.items()
                ),
            )
    return out
