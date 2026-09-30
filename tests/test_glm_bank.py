import json

import numpy as np

from cachalot.glm import bank
from cachalot.glm.experts import tensor_sizes
from cachalot.storage.index import ExpertEntry, ExpertFormat, TensorRange, merge_contiguous_ranges


def _fmt():
    names = ("w1.biases", "w1.scales", "w1.weight")
    shapes = {"w1.biases": (8, 2), "w1.scales": (8, 2), "w1.weight": (8, 8)}
    dtypes = {"w1.biases": "bfloat16", "w1.scales": "bfloat16", "w1.weight": "uint32"}
    return ExpertFormat(kind="affine", bits=4, group_size=64, tensor_names=names, shapes=shapes, dtypes=dtypes)


def test_record_layout_is_slot_order_and_aligned():
    fmt = _fmt()
    layout, record = bank.record_layout(fmt)
    sizes = tensor_sizes(fmt)
    assert list(layout) == list(fmt.tensor_names)
    assert layout["w1.biases"] == (0, sizes["w1.biases"])
    assert layout["w1.scales"][0] == sizes["w1.biases"]
    assert record % bank.ALIGN == 0 and record >= sum(sizes.values())


def test_apply_bank_gives_one_contiguous_range_per_expert(tmp_path, monkeypatch):
    fmt = _fmt()
    layout, record = bank.record_layout(fmt)
    (tmp_path / "layer-003.bin").write_bytes(bytes(record * 2))
    (tmp_path / "bank.json").write_text(json.dumps({
        "version": bank.BANK_VERSION, "tensor_names": list(fmt.tensor_names), "record_bytes": record,
        "layers": {"3": {"file": "layer-003.bin", "experts": 2}}}))
    index = {(4, 0): ExpertEntry(layer=4, expert=0, tensors=())}
    out = bank.apply_bank(fmt, index, tmp_path)
    assert (4, 0) in out  # a layer the bank does not hold keeps its checkpoint entry
    entry = out[(3, 1)]
    ranges = merge_contiguous_ranges(entry)
    assert len(ranges) == 1
    assert ranges[0].start == record and ranges[0].end == record + sum(t.size for t in entry.tensors)


def test_apply_bank_rejects_a_different_format(tmp_path):
    fmt = _fmt()
    _, record = bank.record_layout(fmt)
    (tmp_path / "bank.json").write_text(json.dumps({
        "version": bank.BANK_VERSION, "tensor_names": ["x"], "record_bytes": record, "layers": {}}))
    try:
        bank.apply_bank(fmt, {}, tmp_path)
    except ValueError:
        return
    raise AssertionError("format mismatch was accepted")
