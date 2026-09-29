"""M24 (HANDOFF 18.16): a MiniMax bank whose records are slot images (a byte (scale, bias) index per group and the
pair table, then the weights) fills every kind of slot with the same bytes as the coded and raw records it replaces."""

import json

import numpy as np
from test_minimax_codes import _tiny_bank

from cachalot.minimax import coded_bank as cb
from cachalot.storage.index import ExpertEntry

LAY = cb.BankLayout(weight=8192, scales=2048)


def _slot_bank(src, dst):
    """The tiny bank of `src` rewritten as slot records in `dst`, as benchmarks/minimax_coded_bank.py --to-slot does."""
    meta = json.loads((src / "bank.json").read_text())
    blob, recs = b"", []
    data = (src / "layer-003.bin").read_bytes()
    for r in meta["records"]:
        kind, off = r[4], r[3]
        head = cb.slot_head_from(kind, data[off:off + LAY.head_payload(kind)], LAY)
        assert len(head) == LAY.slot_payload
        weights = data[off + LAY.head(kind):off + LAY.record(kind)]
        recs.append([r[0], r[1], "layer-003-slot.bin", len(blob), "slot"])
        blob += head + b"\0" * (LAY.slot_head - len(head)) + weights
    (dst / "layer-003-slot.bin").write_bytes(blob)
    meta.update(version=2, records=recs)
    (dst / "bank.json").write_text(json.dumps(meta))


def _views(names):
    size = {"weight": LAY.weight, "scales": LAY.scales, "biases": LAY.scales, "codes": LAY.scales // 4,
            "sidx": LAY.groups, "lut": 512, "pidx": LAY.groups, "plut": 1024}
    return {f"{p}.{n}": np.full(size[n], 0xAB, np.uint8) for p in cb.PROJS for n in names}


SLOTS = {
    "bf16": ("weight", "scales", "biases"),
    "codes": ("weight", "scales", "codes"),
    "sidx": ("weight", "sidx", "lut", "codes"),
    "pair": ("weight", "pidx", "plut"),
}


def _fill(reader, e, names):
    v = _views(names)
    reader.read_expert_into(ExpertEntry(layer=3, expert=e, tensors=()), v)
    return v


def test_slot_records_fill_every_slot_kind_like_the_old_records(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    _tiny_bank(old, LAY, distinct=50)
    _slot_bank(old, new)
    a = cb.CodedBankReader(old, bypass_page_cache=False)
    b = cb.CodedBankReader(new, bypass_page_cache=False)
    assert LAY.head_payload("slot") < LAY.head_payload("coded")
    for name, names in SLOTS.items():
        for e in (0, 1):  # a coded record and a raw one
            va, vb = _fill(a, e, names), _fill(b, e, names)
            for k in va:
                assert np.array_equal(va[k], vb[k]), (name, e, k)


def test_compressed_slot_heads_and_stale_version_1_heads(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    _tiny_bank(old, LAY, distinct=50)
    cb.write_zheads(old, level=3, threads=2)
    _slot_bank(old, new)
    # the old bank's heads.zst, written as a version 1 file, next to slot records: never used
    z = json.loads((old / "heads.json").read_text())
    z.update(version=1, records=[r[:4] for r in z["records"]])
    (new / "heads.json").write_text(json.dumps(z))
    (new / "heads.zst").write_bytes((old / "heads.zst").read_bytes())
    assert cb.CodedBankReader(new, bypass_page_cache=False).zheads == {}
    n, raw, _, _ = cb.write_zheads(new, level=3, threads=2)
    assert n == 2 and raw == 2 * LAY.slot_payload
    ref = cb.CodedBankReader(old, bypass_page_cache=False)
    reader = cb.CodedBankReader(new, bypass_page_cache=False)
    assert len(reader.zheads) == 2
    try:
        cb.ZHEADS = 2
        for names in SLOTS.values():
            for e in (0, 1):
                va, vb = _fill(ref, e, names), _fill(reader, e, names)
                assert all(np.array_equal(va[k], vb[k]) for k in va), (names, e)
    finally:
        cb.ZHEADS = 1
    assert reader.zhead_reads == 2 * len(SLOTS)


def test_a_mirror_serves_only_the_records_it_holds_at_the_same_place(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    _tiny_bank(old, LAY, distinct=50)
    _slot_bank(old, new)
    lagging = cb.CodedBankReader(new, old, bypass_page_cache=False)  # the mirror still holds the old records
    assert lagging.mirror_same == set()
    same = cb.CodedBankReader(new, new, bypass_page_cache=False)
    assert same.mirror_same == {(3, 0), (3, 1)}
    lagging.mirror_fraction = 0.5
    ref = cb.CodedBankReader(new, bypass_page_cache=False)
    for e in (0, 1):
        va, vb = _fill(ref, e, SLOTS["pair"]), _fill(lagging, e, SLOTS["pair"])
        assert all(np.array_equal(va[k], vb[k]) for k in va)


def test_a_tail_only_mirror_fills_the_slot_like_the_bank(tmp_path):
    old, new, tail = tmp_path / "old", tmp_path / "new", tmp_path / "tail"
    old.mkdir()
    new.mkdir()
    _tiny_bank(old, LAY, distinct=50)
    _slot_bank(old, new)
    n, written, _ = cb.write_mirror_tail(new, tail, tail=0.5, threads=2)
    assert n == 2 and written == 2 * 3 * (LAY.weight - int(LAY.weight * 0.5) // 4096 * 4096)
    ref = cb.CodedBankReader(new, bypass_page_cache=False)
    reader = cb.CodedBankReader(new, tail, bypass_page_cache=False)
    assert reader.mirror_tail == 0.5 and reader.mirror_same == {(3, 0), (3, 1)}
    for frac in (0.5, 0.9):  # 0.9 is capped at the mirror's tail
        reader.mirror_fraction = frac
        for e in (0, 1):
            va, vb = _fill(ref, e, SLOTS["pair"]), _fill(reader, e, SLOTS["pair"])
            assert all(np.array_equal(va[k], vb[k]) for k in va), (frac, e)


def test_the_mirror_share_follows_the_slower_drive_and_keeps_the_bytes(tmp_path, monkeypatch):
    # HANDOFF 18.24: the share moves towards the split where both drives finish together, capped at the configured
    # fraction, floored so the mirror keeps being measured; the bytes are the bank's whatever the split
    old, new, tail = tmp_path / "old", tmp_path / "new", tmp_path / "tail"
    old.mkdir()
    new.mkdir()
    _tiny_bank(old, LAY, distinct=50)
    _slot_bank(old, new)
    cb.write_mirror_tail(new, tail, tail=0.5, threads=2)
    ref = cb.CodedBankReader(new, bypass_page_cache=False)
    reader = cb.CodedBankReader(new, tail, bypass_page_cache=False)
    reader.mirror_fraction = 0.3
    for _ in range(200):  # a mirror a tenth as fast as the bank
        reader._adapt_share(1.0, 10.0)
    assert abs(reader.mirror_share - 1 / 11) < 1e-3
    for _ in range(400):  # a stalled mirror: down to the floor, not to zero
        reader._adapt_share(1e-3, 10.0)
    assert reader.mirror_share == cb.MIRROR_ADAPT_FLOOR
    for _ in range(400):  # a mirror faster than the bank: back up, but never above the configured fraction
        reader._adapt_share(100.0, 1.0)
    assert reader.mirror_share == 0.3
    reader.mirror_share = 0.05
    for e in (0, 1):
        va, vb = _fill(ref, e, SLOTS["pair"]), _fill(reader, e, SLOTS["pair"])
        assert all(np.array_equal(va[k], vb[k]) for k in va), e
    assert cb.MIRROR_ADAPT_FLOOR <= reader.mirror_share <= 0.3
