"""The bias-free MiniMax bank's encode/decode (cachalot.minimax.coded_bank, HANDOFF 18.4)."""

import numpy as np

from cachalot.minimax.coded_bank import BankLayout, bf16_bits_rne, decode_biases, encode_biases


def _bf16(x):
    return bf16_bits_rne(np.asarray(x, np.float32))


def _f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def test_roundtrip_is_exact_for_whole_multiples():
    rng = np.random.default_rng(0)
    scales = _bf16(rng.uniform(1e-4, 5e-2, 4096))
    k = rng.integers(-6, -2, 4096).astype(np.float32)
    biases = _bf16(k * _f32(scales))
    packed = encode_biases(scales, biases)
    assert packed is not None and packed.nbytes == 4096 // 4
    assert np.array_equal(decode_biases(scales, packed), biases)


def test_decode_writes_into_the_given_buffer():
    scales = _bf16(np.full(8, 0.01))
    biases = _bf16(np.float32(-4) * _f32(scales))
    out = np.zeros(8, np.uint16)
    decode_biases(scales, encode_biases(scales, biases), out)
    assert np.array_equal(out, biases)


def test_out_of_range_or_off_grid_biases_are_refused():
    scales = _bf16(np.full(8, 0.01))
    assert encode_biases(scales, _bf16(np.float32(-7) * _f32(scales))) is None  # k = -7: a raw record
    off = _bf16(np.float32(-4) * _f32(scales) + np.float32(1e-3))
    assert encode_biases(scales, off) is None


def test_rounding_matches_mlx_cast():
    import mlx.core as mx

    x = np.random.default_rng(1).standard_normal(10000).astype(np.float32) * 1e-2
    ours = bf16_bits_rne(x)
    theirs = np.array(mx.array(x).astype(mx.bfloat16).view(mx.uint16))
    assert np.array_equal(ours, theirs)


def test_layout_sizes_match_minimax():
    lay = BankLayout(weight=7077888, scales=589824)
    assert lay.codes == 73728
    assert lay.record("coded") == 23232512  # 22.16 MiB against 23.62 in the checkpoint
    assert lay.record("raw") > lay.record("coded")


def _tiny_bank(tmp_path, lay):
    """A two-record bank (one coded, one raw) in the real on-disk format, and the bytes each slot must hold."""
    import json

    from cachalot.minimax.coded_bank import PROJS, pack_codes

    rng = np.random.default_rng(3)
    groups = lay.scales // 2
    blob, recs, want = b"", [], []
    for e, kind in enumerate(("coded", "raw")):
        v = {}
        for p in PROJS:
            s = _bf16(rng.uniform(1e-3, 2e-2, groups))
            k = rng.integers(-6, -2, groups) if kind == "coded" else rng.integers(-7, -2, groups)
            v[f"{p}.scales"], v[f"{p}.biases"] = s, _bf16(k.astype(np.float32) * _f32(s))
            v[f"{p}.weight"] = rng.integers(0, 256, lay.weight, dtype=np.uint8)
            v[f"{p}.codes"] = pack_codes((k + 6).astype(np.uint8)) if kind == "coded" else None
        if kind == "coded":
            head = b"".join(v[f"{p}.scales"].tobytes() for p in PROJS) + b"".join(v[f"{p}.codes"].tobytes() for p in PROJS)
        else:
            head = b"".join(v[f"{p}.scales"].tobytes() for p in PROJS) + b"".join(v[f"{p}.biases"].tobytes() for p in PROJS)
        head += b"\0" * ((lay.coded_head if kind == "coded" else lay.raw_head) - len(head))
        recs.append([3, e, "layer-003.bin", len(blob), kind])
        blob += head + b"".join(v[f"{p}.weight"].tobytes() for p in PROJS)
        want.append(v)
    (tmp_path / "layer-003.bin").write_bytes(blob)
    (tmp_path / "bank.json").write_text(json.dumps(
        {"version": 1, "layout": {"weight": lay.weight, "scales": lay.scales}, "records": recs}))
    return want


def test_compressed_heads_fill_the_slot_with_the_same_bytes(tmp_path):
    from cachalot.minimax import coded_bank as cb
    from cachalot.storage.index import ExpertEntry

    lay = BankLayout(weight=8192, scales=2048)
    want = _tiny_bank(tmp_path, lay)
    n, raw, packed, _ = cb.write_zheads(tmp_path, level=3, threads=2)
    assert n == 2 and raw == lay.head_payload("coded") + lay.head_payload("raw")
    reader = cb.CodedBankReader(tmp_path, bypass_page_cache=False)
    assert len(reader.zheads) == 2
    names = [f"{p}.{t}" for p in cb.PROJS for t in ("weight", "scales", "biases")]
    try:
        for e in (0, 1):
            slots = []
            for z in (2, 0):
                cb.ZHEADS = z
                views = {k: np.full(lay.weight if k.endswith("weight") else lay.scales, 0xAB, np.uint8) for k in names}
                reader.read_expert_into(ExpertEntry(layer=3, expert=e, tensors=()), views)
                slots.append(views)
            for k in names:
                assert np.array_equal(slots[0][k], slots[1][k]), (e, k)
                assert np.array_equal(slots[0][k].view(np.uint8), want[e][k].view(np.uint8)), (e, k)
    finally:
        cb.ZHEADS = 1
    assert reader.zhead_reads == 2


def test_compressed_heads_only_for_bulk_reads_by_default(tmp_path):
    from cachalot.minimax import coded_bank as cb
    from cachalot.storage.index import ExpertEntry

    lay = BankLayout(weight=8192, scales=2048)
    _tiny_bank(tmp_path, lay)
    cb.write_zheads(tmp_path, level=3, threads=2)
    reader = cb.CodedBankReader(tmp_path, bypass_page_cache=False)
    names = [f"{p}.{t}" for p in cb.PROJS for t in ("weight", "scales", "biases")]
    views = {k: np.zeros(lay.weight if k.endswith("weight") else lay.scales, np.uint8) for k in names}
    assert cb.ZHEADS == 1
    reader.read_expert_into(ExpertEntry(layer=3, expert=0, tensors=()), views)
    assert reader.zhead_reads == 0
    reader.bulk = True
    reader.read_expert_into(ExpertEntry(layer=3, expert=0, tensors=()), views)
    assert reader.zhead_reads == 1
