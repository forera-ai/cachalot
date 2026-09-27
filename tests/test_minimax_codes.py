"""Expert slots holding 4-bit bias codes (cachalot.minimax.codes_qmv, HANDOFF 18.10)."""

import mlx.core as mx
import numpy as np

from cachalot.minimax import codes_qmv
from test_minimax_coded_bank import _bf16, _f32, _tiny_bank


def _nibbles(k):
    c = (np.asarray(k) + 7).astype(np.uint8).reshape(-1)
    return c[0::2] | (c[1::2] << 4)


def test_kernels_match_mlx_bit_for_bit():
    assert codes_qmv.self_check(shapes=((64, 512), (512, 1024), (3072, 6144)), trials=2)


def test_rebuild_matches_the_banks_cpu_table():
    from cachalot.minimax.coded_bank import decode_biases, pack_codes

    rng = np.random.default_rng(0)
    scales = _bf16(rng.uniform(1e-4, 5e-2, 8192))
    k = rng.integers(-6, -2, 8192)
    want = decode_biases(scales, pack_codes((k + 6).astype(np.uint8)))
    got = codes_qmv.rebuild_biases(mx.array(scales).view(mx.bfloat16), mx.array(_nibbles(k)))
    assert np.array_equal(np.array(got.view(mx.uint16)), want)


def test_nibbles_from_bank_codes_and_from_biases():
    from cachalot.minimax.coded_bank import pack_codes

    rng = np.random.default_rng(1)
    k = rng.integers(-6, -2, 4096)
    out = np.zeros(2048, np.uint8)
    codes_qmv.nibbles_from_packed(pack_codes((k + 6).astype(np.uint8)), -6, out)
    assert np.array_equal(out, _nibbles(k))

    k = rng.integers(-7, -2, 4096)  # a raw record's k = -7 groups fit a nibble too
    scales = _bf16(rng.uniform(1e-3, 2e-2, 4096))
    biases = _bf16(k.astype(np.float32) * _f32(scales))
    codes_qmv.nibbles_from_biases(scales, biases, out)
    assert np.array_equal(out, _nibbles(k))


def test_off_grid_bias_is_refused():
    import pytest

    scales = _bf16(np.full(8, 0.01))
    off = _bf16(np.float32(-4) * _f32(scales) + np.float32(1e-3))
    with pytest.raises(ValueError):
        codes_qmv.nibbles_from_biases(scales, off, np.zeros(4, np.uint8))


def test_codes_slot_holds_weights_scales_and_the_codes_of_the_biases(tmp_path):
    from cachalot.minimax import coded_bank as cb
    from cachalot.storage.index import ExpertEntry

    lay = cb.BankLayout(weight=8192, scales=2048)
    want = _tiny_bank(tmp_path, lay)
    reader = cb.CodedBankReader(tmp_path, bypass_page_cache=False)
    for z in (0, 2):  # the plain head and the compressed one
        cb.ZHEADS = z
        try:
            (tmp_path / "heads.json").exists() or z == 0 or cb.write_zheads(tmp_path, level=3, threads=2)
            if z == 2:
                reader = cb.CodedBankReader(tmp_path, bypass_page_cache=False)
            for e in (0, 1):  # coded, raw
                views = {}
                for p in cb.PROJS:
                    views[f"{p}.weight"] = np.zeros(lay.weight, np.uint8)
                    views[f"{p}.scales"] = np.zeros(lay.scales, np.uint8)
                    views[f"{p}.codes"] = np.zeros(lay.scales // 4, np.uint8)
                reader.read_expert_into(ExpertEntry(layer=3, expert=e, tensors=()), views)
                for p in cb.PROJS:
                    assert np.array_equal(views[f"{p}.weight"], want[e][f"{p}.weight"])
                    assert np.array_equal(views[f"{p}.scales"].view(np.uint16), want[e][f"{p}.scales"])
                    s = mx.array(views[f"{p}.scales"].view(np.uint16)).view(mx.bfloat16)
                    b = codes_qmv.rebuild_biases(s, mx.array(views[f"{p}.codes"]))
                    assert np.array_equal(np.array(b.view(mx.uint16)), want[e][f"{p}.biases"]), (z, e, p)
        finally:
            cb.ZHEADS = 1


def test_slot_format_replaces_biases_with_codes():
    from cachalot.glm.experts import tensor_sizes
    from cachalot.minimax.coded_bank import slot_format
    from cachalot.storage.index import ExpertFormat

    shapes, dtypes = {}, {}
    for p, (o, i) in {"w1": (3072, 6144), "w3": (3072, 6144), "w2": (6144, 3072)}.items():
        shapes.update({f"{p}.weight": (o, i * 3 // 32), f"{p}.scales": (o, i // 64), f"{p}.biases": (o, i // 64)})
        dtypes.update({f"{p}.weight": "uint32", f"{p}.scales": "bfloat16", f"{p}.biases": "bfloat16"})
    fmt = ExpertFormat(kind="affine", bits=3, group_size=64, tensor_names=tuple(sorted(shapes)), shapes=shapes,
                       dtypes=dtypes)
    before, after = sum(tensor_sizes(fmt).values()), sum(tensor_sizes(slot_format(fmt)).values())
    assert round(before / 2**20, 2) == 23.62 and round(after / 2**20, 2) == 22.36
    assert "w1.codes" in slot_format(fmt).tensor_names and "w1.biases" not in slot_format(fmt).tensor_names


def test_switch_codes_path_equals_the_bias_path():
    """StreamingSwitchGLU on a codes slot (decode kernel and prefill rebuild) against the same expert with biases."""
    from types import SimpleNamespace

    from cachalot.glm.experts import StreamingSwitchGLU, _typed
    from cachalot.minimax.coded_bank import slot_format
    from cachalot.minimax.language import swiglu_oai
    from cachalot.storage.index import ExpertFormat

    rng = np.random.default_rng(5)
    shapes, dtypes, arrays = {}, {}, {}
    for p, (o, i) in {"w1": (512, 1024), "w3": (512, 1024), "w2": (1024, 512)}.items():
        g = i // 64
        s = _bf16(rng.uniform(1e-3, 2e-2, o * g))
        k = rng.integers(-7, -2, o * g)
        arrays[f"{p}.weight"] = rng.integers(0, 2**32, o * i * 3 // 32, dtype=np.uint32).view(np.uint8)
        arrays[f"{p}.scales"] = s.view(np.uint8)
        arrays[f"{p}.biases"] = _bf16(k.astype(np.float32) * _f32(s)).view(np.uint8)
        arrays[f"{p}.codes"] = _nibbles(k)
        shapes.update({f"{p}.weight": (o, i * 3 // 32), f"{p}.scales": (o, g), f"{p}.biases": (o, g)})
        dtypes.update({f"{p}.weight": "uint32", f"{p}.scales": "bfloat16", f"{p}.biases": "bfloat16"})
    fmt = ExpertFormat(kind="affine", bits=3, group_size=64, tensor_names=tuple(sorted(shapes)), shapes=shapes,
                       dtypes=dtypes)
    cfmt = slot_format(fmt)

    def slot(f):
        return SimpleNamespace(arrays={n: mx.array(arrays[n]) for n in f.tensor_names}, typed={})

    act = lambda up, gate: swiglu_oai(gate, up, 1.702, 7.0)  # noqa: E731
    plain = StreamingSwitchGLU(3, None, {}, fmt, act)
    coded = StreamingSwitchGLU(3, None, {}, cfmt, act)
    coded.codes = True
    sp, sc = slot(fmt), slot(cfmt)
    assert _typed(sc, cfmt, "w1")[2].dtype == mx.uint8
    for rows in (1, 5):  # one decode row (the kernels), a prefill chunk (the rebuild)
        x = mx.array(rng.normal(0, 1, (rows, 1024)).astype(np.float32)).astype(mx.bfloat16)
        assert mx.array_equal(plain._expert_out(x, sp), coded._expert_out(x, sc)).item(), rows


def test_scale_index_slot_format_and_fill(tmp_path):
    """HANDOFF 18.13: a scale-index slot is 21.52 MiB (tensors) at MiniMax's shapes and holds a byte index + table whose
    lookup gives back every bf16 scale of the record, for coded and raw records, plain and compressed heads."""
    from cachalot.glm.experts import tensor_sizes
    from cachalot.minimax import coded_bank as cb
    from cachalot.minimax.coded_bank import slot_format
    from cachalot.storage.index import ExpertEntry, ExpertFormat

    shapes, dtypes = {}, {}
    for p, (o, i) in {"w1": (3072, 6144), "w3": (3072, 6144), "w2": (6144, 3072)}.items():
        shapes.update({f"{p}.weight": (o, i * 3 // 32), f"{p}.scales": (o, i // 64), f"{p}.biases": (o, i // 64)})
        dtypes.update({f"{p}.weight": "uint32", f"{p}.scales": "bfloat16", f"{p}.biases": "bfloat16"})
    fmt = ExpertFormat(kind="affine", bits=3, group_size=64, tensor_names=tuple(sorted(shapes)), shapes=shapes,
                       dtypes=dtypes)
    sf = slot_format(fmt, sidx=True)
    assert round(sum(tensor_sizes(sf).values()) / 2**20, 2) == 21.52
    assert "w1.sidx" in sf.tensor_names and "w1.scales" not in sf.tensor_names

    lay = cb.BankLayout(weight=8192, scales=2048)
    want = _tiny_bank(tmp_path, lay, distinct=170)
    for z in (0, 2):
        cb.ZHEADS = z
        try:
            z == 0 or (tmp_path / "heads.json").exists() or cb.write_zheads(tmp_path, level=3, threads=2)
            reader = cb.CodedBankReader(tmp_path, bypass_page_cache=False)
            for e in (0, 1):
                views = {}
                for p in cb.PROJS:
                    views[f"{p}.weight"] = np.zeros(lay.weight, np.uint8)
                    views[f"{p}.sidx"] = np.zeros(lay.scales // 2, np.uint8)
                    views[f"{p}.lut"] = np.zeros(512, np.uint8)
                    views[f"{p}.codes"] = np.zeros(lay.scales // 4, np.uint8)
                reader.read_expert_into(ExpertEntry(layer=3, expert=e, tensors=()), views)
                for p in cb.PROJS:
                    assert np.array_equal(views[f"{p}.weight"], want[e][f"{p}.weight"])
                    lut = views[f"{p}.lut"].view(np.uint16)
                    assert np.array_equal(lut[views[f"{p}.sidx"]], want[e][f"{p}.scales"]), (z, e, p)
                    s, b = codes_qmv.rebuild_sidx(mx.array(views[f"{p}.sidx"]), mx.array(lut).view(mx.bfloat16),
                                                  mx.array(views[f"{p}.codes"]))
                    assert np.array_equal(np.array(s.view(mx.uint16)), want[e][f"{p}.scales"])
                    assert np.array_equal(np.array(b.view(mx.uint16)), want[e][f"{p}.biases"]), (z, e, p)
        finally:
            cb.ZHEADS = 1


def test_switch_scale_index_path_equals_the_bias_path():
    """StreamingSwitchGLU on a scale-index slot (decode kernel and prefill rebuild) against the bias slot."""
    from types import SimpleNamespace

    from cachalot.glm.experts import StreamingSwitchGLU
    from cachalot.minimax.coded_bank import slot_format
    from cachalot.minimax.language import swiglu_oai
    from cachalot.storage.index import ExpertFormat

    rng = np.random.default_rng(6)
    table = _bf16(rng.uniform(1e-3, 2e-2, 120))
    shapes, dtypes, arrays = {}, {}, {}
    for p, (o, i) in {"w1": (512, 1024), "w3": (512, 1024), "w2": (1024, 512)}.items():
        g = i // 64
        s = table[rng.integers(0, table.size, o * g)]
        k = rng.integers(-7, -2, o * g)
        arrays[f"{p}.weight"] = rng.integers(0, 2**32, o * i * 3 // 32, dtype=np.uint32).view(np.uint8)
        arrays[f"{p}.scales"] = s.view(np.uint8)
        arrays[f"{p}.biases"] = _bf16(k.astype(np.float32) * _f32(s)).view(np.uint8)
        arrays[f"{p}.codes"] = _nibbles(k)
        si, lt = np.zeros(o * g, np.uint8), np.zeros(256, np.uint16)
        codes_qmv.sidx_from_scales(s, si, lt)
        arrays[f"{p}.sidx"], arrays[f"{p}.lut"] = si, lt.view(np.uint8)
        shapes.update({f"{p}.weight": (o, i * 3 // 32), f"{p}.scales": (o, g), f"{p}.biases": (o, g)})
        dtypes.update({f"{p}.weight": "uint32", f"{p}.scales": "bfloat16", f"{p}.biases": "bfloat16"})
    fmt = ExpertFormat(kind="affine", bits=3, group_size=64, tensor_names=tuple(sorted(shapes)), shapes=shapes,
                       dtypes=dtypes)
    sfmt = slot_format(fmt, sidx=True)

    def slot(f):
        return SimpleNamespace(arrays={n: mx.array(arrays[n]) for n in f.tensor_names}, typed={})

    act = lambda up, gate: swiglu_oai(gate, up, 1.702, 7.0)  # noqa: E731
    plain = StreamingSwitchGLU(3, None, {}, fmt, act)
    coded = StreamingSwitchGLU(3, None, {}, sfmt, act)
    coded.codes = True
    sp, sc = slot(fmt), slot(sfmt)
    for rows in (1, 5):
        x = mx.array(rng.normal(0, 1, (rows, 1024)).astype(np.float32)).astype(mx.bfloat16)
        assert mx.array_equal(plain._expert_out(x, sp), coded._expert_out(x, sc)).item(), rows


def test_pair_slot_format_fill_and_switch(tmp_path):
    """HANDOFF 18.13 M23b: a pair slot is 21.10 MiB at MiniMax's shapes; a read fills a byte index + (scale, bias)
    table that gives back every scale and bias of the record (coded and raw); the switch path equals the bias path."""
    from types import SimpleNamespace

    from cachalot.glm.experts import StreamingSwitchGLU, tensor_sizes
    from cachalot.minimax import coded_bank as cb
    from cachalot.minimax.language import swiglu_oai
    from cachalot.storage.index import ExpertEntry, ExpertFormat

    shapes, dtypes = {}, {}
    for p, (o, i) in {"w1": (3072, 6144), "w3": (3072, 6144), "w2": (6144, 3072)}.items():
        shapes.update({f"{p}.weight": (o, i * 3 // 32), f"{p}.scales": (o, i // 64), f"{p}.biases": (o, i // 64)})
        dtypes.update({f"{p}.weight": "uint32", f"{p}.scales": "bfloat16", f"{p}.biases": "bfloat16"})
    fmt = ExpertFormat(kind="affine", bits=3, group_size=64, tensor_names=tuple(sorted(shapes)), shapes=shapes,
                       dtypes=dtypes)
    pf = cb.slot_format(fmt, 2)
    assert round(sum(tensor_sizes(pf).values()) / 2**20, 2) == 21.1
    assert "w1.pidx" in pf.tensor_names and "w1.codes" not in pf.tensor_names

    lay = cb.BankLayout(weight=8192, scales=2048)
    want = _tiny_bank(tmp_path, lay, distinct=50)
    reader = cb.CodedBankReader(tmp_path, bypass_page_cache=False)
    for e in (0, 1):
        v = {}
        for p in cb.PROJS:
            v[f"{p}.weight"] = np.zeros(lay.weight, np.uint8)
            v[f"{p}.pidx"] = np.zeros(lay.scales // 2, np.uint8)
            v[f"{p}.plut"] = np.zeros(1024, np.uint8)
        reader.read_expert_into(ExpertEntry(layer=3, expert=e, tensors=()), v)
        for p in cb.PROJS:
            lut = v[f"{p}.plut"].view(np.uint16)
            assert np.array_equal(lut[0::2][v[f"{p}.pidx"]], want[e][f"{p}.scales"]), (e, p)
            assert np.array_equal(lut[1::2][v[f"{p}.pidx"]], want[e][f"{p}.biases"]), (e, p)

    rng = np.random.default_rng(8)
    table = _bf16(rng.uniform(1e-3, 2e-2, 40))
    shapes, dtypes, arrays = {}, {}, {}
    for p, (o, i) in {"w1": (512, 1024), "w3": (512, 1024), "w2": (1024, 512)}.items():
        g = i // 64
        s = table[rng.integers(0, table.size, o * g)]
        k = rng.integers(-7, -2, o * g)
        arrays[f"{p}.weight"] = rng.integers(0, 2**32, o * i * 3 // 32, dtype=np.uint32).view(np.uint8)
        arrays[f"{p}.scales"] = s.view(np.uint8)
        arrays[f"{p}.biases"] = _bf16(k.astype(np.float32) * _f32(s)).view(np.uint8)
        pi, pl = np.zeros(o * g, np.uint8), np.zeros(512, np.uint16)
        codes_qmv.pair_from(s, (k + 7).astype(np.uint8), pi, pl)
        arrays[f"{p}.pidx"], arrays[f"{p}.plut"] = pi, pl.view(np.uint8)
        shapes.update({f"{p}.weight": (o, i * 3 // 32), f"{p}.scales": (o, g), f"{p}.biases": (o, g)})
        dtypes.update({f"{p}.weight": "uint32", f"{p}.scales": "bfloat16", f"{p}.biases": "bfloat16"})
    fmt = ExpertFormat(kind="affine", bits=3, group_size=64, tensor_names=tuple(sorted(shapes)), shapes=shapes,
                       dtypes=dtypes)
    pfmt = cb.slot_format(fmt, 2)

    def slot(f):
        return SimpleNamespace(arrays={n: mx.array(arrays[n]) for n in f.tensor_names}, typed={})

    act = lambda up, gate: swiglu_oai(gate, up, 1.702, 7.0)  # noqa: E731
    plain = StreamingSwitchGLU(3, None, {}, fmt, act)
    paired = StreamingSwitchGLU(3, None, {}, pfmt, act)
    paired.codes = True
    for rows in (1, 5):
        x = mx.array(rng.normal(0, 1, (rows, 1024)).astype(np.float32)).astype(mx.bfloat16)
        assert mx.array_equal(plain._expert_out(x, slot(fmt)), paired._expert_out(x, slot(pfmt))).item(), rows
