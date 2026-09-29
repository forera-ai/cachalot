"""
A MiniMax-M3 routed-expert bank without the biases (HANDOFF 18.4).

MLX's affine quantization stores, per group of 64 weights, a bf16 scale and a bf16 bias, and the bias is the
group's edge value, which the quantizer makes a whole multiple of the scale. For this checkpoint every one of
its 6.46 billion expert groups satisfies

    bias == bf16_round_to_nearest_even(float32(k) * float32(scale)),   k in {-7, -6, -5, -4, -3}

(benchmarks/minimax_coded_bank.py --scan checks it). So a bias is two bits (k + 6 for k in -6..-3) once the
scale is known, and an expert shrinks from 23.62 MiB (nine pieces in the stacked safetensors) to 22.16 MiB,
stored as one contiguous record per expert:

    head    w1.scales | w3.scales | w2.scales | codes w1 | codes w3 | codes w2     (1.90 MiB, padded to 16 KiB)
    weights w1.weight | w3.weight | w2.weight                                      (20.25 MiB)

The 3.6 % of experts with any k = -7 group keep their biases (a "raw" record: the six scale/bias pieces as the
head, then the weights). A read issues the three weight preads first, reads the head into the slot's scale
views and a code buffer, and rebuilds the biases straight into the slot's bias views through a 256K-entry table
(scale bits, code) -> bias bits while the weights are still arriving. The slot then holds exactly the bytes
the stacked checkpoint would have put there, so nothing downstream changes and outputs are bit-identical.

Layout on disk: `bank.json` (format, the records: layer, expert, file, offset, kind) and one `layer-NNN.bin`
per MoE layer. A bank may cover only some layers; the others are read from the checkpoint as before.
A bank may also hold `heads.zst` + `heads.json` (HANDOFF 18.7): every record's head (scales and codes, or scales
and biases) compressed with zstd, 0.29 of its size, since bf16 scales and 2-bit codes are far from random while the
3-bit weights are not (zstd gets 3 % there). A read then takes the head from that file (4 KiB-aligned blobs) and
the weights from the record as before: 1.4 MB fewer bytes per expert (-6 %), the same bytes in the slot.
Only bulk (prefill) reads use it by default: a decode read waits on the ~1.5 ms decompression and got slower.
`CACHALOT_MINIMAX_ZHEADS=0` never uses it, `=2` always does (ZHEADS, a module int, for TF_ALTERNATE).

`CACHALOT_MINIMAX_BANK` selects the bank directory, `CACHALOT_MINIMAX_BANK_MIRROR` an identical copy on a
second drive (the tail `CACHALOT_MIRROR_FRACTION` of each weight piece is read from it, like the stacked
checkpoint's mirror). ENABLED (a module constant) may be flipped between reads: both paths fill a slot with
the same bytes, which is what makes a token-by-token A/B possible (TF_ALTERNATE).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cachalot.storage.index import ExpertEntry, ExpertFormat
from cachalot.storage.reader import ExpertReader

BANK_VERSION = 2  # 2 may hold slot records (M24); 1 holds only coded and raw ones
BANK_VERSIONS = (1, 2)
ALIGN = 16384
PLUT_BYTES = 1024  # a projection's table of 256 (scale, bias) pairs of bf16
PROJS = ("w1", "w3", "w2")
K_BASE = -6  # code c in 0..3 means k = c - 6
ENABLED = int(os.environ.get("CACHALOT_MINIMAX_BANK_ENABLED", "1"))  # an int, so TF_ALTERNATE can flip it
# Experiment knob: a mirror fraction that overrides the reader's own when >= 0 (TF_ALTERNATE A/Bs, HANDOFF 18.5)
MIRROR_FRACTION = -1.0
# HANDOFF 18.24: the mirror's share follows how fast each drive answers. Each record's weight pieces are timed; the
# share that would make the bank's and the mirror's pieces finish together, rm / (rm + rb) from their bytes per
# second, is averaged over reads (MIRROR_ADAPT_ALPHA) and used for the next record, never above the configured
# fraction (the measured optimum on an idle X10Pro, 18.3-18.14) and never below MIRROR_ADAPT_FLOOR (a small piece
# keeps measuring the mirror). With the X10Pro busy (Spotlight, a copy), a fixed 13 % made every expert wait for its
# USB piece: a 1,874-token side request 11.9 -> 41.4 s, decode reads 4.5 -> 6.3 ms. 0 keeps the share fixed.
MIRROR_ADAPT = int(os.environ.get("CACHALOT_MINIMAX_MIRROR_ADAPT", "1"))
MIRROR_ADAPT_ALPHA = 0.05
MIRROR_ADAPT_FLOOR = 0.02


def _timed_preadv(fd, bufs, pos):
    t0 = time.perf_counter()
    got = os.preadv(fd, bufs, pos)
    return got, time.perf_counter() - t0
# 1: compressed heads for bulk (prefill) reads only; 2: for every read; 0: never. An int, so TF_ALTERNATE can flip
# it. Decode waits on each expert, and the ~1.5 ms decompression is on that path: 18.7 measured it slower there.
ZHEADS = int(os.environ.get("CACHALOT_MINIMAX_ZHEADS", "1"))
# M24 (HANDOFF 18.16): a slot record's head is read into a scratch buffer and copied into the slot (1); read straight into
# the slab views (0) a miss waited 3.33 instead of 2.84 ms in the decode loop. An int, so TF_ALTERNATE can flip it.
SLOT_HEAD_SCRATCH = int(os.environ.get("CACHALOT_MINIMAX_SLOT_HEAD_SCRATCH", "1"))
ZHEADS_VERSION = 2  # 2 records each head's kind; 1 held only coded and raw heads
ZHEADS_VERSIONS = (1, 2)

try:
    from compression import zstd as _zstd  # Python 3.14+
except ImportError:  # pragma: no cover
    _zstd = None


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def bf16_bits_rne(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns, round to nearest even (what MLX's cast does)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bias_table() -> np.ndarray:
    """table[(scale_bits << 2) | code] = bf16 bits of (code + K_BASE) * scale."""
    scale = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    table = np.empty(65536 * 4, np.uint16)
    with np.errstate(all="ignore"):
        for c in range(4):
            table[c::4] = bf16_bits_rne(scale * np.float32(c + K_BASE))
    return table


_TABLE: np.ndarray | None = None


def bias_table() -> np.ndarray:
    global _TABLE
    if _TABLE is None:
        _TABLE = _bias_table()
    return _TABLE


_SHIFTS = np.array([0, 2, 4, 6], np.uint8)


def encode_biases(scales: np.ndarray, biases: np.ndarray) -> np.ndarray | None:
    """Packed 2-bit codes (4 per byte, low bits first) for bf16 bit patterns, or None when any group's bias
    is not (k * scale) rounded for k in -6..-3."""
    s = (scales.astype(np.uint32) << 16).view(np.float32)
    b = (biases.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.round(b / s)
    k = np.where(s == 0, K_BASE, k)
    if k.min() < K_BASE or k.max() > K_BASE + 3:
        return None
    codes = (k - K_BASE).astype(np.uint8)
    if not np.array_equal(decode_biases(scales, pack_codes(codes)), biases):
        return None
    return pack_codes(codes)


def pack_codes(codes: np.ndarray) -> np.ndarray:
    return (codes.reshape(-1, 4) << _SHIFTS).sum(axis=1, dtype=np.uint8)


def decode_biases(scales: np.ndarray, packed: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """bf16 bias bit patterns from bf16 scale bit patterns and packed codes (the read path's rebuild)."""
    codes = ((packed[:, None] >> _SHIFTS) & 3).reshape(-1)
    idx = (scales.astype(np.uint32) << 2) | codes
    if out is None:
        out = np.empty(idx.shape, np.uint16)
    np.take(bias_table(), idx, out=out)
    return out


def slot_head_from(kind: str, payload: bytes, lay: "BankLayout") -> bytes:
    """A coded or raw record's head payload -> a slot record's (M24): pidx w1 | w3 | w2, then plut w1 | w3 | w2,
    exactly what the read path's pair conversion (codes_qmv.pair_from) puts into a pair slot."""
    from cachalot.minimax import codes_qmv

    g = lay.groups
    s = np.frombuffer(payload, np.uint16, 3 * g)
    if kind == "coded":
        packed = np.frombuffer(payload, np.uint8, 3 * lay.codes, 3 * lay.scales)
        n = lay.codes
        cs = [codes_qmv.codes_from_packed(packed[i * n:(i + 1) * n], K_BASE) for i in range(3)]
    elif kind == "raw":
        b = np.frombuffer(payload, np.uint16, 3 * g, 3 * lay.scales)
        cs = [codes_qmv.codes_from_biases(s[i * g:(i + 1) * g], b[i * g:(i + 1) * g]) for i in range(3)]
    else:
        raise ValueError(f"no slot head from a {kind} record")
    pidx = np.empty(3 * g, np.uint8)
    plut = np.empty(3 * PLUT_BYTES // 2, np.uint16)
    half = PLUT_BYTES // 2
    for i in range(3):
        codes_qmv.pair_from(s[i * g:(i + 1) * g], cs[i], pidx[i * g:(i + 1) * g], plut[i * half:(i + 1) * half])
    return pidx.tobytes() + plut.tobytes()


def raw_head_from_slot(payload: bytes, lay: "BankLayout") -> bytes:
    """A slot record's head payload -> the raw head (bf16 scales w1 | w3 | w2, then biases) it stands for, for a
    slot that is not a pair slot."""
    g, half = lay.groups, PLUT_BYTES // 2
    pidx = np.frombuffer(payload, np.uint8, 3 * g).astype(np.intp)
    plut = np.frombuffer(payload, np.uint16, 3 * half, 3 * g)
    scales = np.concatenate([plut[i * half:(i + 1) * half][2 * pidx[i * g:(i + 1) * g]] for i in range(3)])
    biases = np.concatenate([plut[i * half:(i + 1) * half][2 * pidx[i * g:(i + 1) * g] + 1] for i in range(3)])
    return scales.tobytes() + biases.tobytes()


@dataclass(frozen=True)
class BankLayout:
    """Byte sizes of one expert's pieces (from the checkpoint's ExpertFormat)."""

    weight: int  # per projection
    scales: int  # per projection (bf16)

    @property
    def codes(self) -> int:
        return self.scales // 2 // 4

    @property
    def coded_head(self) -> int:
        return _align(3 * (self.scales + self.codes))

    @property
    def raw_head(self) -> int:
        return _align(6 * self.scales)

    @property
    def groups(self) -> int:
        return self.scales // 2

    @property
    def slot_payload(self) -> int:
        """A slot record's head: a byte (scale, bias) index per group, then a 256-pair table, per projection."""
        return 3 * (self.groups + PLUT_BYTES)

    @property
    def slot_head(self) -> int:
        return _align(self.slot_payload)

    def head_payload(self, kind: str) -> int:
        """The head's bytes before padding: what heads.zst compresses."""
        if kind == "slot":
            return self.slot_payload
        return 3 * (self.scales + self.codes) if kind == "coded" else 6 * self.scales

    def head(self, kind: str) -> int:
        """The head's bytes on disk, padded."""
        return {"coded": self.coded_head, "raw": self.raw_head, "slot": self.slot_head}[kind]

    def record(self, kind: str) -> int:
        return self.head(kind) + 3 * self.weight


def layout_from_sizes(sizes: dict[str, int]) -> BankLayout:
    w = {sizes[f"{p}.weight"] for p in PROJS}
    s = {sizes[f"{p}.scales"] for p in PROJS} | {sizes[f"{p}.biases"] for p in PROJS}
    if len(w) != 1 or len(s) != 1:
        raise ValueError(f"projections differ in size: {sizes}")
    return BankLayout(weight=w.pop(), scales=s.pop())


class CodedBankReader(ExpertReader):
    """ExpertReader that serves the experts a coded bank holds from it, the rest from the checkpoint."""

    def __init__(self, bank_dir, mirror_dir=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bank_dir = Path(bank_dir)
        meta = json.loads((self.bank_dir / "bank.json").read_text())
        if meta.get("version") not in BANK_VERSIONS:
            raise ValueError(f"{self.bank_dir}: bank version {meta.get('version')}, expected one of {BANK_VERSIONS}")
        self.layout = BankLayout(**meta["layout"])
        self.records: dict[tuple[int, int], tuple[str, int, str]] = {
            (r[0], r[1]): (r[2], r[3], r[4]) for r in meta["records"]
        }
        self.checkpoint = meta.get("checkpoint")
        self.bank_mirror = Path(mirror_dir) if mirror_dir else None
        # the mirror serves only the records it holds at the same place (a bank rewritten record by record, M24,
        # has a mirror that lags until it is copied)
        self.mirror_same: set[tuple[int, int]] = set()
        self.mirror_tail = 1.0  # a tail-only mirror (write_mirror_tail) holds this fraction of each weight piece
        if self.bank_mirror is not None and not (self.bank_mirror / "bank.json").is_file():
            print(f"[bank] mirror {self.bank_mirror} has no bank.json; bank mirror off", flush=True)
            self.bank_mirror = None
        if self.bank_mirror is not None:
            mirror = json.loads((self.bank_mirror / "bank.json").read_text())
            self.mirror_tail = float(mirror.get("tail", 1.0))
            if mirror.get("layout") == meta["layout"]:
                self.mirror_same = {(r[0], r[1]) for r in mirror["records"]
                                    if self.records.get((r[0], r[1])) == (r[2], r[3], r[4])}
            if len(self.mirror_same) < len(self.records):
                print(f"[bank] mirror {self.bank_mirror} matches {len(self.mirror_same)} of {len(self.records)} "
                      f"records; the rest read from the bank alone", flush=True)
        self.coded_reads = 0
        self.zhead_reads = 0
        self.mirror_share = None  # the adaptive share (MIRROR_ADAPT); None until the first timed record
        self.bulk = False  # set by the store: True while a prefill reads (bandwidth-bound)
        self.zheads: dict[tuple[int, int], tuple[int, int]] = {}
        self.zheads_file = None
        zmeta = self.bank_dir / "heads.json"
        if _zstd is not None and zmeta.is_file():
            z = json.loads(zmeta.read_text())
            if z.get("version") in ZHEADS_VERSIONS and (self.bank_dir / z["file"]).is_file():
                self.zheads_file = self.bank_dir / z["file"]
                # a head compressed from another kind of record than the bank now holds is ignored (version 1
                # heads are coded or raw ones)
                for r in z["records"]:
                    rec = self.records.get((r[0], r[1]))
                    kind = r[4] if len(r) > 4 else None
                    if rec is not None and (kind == rec[2] or (kind is None and rec[2] != "slot")):
                        self.zheads[(r[0], r[1])] = (r[2], r[3])
        bias_table()

    def covers(self, layer: int, expert: int) -> bool:
        return (layer, expert) in self.records

    def read_expert_into(self, entry: ExpertEntry, views) -> int:
        # a codes slot (HANDOFF 18.10) can only be filled from the bank, whatever ENABLED says
        compact = "w1.codes" in views or "w1.pidx" in views
        rec = self.records.get((entry.layer, entry.expert)) if ENABLED or compact else None
        if rec is None:
            if not entry.tensors:
                # a trimmed checkpoint (index_from_bank) has no other copy of this expert
                raise LookupError(f"expert {entry.layer}/{entry.expert}: not in {self.bank_dir} and no checkpoint bytes")
            return super().read_expert_into(entry, views)
        return self._read_record(rec, views, (entry.layer, entry.expert))

    def _head(self, key, kind, fd, offset) -> bytes | None:
        """The head's payload from heads.zst, or None when this bank has none for the expert."""
        use = ZHEADS == 2 or (ZHEADS == 1 and self.bulk)
        z = self.zheads.get(key) if use else None
        if z is None:
            return None
        zoff, zlen = z
        blob = os.pread(self._fd(self.zheads_file), zlen, zoff)
        if len(blob) != zlen:
            raise OSError(f"short compressed head read for {key}: {len(blob)} of {zlen}")
        payload = _zstd.decompress(blob)
        if len(payload) != self.layout.head_payload(kind):
            raise OSError(f"compressed head for {key} is {len(payload)} bytes, expected {self.layout.head_payload(kind)}")
        self.zhead_reads += 1
        return payload

    def _read_record(self, rec, views, key=None) -> int:
        fname, offset, kind = rec
        lay = self.layout
        fd = self._fd(self.bank_dir / fname)
        head = lay.head(kind)
        pool = self._piece_executor()
        mfd = None
        frac = min(self.mirror_fraction if MIRROR_FRACTION < 0 else MIRROR_FRACTION, self.mirror_tail)
        if MIRROR_ADAPT and MIRROR_FRACTION < 0 and self.mirror_share is not None:
            frac = min(frac, self.mirror_share)
        if self.bank_mirror is not None and frac > 0 and key in self.mirror_same:
            mfile = self.bank_mirror / fname
            if mfile.exists():
                mfd = self._fd(mfile)
        # the three weight pieces go first: they are the long pole
        futures = []
        pos = offset + head
        for p in PROJS:
            buf = memoryview(views[f"{p}.weight"]).cast("B")
            if mfd is not None:
                cut = int(lay.weight * (1.0 - frac)) // 4096 * 4096
                futures.append((pool.submit(_timed_preadv, mfd, [buf[cut:]], pos + cut), lay.weight - cut, True, fd, buf[cut:], pos + cut))
                futures.append((pool.submit(_timed_preadv, fd, [buf[:cut]], pos), cut, False, fd, None, 0))
            else:
                futures.append((pool.submit(_timed_preadv, fd, [buf], pos), lay.weight, False, fd, None, 0))
            pos += lay.weight
        scales = [np.asarray(views[f"{p}.scales"]).view(np.uint16) for p in PROJS] if "w1.scales" in views else None
        zhead = self._head(key, kind, fd, offset) if key is not None else None
        if kind == "slot":
            # M24: the head is the pair slot's own bytes; any other slot gets the raw head it stands for
            if "w1.pidx" in views:
                bufs = [memoryview(views[f"{p}.pidx"]).cast("B") for p in PROJS] + [
                    memoryview(views[f"{p}.plut"]).cast("B") for p in PROJS]
                if zhead is None and SLOT_HEAD_SCRATCH:
                    zhead = os.pread(fd, lay.slot_payload, offset)
                if zhead is not None:
                    _scatter(zhead, bufs)
                    got = lay.slot_payload
                else:
                    got = os.preadv(fd, bufs, offset)
                if got != lay.slot_payload:
                    raise OSError(f"short head read from {fname}@{offset}: {got}")
                return self._finish(futures, fname, offset, got)
            if zhead is None:
                zhead = os.pread(fd, lay.slot_payload, offset)
                if len(zhead) != lay.slot_payload:
                    raise OSError(f"short head read from {fname}@{offset}: {len(zhead)}")
            zhead, kind = raw_head_from_slot(zhead, lay), "raw"
        if "w1.codes" in views or "w1.pidx" in views:
            # a slot of 4-bit bias codes (HANDOFF 18.10): the bank's 2-bit codes widened, a raw record's biases coded
            total = self._read_head_codes(kind, fd, offset, fname, views, scales, zhead, pool)
        elif kind == "coded":
            biases = [np.asarray(views[f"{p}.biases"]).view(np.uint16) for p in PROJS]
            codes = bytearray(3 * lay.codes)
            bufs = [memoryview(views[f"{p}.scales"]).cast("B") for p in PROJS] + [memoryview(codes)]
            if zhead is not None:
                _scatter(zhead, bufs)
                got = 3 * (lay.scales + lay.codes)
            else:
                got = os.preadv(fd, bufs, offset)
            if got != 3 * (lay.scales + lay.codes):
                raise OSError(f"short head read from {fname}@{offset}: {got}")
            packed = np.frombuffer(codes, np.uint8)
            n = lay.codes
            rebuilds = [pool.submit(decode_biases, scales[i], packed[i * n:(i + 1) * n], biases[i]) for i in (1, 2)]
            decode_biases(scales[0], packed[:n], biases[0])
            for f in rebuilds:
                f.result()
            total = got
        else:
            biases = [np.asarray(views[f"{p}.biases"]).view(np.uint16) for p in PROJS]
            bufs = [memoryview(views[f"{p}.scales"]).cast("B") for p in PROJS] + [
                memoryview(views[f"{p}.biases"]).cast("B") for p in PROJS
            ]
            if zhead is not None:
                _scatter(zhead, bufs)
                got = 6 * lay.scales
            else:
                got = os.preadv(fd, bufs, offset)
            if got != 6 * lay.scales:
                raise OSError(f"short head read from {fname}@{offset}: {got}")
            total = got
        return self._finish(futures, fname, offset, total)

    def _finish(self, futures, fname, offset, total) -> int:
        """Wait for the weight pieces (a failed mirror piece is read again from the bank); the bytes read."""
        spent = {True: [0, 0.0], False: [0, 0.0]}  # bytes and the slowest piece's seconds, per drive
        for future, size, from_mirror, pfd, pbuf, poff in futures:
            try:
                got, dt = future.result()
                side = spent[from_mirror]
                side[0] += got
                side[1] = max(side[1], dt)
            except OSError as exc:
                if not from_mirror:
                    raise
                print(f"[bank] mirror read failed ({exc}); bank mirror off", flush=True)
                self.bank_mirror = None
                got = os.preadv(pfd, [pbuf], poff)
            if got != size:
                raise OSError(f"short weight read from {fname}@{offset}: {got} of {size}")
            total += got
        self.coded_reads += 1
        if MIRROR_ADAPT and spent[True][0] and spent[False][0]:
            self._adapt_share(spent[True][0] / max(spent[True][1], 1e-6), spent[False][0] / max(spent[False][1], 1e-6))
        return total

    def _adapt_share(self, mirror_rate: float, bank_rate: float) -> None:
        """Move the mirror's share towards the split that has both drives' pieces finish together."""
        cap = min(self.mirror_fraction, self.mirror_tail)
        target = mirror_rate / (mirror_rate + bank_rate)
        share = cap if self.mirror_share is None else self.mirror_share
        share += MIRROR_ADAPT_ALPHA * (target - share)
        self.mirror_share = min(cap, max(min(MIRROR_ADAPT_FLOOR, cap), share))
        # one line a minute at most, and only when the share has moved: it wanders ±0.03 around its level per record
        shown, when = getattr(self, "_share_shown", (cap, 0.0))
        now = time.monotonic()
        if abs(self.mirror_share - shown) >= 0.03 and now - when >= 60:
            print(f"[bank] mirror share {shown:.2f} -> {self.mirror_share:.2f} (configured {cap:.2f})", flush=True)
            self._share_shown = (self.mirror_share, now)

    def _read_head_codes(self, kind, fd, offset, fname, views, scales, zhead, pool) -> int:
        """A record's head into a codes slot: scales into the scale views (or, in a scale-index slot, a byte index
        and a table, M23), bias codes as nibbles (codes_qmv)."""
        from cachalot.minimax import codes_qmv

        lay = self.layout
        pair = "w1.pidx" in views
        nib = None if pair else [np.asarray(views[f"{p}.codes"]) for p in PROJS]
        sidx = "w1.sidx" in views or pair
        if sidx:
            # a scale-index slot (M23): the bf16 scales land in a scratch buffer and become a byte index + table
            scale_buf = bytearray(3 * lay.scales)
            scale_bufs = [memoryview(scale_buf)[i * lay.scales:(i + 1) * lay.scales] for i in range(3)]
            scales = [np.frombuffer(b, np.uint16) for b in scale_bufs]
        else:
            scale_bufs = [memoryview(views[f"{p}.scales"]).cast("B") for p in PROJS]
        if kind == "coded":
            codes = bytearray(3 * lay.codes)
            bufs = scale_bufs + [memoryview(codes)]
            want = 3 * (lay.scales + lay.codes)
        else:
            raw = bytearray(3 * lay.scales)
            bufs = scale_bufs + [memoryview(raw)]
            want = 6 * lay.scales
        if zhead is not None:
            _scatter(zhead, bufs)
            got = want
        else:
            got = os.preadv(fd, bufs, offset)
        if got != want:
            raise OSError(f"short head read from {fname}@{offset}: {got}")
        if pair:
            # M23b: each group's (scale, bias) as a byte index into the projection's table of pairs
            if kind == "coded":
                packed = np.frombuffer(codes, np.uint8)
                n = lay.codes
                cs = [codes_qmv.codes_from_packed(packed[i * n:(i + 1) * n], K_BASE) for i in range(3)]
            else:
                b = np.frombuffer(raw, np.uint16)
                n = lay.scales // 2
                cs = [codes_qmv.codes_from_biases(scales[i], b[i * n:(i + 1) * n]) for i in range(3)]
            jobs = [_sidx_pool().submit(codes_qmv.pair_from, scales[i], cs[i], np.asarray(views[f"{p}.pidx"]),
                                        np.asarray(views[f"{p}.plut"]).view(np.uint16)) for i, p in enumerate(PROJS)]
            for f in jobs:
                f.result()
            return got
        jobs = []
        if sidx:
            jobs = [_sidx_pool().submit(codes_qmv.sidx_from_scales, scales[i], np.asarray(views[f"{p}.sidx"]),
                                np.asarray(views[f"{p}.lut"]).view(np.uint16)) for i, p in enumerate(PROJS)]
        if kind == "coded":
            packed = np.frombuffer(codes, np.uint8)
            n = lay.codes
            for i in range(3):
                codes_qmv.nibbles_from_packed(packed[i * n:(i + 1) * n], K_BASE, nib[i])
            for f in jobs:
                f.result()
        else:
            b = np.frombuffer(raw, np.uint16)
            n = lay.scales // 2
            jobs += [pool.submit(codes_qmv.nibbles_from_biases, scales[i], b[i * n:(i + 1) * n], nib[i]) for i in (1, 2)]
            codes_qmv.nibbles_from_biases(scales[0], b[:n], nib[0])
            for f in jobs:
                f.result()
        return got


_SIDX_POOL = None


def _sidx_pool():
    """Threads for the scale-index conversion (M23), apart from the pread pool so reads cannot queue ahead of it."""
    global _SIDX_POOL
    if _SIDX_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _SIDX_POOL = ThreadPoolExecutor(8, thread_name_prefix="expert-sidx")
    return _SIDX_POOL


def slot_format(fmt: ExpertFormat, sidx: int = 0) -> ExpertFormat:
    """`fmt` with each projection's bf16 biases replaced by 4-bit codes (two groups a byte): a codes slot. With
    `sidx` (HANDOFF 18.13, M23) the bf16 scales are replaced too, by a byte per group indexing the projection's
    256-entry bf16 table (`{p}.lut`): 21.52 MiB a slot instead of 22.36. `sidx=2` (M23b): one byte per group
    indexing the projection's 256 (scale, bias) pairs (`{p}.pidx`, `{p}.plut`), no codes: 21.10 MiB."""
    shapes, dtypes = dict(fmt.shapes), dict(fmt.dtypes)
    for p in PROJS:
        groups = int(np.prod(shapes.pop(f"{p}.biases")))
        dtypes.pop(f"{p}.biases")
        shapes[f"{p}.codes"] = (groups // 2,)
        dtypes[f"{p}.codes"] = "uint8"
        if sidx == 2:
            # a pair slot (HANDOFF 18.13, M23b): a byte per group indexing (scale, bias) pairs; no codes
            shapes[f"{p}.pidx"] = shapes.pop(f"{p}.scales")
            dtypes.pop(f"{p}.scales")
            dtypes[f"{p}.pidx"] = "uint8"
            shapes.pop(f"{p}.codes")
            dtypes.pop(f"{p}.codes")
            shapes[f"{p}.plut"] = (512,)
            dtypes[f"{p}.plut"] = "bfloat16"
        elif sidx:
            shapes[f"{p}.sidx"] = shapes.pop(f"{p}.scales")
            dtypes.pop(f"{p}.scales")
            dtypes[f"{p}.sidx"] = "uint8"
            shapes[f"{p}.lut"] = (256,)
            dtypes[f"{p}.lut"] = "bfloat16"
    names = tuple(sorted(shapes))
    return ExpertFormat(kind=fmt.kind, bits=fmt.bits, group_size=fmt.group_size, tensor_names=names, shapes=shapes,
                        dtypes=dtypes)


def _scatter(payload: bytes, bufs) -> None:
    """Copy a contiguous head payload into the buffers a head preadv would have filled, in order."""
    pos = 0
    for b in bufs:
        b[:] = payload[pos:pos + len(b)]
        pos += len(b)


def write_zheads(bank_dir, level: int = 19, threads: int = 16) -> tuple[int, int, int, float]:
    """Write heads.zst + heads.json into a bank: each record's head payload, zstd at `level`, in bank.json's
    order, each blob 4 KiB-aligned and checked by a round trip. Returns (heads, raw bytes, file bytes, seconds)."""
    import fcntl
    import time
    from concurrent.futures import ThreadPoolExecutor

    bank = Path(bank_dir)
    meta = json.loads((bank / "bank.json").read_text())
    lay = BankLayout(**meta["layout"])
    fds: dict[str, int] = {}
    for r in meta["records"]:
        if r[2] not in fds:
            fds[r[2]] = os.open(bank / r[2], os.O_RDONLY)
            fcntl.fcntl(fds[r[2]], fcntl.F_NOCACHE, 1)

    def job(r):
        n = lay.head_payload(r[4])
        head = os.pread(fds[r[2]], n, r[3])
        blob = _zstd.compress(head, level=level)
        if len(head) != n or _zstd.decompress(blob) != head:
            raise RuntimeError(f"compressed head round trip failed for {r[:2]}")
        return blob, n

    part = bank / "heads.zst.part"
    fd = os.open(part, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    recs, offset, raw_bytes, t0 = [], 0, 0, time.perf_counter()
    try:
        with ThreadPoolExecutor(threads) as pool:
            for r, (blob, n) in zip(meta["records"], pool.map(job, meta["records"]), strict=True):
                os.pwrite(fd, blob, offset)
                recs.append([r[0], r[1], offset, len(blob), r[4]])
                raw_bytes += n
                offset += (len(blob) + 4095) // 4096 * 4096
        os.fsync(fd)
    finally:
        os.close(fd)
        for f in fds.values():
            os.close(f)
    os.replace(part, bank / "heads.zst")
    tmp = bank / "heads.json.part"
    tmp.write_text(json.dumps({"version": ZHEADS_VERSION, "file": "heads.zst", "level": level, "records": recs}))
    os.replace(tmp, bank / "heads.json")
    return len(recs), raw_bytes, offset, time.perf_counter() - t0


def write_mirror_tail(bank_dir, mirror_dir, tail: float = 0.25, threads: int = 8) -> tuple[int, int, float]:
    """A tail-only mirror of a bank (HANDOFF 18.16): every layer file at its full length but sparse, holding only
    the last `tail` of each record's three weight pieces (from the 4 KiB boundary the reader cuts at), and a
    bank.json that records `tail`, so a reader never asks it for more. The reader takes the tail 13 % of each piece
    from the mirror: a full copy (~154 GiB over USB, ~3 h) is not needed. bank.json is written last, so a partial
    mirror is never used. Returns (records, bytes written, seconds)."""
    import fcntl
    import time
    from concurrent.futures import ThreadPoolExecutor

    bank, mirror = Path(bank_dir), Path(mirror_dir)
    meta = json.loads((bank / "bank.json").read_text())
    lay = BankLayout(**meta["layout"])
    cut = int(lay.weight * (1.0 - tail)) // 4096 * 4096
    files: dict[str, list] = {}
    for r in meta["records"]:
        files.setdefault(r[2], []).append(r)
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / "bank.json").unlink(missing_ok=True)
    written, t0 = 0, time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        for fname, recs in sorted(files.items()):
            src = os.open(bank / fname, os.O_RDONLY)
            fcntl.fcntl(src, fcntl.F_NOCACHE, 1)
            part = mirror / (fname + ".part")
            dst = os.open(part, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.fcntl(dst, fcntl.F_NOCACHE, 1)
            os.ftruncate(dst, os.fstat(src).st_size)

            def job(r, src=src, dst=dst):
                n = 0
                for i in range(3):
                    off = r[3] + lay.head(r[4]) + i * lay.weight + cut
                    data = os.pread(src, lay.weight - cut, off)
                    if len(data) != lay.weight - cut:
                        raise OSError(f"short read of {r} at {off}")
                    n += os.pwrite(dst, data, off)
                return n

            written += sum(pool.map(job, recs))
            os.fsync(dst)
            os.close(dst)
            os.close(src)
            os.replace(part, mirror / fname)
    tmp = mirror / "bank.json.part"
    tmp.write_text(json.dumps({**meta, "tail": tail}))
    os.replace(tmp, mirror / "bank.json")
    return len(meta["records"]), written, time.perf_counter() - t0


def format_to_json(fmt: ExpertFormat) -> dict:
    return {"kind": fmt.kind, "bits": fmt.bits, "group_size": fmt.group_size,
            "tensor_names": list(fmt.tensor_names), "shapes": {k: list(v) for k, v in fmt.shapes.items()},
            "dtypes": dict(fmt.dtypes)}


def index_from_bank(bank_dir) -> tuple[ExpertFormat, dict[tuple[int, int], ExpertEntry]]:
    """The expert format and index from a complete bank alone, for a checkpoint trimmed to its non-expert
    weights: the entries carry no checkpoint byte ranges (every read is served by the bank)."""
    meta = json.loads((Path(bank_dir) / "bank.json").read_text())
    if "format" not in meta:
        raise ValueError(f"{bank_dir}/bank.json has no expert format")
    f = meta["format"]
    fmt = ExpertFormat(kind=f["kind"], bits=int(f["bits"]), group_size=int(f["group_size"]),
                       tensor_names=tuple(f["tensor_names"]), shapes={k: tuple(v) for k, v in f["shapes"].items()},
                       dtypes=dict(f["dtypes"]))
    index = {(r[0], r[1]): ExpertEntry(layer=r[0], expert=r[1], tensors=()) for r in meta["records"]}
    return fmt, index


def reader_from_env(**kwargs) -> ExpertReader:
    """The MiniMax store's reader: the coded bank when CACHALOT_MINIMAX_BANK names one, else the checkpoint's."""
    bank = os.environ.get("CACHALOT_MINIMAX_BANK")
    if not bank:
        return ExpertReader(**kwargs)
    return CodedBankReader(bank, os.environ.get("CACHALOT_MINIMAX_BANK_MIRROR") or None, **kwargs)
