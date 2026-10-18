# EXPLORATORY bisect (not a registered cell): run the suite's own copy cell in a fresh process, then vary one thing at a time.
# ruff: noqa  (throwaway exploratory script kept verbatim; see README.md)
import sys, time, statistics as st
sys.path.insert(0, "benchmarks/cuda")
import suite
lim = suite.Limits(max_seconds=60, max_iters=30, max_alloc_mib=2048, warmup=5)
b = suite.TorchBench(0)
c = suite._copy_cell("h2d", "pinned", 9_950_000, lim)
def bw(res): 
    s = sorted(res.arms["events"]); return round(s[len(s)//2], 4), round(9.95/ s[len(s)//2], 1)
print("1 suite _copy cell, fresh process, first:", bw(b.execute(c, suite.Deadline(60), lim)))
print("2 suite _copy cell, again:", bw(b.execute(c, suite.Deadline(60), lim)))
import torch
host = b._pattern(9_950_000, lim.seed, True); dev = torch.empty(9_950_000, dtype=torch.uint8, device=b.dev)
ev, hm, tr = b._timed(lambda: dev.copy_(host, non_blocking=True), 5, 30, suite.Deadline(60))
print("3 same ops via suite._timed, no validation copy, warmup 5:", round(st.median(ev), 4))
ev, hm, tr = b._timed(lambda: dev.copy_(host, non_blocking=True), 60, 60, suite.Deadline(60))
print("4 same, warmup 60 iters 60:", round(st.median(ev), 4))
dst = dev
src = host
dst.copy_(src); torch.cuda.synchronize(); _ = bool(torch.equal(dev.cpu(), host))
ev, hm, tr = b._timed(lambda: dst.copy_(src, non_blocking=True), 5, 30, suite.Deadline(60))
print("5 after the validation round trip (dev.cpu()), warmup 5:", round(st.median(ev), 4))
back = b._pin(torch.empty(9_950_000, dtype=torch.uint8))
ev, hm, tr = b._timed(lambda: dev.copy_(host, non_blocking=True), 5, 30, suite.Deadline(60))
print("6 repeat of 3 at the end:", round(st.median(ev), 4))
