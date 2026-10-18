# EXPLORATORY (not a registered cell): is H2D rate a property of the pinned host buffer?
# ruff: noqa  (throwaway exploratory script kept verbatim; see README.md)
import statistics as st, torch, re
n = 9_950_000
print("THP:", open("/sys/kernel/mm/transparent_hugepage/enabled").read().strip(), "| defrag:", open("/sys/kernel/mm/transparent_hugepage/defrag").read().strip())
dev = torch.empty(n, dtype=torch.uint8, device="cuda")
def timeit(src, iters=25, warm=5):
    for _ in range(warm): dev.copy_(src, non_blocking=True)
    torch.cuda.synchronize(); out = []
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        a.record(); dev.copy_(src, non_blocking=True); b.record(); b.synchronize(); out.append(a.elapsed_time(b))
    return st.median(out)
print("-- 24 separately pinned 9.95 MB buffers (median ms each):")
bufs = [torch.randint(0, 256, (n,), dtype=torch.uint8).pin_memory() for _ in range(24)]
ts = [timeit(x) for x in bufs]
print(" ".join(f"{t:.3f}" for t in ts))
print("fast (<0.25):", sum(t < 0.25 for t in ts), "slow:", sum(t >= 0.25 for t in ts))
print("-- one 24x slab pinned, sliced into 24 records (median ms each):")
slab = torch.randint(0, 256, (24 * n,), dtype=torch.uint8).pin_memory()
ts2 = [timeit(slab[i*n:(i+1)*n]) for i in range(24)]
print(" ".join(f"{t:.3f}" for t in ts2))
# AnonHugePages of this process after allocating
for line in open("/proc/self/smaps_rollup"):
    if line.startswith(("AnonHugePages", "Pss_Anon", "Rss")): print(line.strip())
print("-- same 24 buffers again (stability):")
print(" ".join(f"{timeit(x):.3f}" for x in bufs[:8]))
