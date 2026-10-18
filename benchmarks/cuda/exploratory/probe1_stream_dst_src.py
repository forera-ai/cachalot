# EXPLORATORY probe (not a registered cell): why does a single 9.95 MB pinned H2D copy read 0.33 ms in the S2 cell and 0.20 ms in the replay?
# ruff: noqa  (throwaway exploratory script kept verbatim; see README.md)
import statistics as st, torch
n = 9_950_000
dev = torch.device("cuda")
torch.manual_seed(1)
srcs = [torch.randint(0, 256, (n,), dtype=torch.uint8).pin_memory() for _ in range(8)]
big = torch.empty(64 * n, dtype=torch.uint8, device=dev)        # arena-like
full = torch.empty(n, dtype=torch.uint8, device=dev)            # S2-cell-like destination
side = torch.cuda.Stream()
a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def timed(fn, stream=None, iters=60, warm=10):
    out = []
    for i in range(warm + iters):
        torch.cuda.synchronize()
        a.record(stream); fn(i); b.record(stream); b.synchronize()
        if i >= warm:
            out.append(a.elapsed_time(b))
    return round(st.median(out), 4), round(min(out), 4), round(max(out), 4)

def on(stream, fn):
    def run(i):
        with torch.cuda.stream(stream):
            fn(i)
    return run

rows = {}
rows["A full dst, default stream, same src (S2-cell style)"] = timed(lambda i: full.copy_(srcs[0], non_blocking=True))
rows["B full dst, side stream, same src"] = timed(on(side, lambda i: full.copy_(srcs[0], non_blocking=True)), side)
rows["C arena slice slot 5 (offset misaligned), side stream, same src"] = timed(on(side, lambda i: big[5*n:6*n].copy_(srcs[0], non_blocking=True)), side)
rows["D arena slice offset 0, side stream, same src"] = timed(on(side, lambda i: big[0:n].copy_(srcs[0], non_blocking=True)), side)
rows["E full dst, side stream, rotating src (8)"] = timed(on(side, lambda i: full.copy_(srcs[i % 8], non_blocking=True)), side)
rows["F arena rotating slots, side stream, rotating src"] = timed(on(side, lambda i: big[(i*7 % 64)*n:((i*7 % 64)+1)*n].copy_(srcs[i % 8], non_blocking=True)), side)
rows["G full dst, default stream, rotating src"] = timed(lambda i: full.copy_(srcs[i % 8], non_blocking=True))
for k, v in rows.items():
    print(f"{v[0]:7.4f} ms (min {v[1]}, max {v[2]})  {n/1e9/(v[0]/1e3):5.1f} GB/s  {k}")
