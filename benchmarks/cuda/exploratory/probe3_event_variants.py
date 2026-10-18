# EXPLORATORY bisect 2 (not a registered cell): one thing at a time against the suite's _timed.
# ruff: noqa  (throwaway exploratory script kept verbatim; see README.md)
import time, statistics as st, torch
n = 9_950_000
host = torch.randint(0, 256, (n,), dtype=torch.uint8).pin_memory()
dev = torch.empty(n, dtype=torch.uint8, device="cuda")
def med(xs): return round(st.median(xs), 4)
def run(name, fresh, presync, hostclock, iters=40, warm=8):
    a0, b0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(warm): dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        if presync: torch.cuda.synchronize()
        a, b = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if fresh else (a0, b0)
        if hostclock: h0 = time.perf_counter_ns()
        a.record(); dev.copy_(host, non_blocking=True); b.record(); b.synchronize()
        out.append(a.elapsed_time(b))
    print(f"{med(out):7.4f} ms  {name}")
run("reused events, presync, no host clock", False, True, False)
run("fresh events each iter, presync, no host clock", True, True, False)
run("fresh events, no presync, host clock (= suite._timed)", True, False, True)
run("reused events, no presync, host clock", False, False, True)
run("reused events, presync, host clock", False, True, True)
run("fresh events, no presync, no host clock", True, False, False)
