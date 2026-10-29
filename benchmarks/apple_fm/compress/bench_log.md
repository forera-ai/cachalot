Tool output: run_benchmark --suite decode --model glm-5.3-flash --budget-gib 46 --repeats 5

[00:00:01] loading configuration from /Users/dev/bench/decode.toml
[00:00:01] machine: Mac Studio M3 Ultra, 96 GB, macOS 27.2, sysctl iogpu.wired_limit_mb=88064
[00:00:02] model path /Volumes/X10Pro/GLM-5.3-Flash-MLX-4bit, expert bank enabled, prefetch off
[00:00:02] warming the expert cache, this can take a while on a cold drive
[00:00:41] warm-up done: 38.9 s, 1,912 experts resident, 0 evictions
[00:00:41] starting repeat 1 of 5

The benchmark prints a progress line every 16 tokens. The lines below are summarized by the harness; the full trace is in the results directory.

[00:01:12] repeat 1: 256 tokens, 1.58 s per token, 41 misses per token, drive read 0.97 GB/s
[00:01:43] repeat 2: 256 tokens, 1.61 s per token, 42 misses per token, drive read 0.96 GB/s
[00:02:14] repeat 3: 256 tokens, 1.57 s per token, 40 misses per token, drive read 0.98 GB/s
[00:02:44] repeat 4: 256 tokens, 1.63 s per token, 43 misses per token, drive read 0.95 GB/s
[00:03:15] repeat 5: 256 tokens, 1.60 s per token, 41 misses per token, drive read 0.97 GB/s

[00:03:15] WARNING: memory pressure reached level 2 during repeat 4; the guardian logged it but did not kill the run.
[00:03:15] WARNING: the screensaver was running during repeats 2 and 3; compare those repeats only with each other.

Summary
-------
median seconds per token: 1.60 (min 1.57, max 1.63)
median misses per token: 41
median drive read: 0.97 GB/s
peak resident memory: 71.4 GiB
swap used at the end: 4.6 GiB of 6.0 GiB
energy meter: off
output hash (repeat 1): 9f3c2a71, identical across all five repeats

Notes from the harness
----------------------
The decode rate is bound by the drive: at 41 misses per token and 9.4 MiB per expert record, each token reads about 385 MiB, which at 0.97 GB/s takes about 0.42 s of the 1.60 s. The rest is GPU work and host synchronization. The harness cannot tell the two apart without the token trace, which was not enabled for this run (set CACHALOT_TOKEN_TRACE=4 to sample one token in four).

This run used the default miss budget of 2 for GLM. A previous run with miss budget 0 measured 2.31 s per token on the same drive. Do not compare this run with runs from the internal SSD, where the same model decodes at about 0.3 s per token.

The results directory is benchmarks/results/glm-decode-2026-10-29-1412. It contains trace.jsonl, summary.json and the raw progress log. Keep the directory until the A/B against the prefetch arm is done; deleting it loses the only copy of repeat 4's pressure trace.

Housekeeping: 3 temporary files were removed from /tmp. The next scheduled run is the prefetch arm at 16:00.

Progress log (verbatim, abbreviated)
------------------------------------
token 16: 1.59 s; token 32: 1.58 s; token 48: 1.62 s; token 64: 1.57 s; token 80: 1.60 s; token 96: 1.61 s; token 112: 1.58 s; token 128: 1.59 s; token 144: 1.63 s; token 160: 1.60 s; token 176: 1.58 s; token 192: 1.59 s; token 208: 1.62 s; token 224: 1.61 s; token 240: 1.57 s; token 256: 1.60 s.
The same pattern repeats for repeats 2 to 5 with differences within 0.03 s per token, apart from repeat 4, whose tokens 160 to 224 ran 0.04 to 0.06 s slower while memory pressure was at level 2.
