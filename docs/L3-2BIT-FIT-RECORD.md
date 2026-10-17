# L3 follow-up: an activation-weighted 2-bit fit, experiment record

Written 2026-10-16 against runtime 0.66.7, before any gate run. Kind when finished: research if the prediction below was written first (it was), engineering otherwise. Workload class: W4 (code quality is where a quantization shows first), measured here only on the teacher-forced NLL gate, which is not a W4 measurement.

## 1. Baseline

The shipped internal bank is `~/DeepSeek-V4.1-Flash-q2g128`, 2-bit group 128, fit `search` (from its `config.json`). Existing paired NLL arms on the same 160 tokens (48 prefill tokens, default README text, `nll_expert_precision.py`): fp4 reference mean 2.3004, `requant2g128search` 2.3234, `requant2g128mlx` 17.546 (mx.quantize's fit is destroyed at 2 bits and group 128).

Screen run today (`quant_fit_screen.py --bits 2 --experts 24 --probes 8 --groups 128`, probes from the code activations, weighting fitted on the prose activations, 34.5 s): y-err against `search`: `search+act` -1.5 %, `search+lsq` -5.5 %, `search+lsq+act` -8.1 %, `wide+lsq` -10.1 %, `wide+lsq+act` -13.3 %, `mlx` +19.6 %. Tag: measured, screen only.

## 2. What the record already says (found while writing this, not before the screen)

HANDOFF section 9.3 (2026-09-21): a whole 2-bit g128 bank was built with the unweighted `wide-lsq` fit (115 minutes), whose screen gain was -10.6 %, and gated on the production path at 512 tokens on two texts. Result: paired medians +0.0017 and +0.0001 nats, sign tests not significant, means disagreeing in sign (+0.0216 and -0.1392, each driven by about five tokens). A null. The same section lists activation weighting at 2 bits as "still untried". Activation weighting at 3 bits was built and gated in September and did not improve code (section 9.0). So today's test is the activation-weighted 2-bit fit only, and its screen gain over the known null is 2.7 points (-13.3 % against -10.6 %).

## 3. Hypothesis and prediction

Hypothesis: weighting the fit by what the model multiplies each column by lowers the error on the columns that matter, so the token-level NLL moves even though the unweighted lsq fit did not.

Prediction (written before the run), `wide-lsq` + prose importance against `search`, paired per token on the 160 tokens:
- paired median difference within +-0.01 nats of zero;
- better on 45 % to 55 % of tokens (sign test not significant);
- mean difference within +-0.05 nats, with a paired SE of 0.04 to 0.06 (so the mean cannot decide anything on 160 tokens);
- the mean NLL between 2.28 and 2.36 (fp4 is 2.3004, `search` is 2.3234).

## 4. Falsifier

The null prediction is falsified, in favour of the weighted fit, if the paired median is at least 0.01 nats better AND it is better on at least 60 % of tokens (sign test p below 0.01). It is falsified against the weighted fit if it is worse on at least 60 % of tokens by a median of 0.01 nats or more. Anything else is the null again and the lever closes: a weighted 2-bit bank is not worth a 142 GiB rebuild (about two hours, internal disk).

## 5. Method

`benchmarks/nll_expert_precision.py --experts requant --requant-bits 2 --requant-group 128 --requant-fit wide-lsq --requant-importance benchmarks/results/activations_moe_input.npz --tokens 160`, under `benchmarks/guarded_run.sh --budget-gib 24 --max-seconds 7200`. The `search` and `fp4` arms are reused from the saved JSON files (same script, same default text, same token count); a mismatch in targets between the files is checked before the comparison and would void the pairing. The prose importance file is the one the screen used, so the fit is cross-text with respect to the C# and code probes.

Limits: 160 tokens of README text is not a code test and its power is low (section 9.3.1 puts the paired SE of a 512-token mean at 0.04 to 0.05 nats); a null here does not exclude a small gain, it only says a bank rebuild is not justified.

## 6. Result

**No result: the arm was killed by its own time cap.** `guarded_run.sh --budget-gib 24 --max-seconds 7200` started it at 20:31 and killed it at 22:31 ("timeout after 7200 s", exit 4). The script prints nothing between "requantizing FP4 experts to 2-bit group 128 (wide-lsq fit, activation-weighted ...)" and its final result, so no token count was reached or visible. Footprint: peak wired 54.3 GiB, peak resident 35.8 GiB, peak compressor 11.7 GiB, minimum free 0.1 GiB, swap 0, pressure 1. No `nll_experts_requant2g128wide-lsq*.json` was written, so the prediction in section 3 is neither held nor falsified.

Estimate error (mine, measured): I predicted 60 to 90 minutes with a hard cap of 120, derived from the bank build's 0.45 s an expert (115 minutes for 15,360 experts). The arm needed more than 120 minutes, so that derivation was wrong for this path. The gate re-fits experts on the fly in the NLL process, probably more than once per distinct expert and with the importance weighting; the cause is not established. The memory figures also say the arm ran close to the machine's limits (minimum free 0.1 GiB), which may have slowed it. Tag: measured (the kill), hypothesis (the cause).

**Why it took over two hours (profiled afterwards, 3 timed experts per fit, one process, no guard).** Per expert, 2-bit g128: `mlx` 0.7 ms, `search` 109 ms, `wide-lsq` 388 ms, `wide-lsq` with the prose importance 429 ms (importance itself 1.1 ms; reading the FP4 expert 6 to 25 ms). The NLL gate fits an expert only when it misses its packed cache (`RequantDense.cache_limit` = 8 GiB, about 800 experts at 2 bits); 160 decode tokens make 38,400 routed-expert uses, and far more than 800 distinct experts are touched, so the cache thrashes and experts are fitted again and again. At 0.43 s a fit, 15,000 to 30,000 fits is 1.8 to 3.6 hours, which fits the kill. The earlier `search` arm cost 0.11 s a fit, 4 times less, which is why it finished. Tag: measured (per-fit times), derived (fit counts: not counted in the real run; an attempt to count them from a Hermes routing trace was unreliable because that trace repeats positions across requests). My 60 to 90 minute estimate used the bank build's 0.45 s an expert times roughly 10,000 distinct experts and ignored the cache misses.

Options priced, none run: (a) the same arm with fewer tokens (64 tokens is about 15,000 uses; roughly 40 to 90 minutes, but a paired SE far wider than the 160-token one); (b) make the packed cache large enough or spill it to disk (a code change in the gate, hours of build, tens of GiB); (c) a cheaper weighted fit, `search-lsq` with importance, which the screen puts at -8.1 % y-err (against -13.3 %), about 4 times cheaper per fit.

Not done and not claimed: any quality statement about the weighted `wide-lsq` fit. The only evidence on it remains the screen (-13.3 % y-err against `search`) and the unweighted bank's null (HANDOFF section 9.3).

## 7. Result of the 64-token rerun (2026-10-16, 22:52 to 23:45)

`nll_expert_precision.py --experts requant --requant-bits 2 --requant-group 128 --requant-fit wide-lsq --requant-importance benchmarks/results/activations_moe_input.npz --tokens 64`, under `guarded_run.sh --budget-gib 24 --max-seconds 7200`. Finished in 3,169 s (53 minutes; estimate was 40 to 90). 6,697 experts were fitted at 457 ms each (3,059 s of the 3,083 s decode), 44 % of the 15,360 expert uses; the packed cache was full at 8.0 GiB. That confirms the section 6 explanation: 160 tokens would have needed about 16,000 fits, about two hours. Targets and prefill length match the saved `search` and `fp4` files token for token (checked). Per-token files are copied to `benchmarks/results/l3-2bit-fit/`. Paired on the first 64 tokens (the saved arms ran 160; their first 64 are used):

| comparison | mean | paired SE | median | better on | sign p |
|---|---:|---:|---:|---:|---:|
| `search` minus weighted `wide-lsq` (positive = weighted better) | +0.2148 | 0.1921 | +0.0007 | 35 of 64 (54.7 %) | 0.53 |
| weighted `wide-lsq` minus fp4 (positive = weighted worse) | -0.2732 | 0.1772 | -0.0006 | 31 of 64 | 0.90 |
| `search` minus fp4 | -0.0584 | 0.0982 | +0.0013 | 37 of 64 worse | 0.21 |

Mean NLL on these 64 tokens: fp4 1.977, `search` 1.919, weighted `wide-lsq` 1.704; top-1 39, 37 and 42 of 64; argmax agreement with fp4 52 (`search`) and 47 (weighted). The mean of the weighted arm sits below the fp4 reference it is quantized from, which no real quality gain can do: both means are driven by a handful of tokens (the five largest differences are 7.5, 6.6, 3.1, -2.6 and -2.5 nats against `search`), the same pattern HANDOFF section 9.3.1 found.

**Against the prediction (section 3).** Held: paired median within +-0.01 nats (+0.0007); better on 45 to 55 % of tokens (54.7 %). Missed: mean difference within +-0.05 (it is +0.215), because I took the paired SE from 160-token runs (0.04 to 0.06) and the SE on 64 tokens is 0.19; the mean cannot decide anything, as predicted for the mean at 160. Not applicable: the 2.28 to 2.36 mean-NLL range was for the 160-token run. **The falsifier did not fire** (median far below 0.01 better, better on 55 %, sign p 0.53): the null stands.

**Generalization and bottleneck after.** This is a measurement on the NLL gate, 64 tokens of README text, so it excludes a gain larger than the median's resolution but not a small one, and says nothing about code. Combined with the unweighted `wide-lsq` bank's null at 512 tokens on two texts (HANDOFF section 9.3) and the 3-bit weighted bank's null on code (section 9.0), no evidence supports a 142 GiB rebuild of the 2-bit bank for a better fit. The lever closes unless a code-validity gate (`code_validity.py`) is run on a built bank, which needs the bank (about two hours) and Hamed's call. Nothing shipped; the bottleneck is unchanged (expert drive and cache capacity, section 5 of `docs/L3-QUANT-SWEEP-RECORD.md`). Kind: research (predicted before the run, measured, explained as hypothesis); the first attempt is an engineering miss on the time estimate.
