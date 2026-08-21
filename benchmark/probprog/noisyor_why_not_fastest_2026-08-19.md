# Why noisy-or is not the fastest row for Impulse (2026-08-19)

Question: in the four-system NUTS comparison, noisy_or_topic is the one model
where Impulse (55.2 n_eff/s median) trails Stan (98.6), while leading NumPyro
(50.9) and Turing (4.9). Why?

## Step 1: mixing is identical -- the gap is wall-clock only

From cross_framework_complete_2026-08-02.json, median n_eff over the four
trials: Impulse 6974, NumPyro 6178, Stan 6320, Turing 6500 (within 13%).
Steady wall/trial: Stan 64.1 s vs Impulse 126.3 s vs NumPyro 121.3 s. The two
XLA-based systems sit together at ~2x Stan; the sampler quality is matched.

## Step 2: leapfrog decomposition (noisyor_leapfrog_probe.py, 1 chain,
## 1000 warmup + 4000 draws, same data seed 12345, same 0.8 target accept)

|                    | Stan      | NumPyro   |
|--------------------|----------:|----------:|
| wall               | 33.3 s    | 49.3 s    |
| leapfrogs/draw     | 254.8     | 127.5     |
| per-leapfrog cost  | 26.1 us   | 77.4 us   |

Two opposing factors:
1. Per-gradient, Stan is ~3x cheaper. The relaxed noisy-or density is a
   50-step SEQUENTIAL triangular recursion (topic j's Gumbel-softmax activation
   depends on activations 1..j-1) of tiny ops (~6.4k flops total) plus one
   100x51 matvec. Stan evaluates it as tight compiled scalar C++ with a cheap
   reverse tape; XLA (NumPyro AND Impulse alike) pays tensor-kernel dispatch
   on hundreds of micro-ops per gradient. There is no sample-invariant heavy
   structure for SICM to hoist (the paper already discloses this).
2. Partially compensating, the NumPyro-style NUTS adaptation (which Impulse
   matches by construction) settles on twice the step size: ~127 leapfrogs/draw
   (depth ~7) vs Stan's ~255 (depth ~8), at equal n_eff.

Net: 3x dearer gradients x 0.5x as many gradients ~= the observed ~2x wall gap.

## Reading for the paper

- Impulse already leads its same-substrate peer (NumPyro) on this row; the
  deficit is Stan-vs-XLA structural, in the scalar-sequential-density regime
  where a tensor compiler's dispatch overhead dominates tiny flop counts.
- If we ever want to win the row: the recursion needs a fused/scalarized
  lowering (one XLA while with a fully fused body, or a custom scalar-loop
  kernel), not a SICM-style rewrite. Candidate future-work framing.
- The current CaseStudies.tex framing ("these two rows measure the plain
  compiled NUTS kernel ... not a speedup claim") is accurate; this note gives
  the quantitative backing if a reviewer asks.
