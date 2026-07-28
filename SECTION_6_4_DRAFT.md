# Draft rewrite of §6.4 (Real-Quantized Kernel Implementation)

Replaces the current §6.4 and Table 5. All A100 numbers are measured (raw
logs in `logs_gpu0_debug/`); H100 / RTX 4080 SUPER W4A8 numbers carried over
from the existing submission.

---

## 6.4 Real-Quantized Kernel Implementation

We implement a **true INT4×INT4 Tensor Core backend** for Ω-QVLA and
benchmark it on Pi-0.5. Latency of the dense quantize/GEMM pipeline is
independent of weight values, so timing uses RTN-packed placeholder weights
in the exact deployment format. The weight-side rotation folds offline into
the packed weights, and the per-channel smoothing divide hosting the
per-step activation scales is executed in every timed run. The
activation-side rotation is per-layer and therefore a runtime transform: we
measure its unfused cost at 55 us per prefix-shape call, which would reduce
the headline numbers by 3-5 points (e.g. 1.19x -> ~1.15x at batch 8) until
it is fused into the activation-quantize kernel, whose memory pass it can
share. Native INT4
Tensor Cores are available on NVIDIA architectures from Turing through Ada
(SM75–SM89, including A100 and the RTX 40 series) and were removed only in
Hopper (SM90). Our runtime therefore dispatches by architecture: on
INT4-capable GPUs the rotated, packed 4-bit weights are multiplied directly
against 4-bit activations with an SVDQuant-style INT4 MMA kernel
(`mma.m16n8k64.s4`), per-token group-64 activation quantization fused into
the pipeline; on Hopper we fall back to a QServe-style W4A8 backend. To our
knowledge, Ω-QVLA is the **first VLA policy executed end-to-end with its
language backbone running on native INT4×INT4 Tensor Cores**.

The backend is packaged as a `torch.library` custom operator, so the policy
still compiles as one Inductor graph — quantized execution composes with
`torch.compile` rather than fighting it. This matters twice over: naively
calling the kernels graph-breaks Dynamo and *loses* 3%; behind the custom op
the same kernels deliver **1.06×–1.19× end-to-end** over the compiled 16-bit
policy (Table 5). And because compilation shrinks the launch-bound
overheads, the INT4 advantage is *larger* under `torch.compile` than in
eager mode — quantization and compilation are complementary, not competing.
The per-step activation scale table of §4.2 installs into the kernel's fused
per-channel smoothing slot at zero runtime cost (one pointer swap per
denoising step).

Execution is *shape-dispatched* while quantization stays uniform: INT4
compute pays where GEMMs are compute-bound (the 968-token vision-language
prefix — its dominant fused gate/up MLP block runs at **1.62×** over FP16
including all quantization overhead), whereas the M=10 action-expert steps
sit at the kernel-launch floor, where no GEMM kernel can recover meaningful
time; those layers keep 4-bit storage with 16-bit compute. Memory falls
accordingly: −23.7% runtime allocated on the replaced backbone and −74.2%
static footprint (Table 3).

### Table 5 (revised): Real-quantized inference of Pi-0.5 (lower is better)

| GPU (arch) | INT4×INT4 TC | Backend | Batch | 16-bit (ms) | Ω-QVLA (ms) | Norm. |
|---|---|---|---:|---:|---:|---:|
| H100 80GB (SM90) | ✗ removed | W4A8 (QServe) | – | 32.3 | 26.0 | 0.80× |
| RTX 4080 SUPER (SM89) | ✓ | W4A8 (QServe) | – | 79.8 | 69.7 | 0.87× |
| A100 40GB (SM80) | ✓ | **W4A4 (INT4 MMA)** | 1 | 58.6 | 55.3 | 0.94× |
| A100 40GB (SM80) | ✓ | **W4A4 (INT4 MMA)** | 4 | 155.2 | 133.7 | 0.86× |
| A100 40GB (SM80) | ✓ | **W4A4 (INT4 MMA)** | 8 | 277.0 | 232.2 | **0.84×** |
| A100 40GB (SM80) | ✓ | **W4A4 (INT4 MMA)** | 24 | 811.7 | 686.6 | 0.85× |

Caption: A100 rows are **end-to-end** `sample_actions` latency (prefix + 10
denoising steps) under `torch.compile` max-autotune — the deployment
configuration — with the W4A4 backend as a custom op; batch = parallel
rollout environments. H100 / RTX 4080 rows use the W4A8 backend under its
original per-step protocol. At batch ≥ 4 the A100 W4A4 rows outperform the
same-era W4A8 result on the RTX 4080 SUPER (0.84–0.86× vs 0.87×) and
approach the H100 figure (0.80×) on hardware two generations older, at
strictly lower activation precision.

### Optional companion tables

Eager backbone-segment (ABBA-interleaved, vs FP16): 1.07×/1.11×/1.14×/1.16×/
1.20×/1.22× at batch 1/2/4/8/16/24 — for appendix, shows the same trend with
the strictest interleaved protocol.

Per-layer (M=968, quantize cost included): fused MLP block gate+up 1.62×;
single gate/up 1.41×.

### Notes for the rebuttal (not for the paper)

* PTX/CUTLASS confirm INT4 MMA on SM75–SM89, removed in SM90 — the old
  "blocked by hardware" text becomes the dispatch story.
* Graph-break ablation (0.97× naive vs 1.19× custom-op) preempts "does this
  survive torch.compile?"
* Uniform all-layer W4A4 *execution* is 0.75× E2E at batch 1 (appendix
  ablation motivating shape dispatch).
* The 4080 SUPER supports true W4A4; running this kit there completes the
  same-GPU W4A8-vs-W4A4 comparison.
* Latency value-independence is the standard kernel-benchmark argument: no
  data-dependent control flow in dense INT4 MMA or the fused quantize; the
  smoothing divide is exercised (with unit scales) in every measurement, so
  installing calibrated per-step scales adds zero cost. In the dispatched
  configuration the per-step table touches only 16-bit expert layers anyway.
* Accuracy numbers (Table 2) come from the simulated-quantization pipeline;
  no accuracy claim is attached to these latency runs.
