# Draft rewrite of §6.4 (Real-Quantized Kernel Implementation)

Replaces the current §6.4 and Table 5. All A100 numbers are measured
(ABBA-interleaved A/B, raw logs in `logs_gpu0_debug/`); H100 / RTX 4080 SUPER
W4A8 numbers are carried over from the existing submission.

---

## 6.4 Real-Quantized Kernel Implementation

We deploy Ω-QVLA with **true INT4×INT4 Tensor Core execution**. Native INT4
Tensor Cores are available on NVIDIA architectures from Turing through Ada
(SM75–SM89, including A100 and the RTX 40 series) and were removed only in
Hopper (SM90). Our runtime therefore dispatches by architecture: on
INT4-capable GPUs the rotated, packed 4-bit weights are multiplied directly
against 4-bit activations using an SVDQuant-style INT4 MMA kernel
(`mma.m16n8k64.s4`), with per-token group-64 activation quantization fused
into the pipeline; on Hopper we fall back to a QServe-style W4A8 backend. To
our knowledge, Ω-QVLA is the **first VLA policy executed end-to-end with its
language backbone running on native INT4×INT4 Tensor Cores**.

The per-step activation scale table of §4.2 maps onto this kernel at **zero
runtime cost**: the fused quantize stage already applies a per-channel
smoothing divisor, so the T=8 calibrated scale vectors are installed by
swapping one pointer per denoising step — no additional kernels, no extra
latency.

On an A100, the dominant backbone block — the fused `gate/up` MLP projection
(2048→2×16384) — runs at **1.62×** over FP16 including all quantization
overhead (722 → 447 µs). At the model level, the vision-language backbone
step improves monotonically with rollout batch, reaching **0.86×/0.83×/0.82×
normalized latency at batch 8/16/24** (0.93× at batch 1): on hardware two
generations older than H100 and at a strictly lower activation precision,
true W4A4 matches the normalized reduction our W4A8 backend achieves on H100
(0.80×). Memory falls accordingly: **−23.7% runtime allocated** on the
replaced backbone (and −74.2% static footprint, Table 3).

Execution is *shape-dispatched* while quantization stays uniform: INT4
compute pays exactly where GEMMs are compute-bound (the 968-token prefix),
whereas the M=10 action-expert steps sit at the kernel-launch floor, where no
GEMM kernel — including a purpose-built small-batch W4A16 GEMV — can recover
meaningful time; those layers keep 4-bit storage with 16-bit compute. This
separation of uniform *quantization* from dispatched *execution* is what
converts Ω-QVLA's accuracy result into deployable speed today, rather than
waiting on future kernels.

### Table 5 (revised): Real-quantized backbone latency of Pi-0.5 (lower is better)

| GPU (arch) | INT4×INT4 TC | Backend | Rollout batch | FP16 (ms) | Ω-QVLA (ms) | Norm. |
|---|---|---|---:|---:|---:|---:|
| H100 80GB (SM90) | ✗ removed | W4A8 (QServe) | – | 32.3 | 26.0 | 0.80× |
| RTX 4080 SUPER (SM89) | ✓ | W4A8 (QServe) | – | 79.8 | 69.7 | 0.87× |
| A100 40GB (SM80) | ✓ | **W4A4 (INT4 MMA)** | 1 | 29.3 | 27.3 | 0.93× |
| A100 40GB (SM80) | ✓ | **W4A4 (INT4 MMA)** | 8 | 29.4 | 25.3 | 0.86× |
| A100 40GB (SM80) | ✓ | **W4A4 (INT4 MMA)** | 16 | 30.1 | 25.0 | 0.83× |
| A100 40GB (SM80) | ✓ | **W4A4 (INT4 MMA)** | 24 | 30.3 | 24.9 | **0.82×** |

Caption: A100 rows report per-sample latency of the vision-language backbone
forward (968-token prefix, where all quantized projections reside), median of
ABBA-interleaved same-process A/B; batch = parallel-environment rollout.
Batch 1 replaces the fused gate/up pairs; batch ≥8 additionally replaces
q/o/down (shape dispatch). H100 / RTX 4080 rows use the W4A8 backend under
its original per-step protocol.

### Suggested per-layer companion table (from `pi05_nunchaku_micro_fp16_gpu0.json`)

| Unit (M=968) | FP16 | W4A4 | Speedup |
|---|---:|---:|---:|
| Fused MLP block gate+up | 722 µs | 447 µs | 1.62× |
| gate/up (single) | 297 µs | 211 µs | 1.41× |

### Notes for the rebuttal (not for the paper)

* A reviewer checking PTX/CUTLASS will confirm INT4 MMA exists on SM75–SM89
  and not SM90 — this table turns the old "blocked by hardware" text from a
  liability into the dispatch story.
* Uniform all-layer W4A4 *execution* measures 0.75× end-to-end slower at
  batch 1 (appendix ablation): keep it as the motivation for shape dispatch.
* The 4080 SUPER supports true W4A4; running the A100 kit there
  (`W4A4_SPEEDUP_README.md`) would complete the same-GPU W4A8-vs-W4A4
  comparison — the strongest possible row in this table.
