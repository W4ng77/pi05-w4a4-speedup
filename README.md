# True W4A4 (INT4×INT4) Speedup for pi0.5 — Reproduction Kit

Real INT4×INT4 Tensor Core execution of a uniform-4-bit-weight W4A4 quantization scheme on the pi0.5
LIBERO policy, using Nunchaku's SVDQuant kernels. Runs on any GPU with native
INT4 Tensor Cores: **SM80 (A100), SM86, SM89 (RTX 4080/4090 SUPER)**.
Hopper (SM90/H100) removed INT4 Tensor Cores — use the W4A8 backend there.

All numbers below were measured on an idle A100-SXM4-40GB with a same-process
ABBA-interleaved A/B protocol (both arms alternate within each round, so GPU
clock drift cancels). Raw JSON for every number is in `logs_gpu0_debug/`.

## Headline results (A100, FP16 baseline, lower is better)

Backbone (VLM prefix, 968 tokens/sample) latency per sample:

| Rollout batch | FP16 | W4A4 (ours) | Normalized |
|---:|---:|---:|---:|
| 1 | 29.3 ms | 27.3 ms | 0.93× |
| 8 | 29.4 ms | 25.3 ms | 0.86× |
| 16 | 30.1 ms | 25.0 ms | 0.83× |
| 24 | 30.3 ms | 24.9 ms | 0.82×* |

\* B=24 comes from its own run (`segment_sweep_fp16_b24_32.json`): with both
weight copies resident the consolidated multi-batch process OOMs at B=24.
B=1/8/16 are one process each (`prefix_ab_fp16.json`, `segment_sweep_fp16_final.json`).

Kernel/block level (M=968 prefix shapes, includes activation-quantize cost):

| Unit | FP16 | W4A4 | Speedup |
|---|---:|---:|---:|
| Fused MLP block gate+up (2048→2×16384) | 722 µs | 447 µs | **1.62×** |
| gate/up single (2048→16384) | 297 µs | 211 µs | 1.41× |

Runtime memory: 6.98 → 5.33 GiB steady allocated (−23.7%) with only the 36
MLP projections swapped (drop-in mode; the A/B scripts keep dense weights
resident and do not show this saving).

Configuration ("ours") is shape-dispatched: batch 1 replaces only the fused
`gate/up` pairs; batch ≥ 8 additionally replaces Pali `q/o/down`. The action
expert (M=10 per denoise step) stays 16-bit — it sits at the kernel-launch
floor and no GEMM kernel can win there (we measured W4A4 at 0.18–0.33× and a
purpose-patched W4A16 GEMV at 0.47–0.65× against it).

## Repo layout

```
tools/run_pi05_nunchaku_speed.py     # drop-in replacement smoke (fixed order; memory numbers)
tools/run_pi05_w4a4_ab.py            # interleaved ABBA A/B, batch 1, fused gate/up
tools/run_pi05_w4a4_batch_sweep.py   # ABBA sweep over batches + prefix/denoise segmentation
tools/run_pi05_w4a4_prefix_ab.py     # batch-1 A/B with backbone-segment timing
tools/benchmark_pi05_nunchaku.py     # per-shape microbenchmark (FP16 / GEMM-only / full W4A4)
tools/micro_fused_probe.py           # fused gate_up + fused qkv micro validation
tools/micro_awq_gemv_probe.py        # AWQ W4A16 GEMV probe at expert shapes (M=10)
tools/profile_pi05_breakdown.py      # torch.profiler breakdown of the default compiled policy
patches/nunchaku-w4a4-pi05.patch     # REQUIRED Nunchaku patches (see below)
logs_gpu0_debug/                     # raw JSON + stderr for every reported number
```

## Setup on RTX 4080 SUPER (SM89)

Prereqs: CUDA toolkit ≥ 12.4 with nvcc, PyTorch ≥ 2.5 (CUDA build), Python 3.11.
The Nunchaku build Python and the OpenPI venv Python must share the same
Torch/CUDA ABI (easiest: build inside the OpenPI venv).

```bash
# 1. OpenPI (pinned commit) + its venv
git clone https://github.com/Physical-Intelligence/openpi
cd openpi && git checkout c23745b5ad24e98f66967ea795a07b2588ed6c79
# follow openpi README to create .venv (uv sync)

# 2. pi0.5 LIBERO PyTorch checkpoint
#    -> directory containing config.json + model.safetensors + assets/
#    (same layout as $CHECKPOINTS_ROOT/pi05_libero_pytorch used elsewhere in
#     this repo; ~7.2 GB. Ask us for a copy or convert per openpi docs.)
#    First run downloads the PaliGemma tokenizer (needs network) into
#    ~/.cache/openpi/big_vision/paligemma_tokenizer.model

# 3. Nunchaku (pinned commit) + our patches + build
git clone https://github.com/nunchaku-ai/nunchaku
cd nunchaku
git checkout 8f41840596bd516d434a1f88ac16c86fdb64e74f
git submodule update --init --recursive
git apply /path/to/pi05-w4a4-speedup/patches/nunchaku-w4a4-pi05.patch
NUNCHAKU_INSTALL_MODE=FAST MAX_JOBS=8 \
  /path/to/openpi/.venv/bin/python setup.py build_ext --inplace
# FAST mode auto-detects the local GPU (picks sm_89 on a 4080 SUPER)
```

The patch contains two required changes:

* `src/kernels/zgemm/lora.cuh` — early-return when LoRA rank is 0. Without it
  the fused quantize kernel dereferences empty LoRA buffers (illegal memory
  access). We run rank=0 (no low-rank correction) throughout.
* `src/kernels/awq/gemv_awq.cu` — extends the W4A16 GEMV batch dispatch from
  m≤8 to m≤12 (only needed for `micro_awq_gemv_probe.py`).

## Running

Paths default to our cluster; override with the flags shown.

```bash
VENVPY=/path/to/openpi/.venv/bin/python
COMMON="--openpi-root /path/to/openpi --checkpoint /path/to/pi05_libero_pytorch \
        --nunchaku-root /path/to/nunchaku"
# NOTE: run_pi05_w4a4_ab.py / *_sweep.py / *_prefix_ab.py read the same three
# path constants from run_pi05_nunchaku_speed.py; edit DEFAULT_* there once
# instead of passing flags if you prefer.

# 1. Drop-in smoke + memory numbers (fixed order; treat latency as preliminary)
CUDA_VISIBLE_DEVICES=0 $VENVPY tools/run_pi05_nunchaku_speed.py $COMMON \
  --strategy fastest --compute-dtype fp16 --warmup 2 --iterations 10 \
  --output out/smoke_fp16.json

# 2. Honest batch-1 A/B (ABBA interleave)
CUDA_VISIBLE_DEVICES=0 $VENVPY tools/run_pi05_w4a4_ab.py \
  --compute-dtype fp16 --rounds 10 --output out/ab_fp16.json

# 3. Backbone-segment batch sweep (the paper-table numbers)
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$VENVPY tools/run_pi05_w4a4_batch_sweep.py \
  --compute-dtype fp16 --batches 1,2,4 --rounds 6 --warmup 3 \
  --extra-ops q_proj,o_proj,down_proj --output out/sweep_fp16.json

# 4. Per-shape microbenchmark
CUDA_VISIBLE_DEVICES=0 $VENVPY tools/benchmark_pi05_nunchaku.py \
  --nunchaku-root /path/to/nunchaku --dtype fp16 --scope all \
  --warmup 30 --iterations 100 --repeats 7 > out/micro_fp16.json
```

**16 GB VRAM note (4080 SUPER):** the A/B scripts keep the FP16 weights
(7.2 GB) *and* the packed INT4 weights resident so the two arms can
interleave. Batch 1–4 fits in 16 GB; batches ≥ 8 will OOM — report the
per-sample backbone trend from the batches that fit, or use the drop-in
runner (script 1, ~5.3 GB steady) for larger batches at fixed order.

## Protocol notes / honesty

* Baseline is eager FP16 (autocast) with `torch.compile` disabled in **both**
  arms. Same LIBERO observation, same zero noise, `torch.cuda.synchronize`
  after every inference.
* ABBA interleave matters: a fixed 16bit-then-W4A4 order inflated the
  end-to-end speedup from 1.007× to ~1.05× in our early runs (clock drift).
* Speed-only: weights are RTN-quantized (group-64, rank 0, no rotation, no
  per-step scales). Task success under this exact packed path is not
  evaluated here; pair with your accuracy pipeline for task-success numbers.
* Batch-1 end-to-end gain is small (~1.01×) because eager inference is
  dominated by the launch-bound denoise loop (190 ms of 280 ms); the backbone
  segment — where the quantized layers live — is where W4A4 pays.
```
