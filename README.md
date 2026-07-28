# True W4A4 (INT4×INT4) Speedup for pi0.5 — Reproduction Kit

Real INT4×INT4 Tensor Core execution of a uniform-4-bit-weight W4A4
quantization scheme on the pi0.5 LIBERO policy, using Nunchaku's SVDQuant
kernels. Runs on any GPU with native INT4 Tensor Cores: **SM80 (A100), SM86,
SM89 (RTX 4080/4090 SUPER)**. Hopper (SM90/H100) removed INT4 Tensor Cores —
use a W4A8 backend there.

All numbers measured on an idle A100-SXM4-40GB. Raw JSON for every number is
in `logs_gpu0_debug/`.

## Results

### Deployment mode: torch.compile (max-autotune), end-to-end

W4A4 behind a `torch.library` custom op (`omega::w4a4_linear`), so Inductor
compiles one whole graph — no graph breaks. Full `sample_actions` latency
(vision-language prefix + 10 denoising steps), 18 fused gate/up W4A4 layers,
vs the compiled BF16 policy (OpenPI's default deployment path):

| Batch | Compiled 16-bit | Compiled W4A4 | Speedup | Norm. (lower better) |
|---:|---:|---:|---:|---:|
| 1 | 58.6 ms | 55.3 ms | 1.06× | 0.94× |
| 2 | 101.8 ms | 89.9 ms | 1.13× | 0.88× |
| 4 | 155.2 ms | 133.7 ms | 1.16× | 0.86× |
| 8 | 277.0 ms | 232.2 ms | **1.19×** | **0.84×** |
| 16 | 565.9 ms | 477.3 ms | 1.19× | 0.84× |
| 24 | 811.7 ms | 686.6 ms | 1.18× | 0.85× |

Fixed-order protocol (compiled 16-bit first, then W4A4 + recompile — a
per-sample interleave would recompile for minutes on every switch); the
80–125 ms effects dwarf the ~2 ms clock drift we measured elsewhere.

### Eager mode: backbone (VLM prefix) segment, ABBA-interleaved, vs FP16

Same-process ABBA interleave (arms alternate per round, clock drift cancels),
CUDA-event timing of the 968-token prefix forward — where every quantized
layer lives. Shape-dispatched config: batch ≤ 2 replaces the fused gate/up
pairs only; batch ≥ 4 additionally replaces q/o/down:

| Batch | Prefix 16-bit | Prefix W4A4 | Speedup | Eager E2E speedup |
|---:|---:|---:|---:|---:|
| 1 | 29.1 ms | 27.2 ms | 1.07× | 1.01× |
| 2 | 57.4 ms | 51.8 ms | 1.11× | 1.02× |
| 4 | 117.6 ms | 103.4 ms | 1.14× | 1.04× |
| 8 | 235.2 ms | 202.1 ms | 1.16× | 1.07× |
| 16 | 481.5 ms | 400.4 ms | 1.20× | 1.11× |
| 24 | 726.2 ms | 597.1 ms | **1.22×** | 1.13× |

### Full Omega-QVLA method with real pack parameters (compiled, vs compiled BF16)

Rotated GPTQ weights + activation-scale tables loaded from the released
``quantized.pt`` pack; the complete runtime pipeline (input rotation ->
smooth+INT4 quantize -> INT4x4 GEMM -> output-rotation restore) executes on
every quantized layer. Output rotation + GLU run as our fused single-pass
Triton kernel (``rotglu``, see Kernel design):

| Batch | 1 | 2 | 4 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|
| End-to-end speedup | 1.05x | 1.09x | 1.13x | **1.15x** | 1.14x | 1.14x |

Progression at B=8: eager rotations 0.99x -> rotations traced into the
Inductor graph 1.10x -> fused rotglu kernel 1.15x. Peak batch = 8.

Per-block with real parameters and ALL rotations included:
MLP block (gate/up + GLU) **1.24x** at M=968, **1.30x** at M=7744;
single gate_proj 1.05x / 1.09x (`tools/micro_method_block.py`).

### Kernel design (`tools/triton_block_rotate.py`)

Two single-pass Triton kernels remove the memory-traffic overhead of
DuQuant's runtime rotations:

* ``block_rotate``: grid over (row-tiles, 64-channel blocks). Each program
  loads one 64x64 rotation tile into registers once, gathers its 64 source
  channels (fusing the zigzag permutation into the load for the input side),
  computes a (BM,64)@(64,64) ``tl.dot``, and stores contiguously — one global
  read + one write versus gather/bmm/reshape chains (2.7x over eager on the
  16384-wide output restore).
* ``rotglu``: fuses the *pair* of output-rotation restores for gate and up
  with the GLU nonlinearity: loads both un-restored GEMM output tiles, applies
  the two 64x64 rotations in registers, evaluates overflow-safe tanh-GELU
  (sigmoid form) and the elementwise product, and writes the single fused
  result. Traffic drops from 7 tensor passes (2x rotate read+write + GLU
  2 reads/1 write) to 3; fp32 accumulation throughout. Registered as
  ``torch.library`` custom ops with fake-tensor metas so the policy still
  compiles as one Inductor graph.

The remaining unfused cost is the input-side rotation (~340 us/call at
M=7744; gather-bound, at parity with Inductor codegen). Folding it into the
Nunchaku activation-quantize kernel (which already makes a full pass over
the same tensor) is the identified next step and would close most of the
gap to the rotation-free ceiling (1.18-1.19x).

### Per-layer / per-block (M=968 prefix shapes, includes activation-quantize cost)

| Unit | FP16 | W4A4 | Speedup |
|---|---:|---:|---:|
| Fused MLP block gate+up (2048→2×16384) | 722 µs | 447 µs | **1.62×** |
| gate/up single (2048→16384) | 297 µs | 211 µs | 1.41× |

### Memory

Runtime steady allocated 6.98 → 5.33 GiB (**−23.7%**) with only the 36 MLP
projections swapped (drop-in mode; the A/B scripts keep dense weights
resident for interleaving and do not show this saving).

### Where W4A4 does NOT pay (and why)

The M=10 action-expert denoise steps sit at the kernel-launch floor
(~14 µs/call, of which compute is 1–2 µs): measured W4A4 0.18–0.33× and a
purpose-patched small-batch W4A16 GEMV 0.47–0.65× against FP16 there.
Uniform all-layer W4A4 *execution* is 0.75× end-to-end at batch 1. Keep
4-bit storage everywhere, dispatch compute by shape.

## Repo layout

```
tools/run_pi05_w4a4_compile_v2.py    # HEADLINE: custom-op W4A4 under torch.compile, any batch
tools/nunchaku_compile_ops.py        # omega::w4a4_linear custom-op registration + fused-MLP install
tools/run_pi05_w4a4_batch_sweep.py   # eager ABBA sweep over batches + prefix/denoise segmentation
tools/run_pi05_w4a4_ab.py            # eager batch-1 ABBA A/B, fused gate/up
tools/run_pi05_w4a4_prefix_ab.py     # eager batch-1 A/B with backbone-segment timing
tools/run_pi05_nunchaku_speed.py     # drop-in replacement smoke (memory numbers)
tools/benchmark_pi05_nunchaku.py     # per-shape microbenchmark (FP16 / GEMM-only / full W4A4)
tools/micro_fused_probe.py           # fused gate_up / fused qkv micro validation
tools/micro_awq_gemv_probe.py        # AWQ W4A16 GEMV probe at expert shapes (M=10)
tools/run_pi05_w4a4_compile_ab.py    # compile A/B WITHOUT custom op (graph-break ablation, 0.97x)
tools/profile_pi05_breakdown.py      # torch.profiler breakdown of the compiled policy
patches/nunchaku-w4a4-pi05.patch     # REQUIRED Nunchaku patches (see below)
logs_gpu0_debug/                     # raw JSON + logs for every reported number
```

## Setup on RTX 4080 SUPER (SM89)

Prereqs: CUDA toolkit ≥ 12.4 with nvcc, PyTorch ≥ 2.5 (CUDA build),
Python 3.11. Build Nunchaku with the same Python/Torch as the OpenPI venv.

```bash
# 1. OpenPI (pinned commit) + its venv
git clone https://github.com/Physical-Intelligence/openpi
cd openpi && git checkout c23745b5ad24e98f66967ea795a07b2588ed6c79
# follow openpi README to create .venv (uv sync)

# 2. pi0.5 LIBERO PyTorch checkpoint
#    directory with config.json + model.safetensors + assets/  (~7.2 GB)
#    ask us for a copy, or convert per openpi docs. First run downloads the
#    PaliGemma tokenizer into ~/.cache/openpi/big_vision/ (needs network).

# 3. Nunchaku (pinned commit) + patches + build
git clone https://github.com/nunchaku-ai/nunchaku
cd nunchaku
git checkout 8f41840596bd516d434a1f88ac16c86fdb64e74f
git submodule update --init --recursive
git apply /path/to/pi05-w4a4-speedup/patches/nunchaku-w4a4-pi05.patch
NUNCHAKU_INSTALL_MODE=FAST MAX_JOBS=8 \
  /path/to/openpi/.venv/bin/python setup.py build_ext --inplace
# FAST mode auto-detects the local GPU (sm_89 on a 4080 SUPER)
```

Patch contents (both in one file):

* `src/kernels/zgemm/lora.cuh` — early-return at LoRA rank 0 (REQUIRED; the
  fused quantize kernel otherwise dereferences empty LoRA buffers).
* `src/kernels/awq/gemv_awq.cu` — W4A16 GEMV dispatch extended to m≤12
  (only needed for `micro_awq_gemv_probe.py`).

The three default paths (OpenPI root, checkpoint, Nunchaku root) live at the
top of `tools/run_pi05_nunchaku_speed.py`; every other script imports them
from there. Edit once, or pass the equivalent CLI flags where available.

## Running

```bash
VENVPY=/path/to/openpi/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Headline: compiled (deployment-mode) A/B at a given batch.
# First compile takes minutes; the W4A4 recompile too. Be patient.
$VENVPY tools/run_pi05_w4a4_compile_v2.py --batch 1 --output out/compile_b1.json
$VENVPY tools/run_pi05_w4a4_compile_v2.py --batch 4 --output out/compile_b4.json

# Eager backbone-segment sweep (ABBA; the honest interleaved protocol)
$VENVPY tools/run_pi05_w4a4_batch_sweep.py --compute-dtype fp16 \
  --batches 1,2,4 --rounds 6 --warmup 3 --output out/eager_sweep.json
# batch >= 4: add  --extra-ops q_proj,o_proj,down_proj

# Per-shape microbenchmark
$VENVPY tools/benchmark_pi05_nunchaku.py --nunchaku-root /path/to/nunchaku \
  --dtype fp16 --scope all --warmup 30 --iterations 100 --repeats 7 > out/micro.json
```

**16 GB VRAM (4080 SUPER):** model (7.2 GB fp16/bf16) + packed weights +
activations. Compiled runs (`compile_v2`) keep the dense gate/up weights
resident too — expect batch 1–4 to fit; the eager A/B sweep likewise.
For larger batches use the drop-in runner (~5.3 GB steady) at fixed order.

## Protocol notes

* Two baselines, each matched to its half of the table: eager rows compare
  against eager FP16 (autocast); compiled rows against the compiled BF16
  policy (OpenPI's default). Both arms of any A/B always share dtype,
  observation, noise, and process.
* ABBA interleave matters in eager mode: a fixed measurement order inflated
  batch-1 E2E from 1.007× to ~1.05× in early runs (clock drift).
* Custom-op registration matters in compiled mode: raw kernel calls
  graph-break Dynamo and give 0.97× (slower); the same kernels behind
  `omega::w4a4_linear` give 1.06–1.19× (`run_pi05_w4a4_compile_ab.py` vs
  `run_pi05_w4a4_compile_v2.py`).
* Speed-only: weights are RTN-quantized (group-64, rank 0, no rotation, no
  per-step scales). Task success under this exact packed path is not
  evaluated here.
```
