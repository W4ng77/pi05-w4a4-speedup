"""Micro-validate fused gate_up (N=32768) and fused qkv (N=2560) W4A4 calls.

Compares, at pi0.5 PaliGemma prefix shapes (M=968, K=2048):
  * separate W4A4 gate + up            vs fused single W4A4 GEMM
  * three 16-bit q/k/v linears         vs fused W4A4 qkv (+3 contiguous copies)
  * corresponding 16-bit baselines
Run: CUDA_VISIBLE_DEVICES=0 python tools/micro_fused_probe.py
"""

import importlib.util
import statistics
import sys
from pathlib import Path

import torch

RUNNER = Path(__file__).with_name("run_pi05_nunchaku_speed.py")
spec = importlib.util.spec_from_file_location("pi05_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

M, K = 968, 2048
WARMUP, ITERS, REPEATS = 30, 100, 5


def build(linear_cls, packer, n, dtype, device):
    dense = torch.nn.Linear(K, n, bias=False, dtype=dtype, device=device)
    torch.nn.init.normal_(dense.weight, std=0.02)
    quant = runner.make_nunchaku_linear(dense, linear_cls=linear_cls, packer=packer)
    return dense, quant


def bench(fn):
    medians = []
    for _ in range(REPEATS):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        medians.append(start.elapsed_time(end) / ITERS)
    return statistics.median(medians)


@torch.inference_mode()
def main():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if sys.argv[1:] and sys.argv[1] == "bf16" else torch.float16
    linear_cls, packer_cls = runner.bootstrap_nunchaku(runner.DEFAULT_NUNCHAKU_ROOT)
    packer = packer_cls(bits=4)
    x = torch.randn(M, K, dtype=dtype, device=device)

    gate_d, gate_q = build(linear_cls, packer, 16384, dtype, device)
    up_d, up_q = build(linear_cls, packer, 16384, dtype, device)
    gateup_d, gateup_q = build(linear_cls, packer, 32768, dtype, device)
    q_d, q_q = build(linear_cls, packer, 2048, dtype, device)
    k_d, _ = build(linear_cls, packer, 256, dtype, device)
    v_d, _ = build(linear_cls, packer, 256, dtype, device)
    qkv_d, qkv_q = build(linear_cls, packer, 2560, dtype, device)

    rows = {
        "mlp_16bit_sep": lambda: (gate_d(x), up_d(x)),
        "mlp_16bit_fused": lambda: gateup_d(x),
        "mlp_w4a4_sep": lambda: (gate_q(x), up_q(x)),
        "mlp_w4a4_fused": lambda: gateup_q(x),
        "qkv_16bit_sep": lambda: (q_d(x), k_d(x), v_d(x)),
        "qkv_w4a4_fused_sliced": lambda: (
            lambda o: (o[:, :2048].contiguous(), o[:, 2048:2304].contiguous(), o[:, 2304:].contiguous())
        )(qkv_q(x)),
        "qkv_w4a4_fused_raw": lambda: qkv_q(x),
        "qkv_16bit_fused_sliced": lambda: (
            lambda o: (o[:, :2048].contiguous(), o[:, 2048:2304].contiguous(), o[:, 2304:].contiguous())
        )(qkv_d(x)),
    }
    print(f"dtype={dtype}, M={M}, K={K}, warmup={WARMUP}, iters={ITERS}, repeats={REPEATS}")
    for name, fn in rows.items():
        print(f"{name:<26}{bench(fn) * 1000:>10.1f} us")


if __name__ == "__main__":
    main()
