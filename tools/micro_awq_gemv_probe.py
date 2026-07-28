"""Micro-bench Nunchaku AWQ W4A16 GEMV vs FP16 F.linear at expert shapes (M=10).

Speed-only: buffers are layout-valid but numerically arbitrary.
Run: CUDA_VISIBLE_DEVICES=0 python tools/micro_awq_gemv_probe.py
"""

import importlib.util
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

RUNNER = Path(__file__).with_name("run_pi05_nunchaku_speed.py")
spec = importlib.util.spec_from_file_location("pi05_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.bootstrap_nunchaku(runner.DEFAULT_NUNCHAKU_ROOT)
from nunchaku.ops.gemv import awq_gemv_w4a16_cuda  # noqa: E402

M = 10
SHAPES = [  # (name, k, n, calls_per_action)
    ("expert/q_proj", 1024, 2048, 180),
    ("expert/k_or_v", 1024, 256, 360),
    ("expert/o_proj", 2048, 1024, 180),
    ("expert/gate_up", 1024, 4096, 360),
    ("expert/down", 4096, 1024, 180),
]
WARMUP, ITERS, REPEATS = 50, 200, 5


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
        medians.append(start.elapsed_time(end) / ITERS * 1000.0)
    return statistics.median(medians)


@torch.inference_mode()
def main():
    device = torch.device("cuda:0")
    dtype = torch.float16 if (len(sys.argv) < 2 or sys.argv[1] == "fp16") else torch.bfloat16
    total_fp16 = total_awq = 0.0
    print(f"dtype={dtype}, M={M}")
    print(f"{'shape':<18}{'fp16_us':>9}{'awq_us':>9}{'speedup':>9}")
    for name, k, n, calls in SHAPES:
        x = torch.randn(M, k, dtype=dtype, device=device)
        w = torch.randn(n, k, dtype=dtype, device=device)
        qweight = torch.randint(-(2**31), 2**31 - 1, (n // 4, k // 2), dtype=torch.int32, device=device)
        wscales = torch.full((k // 64, n), 0.01, dtype=dtype, device=device)
        wzeros = torch.zeros((k // 64, n), dtype=dtype, device=device)

        out = awq_gemv_w4a16_cuda(x, qweight, wscales, wzeros, M, n, k, 64)
        assert out.shape == (M, n), out.shape

        t_fp16 = bench(lambda: F.linear(x, w))
        t_awq = bench(lambda: awq_gemv_w4a16_cuda(x, qweight, wscales, wzeros, M, n, k, 64))
        total_fp16 += t_fp16 * calls / 1000.0
        total_awq += t_awq * calls / 1000.0
        print(f"{name:<18}{t_fp16:>9.1f}{t_awq:>9.1f}{t_fp16 / t_awq:>9.3f}")
    print(f"modeled expert linear ms/action: fp16={total_fp16:.2f}  awq={total_awq:.2f}  "
          f"speedup={total_fp16 / total_awq:.3f}")


if __name__ == "__main__":
    main()
