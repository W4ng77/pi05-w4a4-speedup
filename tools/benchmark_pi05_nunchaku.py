#!/usr/bin/env python3
"""Benchmark Nunchaku W4A4 on the exact Linear shapes used by local pi0.5."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Shape:
    scope: str
    name: str
    m: int
    k: int
    n: int
    calls_per_action: int


# Counts come from a real pi05_libero forward: one 968-token Pali prefix and
# ten denoise steps with a 10-token Expert suffix, across 18 Gemma layers.
SHAPES = (
    Shape("expert", "q_proj", 10, 1024, 2048, 180),
    Shape("expert", "k_v_proj", 10, 1024, 256, 360),
    Shape("expert", "o_proj", 10, 2048, 1024, 180),
    Shape("expert", "gate_up_proj", 10, 1024, 4096, 360),
    Shape("expert", "down_proj", 10, 4096, 1024, 180),
    Shape("pali", "q_o_proj", 968, 2048, 2048, 36),
    Shape("pali", "k_v_proj", 968, 2048, 256, 36),
    Shape("pali", "gate_up_proj", 968, 2048, 16384, 36),
    Shape("pali", "down_proj", 968, 16384, 2048, 18),
)


def bootstrap_nunchaku(root: Path):
    """Import the standalone Linear path without importing Diffusers models."""
    package_root = root.resolve() / "nunchaku"
    if not (package_root / "_C.cpython-311-x86_64-linux-gnu.so").exists():
        candidates = list(package_root.glob("_C*.so"))
        if not candidates:
            raise FileNotFoundError(f"Nunchaku extension is not built under {package_root}")

    package = types.ModuleType("nunchaku")
    package.__path__ = [str(package_root)]
    package.__package__ = "nunchaku"
    sys.modules["nunchaku"] = package

    models = types.ModuleType("nunchaku.models")
    models.__path__ = [str(package_root / "models")]
    models.__package__ = "nunchaku.models"
    sys.modules["nunchaku.models"] = models

    importlib.import_module("nunchaku._C")
    return importlib.import_module("nunchaku.models.linear").SVDQW4A4Linear


def timed_ms(
    op: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[float, float, list[float]]:
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            op()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / iterations)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return statistics.median(samples), ordered[p95_index], samples


@torch.inference_mode()
def benchmark_shape(
    linear_cls,
    shape: Shape,
    *,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict:
    x2d = torch.randn((shape.m, shape.k), dtype=dtype, device=device)
    x3d = x2d.unsqueeze(0)
    dense_weight = torch.zeros((shape.n, shape.k), dtype=dtype, device=device)

    quant = linear_cls(
        shape.k,
        shape.n,
        rank=0,
        bias=False,
        precision="int4",
        torch_dtype=dtype,
        device=device,
    )
    quant.qweight.zero_()
    quant.wscales.fill_(1)
    quant.smooth_factor.fill_(1)
    quant.smooth_factor_orig.fill_(1)

    qact, ascales, lora_act = quant.quantize(x2d)
    quant_out = torch.empty((shape.m, shape.n), dtype=dtype, device=device)

    # Warmup also doubles as a CUDA smoke test for the patched rank-0 path.
    full_check = quant(x3d)
    quant.forward_quant(qact, ascales, lora_act, output=quant_out)
    dense_check = F.linear(x2d, dense_weight)
    torch.cuda.synchronize()
    finite = bool(full_check.isfinite().all() and dense_check.isfinite().all())

    fp16_ms, fp16_p95, fp16_samples = timed_ms(
        lambda: F.linear(x2d, dense_weight),
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    gemm_ms, gemm_p95, gemm_samples = timed_ms(
        lambda: quant.forward_quant(qact, ascales, lora_act, output=quant_out),
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    full_ms, full_p95, full_samples = timed_ms(
        lambda: quant(x3d),
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )

    result = asdict(shape)
    result.update(
        {
            "m_padded_nunchaku": math.ceil(shape.m / 256) * 256,
            "finite_smoke": finite,
            "fp16_linear_ms": fp16_ms,
            "fp16_linear_p95_ms": fp16_p95,
            "nunchaku_gemm_only_ms": gemm_ms,
            "nunchaku_gemm_only_p95_ms": gemm_p95,
            "nunchaku_full_ms": full_ms,
            "nunchaku_full_p95_ms": full_p95,
            "nunchaku_gemm_speedup_vs_fp16": fp16_ms / gemm_ms,
            "nunchaku_full_speedup_vs_fp16": fp16_ms / full_ms,
            "samples_ms": {
                "fp16_linear": fp16_samples,
                "nunchaku_gemm_only": gemm_samples,
                "nunchaku_full": full_samples,
            },
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nunchaku-root",
        type=Path,
        default=Path("/ceph/workspace/xinyu/Nunchaku"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--scope", choices=("all", "pali", "expert"), default="all")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    linear_cls = bootstrap_nunchaku(args.nunchaku_root)

    selected = [s for s in SHAPES if args.scope == "all" or s.scope == args.scope]
    results = []
    for index, shape in enumerate(selected, 1):
        print(
            f"[{index}/{len(selected)}] {shape.scope}.{shape.name} "
            f"MKN={shape.m}x{shape.k}x{shape.n}",
            file=sys.stderr,
            flush=True,
        )
        results.append(
            benchmark_shape(
                linear_cls,
                shape,
                dtype=dtype,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
        )
        torch.cuda.empty_cache()

    aggregate = {}
    for scope in ("pali", "expert", "all"):
        rows = [r for r in results if scope == "all" or r["scope"] == scope]
        if not rows:
            continue
        fp16 = sum(r["fp16_linear_ms"] * r["calls_per_action"] for r in rows)
        gemm = sum(r["nunchaku_gemm_only_ms"] * r["calls_per_action"] for r in rows)
        full = sum(r["nunchaku_full_ms"] * r["calls_per_action"] for r in rows)
        aggregate[scope] = {
            "modeled_fp16_linear_ms_per_action": fp16,
            "modeled_nunchaku_gemm_only_ms_per_action": gemm,
            "modeled_nunchaku_full_ms_per_action": full,
            "modeled_gemm_speedup_vs_fp16": fp16 / gemm,
            "modeled_full_speedup_vs_fp16": fp16 / full,
        }

    props = torch.cuda.get_device_properties(device)
    payload = {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dtype": args.dtype,
        "rank": 0,
        "bias": False,
        "per_step_scale_table": False,
        "activation_scale": "Nunchaku runtime per-token group64 (required by W4A4)",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "results": results,
        "aggregate": aggregate,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
