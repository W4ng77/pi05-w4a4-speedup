#!/usr/bin/env python3
"""Run a real pi0.5 LIBERO policy smoke with Nunchaku W4A4 Linear layers.

This is intentionally a speed-only experiment:

* the baseline is the eager BF16 policy from the local PyTorch checkpoint;
* selected Gemma projection weights are quantized with symmetric group-64 RTN;
* the Nunchaku low-rank correction is disabled (rank=0);
* SmoothQuant is disabled (``smooth_factor == 1``);
* no timestep/per-step activation-scale table is installed or consulted.

Nunchaku still computes the activation scales required by its W4A4 kernel at
runtime.  That is intrinsic to the kernel and is not a pi0.5 denoising-step
scale table.

Run with the OpenPI virtual environment, for example:

    /ceph/workspace/xinyu/openpi/.venv/bin/python \
      tools/run_pi05_nunchaku_speed.py --strategy fastest --output /tmp/pi05.json

The measured-fastest strategy replaces only the 18 PaliGemma ``gate_proj`` and
18 ``up_proj`` layers.  Use ``--strategy scope --scope all`` for the deliberately
slower all-projection experiment, or ``--scope pali --ops gate_proj,up_proj``
for an explicit custom selection.

The script bootstraps only Nunchaku's compiled extension, standalone Linear
module, and official weight packer.  It deliberately bypasses Nunchaku's
top-level package initializers, which otherwise import Diffusers models.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import json
import math
import re
import statistics
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_OPENPI_ROOT = Path("/ceph/workspace/xinyu/openpi")
DEFAULT_CHECKPOINT = Path("/ceph/workspace/xinyu/models/pi05_libero_pytorch")
DEFAULT_NUNCHAKU_ROOT = Path("/ceph/workspace/xinyu/Nunchaku")

ALL_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
FASTEST_PROJECTIONS = frozenset(("gate_proj", "up_proj"))
PROJECTION_NAMES = rf"(?:{'|'.join(ALL_PROJECTIONS)})"
SCOPE_PATTERNS = {
    "pali": re.compile(
        rf"^paligemma_with_expert\.paligemma\.model\.language_model"
        rf"\.layers\.(?P<layer>\d+)\.(?:self_attn|mlp)\.(?P<projection>{PROJECTION_NAMES})$"
    ),
    "expert": re.compile(
        rf"^paligemma_with_expert\.gemma_expert\.model"
        rf"\.layers\.(?P<layer>\d+)\.(?:self_attn|mlp)\.(?P<projection>{PROJECTION_NAMES})$"
    ),
}


def _ceil_divide(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _pad(
    tensor: torch.Tensor | None,
    divisor: int | tuple[int, ...],
    dim: int | tuple[int, ...],
    fill_value: float | int = 0,
) -> torch.Tensor | None:
    """Small dependency-free equivalent of nunchaku.lora.flux.utils.pad."""
    if tensor is None:
        return None
    dimensions = (dim,) if isinstance(dim, int) else dim
    divisors = (divisor,) * len(dimensions) if isinstance(divisor, int) else divisor
    shape = list(tensor.shape)
    for axis, multiple in zip(dimensions, divisors, strict=True):
        if multiple > 1:
            shape[axis] = _ceil_divide(shape[axis], multiple) * multiple
    if tuple(shape) == tuple(tensor.shape):
        return tensor
    output = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    output[tuple(slice(0, extent) for extent in tensor.shape)] = tensor
    return output


def bootstrap_nunchaku(nunchaku_root: Path) -> tuple[type[nn.Module], type]:
    """Load only Nunchaku ``_C``, ``SVDQW4A4Linear``, and its official packer."""
    package_root = nunchaku_root.expanduser().resolve() / "nunchaku"
    if not package_root.is_dir():
        raise FileNotFoundError(f"Nunchaku package directory not found: {package_root}")
    extensions = sorted(package_root.glob("_C*.so"))
    if not extensions:
        raise FileNotFoundError(
            f"Nunchaku extension is not built under {package_root}; expected _C*.so"
        )

    # OpenPI does not import Nunchaku.  Clearing a possible partial import makes
    # this bootstrap deterministic when the script is run interactively.
    for module_name in tuple(sys.modules):
        if module_name == "nunchaku" or module_name.startswith("nunchaku."):
            del sys.modules[module_name]

    def install_package(name: str, path: Path) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module
        return module

    install_package("nunchaku", package_root)
    install_package("nunchaku.ops", package_root / "ops")
    install_package("nunchaku.models", package_root / "models")
    install_package("nunchaku.lora", package_root / "lora")
    install_package("nunchaku.lora.flux", package_root / "lora" / "flux")

    # The standalone CUDA wrappers and packer need only these tiny helpers.
    # Supplying them here avoids importing nunchaku.utils (HF Hub/safetensors)
    # and nunchaku.lora.flux.__init__ (Diffusers converters).
    nunchaku_utils = types.ModuleType("nunchaku.utils")
    nunchaku_utils.ceil_divide = _ceil_divide
    sys.modules["nunchaku.utils"] = nunchaku_utils

    flux_utils = types.ModuleType("nunchaku.lora.flux.utils")
    flux_utils.pad = _pad
    sys.modules["nunchaku.lora.flux.utils"] = flux_utils

    importlib.import_module("nunchaku._C")
    linear_cls = importlib.import_module("nunchaku.models.linear").SVDQW4A4Linear
    packer_cls = importlib.import_module("nunchaku.lora.flux.packer").NunchakuWeightPacker
    return linear_cls, packer_cls


class NunchakuLinearWrapper(nn.Module):
    """Shape-compatible facade around Nunchaku's 3-D-only W4A4 Linear."""

    def __init__(self, kernel: nn.Module, weight_dtype: torch.dtype):
        super().__init__()
        self.kernel = kernel
        self.in_features = int(kernel.in_features)
        self.out_features = int(kernel.out_features)
        # OpenPI uses ``projection.weight.dtype`` to select residual dtypes.
        # A zero-element non-persistent sentinel preserves that interface
        # without retaining the original dense matrix.
        self.register_buffer(
            "_weight_dtype_sentinel",
            torch.empty(0, dtype=weight_dtype, device=kernel.qweight.device),
            persistent=False,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self._weight_dtype_sentinel

    @property
    def bias(self) -> torch.Tensor | None:
        return self.kernel.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 2:
            raise ValueError(f"Nunchaku Linear expects at least 2-D input, got {inputs.shape}")
        leading_shape = inputs.shape[:-1]
        inputs_3d = inputs.reshape(1, -1, self.in_features)
        output_3d = self.kernel(inputs_3d)
        return output_3d.reshape(*leading_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            "W4A4=group64, rank=0"
        )


def _matches_scope(
    name: str,
    scope: str,
    projections: frozenset[str] | None = None,
) -> re.Match[str] | None:
    if scope == "all":
        match = SCOPE_PATTERNS["pali"].fullmatch(name) or SCOPE_PATTERNS["expert"].fullmatch(name)
    else:
        match = SCOPE_PATTERNS[scope].fullmatch(name)
    if match is not None and projections is not None and match.group("projection") not in projections:
        return None
    return match


def _parameter_bytes(module: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


@torch.inference_mode()
def make_nunchaku_linear(
    dense: nn.Linear,
    *,
    linear_cls: type[nn.Module],
    packer: Any,
) -> NunchakuLinearWrapper:
    """Quantize one real BF16 Linear and pack it in Nunchaku's MMA layout."""
    if dense.weight.device.type != "cuda":
        raise ValueError(f"target Linear is not on CUDA: {dense.weight.device}")
    if dense.in_features % 128 or dense.out_features % 128:
        raise ValueError(
            "Nunchaku INT4 packer requires in/out features divisible by 128, "
            f"got {dense.in_features}x{dense.out_features}"
        )

    dtype = dense.weight.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Nunchaku W4A4 requires FP16/BF16 source weights, got {dtype}")

    kernel = linear_cls(
        dense.in_features,
        dense.out_features,
        rank=0,
        bias=dense.bias is not None,
        precision="int4",
        act_unsigned=False,
        torch_dtype=dtype,
        device=dense.weight.device,
    )

    # Symmetric signed INT4 RTN, independently scaled for each output
    # channel and each contiguous group of 64 input channels.
    grouped = dense.weight.detach().float().reshape(dense.out_features, -1, 64)
    scales_fp32 = grouped.abs().amax(dim=-1).div_(7.0)
    scales_fp32 = torch.where(scales_fp32 > 0, scales_fp32, torch.ones_like(scales_fp32))
    quantized = (
        torch.round(grouped / scales_fp32.unsqueeze(-1))
        .clamp_(-7, 7)
        .to(torch.int32)
        .reshape(dense.out_features, dense.in_features)
    )
    scales = scales_fp32.to(dtype=dtype)

    packed_weight = packer.pack_weight(quantized)
    packed_scales = packer.pack_scale(scales, group_size=64)
    if packed_weight.shape != kernel.qweight.shape:
        raise RuntimeError(
            f"packed weight shape {packed_weight.shape} != kernel shape {kernel.qweight.shape}"
        )
    if packed_scales.shape != kernel.wscales.shape:
        raise RuntimeError(
            f"packed scale shape {packed_scales.shape} != kernel shape {kernel.wscales.shape}"
        )

    kernel.qweight.copy_(packed_weight)
    kernel.wscales.copy_(packed_scales)
    kernel.smooth_factor.fill_(1)
    kernel.smooth_factor_orig.fill_(1)
    if kernel.bias is not None:
        kernel.bias.copy_(dense.bias.detach().to(dtype=dtype))
        kernel.bias.requires_grad_(False)
    kernel.proj_down.requires_grad_(False)
    kernel.proj_up.requires_grad_(False)
    kernel.eval()

    del grouped, scales_fp32, quantized, scales, packed_weight, packed_scales
    return NunchakuLinearWrapper(kernel, weight_dtype=dtype).eval()


@torch.inference_mode()
def replace_target_linears(
    model: nn.Module,
    *,
    scope: str,
    projections: frozenset[str],
    linear_cls: type[nn.Module],
    packer_cls: type,
) -> dict[str, Any]:
    target_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches_scope(name, scope, projections)
    ]
    selected_sides = 2 if scope == "all" else 1
    expected = selected_sides * 18 * len(projections)
    if len(target_names) != expected:
        raise RuntimeError(
            f"scope={scope!r} matched {len(target_names)} Linear layers; expected {expected} "
            f"(18 Gemma blocks x {len(projections)} selected projections x {selected_sides} side(s))"
        )

    packer = packer_cls(bits=4)
    dense_bytes = 0
    packed_bytes = 0
    projection_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    started = time.perf_counter()

    for index, name in enumerate(target_names, 1):
        dense = model.get_submodule(name)
        if not isinstance(dense, nn.Linear):
            raise RuntimeError(f"{name} changed type while replacing layers: {type(dense)}")
        dense_bytes += _parameter_bytes(dense)

        match = _matches_scope(name, scope, projections)
        assert match is not None
        projection_counts[match.group("projection")] += 1
        shape_counts[f"{dense.in_features}x{dense.out_features}"] += 1

        quantized = make_nunchaku_linear(dense, linear_cls=linear_cls, packer=packer)
        packed_bytes += _parameter_bytes(quantized)
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, quantized)

        del dense, quantized
        if index % 8 == 0 or index == len(target_names):
            gc.collect()
            torch.cuda.empty_cache()
        if index == 1 or index % 18 == 0 or index == len(target_names):
            print(f"[pack] {index}/{len(target_names)} layers", file=sys.stderr, flush=True)

    torch.cuda.synchronize()
    return {
        "scope": scope,
        "projections": sorted(projections),
        "count": len(target_names),
        "expected_count": expected,
        "projection_counts": dict(sorted(projection_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "dense_parameter_bytes_replaced": dense_bytes,
        "nunchaku_parameter_bytes": packed_bytes,
        "parameter_compression_ratio": dense_bytes / packed_bytes,
        "pack_seconds": time.perf_counter() - started,
        "rank": 0,
        "weight_quantization": "symmetric signed INT4 RTN, group_size=64",
        "smooth_factor": 1.0,
        "per_step_scale_table": False,
    }


def _latency_summary(samples_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[p95_index],
        "samples_ms": samples_ms,
    }


@torch.inference_mode()
def time_policy(
    policy: Any,
    observation: dict[str, Any],
    noise: np.ndarray,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def infer() -> dict[str, Any]:
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype or torch.float16,
            enabled=autocast_dtype is not None,
        ):
            return policy.infer(observation, noise=noise)

    output = None
    for _ in range(warmup):
        output = infer()
        torch.cuda.synchronize(device)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = infer()
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    assert output is not None
    actions = np.asarray(output["actions"])
    smoke = {
        "finite": bool(np.isfinite(actions).all()),
        "action_shape": list(actions.shape),
        "action_dtype": str(actions.dtype),
    }
    memory = {
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": torch.cuda.memory_allocated(device),
        "reserved_before_bytes": reserved_before,
        "reserved_after_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    return {**_latency_summary(samples_ms), **smoke}, memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--nunchaku-root", type=Path, default=DEFAULT_NUNCHAKU_ROOT)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compute-dtype",
        choices=("native", "fp16"),
        default="native",
        help=(
            "native keeps the checkpoint's BF16 path; fp16 casts the eager model "
            "to FP16 and uses CUDA autocast for a speed-only FP16 comparison"
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=("fastest", "scope"),
        default="fastest",
        help=(
            "fastest: replace only Pali gate/up (measured winners); "
            "scope: replace all seven projections selected by --scope"
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("pali", "expert", "all"),
        default="pali",
        help="Scope used by --strategy scope, or by an explicit --ops selection",
    )
    parser.add_argument(
        "--ops",
        default=None,
        help=(
            "Comma-separated projection override, e.g. gate_proj,up_proj. "
            "When set, --scope selects where these ops are replaced."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Also write the JSON payload here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be >= 0 and --iterations must be >= 1")
    if not (args.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {args.checkpoint}")

    if args.ops is not None:
        requested_ops = [item.strip() for item in args.ops.split(",") if item.strip()]
        invalid_ops = sorted(set(requested_ops).difference(ALL_PROJECTIONS))
        if not requested_ops or invalid_ops:
            raise ValueError(
                f"--ops must be a non-empty comma-separated subset of {ALL_PROJECTIONS}; "
                f"invalid={invalid_ops}"
            )
        effective_scope = args.scope
        effective_projections = frozenset(requested_ops)
        effective_strategy = "custom_ops"
    elif args.strategy == "fastest":
        effective_scope = "pali"
        effective_projections = FASTEST_PROJECTIONS
        effective_strategy = "fastest"
    else:
        effective_scope = args.scope
        effective_projections = frozenset(ALL_PROJECTIONS)
        effective_strategy = "scope"

    openpi_root = args.openpi_root.expanduser().resolve()
    sys.path.insert(0, str(openpi_root / "src"))
    sys.path.insert(0, str(openpi_root))

    from openpi.policies import libero_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Disable torch.compile for both paths.  This keeps the comparison eager and
    # prevents Dynamo compilation/recompilation from contaminating this smoke.
    train_config = _config.get_config(args.config)
    model_config = dataclasses.replace(train_config.model, pytorch_compile_mode=None)
    train_config = dataclasses.replace(train_config, model=model_config)

    print(f"[load] {args.checkpoint}", file=sys.stderr, flush=True)
    load_started = time.perf_counter()
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        pytorch_device=str(device),
    )
    load_seconds = time.perf_counter() - load_started
    model = policy._model  # noqa: SLF001 - benchmark needs in-place module replacement.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    autocast_dtype = None
    if args.compute_dtype == "fp16":
        model.to(dtype=torch.float16)
        autocast_dtype = torch.float16

    observation = libero_policy.make_libero_example()
    noise = np.zeros(
        (train_config.model.action_horizon, train_config.model.action_dim),
        dtype=np.float32,
    )

    baseline_label = "FP16 autocast" if autocast_dtype == torch.float16 else "eager BF16"
    print(f"[run] synchronized {baseline_label} baseline", file=sys.stderr, flush=True)
    baseline, baseline_memory = time_policy(
        policy,
        observation,
        noise,
        device=device,
        autocast_dtype=autocast_dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    linear_cls, packer_cls = bootstrap_nunchaku(args.nunchaku_root)
    print(
        f"[replace] strategy={effective_strategy} scope={effective_scope} "
        f"ops={','.join(sorted(effective_projections))}",
        file=sys.stderr,
        flush=True,
    )
    replacement = replace_target_linears(
        model,
        scope=effective_scope,
        projections=effective_projections,
        linear_cls=linear_cls,
        packer_cls=packer_cls,
    )

    print("[run] synchronized Nunchaku W4A4 policy", file=sys.stderr, flush=True)
    quantized, quantized_memory = time_policy(
        policy,
        observation,
        noise,
        device=device,
        autocast_dtype=autocast_dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    properties = torch.cuda.get_device_properties(device)
    payload = {
        "experiment": "pi05_libero_nunchaku_w4a4_e2e_speed_smoke",
        "paths": {
            "openpi_root": str(openpi_root),
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "nunchaku_root": str(args.nunchaku_root.expanduser().resolve()),
        },
        "config": {
            "name": args.config,
            "strategy": effective_strategy,
            "requested_strategy": args.strategy,
            "requested_scope": args.scope,
            "scope": effective_scope,
            "projections": sorted(effective_projections),
            "action_horizon": train_config.model.action_horizon,
            "action_dim_internal": train_config.model.action_dim,
            "dtype": train_config.model.dtype,
            "compute_dtype": args.compute_dtype,
            "denoise_steps": 10,
            "torch_compile": False,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "per_step_scale_table": False,
            "activation_scale": "Nunchaku runtime group64 scale (kernel-required; no timestep table)",
        },
        "environment": {
            "gpu": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version.split()[0],
        },
        "load_seconds": load_seconds,
        "replacement": replacement,
        "baseline_16bit": {
            **baseline,
            "compute_dtype": "fp16" if autocast_dtype == torch.float16 else "bfloat16",
            "memory": baseline_memory,
        },
        "nunchaku_w4a4": {**quantized, "memory": quantized_memory},
        "speedup_vs_16bit": baseline["median_ms"] / quantized["median_ms"],
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"[done] wrote {output_path}", file=sys.stderr, flush=True)
    print(rendered)


if __name__ == "__main__":
    main()
