#!/usr/bin/env python3
"""Latency of the REAL Omega-QVLA method (pack params) on the INT4 backend.

Loads a released ``quantized.pt`` pack and measures, under torch.compile
(max-autotune, deployment config), the compiled 16-bit baseline vs the
real-pack method pipeline on a selectable set of Pali and action-expert
projections:

    input rotation (perm + block bmm) -> smooth + INT4 quantize -> INT4x4
    GEMM (pack's rotated GPTQ weights) -> output rotation restore

``--mode`` ablates rotation cost: 0 full, 1 no out-rot, 2 no rotations.

Run inside the OpenPI venv:
    CUDA_VISIBLE_DEVICES=0 python tools/run_pi05_omega_method_speed.py \
        --batch 8 --mode 0 --output logs_gpu0_debug/method_b8_m0.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("pi05_runner", TOOLS / "run_pi05_nunchaku_speed.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)

DEFAULT_PACK = Path("/ceph/workspace/xinyu/omega_packs/pi05_object/quantized.pt")

_SCOPE_CONFIGS = {
    "pali_gateup": (("pali",), ("gate_proj", "up_proj")),
    "pali_all": (
        ("pali",),
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ),
    "all": (
        ("pali", "expert"),
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ),
}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--scope", choices=tuple(_SCOPE_CONFIGS), default="pali_gateup",
        help="pali_gateup: historical 36-layer path; pali_all: 126 Pali linears; "
             "all: Pali + action expert, 252 linears",
    )
    parser.add_argument("--mode", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument(
        "--rot-impl", choices=("graph", "triton", "rotglu", "rotglu2", "rotglu3", "rotglu4", "op"), default="graph",
        help="graph: trace both rotations; triton: custom Triton output rotation; "
             "rotglu: Triton output rotations fused with MLP GLU; "
             "rotglu2: rotglu + single-pass Triton input rotations everywhere; "
             "op: rotations eager inside an opaque custom op",
    )
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(_runner.DEFAULT_OPENPI_ROOT / "src"))
    import jax
    from openpi.models import model as _model
    from openpi.policies import libero_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    train_config = _config.get_config("pi05_libero")
    print(f"[load] policy, compile={train_config.model.pytorch_compile_mode!r}", file=sys.stderr, flush=True)
    policy = _policy_config.create_trained_policy(
        train_config, _runner.DEFAULT_CHECKPOINT, pytorch_device=str(device)
    )
    model = policy._model  # noqa: SLF001
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    example = libero_policy.make_libero_example()
    inputs = jax.tree.map(lambda x: x, example)
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    tiled = jax.tree.map(
        lambda x: x.expand(args.batch, *x.shape[1:]).contiguous() if isinstance(x, torch.Tensor) else x,
        inputs,
    )
    observation = _model.Observation.from_dict(tiled)
    noise = torch.zeros(
        (args.batch, train_config.model.action_horizon, train_config.model.action_dim),
        dtype=torch.float32, device=device,
    )

    def infer() -> torch.Tensor:
        return model.sample_actions(device, observation, noise=noise, num_steps=10)

    def measure(label: str) -> dict:
        warmups = []
        for _ in range(args.warmup):
            t0 = time.perf_counter()
            actions = infer()
            torch.cuda.synchronize(device)
            warmups.append((time.perf_counter() - t0) * 1000.0)
        samples = []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            actions = infer()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - t0) * 1000.0)
        summary = {
            **_runner._latency_summary(samples),  # noqa: SLF001
            "warmup_ms": warmups,
            "finite": bool(torch.isfinite(actions).all().item()),
        }
        print(f"[{label}] median={summary['median_ms']:.2f}ms (first warmup {warmups[0]:.0f}ms)",
              file=sys.stderr, flush=True)
        return summary

    baseline = measure("compiled 16bit")

    print(f"[pack] {args.pack}", file=sys.stderr, flush=True)
    pack = torch.load(args.pack, map_location="cpu", weights_only=False)
    linear_cls, packer_cls = _runner.bootstrap_nunchaku(_runner.DEFAULT_NUNCHAKU_ROOT)
    if "omega_method_ops" in sys.modules:
        ops_module = sys.modules["omega_method_ops"]
    else:
        ops_spec = importlib.util.spec_from_file_location("omega_method_ops", TOOLS / "omega_method_ops.py")
        ops_module = importlib.util.module_from_spec(ops_spec)
        sys.modules["omega_method_ops"] = ops_module
        ops_spec.loader.exec_module(ops_module)
    scopes, projections = _SCOPE_CONFIGS[args.scope]
    replacement = ops_module.install_method_layers(
        model, pack, _runner, linear_cls, packer_cls,
        scopes=scopes, projections=projections, mode=args.mode, rot_impl=args.rot_impl,
    )
    print(f"[replace] {replacement}", file=sys.stderr, flush=True)

    quantized = measure("compiled omega-method")

    payload = {
        "experiment": "pi05_libero_omega_method_real_params_speed",
        "config": {
            "pack": str(args.pack),
            "mode": args.mode,
            "rot_impl": args.rot_impl,
            "scope": args.scope,
            "batch": args.batch,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "compile_mode": train_config.model.pytorch_compile_mode,
            "timing_scope": "model.sample_actions",
            "note": "fixed order (16bit then method), same process",
        },
        "replacement": replacement,
        "baseline_compiled_16bit": baseline,
        "omega_method_compiled": quantized,
        "speedup_vs_compiled_16bit": baseline["median_ms"] / quantized["median_ms"],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
