#!/usr/bin/env python3
"""Compile-mode A/B v2: Nunchaku W4A4 behind a torch.library custom op.

Unlike run_pi05_w4a4_compile_ab.py (whose raw kernel calls graph-break
Dynamo), this version registers ``omega::w4a4_linear`` and swaps each Pali
GemmaMLP.forward for a traceable fused form, so Inductor keeps one whole
graph around the INT4 GEMMs.

Order is fixed (compiled 16-bit first, then W4A4 + recompile) — recompiling
per sample would cost minutes per switch.  ``--batch N`` times batched
``model.sample_actions`` directly (tiled observation) instead of
``policy.infer``.

Run inside the OpenPI venv:
    CUDA_VISIBLE_DEVICES=0 python tools/run_pi05_w4a4_compile_v2.py \
        --batch 1 --output logs_gpu0_debug/compile_v2_b1.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "pi05_runner", TOOLS / "run_pi05_nunchaku_speed.py"
)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
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
    compile_mode = train_config.model.pytorch_compile_mode
    print(f"[load] policy with compile mode {compile_mode!r}", file=sys.stderr, flush=True)
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
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs
    )
    tiled = jax.tree.map(
        lambda x: x.expand(args.batch, *x.shape[1:]).contiguous()
        if isinstance(x, torch.Tensor)
        else x,
        inputs,
    )
    observation = _model.Observation.from_dict(tiled)
    noise = torch.zeros(
        (args.batch, train_config.model.action_horizon, train_config.model.action_dim),
        dtype=torch.float32,
        device=device,
    )

    def infer() -> torch.Tensor:
        return model.sample_actions(device, observation, noise=noise, num_steps=10)

    def measure(label: str) -> dict:
        warmup_ms = []
        for _ in range(args.warmup):
            started = time.perf_counter()
            actions = infer()
            torch.cuda.synchronize(device)
            warmup_ms.append((time.perf_counter() - started) * 1000.0)
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            actions = infer()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
        summary = {
            **_runner._latency_summary(samples),  # noqa: SLF001
            "warmup_ms": warmup_ms,
            "finite": bool(torch.isfinite(actions).all().item()),
        }
        print(
            f"[{label}] median={summary['median_ms']:.2f}ms "
            f"(first warmup {warmup_ms[0]:.0f}ms, last {warmup_ms[-1]:.0f}ms)",
            file=sys.stderr,
            flush=True,
        )
        return summary

    baseline = measure("compiled 16bit")

    linear_cls, packer_cls = _runner.bootstrap_nunchaku(_runner.DEFAULT_NUNCHAKU_ROOT)
    if "nunchaku_compile_ops" in sys.modules:
        ops_module = sys.modules["nunchaku_compile_ops"]
    else:
        ops_spec = importlib.util.spec_from_file_location(
            "nunchaku_compile_ops", TOOLS / "nunchaku_compile_ops.py"
        )
        ops_module = importlib.util.module_from_spec(ops_spec)
        sys.modules["nunchaku_compile_ops"] = ops_module
        ops_spec.loader.exec_module(ops_module)
    replacement = ops_module.install_fused_mlp_custom_op(
        model, _runner, linear_cls, packer_cls
    )
    print(f"[replace] {replacement}", file=sys.stderr, flush=True)

    quantized = measure("compiled w4a4 custom-op")

    payload = {
        "experiment": "pi05_libero_w4a4_custom_op_under_torch_compile",
        "compile_mode": compile_mode,
        "config": {
            "note": "fixed order (16bit then w4a4), same process; custom-op integration",
            "ours": "Pali 18x fused gate+up W4A4 behind omega::w4a4_linear custom op",
            "batch": args.batch,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timing_scope": "model.sample_actions (input transforms excluded)",
        },
        "replacement": replacement,
        "baseline_compiled_16bit": baseline,
        "w4a4_compiled_custom_op": quantized,
        "speedup_vs_compiled_16bit": baseline["median_ms"] / quantized["median_ms"],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
