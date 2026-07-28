#!/usr/bin/env python3
"""Does W4A4 still pay under torch.compile (max-autotune)?

Loads the pi0.5 policy with its DEFAULT compile mode (unlike every other
script here, which disables compile), measures the compiled 16-bit baseline,
then swaps the 18x2 Pali gate/up projections to Nunchaku W4A4 in place
(unfused wrappers - the fused stash pattern graph-breaks Dynamo), lets
Dynamo recompile, and measures again.

Fixed order (baseline then W4A4) - this answers the coarse question
"is compile+W4A4 viable and what is the sign of the delta", not a
percent-level claim.  Expect Dynamo graph breaks around the Nunchaku
custom op; part of the question is how much they cost under max-autotune.

Run inside the OpenPI venv:
    CUDA_VISIBLE_DEVICES=0 python tools/run_pi05_w4a4_compile_ab.py \
        --output logs_gpu0_debug/compile_ab.json
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

RUNNER_PATH = Path(__file__).with_name("run_pi05_nunchaku_speed.py")
_spec = importlib.util.spec_from_file_location("pi05_runner", RUNNER_PATH)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(_runner.DEFAULT_OPENPI_ROOT / "src"))
    from openpi.policies import libero_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    # Default config: keeps pytorch_compile_mode = "max-autotune".
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

    observation = libero_policy.make_libero_example()
    noise = np.zeros(
        (train_config.model.action_horizon, train_config.model.action_dim), dtype=np.float32
    )

    def measure(label: str) -> dict:
        warmup_ms = []
        for _ in range(args.warmup):
            started = time.perf_counter()
            policy.infer(observation, noise=noise)
            torch.cuda.synchronize(device)
            warmup_ms.append((time.perf_counter() - started) * 1000.0)
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            output = policy.infer(observation, noise=noise)
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
        actions = np.asarray(output["actions"])
        summary = {
            **_runner._latency_summary(samples),  # noqa: SLF001
            "warmup_ms": warmup_ms,
            "finite": bool(np.isfinite(actions).all()),
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
    replacement = _runner.replace_target_linears(
        model,
        scope="pali",
        projections=frozenset(("gate_proj", "up_proj")),
        linear_cls=linear_cls,
        packer_cls=packer_cls,
    )
    quantized = measure("compiled w4a4")

    payload = {
        "experiment": "pi05_libero_w4a4_under_torch_compile",
        "compile_mode": compile_mode,
        "config": {
            "note": "fixed order (16bit then w4a4), same process; coarse viability test",
            "ours": "Pali 18x gate + 18x up unfused W4A4",
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "replacement": replacement,
        "baseline_compiled_16bit": baseline,
        "w4a4_compiled": quantized,
        "speedup_vs_compiled_16bit": baseline["median_ms"] / quantized["median_ms"],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
