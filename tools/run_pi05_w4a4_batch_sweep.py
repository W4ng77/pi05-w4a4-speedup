#!/usr/bin/env python3
"""Batched rollout A/B: 16-bit vs fused-gate/up W4A4 across batch sizes.

Setting: parallel-environment rollout (e.g. LIBERO evaluation with B
simulators stepping the same policy).  The observation is tiled to batch B
and ``model.sample_actions`` is timed directly, skipping the CPU-side
input/output transforms so the number is the policy forward itself.

Same interleaved ABBA protocol and fused configuration as
run_pi05_w4a4_ab.py; dense weights stay resident (latency-only).

Run inside the OpenPI venv:
    CUDA_VISIBLE_DEVICES=0 python tools/run_pi05_w4a4_batch_sweep.py \
        --compute-dtype fp16 --batches 1,4,8,16 \
        --output logs_gpu0_debug/batch_sweep_fp16.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import numpy as np
import torch

AB_PATH = Path(__file__).with_name("run_pi05_w4a4_ab.py")
_spec = importlib.util.spec_from_file_location("pi05_ab", AB_PATH)
_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ab)
_runner = _ab._runner  # noqa: SLF001
MODE = _ab.MODE


class DualLinear(torch.nn.Module):
    """Mode-switchable pair of one dense Linear and its W4A4 replacement."""

    def __init__(self, dense: torch.nn.Linear, quant: torch.nn.Module):
        super().__init__()
        self.dense = dense
        self.quant = quant

    @property
    def weight(self) -> torch.Tensor:
        return self.dense.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.dense.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant(x) if MODE["w4a4"] else self.dense(x)


DUAL_PREFIXES = {
    "pali": "paligemma_with_expert.paligemma.model.language_model.layers",
    "expert": "paligemma_with_expert.gemma_expert.model.layers",
}


@torch.inference_mode()
def install_dual_projections(
    model: torch.nn.Module,
    projections: list[str],
    linear_cls: type,
    packer_cls: type,
    scope: str = "pali",
) -> int:
    packer = packer_cls(bits=4)
    prefix = DUAL_PREFIXES[scope]
    count = 0
    for layer_index in range(18):
        for projection in projections:
            parent_name = "self_attn" if projection in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            parent = model.get_submodule(f"{prefix}.{layer_index}.{parent_name}")
            dense = getattr(parent, projection)
            if not isinstance(dense, torch.nn.Linear):
                raise RuntimeError(f"layer {layer_index}.{projection} is not a plain Linear")
            quant = _runner.make_nunchaku_linear(dense, linear_cls=linear_cls, packer=packer)
            setattr(parent, projection, DualLinear(dense, quant).eval())
            count += 1
        if layer_index % 6 == 5:
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return count


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-dtype", choices=("native", "fp16"), default="fp16")
    parser.add_argument("--batches", default="1,4,8,16")
    parser.add_argument(
        "--extra-ops",
        default=None,
        help=(
            "Comma-separated extra Pali projections to also replace with "
            "mode-switchable unfused W4A4 (e.g. q_proj,o_proj,down_proj)"
        ),
    )
    parser.add_argument(
        "--expert-ops",
        default=None,
        help=(
            "Comma-separated expert projections to also replace with "
            "mode-switchable unfused W4A4; pass all seven for uniform W4A4"
        ),
    )
    parser.add_argument("--rounds", type=int, default=6, help="ABBA rounds per batch size")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch_sizes = [int(b) for b in args.batches.split(",")]

    sys.path.insert(0, str(_runner.DEFAULT_OPENPI_ROOT / "src"))
    from openpi.models import model as _model
    from openpi.policies import libero_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_config = _config.get_config("pi05_libero")
    model_config = dataclasses.replace(train_config.model, pytorch_compile_mode=None)
    train_config = dataclasses.replace(train_config, model=model_config)

    print("[load] policy", file=sys.stderr, flush=True)
    policy = _policy_config.create_trained_policy(
        train_config, _runner.DEFAULT_CHECKPOINT, pytorch_device=str(device)
    )
    model = policy._model  # noqa: SLF001
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.compute_dtype == "fp16":
        model.to(dtype=torch.float16)

    linear_cls, packer_cls = _runner.bootstrap_nunchaku(_runner.DEFAULT_NUNCHAKU_ROOT)
    replacement = _ab.install_fused_gate_up(model, linear_cls, packer_cls)
    extra_ops = []
    if args.extra_ops:
        extra_ops = [op.strip() for op in args.extra_ops.split(",") if op.strip()]
        replacement["extra_dual_projections"] = extra_ops
        replacement["extra_dual_count"] = install_dual_projections(
            model, extra_ops, linear_cls, packer_cls, scope="pali"
        )
    if args.expert_ops:
        expert_ops = [op.strip() for op in args.expert_ops.split(",") if op.strip()]
        replacement["expert_dual_projections"] = expert_ops
        replacement["expert_dual_count"] = install_dual_projections(
            model, expert_ops, linear_cls, packer_cls, scope="expert"
        )
    print(f"[replace] {replacement}", file=sys.stderr, flush=True)

    # CUDA-event instrumentation: call 0 per inference is the VLM prefix
    # forward (where all quantized layers live), calls 1..10 the denoise steps.
    backbone = model.paligemma_with_expert
    original_forward = backbone.forward
    call_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def timed_forward(*call_args, **call_kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_forward(*call_args, **call_kwargs)
        end.record()
        call_events.append((start, end))
        return result

    backbone.forward = timed_forward

    # One pass through the real input transforms, then tile tensors to batch B.
    example = libero_policy.make_libero_example()
    inputs = jax.tree.map(lambda x: x, example)
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs
    )

    autocast_enabled = args.compute_dtype == "fp16"

    def run(batch: int) -> torch.Tensor:
        tiled = jax.tree.map(
            lambda x: x.expand(batch, *x.shape[1:]).contiguous()
            if isinstance(x, torch.Tensor)
            else x,
            inputs,
        )
        observation = _model.Observation.from_dict(tiled)
        noise = torch.zeros(
            (batch, train_config.model.action_horizon, train_config.model.action_dim),
            dtype=torch.float32,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            return model.sample_actions(device, observation, noise=noise, num_steps=10)

    segments = ("prefix", "denoise", "e2e")
    results = []
    for batch in batch_sizes:
        try:
            samples = {
                arm: {segment: [] for segment in segments} for arm in ("16bit", "w4a4")
            }

            def timed_run(batch_size: int, arm: str) -> None:
                call_events.clear()
                started = time.perf_counter()
                actions = run(batch_size)
                torch.cuda.synchronize(device)
                e2e_ms = (time.perf_counter() - started) * 1000.0
                if not torch.isfinite(actions).all():
                    raise RuntimeError(f"non-finite actions: batch={batch_size} arm={arm}")
                call_ms = [start.elapsed_time(end) for start, end in call_events]
                samples[arm]["prefix"].append(call_ms[0])
                samples[arm]["denoise"].append(sum(call_ms[1:]))
                samples[arm]["e2e"].append(e2e_ms)

            for arm in ("16bit", "w4a4"):
                MODE["w4a4"] = arm == "w4a4"
                for _ in range(args.warmup):
                    timed_run(batch, arm)
                for segment in segments:
                    samples[arm][segment].clear()
            for round_index in range(args.rounds):
                order = ("16bit", "w4a4", "w4a4", "16bit") if round_index % 2 == 0 else (
                    "w4a4", "16bit", "16bit", "w4a4"
                )
                for arm in order:
                    MODE["w4a4"] = arm == "w4a4"
                    timed_run(batch, arm)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"batch": batch, "error": "cuda_out_of_memory"})
            print(f"[batch {batch}] OOM, skipped", file=sys.stderr, flush=True)
            continue
        row = {
            "batch": batch,
            "baseline_16bit": {
                segment: _runner._latency_summary(samples["16bit"][segment])  # noqa: SLF001
                for segment in segments
            },
            "ours_w4a4": {
                segment: _runner._latency_summary(samples["w4a4"][segment])  # noqa: SLF001
                for segment in segments
            },
        }
        row["speedup"] = {
            segment: (
                row["baseline_16bit"][segment]["median_ms"]
                / row["ours_w4a4"][segment]["median_ms"]
            )
            for segment in segments
        }
        results.append(row)
        print(
            f"[batch {batch}] prefix 16bit={row['baseline_16bit']['prefix']['median_ms']:.1f}ms "
            f"w4a4={row['ours_w4a4']['prefix']['median_ms']:.1f}ms "
            f"speedup prefix={row['speedup']['prefix']:.3f} "
            f"denoise={row['speedup']['denoise']:.3f} e2e={row['speedup']['e2e']:.3f}",
            file=sys.stderr,
            flush=True,
        )

    properties = torch.cuda.get_device_properties(device)
    payload = {
        "experiment": "pi05_libero_w4a4_fused_gateup_batch_sweep",
        "config": {
            "compute_dtype": args.compute_dtype,
            "baseline": "FP16 autocast" if autocast_enabled else "eager BF16",
            "ours": "Pali 18x fused gate+up W4A4 (N=32768); rest 16-bit",
            "protocol": "same-process ABBA interleave per batch size; sample_actions timed directly",
            "denoise_steps": 10,
            "rounds": args.rounds,
            "samples_per_arm_per_batch": 2 * args.rounds,
            "warmup_per_arm": args.warmup,
            "torch_compile": False,
            "seed": args.seed,
        },
        "environment": {
            "gpu": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            **_ab.gpu_clocks(),
        },
        "replacement": replacement,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
