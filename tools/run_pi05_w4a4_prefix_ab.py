#!/usr/bin/env python3
"""Backbone-step latency A/B: FP16 vs fused-gate/up W4A4, per model segment.

Instruments ``paligemma_with_expert.forward`` with CUDA events during real
``policy.infer`` calls and splits each inference into:

* ``prefix``  - the VLM backbone forward over the 968-token prefix
                (call 0; this is where every quantized layer lives);
* ``denoise`` - the 10 action-expert denoising steps (calls 1..10, 16-bit
                in both arms).

Same fused configuration and same-process ABBA interleave as
run_pi05_w4a4_ab.py.  Latency-only; dense weights stay resident.

Run inside the OpenPI venv:
    CUDA_VISIBLE_DEVICES=0 python tools/run_pi05_w4a4_prefix_ab.py \
        --compute-dtype fp16 --rounds 10 --output logs_gpu0_debug/prefix_ab_fp16.json
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

import numpy as np
import torch

AB_PATH = Path(__file__).with_name("run_pi05_w4a4_ab.py")
_spec = importlib.util.spec_from_file_location("pi05_ab", AB_PATH)
_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ab)
_runner = _ab._runner  # noqa: SLF001
MODE = _ab.MODE


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-dtype", choices=("native", "fp16"), default="fp16")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(_runner.DEFAULT_OPENPI_ROOT / "src"))
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
    autocast_dtype = None
    if args.compute_dtype == "fp16":
        model.to(dtype=torch.float16)
        autocast_dtype = torch.float16

    linear_cls, packer_cls = _runner.bootstrap_nunchaku(_runner.DEFAULT_NUNCHAKU_ROOT)
    replacement = _ab.install_fused_gate_up(model, linear_cls, packer_cls)
    print(f"[replace] {replacement}", file=sys.stderr, flush=True)

    # CUDA-event instrumentation of every backbone forward call.
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

    observation = libero_policy.make_libero_example()
    noise = np.zeros(
        (train_config.model.action_horizon, train_config.model.action_dim), dtype=np.float32
    )

    def infer_segmented() -> tuple[float, float, float, int]:
        call_events.clear()
        started = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype or torch.float16,
            enabled=autocast_dtype is not None,
        ):
            policy.infer(observation, noise=noise)
        torch.cuda.synchronize(device)
        e2e_ms = (time.perf_counter() - started) * 1000.0
        call_ms = [start.elapsed_time(end) for start, end in call_events]
        return call_ms[0], sum(call_ms[1:]), e2e_ms, len(call_ms)

    for arm in ("16bit", "w4a4"):
        MODE["w4a4"] = arm == "w4a4"
        for _ in range(max(1, args.warmup)):
            prefix_ms, denoise_ms, e2e_ms, calls = infer_segmented()
        print(
            f"[warmup {arm}] prefix={prefix_ms:.2f}ms denoise={denoise_ms:.2f}ms "
            f"e2e={e2e_ms:.2f}ms backbone_calls={calls}",
            file=sys.stderr,
            flush=True,
        )

    samples: dict[str, dict[str, list[float]]] = {
        arm: {"prefix": [], "denoise": [], "e2e": []} for arm in ("16bit", "w4a4")
    }
    for round_index in range(args.rounds):
        order = ("16bit", "w4a4", "w4a4", "16bit") if round_index % 2 == 0 else (
            "w4a4", "16bit", "16bit", "w4a4"
        )
        for arm in order:
            MODE["w4a4"] = arm == "w4a4"
            prefix_ms, denoise_ms, e2e_ms, _ = infer_segmented()
            samples[arm]["prefix"].append(prefix_ms)
            samples[arm]["denoise"].append(denoise_ms)
            samples[arm]["e2e"].append(e2e_ms)
        print(
            f"[round {round_index + 1}/{args.rounds}] "
            f"prefix 16bit={statistics.median(samples['16bit']['prefix']):.2f} "
            f"w4a4={statistics.median(samples['w4a4']['prefix']):.2f}",
            file=sys.stderr,
            flush=True,
        )

    def summarize(arm: str) -> dict:
        return {
            segment: _runner._latency_summary(values)  # noqa: SLF001
            for segment, values in samples[arm].items()
        }

    summary = {arm: summarize(arm) for arm in ("16bit", "w4a4")}
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "experiment": "pi05_libero_w4a4_backbone_segment_interleaved_ab",
        "config": {
            "compute_dtype": args.compute_dtype,
            "baseline": "FP16 autocast" if autocast_dtype else "eager BF16",
            "ours": "Pali 18x fused gate+up W4A4 (N=32768); rest 16-bit",
            "protocol": "same-process ABBA interleave; CUDA events around paligemma_with_expert.forward",
            "segments": {
                "prefix": "VLM backbone forward, 968-token prefix (all quantized layers)",
                "denoise": "sum of 10 action-expert denoise forwards (16-bit both arms)",
                "e2e": "full policy.infer wall time",
            },
            "rounds": args.rounds,
            "samples_per_arm": 2 * args.rounds,
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
        "baseline_16bit": summary["16bit"],
        "ours_w4a4": summary["w4a4"],
        "speedup": {
            segment: summary["16bit"][segment]["median_ms"] / summary["w4a4"][segment]["median_ms"]
            for segment in ("prefix", "denoise", "e2e")
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
