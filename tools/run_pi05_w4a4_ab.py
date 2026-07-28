#!/usr/bin/env python3
"""Interleaved A/B: pi0.5 16-bit baseline vs fused-gate/up Nunchaku W4A4.

Measured-best A100 configuration from the GPU0 microbenchmarks:

* the 18 PaliGemma ``gate_proj``/``up_proj`` pairs are fused into a single
  W4A4 GEMM (N=32768) sharing one activation-quantize launch;
* every other Linear (Pali q/k/v/o/down, the whole expert) stays 16-bit,
  where W4A4 measurably loses on this GPU;
* QKV fusion is deliberately excluded: the contiguous copies required by
  the downstream ``.view()`` cost more than the fused GEMM saves.

Protocol: both arms live in the same process.  The dense gate/up weights are
kept resident next to the packed W4A4 weights and a global mode switch picks
the execution path, so samples interleave in ABBA order per round.  This
removes the fixed 16-bit-then-W4A4 ordering that made the earlier smoke
numbers preliminary.  Because the dense weights stay resident, this script
measures latency only; memory savings come from the drop-in runner.

Run inside the OpenPI venv:
    CUDA_VISIBLE_DEVICES=0 python tools/run_pi05_w4a4_ab.py \
        --compute-dtype fp16 --rounds 10 --output logs_gpu0_debug/ab_fp16.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

RUNNER_PATH = Path(__file__).with_name("run_pi05_nunchaku_speed.py")
_spec = importlib.util.spec_from_file_location("pi05_runner", RUNNER_PATH)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)

MODE = {"w4a4": False}
INTERMEDIATE = 16384


class FusedGateUpLead(nn.Module):
    """Replaces ``gate_proj``: runs the fused gate+up W4A4 GEMM, stashes up."""

    def __init__(self, dense_gate: nn.Linear, dense_up: nn.Linear, fused: nn.Module):
        super().__init__()
        self.dense_gate = dense_gate
        self.dense_up = dense_up
        self.fused = fused
        self._stash: tuple[int, torch.Tensor] | None = None

    @property
    def weight(self) -> torch.Tensor:
        return self.dense_gate.weight

    @property
    def bias(self) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if MODE["w4a4"]:
            out = self.fused(x)
            self._stash = (x.data_ptr(), out)
            return out[..., :INTERMEDIATE]
        return self.dense_gate(x)


class FusedGateUpFollower(nn.Module):
    """Replaces ``up_proj``: returns the up half stashed by the lead."""

    def __init__(self, lead: FusedGateUpLead):
        super().__init__()
        self.lead = lead

    @property
    def weight(self) -> torch.Tensor:
        return self.lead.dense_up.weight

    @property
    def bias(self) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if MODE["w4a4"]:
            stash = self.lead._stash
            if stash is None or stash[0] != x.data_ptr():
                raise RuntimeError("gate/up fusion stash mismatch: unexpected call order")
            return stash[1][..., INTERMEDIATE:]
        return self.lead.dense_up(x)


@torch.inference_mode()
def install_fused_gate_up(model: nn.Module, linear_cls: type[nn.Module], packer_cls: type) -> dict[str, Any]:
    packer = packer_cls(bits=4)
    prefix = "paligemma_with_expert.paligemma.model.language_model.layers"
    started = time.perf_counter()
    count = 0
    for layer_index in range(18):
        mlp = model.get_submodule(f"{prefix}.{layer_index}.mlp")
        gate, up = mlp.gate_proj, mlp.up_proj
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear):
            raise RuntimeError(f"layer {layer_index}: gate/up already replaced")
        fused_dense = nn.Linear(
            gate.in_features, 2 * INTERMEDIATE, bias=False,
            dtype=gate.weight.dtype, device=gate.weight.device,
        )
        fused_dense.weight.copy_(torch.cat([gate.weight, up.weight], dim=0))
        fused_quant = _runner.make_nunchaku_linear(fused_dense, linear_cls=linear_cls, packer=packer)
        del fused_dense
        lead = FusedGateUpLead(gate, up, fused_quant).eval()
        mlp.gate_proj = lead
        mlp.up_proj = FusedGateUpFollower(lead).eval()
        count += 1
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {
        "fused_layers": count,
        "fused_shape": f"2048x{2 * INTERMEDIATE}",
        "pack_seconds": time.perf_counter() - started,
        "rank": 0,
        "weight_quantization": "symmetric signed INT4 RTN, group_size=64",
        "per_step_scale_table": False,
    }


def gpu_clocks() -> dict[str, str]:
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw",
             "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - archival only
        raw = f"unavailable: {error}"
    return {"gpu0_uuid_smclock_memclock_temp_power": raw}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-dtype", choices=("native", "fp16"), default="fp16")
    parser.add_argument("--rounds", type=int, default=10, help="ABBA rounds; 2 samples/arm/round")
    parser.add_argument("--warmup", type=int, default=3, help="warmup inferences per arm")
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
    replacement = install_fused_gate_up(model, linear_cls, packer_cls)
    print(f"[replace] {replacement}", file=sys.stderr, flush=True)

    observation = libero_policy.make_libero_example()
    noise = np.zeros(
        (train_config.model.action_horizon, train_config.model.action_dim), dtype=np.float32
    )

    def infer() -> dict[str, Any]:
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype or torch.float16,
            enabled=autocast_dtype is not None,
        ):
            return policy.infer(observation, noise=noise)

    actions: dict[str, np.ndarray] = {}
    for arm in ("16bit", "w4a4"):
        MODE["w4a4"] = arm == "w4a4"
        output = None
        for _ in range(max(1, args.warmup)):
            output = infer()
            torch.cuda.synchronize(device)
        actions[arm] = np.asarray(output["actions"])
        if not np.isfinite(actions[arm]).all():
            raise RuntimeError(f"non-finite actions in arm {arm}")

    samples: dict[str, list[float]] = {"16bit": [], "w4a4": []}
    schedule: list[str] = []
    for round_index in range(args.rounds):
        order = ("16bit", "w4a4", "w4a4", "16bit") if round_index % 2 == 0 else (
            "w4a4", "16bit", "16bit", "w4a4"
        )
        for arm in order:
            MODE["w4a4"] = arm == "w4a4"
            started = time.perf_counter()
            infer()
            torch.cuda.synchronize(device)
            samples[arm].append((time.perf_counter() - started) * 1000.0)
            schedule.append(arm)
        print(
            f"[round {round_index + 1}/{args.rounds}] "
            f"16bit={statistics.median(samples['16bit']):.2f}ms "
            f"w4a4={statistics.median(samples['w4a4']):.2f}ms",
            file=sys.stderr,
            flush=True,
        )

    summary = {arm: _runner._latency_summary(values) for arm, values in samples.items()}  # noqa: SLF001
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "experiment": "pi05_libero_w4a4_fused_gateup_interleaved_ab",
        "config": {
            "compute_dtype": args.compute_dtype,
            "baseline": "FP16 autocast" if autocast_dtype else "eager BF16",
            "ours": "Pali 18x fused gate+up W4A4 (N=32768, one shared activation quantize); rest 16-bit",
            "protocol": "same-process ABBA interleave, dense weights resident (latency-only)",
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
            **gpu_clocks(),
        },
        "replacement": replacement,
        "schedule": schedule,
        "baseline_16bit": summary["16bit"],
        "ours_w4a4": summary["w4a4"],
        "speedup_ours_vs_16bit": summary["16bit"]["median_ms"] / summary["w4a4"]["median_ms"],
        "action_max_abs_diff_16bit_vs_w4a4": float(
            np.abs(actions["16bit"] - actions["w4a4"]).max()
        ),
        "action_shape": list(actions["w4a4"].shape),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
