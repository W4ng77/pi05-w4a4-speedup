"""Real Omega-QVLA method execution behind a torch.library custom op.

Loads per-layer records from an Omega-QVLA ``quantized.pt`` pack
(``dit_svdquant_v1``) and installs, for each selected Pali projection, the
FAITHFUL deployment pipeline of the method:

    x -> input rotation (zigzag perm + block-diag(64) bmm)
      -> Nunchaku fused quantize (per-channel act-scale in the smooth slot,
         per-token group-64 dynamic INT4)
      -> INT4x4 GEMM with the pack's rotated GPTQ weights (repacked group-64)
      -> output rotation restore (block-diag bmm)

registered as ``omega::method_linear(x, handle, out_features, mode)``.
mode: 0 = full method; 1 = skip output rotation; 2 = skip both rotations
(ablation to localize rotation cost). All parameters are the real pack
values; latency is what deployment would pay with unfused rotations.
"""

from __future__ import annotations

import time

import torch
from torch import nn

if not hasattr(torch, "_omega_method_entries"):
    torch._omega_method_entries = []
_ENTRIES: list = torch._omega_method_entries

_PALI_PREFIX = "paligemma_with_expert.paligemma.model.language_model.layers"


def _block_rotate(x2d: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    n_blocks, bsz, _ = blocks.shape
    xb = x2d.reshape(-1, n_blocks, bsz).transpose(0, 1)
    out = torch.bmm(xb, blocks)
    return out.transpose(0, 1).reshape(x2d.shape[0], n_blocks * bsz)


@torch.library.custom_op("omega::method_linear", mutates_args=())
def method_linear(x: torch.Tensor, handle: int, out_features: int, mode: int) -> torch.Tensor:
    entry = _ENTRIES[handle]
    leading = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    if mode < 2:
        x2d = x2d.index_select(-1, entry["perm"])
        x2d = _block_rotate(x2d, entry["in_blocks"].to(dtype=x.dtype))
    y = entry["kernel"](x2d.reshape(1, -1, x2d.shape[-1])).reshape(-1, out_features)
    if mode == 0 and entry["out_blocks"] is not None:
        y = _block_rotate(y, entry["out_blocks"].to(dtype=y.dtype))
    return y.reshape(*leading, out_features)


@method_linear.register_fake
def _method_linear_fake(x: torch.Tensor, handle: int, out_features: int, mode: int) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], out_features)


class OmegaMethodLinear(nn.Module):
    def __init__(self, handle: int, in_features: int, out_features: int,
                 mode: int, weight_dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.handle = handle
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.register_buffer(
            "_dtype_sentinel", torch.empty(0, dtype=weight_dtype, device=device), persistent=False
        )

    @property
    def weight(self) -> torch.Tensor:
        return self._dtype_sentinel

    @property
    def bias(self) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.omega.method_linear(x, self.handle, self.out_features, self.mode)


@torch.inference_mode()
def install_method_layers(
    model: nn.Module,
    pack: dict,
    runner_module,
    linear_cls: type[nn.Module],
    packer_cls: type,
    projections: tuple[str, ...] = ("gate_proj", "up_proj"),
    mode: int = 0,
    device: torch.device | None = None,
) -> dict:
    packer = packer_cls(bits=4)
    started = time.perf_counter()
    count = 0
    for layer_index in range(18):
        for projection in projections:
            parent_name = "mlp" if projection in ("gate_proj", "up_proj", "down_proj") else "self_attn"
            name = f"{_PALI_PREFIX}.{layer_index}.{parent_name}.{projection}"
            record = pack[name]
            parent = model.get_submodule(f"{_PALI_PREFIX}.{layer_index}.{parent_name}")
            dense = getattr(parent, projection)
            if not isinstance(dense, nn.Linear):
                raise RuntimeError(f"{name} already replaced")
            dev = device or dense.weight.device
            dtype = dense.weight.dtype

            carrier = nn.Linear(
                record["in_features"], record["out_features"], bias=False, dtype=dtype, device=dev
            )
            carrier.weight.copy_(record["weight_res_q"].to(dtype))
            wrapper = runner_module.make_nunchaku_linear(carrier, linear_cls=linear_cls, packer=packer)
            del carrier
            kernel = wrapper.kernel
            act_scale = record["act_scale_table"].mean(dim=0).to(dtype=dtype, device=dev)
            act_scale = torch.where(act_scale > 0, act_scale, torch.ones_like(act_scale))
            kernel.smooth_factor.copy_(act_scale)
            kernel.smooth_factor_orig.copy_(act_scale)

            entry = {
                "kernel": kernel,
                "perm": record["duquant_rotation_perm"].to(device=dev),
                "in_blocks": record["duquant_rotation_blocks"].to(dtype=dtype, device=dev),
                "out_blocks": (
                    record["duquant_rotation_out_blocks"].to(dtype=dtype, device=dev)
                    if "duquant_rotation_out_blocks" in record
                    else None
                ),
            }
            handle = len(_ENTRIES)
            _ENTRIES.append(entry)
            setattr(
                parent,
                projection,
                OmegaMethodLinear(
                    handle, record["in_features"], record["out_features"], mode, dtype, dev
                ).eval(),
            )
            count += 1
        if layer_index % 6 == 5:
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {
        "installed_layers": count,
        "projections": list(projections),
        "mode": {0: "full method (in-rot + smooth + W4A4 + out-rot)",
                 1: "no out-rot", 2: "no rotations (smooth + W4A4 only)"}[mode],
        "custom_op": "omega::method_linear",
        "pack_seconds": time.perf_counter() - started,
        "weights": "pack weight_res_q (rotated GPTQ), repacked RTN group-64",
        "act_scale": "pack act_scale_table -> kernel smooth slot",
    }
