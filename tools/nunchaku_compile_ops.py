"""torch.library custom-op wrapper for Nunchaku W4A4 — compile-friendly.

Registers ``omega::w4a4_linear`` so Dynamo/Inductor treat the Nunchaku
quantize+GEMM pair as one opaque graph node instead of graph-breaking.
The op is dispatched through a process-local kernel registry (the packed
weights live inside SVDQW4A4Linear modules; the graph only carries an
integer handle, which is constant-folded per call site).

Also provides ``install_fused_mlp_custom_op``: replaces each PaliGemma
GemmaMLP.forward with

    down_proj(act_fn(fused[..., :I]) * fused[..., I:])   # fused = one W4A4 GEMM

which is fully traceable (plain slicing, no stash/data_ptr tricks).
"""

from __future__ import annotations

import types

import torch
from torch import nn

# Registry lives on the torch module: unique per process even if this file
# is exec'd more than once (importlib spec loading does not dedupe).
if not hasattr(torch, "_omega_w4a4_kernels"):
    torch._omega_w4a4_kernels = []
_KERNELS: list[nn.Module] = torch._omega_w4a4_kernels
_INTERMEDIATE = 16384
_PALI_PREFIX = "paligemma_with_expert.paligemma.model.language_model.layers"


@torch.library.custom_op("omega::w4a4_linear", mutates_args=())
def w4a4_linear(x: torch.Tensor, handle: int, out_features: int) -> torch.Tensor:
    kernel = _KERNELS[handle]
    leading = x.shape[:-1]
    out = kernel(x.reshape(1, -1, x.shape[-1]))
    return out.reshape(*leading, out_features)


@w4a4_linear.register_fake
def _w4a4_linear_fake(x: torch.Tensor, handle: int, out_features: int) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], out_features)


@torch.inference_mode()
def install_fused_mlp_custom_op(
    model: nn.Module,
    runner_module,
    linear_cls: type[nn.Module],
    packer_cls: type,
) -> dict:
    """Quantize+fuse the 18 Pali gate/up pairs behind the custom op."""
    import time

    packer = packer_cls(bits=4)
    started = time.perf_counter()
    count = 0
    for layer_index in range(18):
        mlp = model.get_submodule(f"{_PALI_PREFIX}.{layer_index}.mlp")
        gate, up = mlp.gate_proj, mlp.up_proj
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear):
            raise RuntimeError(f"layer {layer_index}: gate/up already replaced")
        fused_dense = nn.Linear(
            gate.in_features,
            2 * _INTERMEDIATE,
            bias=False,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        fused_dense.weight.copy_(torch.cat([gate.weight, up.weight], dim=0))
        wrapper = runner_module.make_nunchaku_linear(
            fused_dense, linear_cls=linear_cls, packer=packer
        )
        del fused_dense
        handle = len(_KERNELS)
        _KERNELS.append(wrapper.kernel)

        def fused_forward(
            self,
            x: torch.Tensor,
            _h: int = handle,
            _n: int = 2 * _INTERMEDIATE,
            _i: int = _INTERMEDIATE,
        ) -> torch.Tensor:
            fused = torch.ops.omega.w4a4_linear(x, _h, _n)
            return self.down_proj(self.act_fn(fused[..., :_i]) * fused[..., _i:])

        mlp.forward = types.MethodType(fused_forward, mlp)
        count += 1
        if layer_index % 6 == 5:
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {
        "fused_layers": count,
        "custom_op": "omega::w4a4_linear",
        "fused_shape": f"2048x{2 * _INTERMEDIATE}",
        "pack_seconds": time.perf_counter() - started,
        "rank": 0,
        "weight_quantization": "symmetric signed INT4 RTN, group_size=64",
    }
