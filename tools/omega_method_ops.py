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

import importlib.util
import sys
import time
import types
from pathlib import Path

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


if not hasattr(torch, "_omega_outrot_entries"):
    torch._omega_outrot_entries = []
_OUTROT: list = torch._omega_outrot_entries


@torch.library.custom_op("omega::triton_out_rotate", mutates_args=())
def triton_out_rotate(y: torch.Tensor, handle: int) -> torch.Tensor:
    from triton_block_rotate import block_rotate_triton

    leading = y.shape[:-1]
    y2 = y.reshape(-1, y.shape[-1])
    out = block_rotate_triton(y2.contiguous(), _OUTROT[handle])
    return out.reshape(*leading, y.shape[-1])


@triton_out_rotate.register_fake
def _triton_out_rotate_fake(y: torch.Tensor, handle: int) -> torch.Tensor:
    return torch.empty_like(y)


if not hasattr(torch, "_omega_rotglu_entries"):
    torch._omega_rotglu_entries = []
_ROTGLU: list = torch._omega_rotglu_entries


@torch.library.custom_op("omega::rotglu", mutates_args=())
def rotglu(g: torch.Tensor, u: torch.Tensor, handle: int) -> torch.Tensor:
    from triton_block_rotate import rotglu_triton

    rg, ru = _ROTGLU[handle]
    leading = g.shape[:-1]
    g2 = g.reshape(-1, g.shape[-1]).contiguous()
    u2 = u.reshape(-1, u.shape[-1]).contiguous()
    return rotglu_triton(g2, u2, rg, ru).reshape(*leading, g.shape[-1])


@rotglu.register_fake
def _rotglu_fake(g: torch.Tensor, u: torch.Tensor, handle: int) -> torch.Tensor:
    return torch.empty_like(g)


class OmegaMethodLinearGraphRot(nn.Module):
    """Rotations as traceable graph ops (Inductor-fused); only quantize+GEMM
    stays inside the opaque custom op (``omega::w4a4_linear``).
    With ``triton_outrot=True`` the output rotation runs as the single-pass
    Triton kernel instead of the Inductor einsum."""

    def __init__(self, w4a4_handle: int, in_features: int, out_features: int, mode: int,
                 perm: torch.Tensor, in_blocks: torch.Tensor, out_blocks: torch.Tensor | None,
                 weight_dtype: torch.dtype, device: torch.device,
                 triton_outrot: bool = False):
        super().__init__()
        self.w4a4_handle = w4a4_handle
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.n_in = in_features // 64
        self.n_out = out_features // 64
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("in_blocks", in_blocks, persistent=False)
        self.has_out = out_blocks is not None
        self.triton_outrot = triton_outrot
        if self.has_out:
            self.register_buffer("out_blocks", out_blocks, persistent=False)
            if triton_outrot:
                self.outrot_handle = len(_OUTROT)
                _OUTROT.append(out_blocks.contiguous())
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
        leading = x.shape[:-1]
        if self.mode < 2:
            xp = x.index_select(-1, self.perm)
            x2 = xp.reshape(-1, self.n_in, 64)
            x2 = torch.einsum("mnk,nkh->mnh", x2, self.in_blocks.to(x.dtype))
            x = x2.reshape(*leading, self.in_features)
        y = torch.ops.omega.w4a4_linear(x, self.w4a4_handle, self.out_features)
        if self.mode == 0 and self.has_out:
            if self.triton_outrot:
                y = torch.ops.omega.triton_out_rotate(y, self.outrot_handle)
            else:
                y2 = y.reshape(-1, self.n_out, 64)
                y2 = torch.einsum("mnk,nkh->mnh", y2, self.out_blocks.to(y.dtype))
                y = y2.reshape(*leading, self.out_features)
        return y


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
    rot_impl: str = "graph",  # graph | triton | op
    device: torch.device | None = None,
) -> dict:
    if rot_impl in ("graph", "triton", "rotglu"):
        if rot_impl == "rotglu" and tuple(projections) != ("gate_proj", "up_proj"):
            raise ValueError("rotglu fusion requires projections=(gate_proj, up_proj)")
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))
        # Ensure omega::w4a4_linear + its kernel registry exist.
        if "nunchaku_compile_ops" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                "nunchaku_compile_ops", Path(__file__).parent / "nunchaku_compile_ops.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["nunchaku_compile_ops"] = module
            spec.loader.exec_module(module)
        w4a4_kernels = torch._omega_w4a4_kernels
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
            if rot_impl in ("graph", "triton", "rotglu"):
                w4a4_handle = len(w4a4_kernels)
                w4a4_kernels.append(kernel)
                # rotglu: the out-rot moves into the fused rotglu kernel, so
                # the per-projection module runs mode>=1 internally.
                eff_mode = max(mode, 1) if rot_impl == "rotglu" else mode
                new_module = OmegaMethodLinearGraphRot(
                    w4a4_handle, record["in_features"], record["out_features"], eff_mode,
                    entry["perm"], entry["in_blocks"], entry["out_blocks"], dtype, dev,
                    triton_outrot=rot_impl == "triton",
                ).eval()
            else:
                handle = len(_ENTRIES)
                _ENTRIES.append(entry)
                new_module = OmegaMethodLinear(
                    handle, record["in_features"], record["out_features"], mode, dtype, dev
                ).eval()
            setattr(parent, projection, new_module)
            count += 1
        if rot_impl == "rotglu" and mode == 0:
            mlp = model.get_submodule(f"{_PALI_PREFIX}.{layer_index}.mlp")
            rec_g = pack[f"{_PALI_PREFIX}.{layer_index}.mlp.gate_proj"]
            rec_u = pack[f"{_PALI_PREFIX}.{layer_index}.mlp.up_proj"]
            dev0 = device or mlp.down_proj.weight.device
            dt0 = mlp.down_proj.weight.dtype
            rotglu_handle = len(_ROTGLU)
            _ROTGLU.append((
                rec_g["duquant_rotation_out_blocks"].to(dtype=dt0, device=dev0).contiguous(),
                rec_u["duquant_rotation_out_blocks"].to(dtype=dt0, device=dev0).contiguous(),
            ))

            def _mlp_forward(self, x: torch.Tensor, _h: int = rotglu_handle) -> torch.Tensor:
                g = self.gate_proj(x)
                u = self.up_proj(x)
                return self.down_proj(torch.ops.omega.rotglu(g, u, _h))

            mlp.forward = types.MethodType(_mlp_forward, mlp)
        if layer_index % 6 == 5:
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {
        "installed_layers": count,
        "rot_impl": rot_impl,
        "projections": list(projections),
        "mode": {0: "full method (in-rot + smooth + W4A4 + out-rot)",
                 1: "no out-rot", 2: "no rotations (smooth + W4A4 only)"}[mode],
        "custom_op": "omega::method_linear",
        "pack_seconds": time.perf_counter() - started,
        "weights": "pack weight_res_q (rotated GPTQ), repacked RTN group-64",
        "act_scale": "pack act_scale_table -> kernel smooth slot",
    }
