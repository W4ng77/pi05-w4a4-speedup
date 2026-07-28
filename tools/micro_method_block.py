"""Per-layer / per-block latency of the full Omega-QVLA method unit.

Uses real pack parameters (layer 0) and the best implementations:
  * single projection: in-rot (index_select+einsum) -> Nunchaku quantize+GEMM
    (smooth slot = pack act scale) -> out-rot (Triton single-pass)
  * MLP block: in-rot x2 -> W4A4 x2 -> fused rotglu Triton kernel
against the 16-bit baselines (F.linear, and gate/up + tanh-gelu*up for the
block). Shapes: M=968 (batch-1 prefix) and M=7744 (batch-8 rollout).

Run: CUDA_VISIBLE_DEVICES=0 python tools/micro_method_block.py
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))
_spec = importlib.util.spec_from_file_location("pi05_runner", TOOLS / "run_pi05_nunchaku_speed.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

from triton_block_rotate import block_rotate_triton, rotglu_triton  # noqa: E402

PACK = "/ceph/workspace/xinyu/omega_packs/pi05_object/quantized.pt"
PALI = "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp"


def bench(fn, repeats=5, iters=100, warmup=30):
    meds = []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        meds.append(s.elapsed_time(e) / iters * 1000)
    return statistics.median(meds)


def make_kernel(record, linear_cls, packer, dtype, device):
    carrier = torch.nn.Linear(
        record["in_features"], record["out_features"], bias=False, dtype=dtype, device=device
    )
    carrier.weight.copy_(record["weight_res_q"].to(dtype))
    wrapper = runner.make_nunchaku_linear(carrier, linear_cls=linear_cls, packer=packer)
    kernel = wrapper.kernel
    scale = record["act_scale_table"].mean(dim=0).to(dtype=dtype, device=device)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    kernel.smooth_factor.copy_(scale)
    kernel.smooth_factor_orig.copy_(scale)
    return kernel


def in_rot(x, perm, blocks):
    xp = x.index_select(-1, perm)
    n = blocks.shape[0]
    return torch.einsum("mnk,nkh->mnh", xp.reshape(-1, n, 64), blocks).reshape(x.shape[0], -1)


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pack = torch.load(PACK, map_location="cpu", weights_only=False)
    linear_cls, packer_cls = runner.bootstrap_nunchaku(runner.DEFAULT_NUNCHAKU_ROOT)
    packer = packer_cls(bits=4)

    rec_g = pack[f"{PALI}.gate_proj"]
    rec_u = pack[f"{PALI}.up_proj"]
    kg = make_kernel(rec_g, linear_cls, packer, dtype, device)
    ku = make_kernel(rec_u, linear_cls, packer, dtype, device)
    perm_g = rec_g["duquant_rotation_perm"].to(device)
    perm_u = rec_u["duquant_rotation_perm"].to(device)
    rin_g = rec_g["duquant_rotation_blocks"].to(dtype=dtype, device=device)
    rin_u = rec_u["duquant_rotation_blocks"].to(dtype=dtype, device=device)
    rout_g = rec_g["duquant_rotation_out_blocks"].to(dtype=dtype, device=device).contiguous()
    rout_u = rec_u["duquant_rotation_out_blocks"].to(dtype=dtype, device=device).contiguous()

    wg16 = torch.randn(16384, 2048, dtype=dtype, device=device) * 0.02
    wu16 = torch.randn(16384, 2048, dtype=dtype, device=device) * 0.02

    for M in (968, 7744):
        x = torch.randn(M, 2048, dtype=dtype, device=device)

        def gate16():
            return F.linear(x, wg16)

        def block16():
            g = F.linear(x, wg16)
            u = F.linear(x, wu16)
            return F.gelu(g, approximate="tanh") * u

        def gate_method():
            xr = in_rot(x, perm_g, rin_g)
            y = kg(xr.reshape(1, M, 2048)).reshape(M, 16384)
            return block_rotate_triton(y.contiguous(), rout_g)

        def block_method():
            xg = in_rot(x, perm_g, rin_g)
            xu = in_rot(x, perm_u, rin_u)
            g = kg(xg.reshape(1, M, 2048)).reshape(M, 16384)
            u = ku(xu.reshape(1, M, 2048)).reshape(M, 16384)
            return rotglu_triton(g.contiguous(), u.contiguous(), rout_g, rout_u)

        assert torch.isfinite(block_method()).all()
        t_g16, t_gm = bench(gate16), bench(gate_method)
        t_b16, t_bm = bench(block16), bench(block_method)
        print(f"M={M}: gate_proj  16bit {t_g16:8.1f} us  method {t_gm:8.1f} us  speedup {t_g16 / t_gm:.3f}x")
        print(f"M={M}: MLP block  16bit {t_b16:8.1f} us  method {t_bm:8.1f} us  speedup {t_b16 / t_bm:.3f}x")


if __name__ == "__main__":
    main()
