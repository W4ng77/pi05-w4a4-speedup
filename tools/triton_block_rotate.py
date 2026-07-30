"""Single-pass Triton kernel for DuQuant block-diagonal rotation.

Computes ``out[:, b*64:(b+1)*64] = x[:, perm[b*64:(b+1)*64]] @ R_b`` in one
memory pass (optional gather fused with the per-block 64x64 matmul), versus
the eager/Inductor path which materializes the permuted tensor first.

Run as a script to microbenchmark against index_select+einsum:
    CUDA_VISIBLE_DEVICES=1 python tools/triton_block_rotate.py
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _block_rotate_kernel(
    x_ptr, r_ptr, out_ptr, perm_ptr,
    M, stride_xm, stride_om,
    HAS_PERM: tl.constexpr, BM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, 64)
    row_mask = rows < M

    r_offs = pid_b * 64 * 64 + cols[:, None] * 64 + cols[None, :]
    r = tl.load(r_ptr + r_offs)

    if HAS_PERM:
        src = tl.load(perm_ptr + pid_b * 64 + cols)
    else:
        src = pid_b * 64 + cols
    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + src[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    y = tl.dot(x, r)
    dst = pid_b * 64 + cols
    tl.store(
        out_ptr + rows[:, None] * stride_om + dst[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


@triton.jit
def _rotglu_kernel(
    g_ptr, u_ptr, rg_ptr, ru_ptr, out_ptr, dst_idx_ptr,
    M, stride_m,
    HAS_DST: tl.constexpr, BM: tl.constexpr,
):
    """out[:, b*64:(b+1)*64] = silu(g @ Rg_b) * (u @ Ru_b), single pass."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, 64)
    row_mask = rows < M
    r_offs = pid_b * 64 * 64 + cols[:, None] * 64 + cols[None, :]
    rg = tl.load(rg_ptr + r_offs)
    ru = tl.load(ru_ptr + r_offs)
    offs = rows[:, None] * stride_m + (pid_b * 64 + cols)[None, :]
    g = tl.load(g_ptr + offs, mask=row_mask[:, None], other=0.0)
    u = tl.load(u_ptr + offs, mask=row_mask[:, None], other=0.0)
    gr = tl.dot(g, rg)
    ur = tl.dot(u, ru)
    # Gemma uses gelu_pytorch_tanh; for latency parity any elementwise gate is
    # equivalent cost — use exact tanh-gelu to keep numerics meaningful.
    c0 = 0.7978845608028654
    c1 = 0.044715
    inner = c0 * (gr + c1 * gr * gr * gr)
    # tanh(z) = 2*sigmoid(2z) - 1: overflow-safe for large |z|
    act = 0.5 * gr * (2.0 * tl.sigmoid(2.0 * inner))
    y = act * ur
    if HAS_DST:
        dst = tl.load(dst_idx_ptr + pid_b * 64 + cols)
        out_offs = rows[:, None] * stride_m + dst[None, :]
    else:
        out_offs = offs
    tl.store(out_ptr + out_offs, y.to(out_ptr.dtype.element_ty), mask=row_mask[:, None])


def rotglu_triton(g: torch.Tensor, u: torch.Tensor, rg: torch.Tensor, ru: torch.Tensor,
                  dst_index: torch.Tensor | None = None, BM: int = 64) -> torch.Tensor:
    """g, u: (M, F) un-restored GEMM outputs; rg, ru: (F//64, 64, 64).

    ``dst_index`` (len F) scatters the store so the result comes out already
    permuted for the NEXT layer's input rotation (down_proj), letting that
    rotation skip its gather pass."""
    M, F = g.shape
    out = torch.empty_like(g)
    grid = (triton.cdiv(M, BM), F // 64)
    _rotglu_kernel[grid](
        g, u, rg, ru, out,
        dst_index if dst_index is not None else g,
        M, g.stride(0),
        HAS_DST=dst_index is not None, BM=BM, num_warps=4,
    )
    return out


def block_rotate_triton(
    x2d: torch.Tensor, blocks: torch.Tensor, perm: torch.Tensor | None = None, BM: int = 64
) -> torch.Tensor:
    """x2d: (M, F) contiguous; blocks: (F//64, 64, 64); perm: (F,) int or None."""
    M, F = x2d.shape
    out = torch.empty_like(x2d)
    grid = (triton.cdiv(M, BM), F // 64)
    _block_rotate_kernel[grid](
        x2d, blocks, out,
        perm if perm is not None else x2d,  # dummy ptr when unused
        M, x2d.stride(0), out.stride(0),
        HAS_PERM=perm is not None, BM=BM,
        num_warps=4,
    )
    return out


def _reference(x2d, blocks, perm):
    xp = x2d.index_select(-1, perm) if perm is not None else x2d
    n, b, _ = blocks.shape
    return torch.einsum(
        "mnk,nkh->mnh", xp.reshape(-1, n, b), blocks
    ).reshape(x2d.shape[0], -1)


def _bench(fn, repeats=5, iters=200, warmup=50):
    import statistics
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


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    for dtype in (torch.bfloat16,):
        for name, M, F, has_perm in (
            ("in-rot  M=7744 F=2048", 7744, 2048, True),
            ("out-rot M=7744 F=16384", 7744, 16384, False),
            ("in-rot  M=968  F=2048", 968, 2048, True),
            ("out-rot M=968  F=16384", 968, 16384, False),
        ):
            x = torch.randn(M, F, dtype=dtype, device=device)
            blocks = (torch.randn(F // 64, 64, 64, dtype=dtype, device=device) * 0.125)
            perm = torch.randperm(F, device=device) if has_perm else None
            ref = _reference(x, blocks, perm)
            got = block_rotate_triton(x, blocks, perm)
            err = (ref.float() - got.float()).abs().max().item()
            t_ref = _bench(lambda: _reference(x, blocks, perm))
            t_tri = _bench(lambda: block_rotate_triton(x, blocks, perm))
            print(f"{name} {dtype}: eager {t_ref:8.1f} us  triton {t_tri:8.1f} us  "
                  f"speedup {t_ref / t_tri:.2f}x  maxerr {err:.4f}")


if __name__ == "__main__":
    main()
