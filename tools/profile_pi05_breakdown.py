"""Coarse profile of a real pi0.5 LIBERO policy.infer to locate non-Linear time.

Loads the BF16 policy exactly like run_pi05_nunchaku_speed.py, warms up,
then profiles one infer with torch.profiler and prints:
  * total CUDA kernel time vs wall time (launch/CPU-bound gap)
  * top ops by CUDA time
  * number of kernel launches
Run inside the OpenPI venv:
  CUDA_VISIBLE_DEVICES=0 python tools/profile_pi05_breakdown.py
"""

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--openpi-root", default="/ceph/workspace/xinyu/openpi")
    p.add_argument(
        "--checkpoint", default="/ceph/workspace/xinyu/models/pi05_libero_pytorch"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rows", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, f"{args.openpi_root}/src")
    sys.path.insert(0, f"{args.openpi_root}/packages/openpi-client/src")

    import torch
    from openpi.policies import libero_policy, policy_config
    from openpi.training import config as train_config

    cfg = train_config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(
        cfg, args.checkpoint, default_prompt=None, pytorch_device=args.device
    )
    example = libero_policy.make_libero_example()

    for _ in range(2):
        policy.infer(example)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    policy.infer(example)
    torch.cuda.synchronize()
    wall_unprofiled = (time.perf_counter() - t0) * 1e3

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        t0 = time.perf_counter()
        policy.infer(example)
        torch.cuda.synchronize()
        wall_profiled = (time.perf_counter() - t0) * 1e3

    events = prof.key_averages()
    cuda_total_us = sum(
        e.self_device_time_total for e in events if e.self_device_time_total > 0
    )
    n_kernels = sum(
        e.count
        for e in events
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
    )
    print(f"wall unprofiled: {wall_unprofiled:.1f} ms")
    print(f"wall profiled:   {wall_profiled:.1f} ms")
    print(f"sum of self CUDA kernel time: {cuda_total_us / 1e3:.1f} ms")
    print(f"kernel-launch-ish event count (profiled infer): {n_kernels}")
    print()
    print(events.table(sort_by="self_cuda_time_total", row_limit=args.rows))


if __name__ == "__main__":
    main()
