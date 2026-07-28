# pi0.5 W4A4 RealQuant kernel 加速报告

日期：2026-07-28

## 结论

在这台 NVIDIA A100-SXM4-40GB（SM80）上，最终可用的最快方案是：

> **Nunchaku INT4 W4A4，只替换 PaliGemma 的 18 个 `gate_proj` 和 18 个
> `up_proj`，其余 Pali 层和整个 Expert 保持 16-bit。**

原因不是所有 W4A4 都快，而是 pi0.5 的两条路径形状差异很大：

- PaliGemma 的 `gate/up` 是 `M=968, K=2048, N=16384`，Nunchaku 完整
  `activation quantize + W4A4 GEMM` 比 FP16 `F.linear` 快 **1.400×**，
  比原生 BF16 快 **1.358×**。
- Expert 每次只有 `M=10`，Nunchaku 会补到 256 行；完整 W4A4 只有 FP16
  的约 **0.18–0.25×**，所以不应该替换。
- Atom 即使只测预打包后的 GEMM ceiling，所有 pi0.5 目标层加权汇总也只有
  FP16 的 **0.587×**。它缺少适用于 K=1024/2048/16384 的通用在线
  activation pack，接入后只会更慢。

真实 pi0.5 policy 的同步端到端结果：

| 计算路径 | 16-bit 中位数 | 混合 Nunchaku 中位数 | 加速 |
|---|---:|---:|---:|
| 原生 BF16，A/B 第 1 轮（各 5 次） | 269.581 ms | 258.680 ms | **1.042×** |
| 原生 BF16，A/B 第 2 轮（各 10 次） | 271.581 ms | 262.519 ms | **1.035×** |
| speed-only FP16 autocast（各 5 次） | 290.336 ms | 274.954 ms | **1.056×** |

三轮输出均为 `(10, 7)`，且全部 finite。没有评测 LIBERO 成功率或动作质量；
这是用户指定的纯速度实验。

## 代码与环境

| 项目 | 值 |
|---|---|
| Omega-QVLA | `3727e2203568db43fc5fba06ee8686c1b47c044f` |
| Nunchaku | `8f41840596bd516d434a1f88ac16c86fdb64e74f` |
| Atom | `7e3618b1a7a7c86e1c93cc909b1510c046d76ac6` |
| OpenPI | `c23745b5ad24e98f66967ea795a07b2588ed6c79` |
| checkpoint | `models/pi05_libero_pytorch/model.safetensors` |
| GPU | NVIDIA A100-SXM4-40GB, compute capability 8.0 |
| Python / PyTorch | 3.11.15 / 2.7.1+cu126 |
| 编译 toolkit | CUDA 12.8.93，目标 `sm_80` |
| pi0.5 dtype | BF16 |
| action horizon / 内部 action dim | 10 / 32 |
| denoise steps | 10 |

代码来源：

- [Omega-QVLA](https://github.com/UCMP13753/Omega-QVLA)
- [Nunchaku](https://github.com/nunchaku-ai/nunchaku)；
  [官方安装文档](https://nunchaku.tech/docs/nunchaku/installation/installation.html)
  明确列出 SM80/A100 支持。
- [Atom](https://github.com/efeslab/Atom)；
  [MLSys 2024 论文](https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf)。

## pi0.5 的真实 Linear 调度

本地 checkpoint 为 PaliGemma Gemma-2B + Gemma-300M Expert，共 18 层。
真实 policy forward 的目标调用如下：

| 分支 | 聚合算子 | M | K → N | 每个 action 的调用数 |
|---|---|---:|---:|---:|
| Pali | q/o | 968 | 2048 → 2048 | 36 |
| Pali | k/v | 968 | 2048 → 256 | 36 |
| Pali | gate/up | 968 | 2048 → 16384 | 36 |
| Pali | down | 968 | 16384 → 2048 | 18 |
| Expert | q | 10 | 1024 → 2048 | 180 |
| Expert | k/v | 10 | 1024 → 256 | 360 |
| Expert | o | 10 | 2048 → 1024 | 180 |
| Expert | gate/up | 10 | 1024 → 4096 | 360 |
| Expert | down | 10 | 4096 → 1024 | 180 |

Pali 的 `M=968` 来自 3×256 图像 token + 200 个固定长度语言 token，只跑一次
prefix；Expert 的 `M=10` 每个 denoise step 都运行一次，共 10 次。

## 测试方法

### Atom

官方 Atom kernel/CMake 写死 `sm_86`，README 也说明当前 kernel 只针对 RTX 4090
优化。本实验：

1. 移除显式 `sm_86`，以 `sm_80` 重编；
2. 新增不依赖旧 NVBench/Thrust 的 `ATOM_PI05_ONLY` 构建；
3. 在同一可执行文件里用 CUDA Events 比较 Atom 预打包 GEMM 与 cuBLAS FP16；
4. 每个形状热身 20 次，计时循环重复 7 轮，报告中位数和 p95。

Atom 数据是 **GEMM-only ceiling**：不含 FP16/BF16 activation 的在线 reorder、
动态量化和 pack。它因此不能与 Nunchaku 的完整 forward 混为一个指标。

### Nunchaku

Nunchaku 使用真正的 `mma.sync ... s4.s4.s32` Tensor Core 路径：

- weight：signed INT4，group size 64，离线 pack；
- activation：运行时逐 token、group-64 动态量化；
- 输出：FP16 或 BF16；
- 关闭 SVD 低秩修正，即 `rank=0`；
- `smooth_factor=1`；
- 无 bias（目标 Gemma projection 本身无 bias）；
- 不安装、不查询任何 denoise per-step scale table。

Nunchaku 上游的 fused quantize kernel 在 `rank=0` 时仍会预读空 LoRA 指针。
本地增加了 `rank == 0` early return，重编后 rank-0 FP16/BF16 CUDA smoke 均通过。

微基准对每个形状热身 30 次，每轮 100 次，重复 7 轮，CUDA Events 报中位数和
p95。`full` 包含在线 activation quantize 和 W4A4 GEMM；`gemm-only` 使用预量化
activation。

### 真实 policy

- `torch.compile` 关闭，使用 eager，避免编译时间污染；
- 使用 `make_libero_example()` 和固定全零 noise；
- 每次 `policy.infer` 后执行 `torch.cuda.synchronize()`；
- 同一进程先测 16-bit，再从真实 checkpoint 权重做 group-64 INT4 RTN、官方
  Nunchaku MMA layout pack，最后测 W4A4；
- BF16 做两轮独立加载复验；另做一轮 speed-only FP16 cast + autocast；
- 权重 pack 是一次性初始化成本，不计入推理 latency。

## Atom 结果

以下全部是对 Atom 有利的预打包 GEMM-only 数据：

| 分支/算子 | M×K×N | Atom ms | FP16 ms | Atom / FP16 speedup |
|---|---:|---:|---:|---:|
| Expert q | 10×1024×2048 | 0.020777 | 0.011397 | 0.549× |
| Expert k/v | 10×1024×256 | 0.020531 | 0.010680 | 0.520× |
| Expert o | 10×2048×1024 | 0.033782 | 0.011837 | 0.350× |
| Expert gate/up | 10×1024×4096 | 0.020808 | 0.011489 | 0.552× |
| Expert down | 10×4096×1024 | 0.059894 | 0.014746 | 0.246× |
| Pali q/o | 968×2048×2048 | 0.052654 | 0.042742 | 0.812× |
| Pali k/v | 968×2048×256 | 0.027044 | 0.013363 | 0.494× |
| Pali gate/up | 968×2048×16384 | 0.252232 | 0.270746 | 1.073× |
| Pali down | 968×16384×2048 | 0.339722 | 0.269722 | 0.794× |

按真实调用数加权：

| 范围 | Atom GEMM-only | FP16 | 加速 |
|---|---:|---:|---:|
| Pali | 18.064 ms | 16.622 ms | 0.920× |
| Expert | 35.484 ms | 14.817 ms | 0.418× |
| 全部目标 Linear | 53.548 ms | 31.439 ms | **0.587×** |

因此 Atom 在这台 A100 和 pi0.5 shape 上淘汰。即使唯一获胜的 Pali gate/up
也只有 1.073× GEMM ceiling；加上 Atom 当前缺失的通用在线 activation pack 后，
没有端到端获胜空间。

## Nunchaku 微基准结果

### FP16

| 分支/算子 | FP16 ms | W4A4 GEMM-only ms | W4A4 full ms | full / FP16 speedup |
|---|---:|---:|---:|---:|
| Expert q | 0.011438 | 0.027085 | 0.059781 | 0.191× |
| Expert k/v | 0.014858 | 0.026870 | 0.059781 | 0.249× |
| Expert o | 0.014930 | 0.048773 | 0.060877 | 0.245× |
| Expert gate/up | 0.011581 | 0.027146 | 0.059679 | 0.194× |
| Expert down | 0.014684 | 0.071731 | 0.081183 | 0.181× |
| Pali q/o | 0.042752 | 0.038676 | 0.060180 | 0.710× |
| Pali k/v | 0.013302 | 0.038298 | 0.060303 | 0.221× |
| Pali gate/up | 0.286607 | 0.196700 | 0.204698 | **1.400×** |
| Pali down | 0.284140 | 0.279675 | 0.344228 | 0.825× |

若盲目替换全部目标 Linear，真实调用加权后的 full W4A4 只有 FP16 的
0.353×。这个数据说明必须按 shape 选择，而不是按模型统一开启 W4A4。

### pi0.5 原生 BF16（Pali）

| 算子 | BF16 ms | W4A4 full ms | full / BF16 speedup |
|---|---:|---:|---:|
| q/o | 0.071322 | 0.070451 | 1.012× |
| k/v | 0.017203 | 0.065004 | 0.265× |
| gate/up | 0.284447 | 0.209449 | **1.358×** |
| down | 0.293837 | 0.346593 | 0.848× |

最终只选择了稳定且收益明显的 gate/up；q/o 的 1.012× 太接近噪声，不纳入。

## 真实 pi0.5 端到端结果

### BF16 复验

| 轮次 | 热身 / 采样 | BF16 median | W4A4 median | speedup | 输出 |
|---|---:|---:|---:|---:|---|
| 1 | 1 / 5 | 269.581 ms | 258.680 ms | 1.042× | `(10,7)`, finite |
| 2 | 2 / 10 | 271.581 ms | 262.519 ms | 1.035× | `(10,7)`, finite |

两轮平均加速比为 1.038×。第二轮的 p95 分别为 276.815 ms 和 276.351 ms，
说明尾延迟收益不稳定；结论只取中位数。

### FP16 speed-only 对照

模型转 FP16 并在 CUDA autocast 下运行：

| FP16 median | W4A4 median | speedup | 输出 |
|---:|---:|---:|---|
| 290.336 ms | 274.954 ms | **1.056×** | `(10,7)`, finite |

OpenPI 的正式 pi0.5 配置是 BF16；此 FP16 路径只用于回答“是否快于 FP16”，
不代表官方部署配置或精度保证。

### 显存与初始化

BF16 端到端测量：

- peak allocated：7.120 GiB → 5.469 GiB，减少 **1.651 GiB**；
- steady allocated：6.979 GiB → 5.327 GiB，减少 **23.67%**；
- 36 个 dense weight 参数：2,415,919,104 bytes；
- 对应 Nunchaku 参数：642,023,424 bytes；
- 局部参数压缩比：**3.763×**；
- 运行时 pack：1.19–1.41 s，一次性成本。

## 修改内容

Omega-QVLA：

- `tools/benchmark_pi05_nunchaku.py`：真实 shape 的 FP16/BF16、GEMM-only/full
  Nunchaku 微基准；
- `tools/run_pi05_nunchaku_speed.py`：真实 checkpoint 权重转换、选择性模块替换、
  BF16/FP16 同步端到端 A/B；
- 本报告。

Nunchaku：

- `src/kernels/zgemm/lora.cuh`：rank-0 LoRA-down 空指针 early return；
- 已生成本机 `sm_80` 扩展。

Atom：

- `kernels/CMakeLists.txt`：可选 `sm_80` 与 `ATOM_PI05_ONLY`；
- `kernels/src/GEMM/bench_pi05_atom.cu`：pi0.5 实际矩形 shape、Atom vs cuBLAS
  FP16、7 轮 median/p95。

## 复现

### 1. 编译 Nunchaku SM80

```bash
cd /ceph/workspace/xinyu/Nunchaku
CUDA_VISIBLE_DEVICES=1 \
CUDA_HOME=/home/xinyu/avatar_image_work/rootfs/usr/local/cuda-12.8 \
PATH=/home/xinyu/avatar_image_work/rootfs/usr/local/cuda-12.8/bin:/home/xinyu/verl_venv/bin:/usr/bin:/bin \
NUNCHAKU_INSTALL_MODE=FAST MAX_JOBS=8 \
/home/xinyu/verl_venv/bin/python setup.py build_ext --inplace
```

### 2. Nunchaku pi0.5 shape benchmark

```bash
cd /ceph/workspace/xinyu/Omega-QVLA
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
/ceph/workspace/xinyu/openpi/.venv/bin/python \
tools/benchmark_pi05_nunchaku.py \
  --nunchaku-root /ceph/workspace/xinyu/Nunchaku \
  --dtype fp16 --scope all \
  --warmup 30 --iterations 100 --repeats 7
```

### 3. 真实 pi0.5 最快策略

```bash
cd /ceph/workspace/xinyu/Omega-QVLA
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
/ceph/workspace/xinyu/openpi/.venv/bin/python \
tools/run_pi05_nunchaku_speed.py \
  --strategy fastest --compute-dtype native \
  --warmup 2 --iterations 10
```

FP16 speed-only 对照把 `--compute-dtype native` 改为
`--compute-dtype fp16`。

### 4. Atom SM80 benchmark

```bash
cd /ceph/workspace/xinyu/Atom/kernels
/home/xinyu/avatar_image_work/rootfs/opt/conda/lib/python3.11/site-packages/cmake/data/bin/cmake \
  -S . -B build-sm80 -G Ninja \
  -DATOM_PI05_ONLY=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/home/xinyu/avatar_image_work/rootfs/usr/local/cuda-12.8/bin/nvcc
/home/xinyu/avatar_image_work/rootfs/opt/conda/lib/python3.11/site-packages/cmake/data/bin/cmake \
  --build build-sm80 --target bench_pi05_atom -j 8
LD_LIBRARY_PATH=/home/xinyu/avatar_image_work/rootfs/usr/local/cuda-12.8/lib64 \
  ./build-sm80/bench_pi05_atom 1
```

## 限制与下一步

1. 本实验按要求不评估 task performance。rank-0、无 SmoothQuant/per-step table 会影响
   动作质量；finite 只证明 CUDA/policy 路径可运行。
2. 不能把 activation scale 完全删除：Nunchaku 的 group-64 runtime scale 是 W4A4
   数学本身需要的。删除的是 Omega 的 denoise per-step table/context 和额外平滑分支。
3. 当前 gate 和 up 各自重复量化同一份 activation。若做一个 fused `gate+up` wrapper，
   共享一次 activation quantize，理论上还可进一步提升。
4. LIBERO prompt 实际只有约 6–21 个有效语言 token，却固定补到 200；第三个
   right-wrist image 在当前数据中恒为 mask=False。裁语言 padding、跳过全局无效图像
   的潜在收益可能大于换 kernel，但本次没有改动模型语义路径。
5. 若要进入正式部署，需要离线保存 packed checkpoint、增加量化误差/任务成功率测试，
   并在目标并发、CUDA Graph 和真实 observation 分布下复测。
