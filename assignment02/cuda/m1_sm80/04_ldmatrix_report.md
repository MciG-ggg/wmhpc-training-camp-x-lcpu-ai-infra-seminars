# HW2 1.4 ldmatrix report — smem → fragment 段指令对比

## 测试结果

```
seed=1      manual PASS(0)  ldsm PASS(0)
seed=7      manual PASS(0)  ldsm PASS(0)
seed=42     manual PASS(0)  ldsm PASS(0)
```

两条路径都 PASS。

## smem → fragment 段指令构成

`make ptx/m1_sm80/04_ldmatrix` 生成的 PTX，从 `bar.warp.sync` 起到 `mma.sync` 之间的 smem→fragment 装载段，逐 op 数（不含 mma 本身）：

### manual 路径（load_manual）

| 类别             | 条数 |
|------------------|------|
| `ld.shared.u8`   | 24   |
| `mul.wide.u16`   | 6    |
| `shl.b32`        | 15   |
| `or.b32`         | 19   |
| 地址计算 (and/shl/shr/add/mov) | 9    |
| **合计**         | **73** |

24 个 byte load：A 需要 16 byte/lane（4 个 4-byte b32 regs × 4 byte），B 需要 8 byte/lane（2 regs × 4 byte）。每个 byte 还要再 `shl + or` 才能塞进 b32 的对应字节位置，所以 packing ALU 占了 50 多条。

### ldsm 路径（load_ldsm）

| 类别             | 条数 |
|------------------|------|
| `ldmatrix.x4.shared::cta.b16` (A) | 1 |
| `ldmatrix.x2.shared::cta.b16` (B) | 1 |
| 地址计算 (and/shl/add/mov)       | 12 |
| **合计**         | **14** |

A 一次 `.x4` 直接读 4 个 8x8 子矩阵到 4 个 b32 regs；B 一次 `.x2` 读 2 个子矩阵到 2 个 b32 regs。

## ldmatrix 消掉的是哪部分工作？

ldmatrix 消掉的是**逐 byte 的搬运 + 字节到 b16/b32 的手工 packing**。

PTX `ld.shared.u8` 一次只取 1 byte，每个 lane 要发 16 条（A）+ 8 条（B）才能攒齐 fragment，然后还得用 ~50 条 shl/or 把它打包成 6 个 b32。这种做法从 smem 总带宽看是 24 byte/lane = 96 byte/lane 单 broadcast（每 lane 自己访问自己的字节，但 mma 期望的 fragment 排布并不需要每个 lane 单独去取全部字节）。

`ldmatrix` 的工作方式：硬件识别出 mma 期望的 fragment 排布，把 4（或 2）个 8x8 子矩阵的 smem 行/列直接搬到目标 b32 regs，并完成 byte→b32 的重排。**对一个 warp 来说，子矩阵的 8 行被 8 个 lane 同时读，每 lane 提供 16 byte base、硬件取回一行并按 lane-id 派发**，所以总 smem 带宽只有 8 行 × 16 byte / sub × 4 (x4) = 512 byte/A load + 256 byte/B load = 768 byte，整个 warp 分摊到 32 lane = 24 byte/lane 等价吞吐，但**全部由硬件调度，没有 packing 开销**。

## 为什么 manual 路径绕不开它？

manual 路径是按 1.1 里的 mma fragment 排布直接展开的：`a_row_of/a_col_of` 给定每个 byte 在 smem 里的偏移，代码必须老老实实去 load 然后 pack。这个映射是 mma layout 的直接镜像，没有捷径——要喂 16 个 fp8 给 a[]，只能一次 1 byte 拿，再一位一位拼起来。

绕不开的核心原因是 **1 byte 级别的 mma fragment 描述天然就是逐元素的**：mma 期望的 fragment 把每个 lane 分到的 byte 散落在 smem 的不同位置（FP8 比 fp16/fp32 排得更稀疏），手工路径必须把这个散落关系一一实现。ldmatrix 是把这个散落关系**内置到硬件**的指令，省掉了 software 端的所有 packing。

## 变体选择

- A: `ldmatrix.x4.shared::cta.b16`（无 `.trans`）。`.x4` 是 mma.m16n8k32 的 A 端要求的最低变体（4 个 8x8 子矩阵），`.trans` 不需要——A fragment 期望 smem 的 4 子矩阵在 [16][32] 行主序里就是直读。
- B: `ldmatrix.x2.shared::cta.b16`（无 `.trans`）。`.x2` 覆盖 2 个 8x8 子矩阵（K 两半），B 的 8 列 N 用 sBn 的 n-major 排布恰好对齐——`sBn[n*32 + k]` 中固定 n 后 32 byte 是连续 K 字节，前 16 是 sub 0、后 16 是 sub 1，sub 内 stride = 32 byte 对应一行。