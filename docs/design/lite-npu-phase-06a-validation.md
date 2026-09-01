# Lite NPU 阶段 6A：Grouped MoE Torch 与 Packed Loader 验证记录

## 1. 验证范围

本记录验证阶段 6A 的 Lite Grouped MoE 基线：canonical packed expert 参数、严格 EP loader、四组 Torch
路由、real/identity expert、shared expert 和 N=1 reference forward。Ascend fused TopK/local expert、
EP8 collective、Decode graph、A/B/C placement 和完整服务不在本子阶段范围内。

提交链为：

| 类型 | 提交 |
| --- | --- |
| Grouped MoE 总体设计 | `0fcf6ec63120e6bfbf3a674da4fc2c39ba6bdba3` |
| 6A 设计 | `bca2144dacea395cde7bf1db016107e1a376fe2f` |
| 6A 实现与测试 | `f5fa76c82556bb3a2f9085886e1c9d8947848d73` |

## 2. Exact-source 与环境

已推送的实现提交通过 `git archive` 生成只含 tracked source 的验证包：

```text
commit:  f5fa76c82556bb3a2f9085886e1c9d8947848d73
archive: 311a8e5fa2e6fa63aa6eb633c9fb2a0a79efd0484bc8102b1f55ad38f52adf38
```

目标机收到 archive 后先复算相同 SHA-256，再解包到全新目录。验证只暴露一张 Ascend 910B NPU，源码
经 `PYTHONPATH` 加载并使用既有隔离测试环境；基础镜像、系统 Python 和 package 均未修改。

目标环境版本为 `torch 2.9.0+cpu`、`torch_npu 2.9.0.post2+git912882d`，进程可见 NPU 数为 1。

## 3. Packed 参数与 Loader

实现把每层 1536 个逐 expert 参数对象替换为两个 canonical target：

```text
experts.w13_weight = [num_local_experts, 2 * I, D]
experts.w2_weight  = [num_local_experts, D, I]
```

真实配置 `EP8,D=768,I=512` 下每 rank 为 `[192,1024,768]` 与 `[192,768,512]`。参数由通用
`MoELayerSpec/create_dense_weight_pair` 创建；6A 不实例化会选择设备执行计划的完整 `MoELayer`。

严格 checkpoint loader 复用 `build_moe_checkpoint_loader`：

- gate 写入 W13 前半，up 写入 W13 后半，down 写入 W2；
- flattened real expert 按 contiguous EP owner 映射到 local `0..191`；
- global expert name plan 识别非本 rank source，但不写本地参数；
- source duplicate 仍被外层 strict loader拒绝；
- gate/up 合法共享 W13 target，不再被误判成重复 target；
- 每层 W13/W2 都进入最终 loaded-target coverage。

小配置 sentinel 对 rank1 的 global expert 4 验证为：W13 前半全 1、后半全 2、W2 全 3。通用 EP planner
同时覆盖 rank0、rank3、rank7 的 first/last global ID 与 local ID，证明 ownership 无重无漏。

## 4. CPU/meta 导入边界

审计确认 loader/unquant owner 原本被四个无关 eager import 绑到加速器平台：MoE executor export、全部
quant creator、FP8 quant helper，以及只在 fused-local loader 分支使用的 model-loader helper。

实现仅把这些 import 下沉到真实使用分支，没有新增 fallback 或改变 accelerator kernel 选择。fresh CPU
子进程依次导入：

```text
tokenspeed.runtime.layers.moe.loader
tokenspeed.runtime.layers.moe.weights.unquant
```

随后断言 `tokenspeed_kernel` 不在 `sys.modules`，结果通过。目标 NPU 环境再从 package lazy export 导入
`MoELayer`，并导入 loader、`create_layer_weights` 与 Lite owner，四项均成功，证明延迟导入没有破坏真实
accelerator API。

N=1/N=8 meta 构造分别得到 32/4 个 local expert，W13/W2 shape 与 dtype正确；strict source/target
coverage 在 meta target 上通过且不 materialize payload。

## 5. 路由与数值语义

### 5.1 路由

四组 router 分别执行 FP32 softmax。测试把 classifier 置零、给 correction bias 不同单调值，验证：

- ID 与 `topk(softmax(logits)+bias)` 逐元素一致；
- weight 始终取无 bias softmax 的 gather；
- route scale 为 5 的测试中每条 weight 为 `5/12`；
- TopK2 weight 和为 `10/12`，证明没有 TopK 内 renorm；
- 输出 ID 为 INT32，weight/logits 为 FP32。

### 5.2 Real、identity 与 shared

单个 N=1 用例同时覆盖：

- 同 token 两次选择同一个 real expert；
- 两条 route 都是 identity expert；
- 一条 real 加一条 identity 的 mixed route；
- 四个 group 的 group-major flattened expert ID；
- shared expert 消费原始 hidden 并只累加一次。

实现输出对独立逐 token/route loop oracle 满足 `atol=2e-2,rtol=2e-2`，dtype 为 BF16且全部 finite。
T0 直接返回 `[0,H]`；N=8 forward 在 collective 尚未接入时明确抛出指向 6C 的 `NotImplementedError`。

## 6. 测试汇总

| 门禁 | 结果 |
| --- | --- |
| 本地 Grouped MoE/loader 首轮 | `20 passed, 2 skipped` |
| 本地累计 Lite KDA/MLA/cache/public-kernel focused | `34 passed, 27 skipped`（NPU-only skip） |
| exact-source NPU0 累计 focused | `61 passed` |
| 全仓 `pre-commit run --all-files` | 全部通过 |

NPU0 累计用例除 6A CPU/meta 数学外，还回归了既有 Lite KDA、MLA、hybrid cache 与 public-kernel
registration。61 项无 skip、无失败。

目标 wheel 未编译 TorchAir和 internal-format 关闭的两条 warning 为既有环境提示；相关回归均真实完成，
没有改变数值结论。

## 7. 结论与后续

阶段 6A 准入通过。Lite expert 参数已统一为 canonical W13/W2，checkpoint source/EP ownership/packed
顺序严格覆盖；N=1 四组 Torch 数学成为后续 NPU leaf 和 EP8 的统一 oracle；N=8 继续 fail closed。

下一阶段 6B 只新增 Ascend softmax+bias TopK facade 和 precomputed-TopK local expert leaf，并在 NPU0
验证 active-row mask、ND/NZ、BS1/BS2 graph。Prefill/Decode EP8 collective 与 A/B/C winner 仍留在
阶段 6C。
