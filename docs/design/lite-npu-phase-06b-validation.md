# Lite NPU 阶段 6B：Ascend Grouped MoE Leaf 验证记录

## 1. 验证范围

本记录验证 Lite Grouped MoE 的 Ascend biased-softmax TopK、canonical 权重 post-load 转换、单 rank
local-expert leaf、fixed-capacity 空路由和 BS1/BS2 NPUGraph。EP8 collective、P/D 并行、A/B/C placement
和完整模型服务属于阶段 6C/6D，不在本阶段提前接线。

提交链为：

| 类型 | 提交 |
| --- | --- |
| Grouped MoE 总体设计 | `0fcf6ec63120e6bfbf3a674da4fc2c39ba6bdba3` |
| 6B 设计 | `93ed0e3bd1ea69c6b4893c105fab13e7ba482cc1` |
| 6B 实现与测试 | `1bff97921c9da0aca81a6b9cafaeb234b79cbf96` |

## 2. Exact-source 与环境

验证从已推送的实现提交生成全新 `git archive`，传输前后 SHA-256 一致：

```text
commit:  1bff97921c9da0aca81a6b9cafaeb234b79cbf96
archive: 78006f3ab90fea6b47099379ea0902ca51903b11ab2193d2e9b958c5e8664b5e
```

目标环境只暴露 NPU0，加载 CANN 9.0.0，并通过源码 `PYTHONPATH` 使用隔离测试环境。累计 KDA 回归
需要的生成物不在 Git archive 中，因此按阶段 4D 的既有流程补入阶段 4E 已验证的 package-local
artifact；loader 在注册前重新校验 manifest、binding、op-api 和 tiling library digest。该操作没有覆盖
exact archive 中的源码、测试或文档。

## 3. TopK 语义

统一入口执行：

```text
probability = softmax(router_logits, fp32)
ids = topk(probability + correction_bias, 12)
weights = gather(probability, ids) * 6
```

因此 correction bias 只影响 expert ID，不进入 route weight，也不做 TopK 内 renorm。T1/T2/T32 的
pytest 和独立 T1024 probe 均与 Torch ID exact；T1024 weight 最大绝对误差为 `5.96046448e-08`。
空 token 返回 `[0,K]` 的 FP32 weight 和 INT32 ID，shape/dtype/trust-boundary 用例全部通过。

## 4. Local-expert leaf

实现复用目标 wheel 的公开高融合链：

```text
init_routing_v2 -> grouped_matmul -> SwiGLU -> grouped_matmul -> finalize_routing
```

canonical `W13=[E,2I,H]`、`W2=[E,H,I]` 只在 post-load 转为 GMM 所需视图，重复调用保持幂等。
fixed-capacity routing 通过 device-side active-row mask 在每个 stage 后清零无效尾行，没有 `.item()`、
D2H 或 token Python 循环。exact CANN 9 的 token-major combine 固定使用
`row_idx_type=0`、`expanded=[R,1,H]` 和 `drop_pad_mode=3`。

NPU0 覆盖 48 和 192 local experts、all/partial/zero local route、duplicate ID、首末 local ID、非 local
real ID、identity/invalid sentinel。输出全部 finite，并满足独立 FP32 route-accumulation oracle 的
`atol=2e-2`、`rtol=2e-2` 和 rel-L2 `<=5e-3`；zero-local-route 输出严格为零。

BS1/BS2 在固定地址预热后完成 NPUGraph capture。两次 replay 之间同时更新 hidden、weight 和 route ID，
输出随输入改变，并继续满足同一 oracle；没有创建新 specialization。

## 5. 格式与性能记录

当前环境关闭 internal format。对 post-load 权重查询得到 `ND`；此前 `npu_format_cast(..., 29)` 前后也
都保持 base format。因此本阶段生产 leaf 使用 ND，不把无效 cast 伪装成 NZ A/B。

对 `T=4,H=768,I=512,localE=48,K=12` 的完整 local leaf 预热 20 次后，以 50 次 NPU event 记录得到
warm median `333.220 us`。本阶段没有性能准入目标；该值作为阶段 6C 同机相邻 A/B/C placement 的基线，
不外推完整服务 TPOT。

## 6. 测试汇总

| 门禁 | 结果 |
| --- | --- |
| exact-source Lite runtime 累计 focused | `57 passed` |
| exact-source kernel/NPU 累计 focused | `51 passed` |
| T1024、duplicate/boundary 与 warm-median probe | 全部通过 |
| 全仓 `pre-commit run --all-files` | 全部通过 |

两个 warning 分别来自目标 wheel 未编译 TorchAir和 internal-format 关闭；相关 NPU 算子均真实执行，
不改变数值、graph 或格式结论。

## 7. 已拒绝路径

- 合法 2D `drop_pad_mode=0` 对同一 token-major metadata 的 rel-L2 约为 `1.285`，不能替换 mode 3；
- internal-format 关闭时无法形成 NZ，因此不增加无效 format-cast；
- identity/invalid route 不进入 GMM，而是规范化成 non-local documented input，最终 identity 累加仍由
  后续 Grouped MoE wrapper 完成；
- MC2 的公开 EP size/expert-count 约束不覆盖 EP8、192 experts/rank，本阶段不增加第二套 executor。

## 8. 结论

阶段 6B 准入通过。通用 biased TopK 与 Ascend local leaf 已在 exact source 上完成数值、空路由、
192-local-expert 和 Decode graph 验证；runtime、loader 和 collective 未被旁路。下一步进入阶段 6C，
使用八张卡对 A=`4xEP8`、B=flattened-interleaved、C=flattened-group-major 做同输入、同权重的相邻实验，
只保留唯一胜出布局。
