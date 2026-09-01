# Lite NPU 阶段 8A：Decoder 数值执行验证记录

## 1. 结论

阶段 8A 的 decoder、model 与 logits 数值接线已通过独立验收：

- Lite 使用标准双 residual，KDA output 只沿 linear-attention TP 域归约一次，MLA 不重复归约；
- Prefill 只向 Grouped MoE 透传全局 token split，Decode 与单 rank 不传该 metadata；
- embedding、OE merge、所有 decoder layer、final RMSNorm、dense vocab head 和统一
  `LogitsProcessor` 已连成一条生产数据流；
- parameter 名称及 strict checkpoint loader coverage 未变化；
- 单 NPU 上的 residual 组合与 OE/embedding/norm/logits 组合均和独立 CPU BF16 oracle 对齐，所有输出
  finite；
- 阶段 1--8A 的 80 项 exact-source 累计回归全部通过。

本阶段仍不声明完整 checkpoint、P8/D8 role、CP8/KVP8 cache 或 HTTP 服务可用；这些边界分别属于后续
阶段 8B、9 和 11。

## 2. 提交与 exact source

| 内容 | 提交 |
| --- | --- |
| Phase 8A 设计 | `83f85b940a47710af61ee57df10ebcda9111af9f` |
| decoder/model/logits 接线 | `eab5c69c36999f52b34049da36c388b242b5c363` |
| NPU 数值 oracle 补强 | `dd5946e91b2f9f40387515d748386ab243553672` |

最终上板源码由已推送的 `dd5946e9` 生成全新 tracked archive：

```text
archive SHA-256: 131a0dbedce797f8923e8e565d5312291e2580c8c378207209f74146734b9fc3
```

最终 focused test 文件的本地与远端 SHA-256 均为：

```text
29e92fc5df426afcfd73042729340210ac5d9ecb6d143c4ce9a08dd34aafa25b
```

动态资源、凭据、主机地址和 checkpoint 路径均未进入提交。

## 3. 实现边界复核

实现保持已有模型层级和参数 ownership，未增加 checkpoint alias。顶层执行顺序为：

```text
input IDs
  -> dense vocab embedding
  -> prepared OE merge
  -> 28 decoder layers
  -> final RMSNorm
  -> dense vocab head
  -> shared LogitsProcessor
```

单层执行遵循 `residual + attention(norm(x))`、再
`residual + grouped_moe(norm(residual))`。Idle layer 返回原 tensor；KDA 与 MLA 的 collective ownership
分别由测试冻结，避免 decoder 对 MLA 做第二次 AllReduce。

设计中原计划直接复用 `VocabParallelEmbedding`。审计发现其模块级导入会在无 accelerator 的 CPU
config/loader 测试中提前触发 platform detection。最终只保留一个最小、CPU-safe 的 vocab shard
Parameter owner；accelerator 路径的 RMSNorm、AllReduce 和 logits 仍通过统一 kernel/runtime 边界复用，
没有从 model runtime 直接导入 vendor 包。

## 4. 本地验证

本地 Miniforge 环境运行 decoder execution、loader、Grouped MoE、KDA、MLA 和 OE focused 集合：

```text
47 passed, 23 skipped
```

23 项 skip 均是本机没有 Ascend accelerator 或分布式 NPU runtime 的环境门禁，不是功能跳过。CPU 测试
覆盖双 residual oracle、KDA/MLA 归约归属、P/D MoE metadata、OE merge、dense TP8
embedding/head ownership、统一 logits processor 和参数层级。

## 5. Exact-source NPU 数值验证

最终 exact source 在单张 Ascend NPU 上运行 Phase 8A focused 文件：

```text
10 passed, 1 warning, 0 skipped
```

其中两项是真实 NPU 数值用例：

| 用例 | 生产数据流 | 独立 oracle 与门槛 | 结果 |
| --- | --- | --- | --- |
| decoder residual | NPU RMSNorm、attention residual、第二次 RMSNorm、MoE residual | CPU FP32 RMS 公式并按 BF16 舍入；`atol=0.02, rtol=0.02` | shape 对齐、finite、pass |
| OE to logits | NPU embedding、OE merge、final RMSNorm、统一 logits processor | CPU BF16 embedding/OE/RMS/head 公式；`atol=0.02, rtol=0.02` | logits `[2,120]`、finite、pass |

这一验证比较的是组合层输出，不只检查字符串、shape 或最终 finite，因此可发现 residual 舍入点、OE
归一化、RMSNorm 和 lm-head 接线错误。warning 仅为目标镜像未编译 TorchAir；本阶段明确使用 eager，
不影响结论。

## 6. 累计回归

同一 exact source、同一 NPU 环境运行 loader、hybrid cache、KDA、MLA、Grouped MoE、OE 和 decoder
execution 的累计集合：

```text
80 passed, 2 warnings, 0 skipped
```

两个 warning 分别是 TorchAir 未编译提示和已知的 base-format tensor 提示。没有失败、NaN/Inf、异常
退出或源码回退。

设计中的通用 `--check-finite` 层级 hook 没有在 8A production 热路径提前实现。8A 已由两个组合 NPU
oracle 对所有本阶段输出执行 finite 检查；完整 checkpoint 的逐层 hook 必须和 P8/D8 rank-local probe
共用，避免现在新增一个随后被替换的调试入口，因此移入阶段 8B 的 exact probe，不降低最终验收门槛。

## 7. 清理与后续边界

最终验证退出后，16 张卡均报告无运行进程，exact-source 临时目录已不存在。源码仍可由已推送提交完整
恢复。

阶段 8B 将先提交独立设计，再验证目标 checkpoint 的 P8/D8 shard ownership、16 rank 同时加载、
role-local eager、逐层 finite hook 和 HBM 账本。CP8/KVP8 cache ownership、PD one-copy 和服务验收仍
不会提前并入阶段 8B。
