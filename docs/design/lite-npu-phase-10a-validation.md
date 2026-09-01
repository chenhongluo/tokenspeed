# Lite NPU 阶段 10A：Decode Graph BS1/BS2 验证记录

## 1. 验证范围

本记录闭环 Lite 8P8D bounded 服务的 Decode graph BS1/BS2：Prefill 保持 eager 且关闭 overlap，Decode
只捕获 BS1/BS2 并关闭 overlap。阶段 10A 不启用 Decode overlap、不改变模型数学、checkpoint、PD wire、
cache recipe、Grouped MoE 或采样语义。

提交链为：

| 类型 | 提交 |
| --- | --- |
| Decode graph 总体设计 | `7cf00e6c` |
| launcher policy 与测试 | `55918428` |
| external graph token width 设计/修复 | `0ed03eb1` / `3f82960e` |
| graph state publication 诊断设计 | `ebf8a057` |
| 跨请求 prefix cache 策略/实现 | `810102fa` / `1fc4f569` |
| MLA continuation 设计/实现 | `0044df5c` / `a54cfe36` |
| MLA continuation 验证 | `d10fe911` |

graph state 诊断最终排除了 KDA/MLA graph replay 后 state 未发布的假设。服务输出差异在首次本地
Decode 前已经存在，首个可见边界位于 Prefill layer 0 attention collective 之后；因此没有提交盲同步、
第二份 state cache 或 Lite 专用 graph patch。

## 2. Exact source 与累计测试

服务使用 exact tracked source：

```text
commit:  a54cfe366c8ce45409de141944e0d44373d47d1a
archive: 1b838d1e965f5fb085264c9d94f0f9d6da4bdd0117a16d1ece65d4cdad9515dd
```

验证节点为单节点16张 Ascend 910B、CANN 9.0。目标 checkpoint 与依赖从节点已有只读目录加载，
系统环境未修改。

| 门禁 | 结果 |
| --- | --- |
| 本地 launcher + MLA cumulative | `18 passed, 7 skipped` |
| exact-source NPU executor cache state | `5 passed` |
| exact-source NPU MLA page64 focused | `12 passed` |
| 验证文档前 `pre-commit run --all-files` | 全部通过 |

本机没有支持的 accelerator platform，单独收集 executor 测试会在 package import 时 fail closed；该文件
因此在 exact NPU source 上运行，没有为测试增加 CPU 假平台。

## 3. Role policy 与 capture

最终日志解析结果如下：

| 项目 | Prefill | Decode |
| --- | --- | --- |
| `enforce_eager` | `true` | `false` |
| overlap | disabled | disabled |
| capture sizes | 无 | `[1, 2]` |
| capture 启动次数 | 0 | 1 |
| capture rank | 不适用 | 8/8 |

Decode 只报告一次`Capturing batches: [1, 2]`，进度日志约3.8秒完成；服务 ready 后未再次 capture。
没有未声明的更大 bucket，也没有 eager fallback。

capture 日志给出的 available memory 从46.12 GB降至44.92 GB，对应本阶段两个 graph 合计约1.20 GB
HBM 增量。该值来自 graph wrapper 同一 rank、同一 capture 过程的观测，不等同于整卡进程总显存。

## 4. 服务与 state 准入

exact graph 服务完成 P/D/gateway readiness。1102-token 长请求在 Prefill 形成`1024 + 78`两个 batch，
随后 Decode 返回 HTTP 200并生成4个 tokens；`cached_tokens=0`、`prefix_replay_tokens=0`。

诊断阶段已有以下组件证据：

- KDA causal-conv/recurrent state 的 padding-to-live 和连续换页 graph 对 eager逐元素 exact；
- MLA latent write、动态 page/length、下一步 paged read 对 eager逐元素 exact；
- 12次请求的8个 Decode rank中，9,408组 replay 后 state publication 指纹逐项 exact；
- OE resolved token、3-token context、12路 n-gram ID 与更新后 context 在分歧前一致；
- 日志未命中 Traceback、RuntimeError、NotImplementedError、NaN、Inf 或 non-finite。

这些证据排除了 graph state 丢失、固定坏页、OE context 污染和缺少全设备 fence。已知 Prefill
collective 低位波动不被误报为 Decode graph state bug。

## 5. GSM8K first100 消融

eager 与 graph 使用相同 EvalScope 1.9.1 配置：GSM8K `main/test`前100条、4-shot、seed 42、API
batch 2、greedy、max tokens 512、真实换行 stop、`ignore_eos=true`和`no_stop_trim=true`。

| 指标 | Decode eager | Decode graph |
| --- | ---: | ---: |
| 完成请求 | 100/100 | 100/100 |
| API error / 空输出 | 0 / 0 | 0 / 0 |
| GSM8K accuracy | 14/100 | 16/100 |
| 平均延迟 | 26.939 s | 41.6887 s |
| 平均吞吐 | 7.17 tok/s | 4.98 tok/s |

逐样本比较有59条完整输出 exact、71条抽取答案 exact、98条判分一致；剩余2条仅 graph 正确，没有
仅 eager 正确的样本。本次100条消融未观察到 graph 精度退化，但不声明 bitwise 等价。

第一次 eager run 因 stop 被多转义成字面量反斜杠而全部跑满512 tokens。该轮只保留为负证据，
未进入上述比较。

## 6. 性能观测

Decode 周期日志按实际 running requests 分组后的生成吞吐如下：

| 模式 | BS1 中位数 | BS2 中位数 |
| --- | ---: | ---: |
| eager | 12.41 tok/s | 16.13 tok/s |
| graph | 9.07 tok/s | 10.56 tok/s |

周期日志来自两轮真实服务流量，不是同输入、同长度的受控微基准；只能说明它与 EvalScope 的 graph
变慢方向一致。本阶段没有性能准入目标，但该缺口必须保留给后续独立性能定位，不能作为 graph 已优化的
证据。

## 7. Artifact 与清理

停服后的 eager/graph 日志已经拉回并与远端逐文件 exact，聚合 SHA256 分别为：

```text
eager: 06b615c53468357bc060ae2b1fd928944dac9df1b65e6a2ef1a9c0cd1a36946d
graph: 019105bdee831b43969f04ad570310425ac23b133c87e9efc469be1847198ae2
```

服务按精确 PGID发送 TERM，未使用 KILL 或宽泛名称匹配。最终 process group、固定 listeners、
source/run markers、本地 tunnel 和16张卡的 device fd holder 均为0。

## 8. 结论与边界

阶段 10A 功能与精度准入通过：P eager、D graph BS1/BS2、两侧 overlap-off 的 exact 8P8D 服务完成
长请求和 GSM8K first100，未观察到 graph 引入的精度退化或非有限值。

graph 相对 eager 的性能退化仍然存在；阶段 10B 只能单独启用 Decode overlap并做同口径 A/B，不能把
overlap 当作掩盖 graph 基线缺口的修复。跨请求 prefix cache、分布式 page ownership和最终容量仍不在
本阶段准入范围内。
