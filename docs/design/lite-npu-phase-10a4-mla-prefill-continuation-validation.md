# Lite NPU 阶段 10A.4：MLA Prefill Continuation 验证记录

## 1. 验证对象

本记录验证 Lite 在关闭跨请求 prefix cache 后，仍可通过已有 absorbed MLA Extend 数据流处理同一请求
超过 1024 tokens 的后续 Prefill chunk，并分别完成 Decode eager 与 Decode graph BS1/BS2 的服务和
GSM8K first100 验收。

提交链为：

| 类型 | 提交 |
| --- | --- |
| 跨请求 prefix cache 策略 | `810102facb2455eebe6f4f2641b4cc93a4c78145` |
| launcher fail-closed | `1fc4f569ead08a13f5f8c026c35e70965297dd65` |
| page64 continuation 设计 | `0044df5c0b9b4a634b0a453abdbba60e1fa0691d` |
| page64 capability 与测试 | `a54cfe366c8ce45409de141944e0d44373d47d1a` |

实现只把 Ascend `mla_extend_with_kvcache` 注册的 page-size trait 从 `{128}`修正为`{64, 128}`，未增加
Lite replay 循环、fallback、额外 cache 或更大的 Prefill chunk，也未修改 KDA、OE、Grouped MoE、
PD wire、Decode graph 或采样逻辑。

## 2. 根因与 exact source

关闭 prefix cache 只能禁止跨请求复用。`chunked_prefill_size=1024` 时，同一请求的第二个 chunk
仍带有已计算 prefix。Lite 已有 absorbed Extend 数据流，但 Ascend registry 只声明 page128，导致
bounded 服务使用 page64 时 capability 查询返回 false，并在普通 explicit Prefill 路径 fail loud。

Ascend leaf 本身按 live cache 的真实 page size调用官方 attention 算子，不存在 page128 假设。因此
最小根修复是扩大共享 capability 声明，而不是增加模型特例。

实现提交生成 tracked-only archive，并在验证节点复算相同摘要：

```text
commit:  a54cfe366c8ce45409de141944e0d44373d47d1a
archive: 1b838d1e965f5fb085264c9d94f0f9d6da4bdd0117a16d1ece65d4cdad9515dd
```

验证使用单节点 16 张 Ascend 910B、CANN 9.0、P8+D8 bounded topology。P 固定 eager 且关闭
overlap；D 分别使用 eager 和 graph BS1/BS2，均关闭 overlap。目标 checkpoint 与依赖从节点已有只读
目录加载，系统环境未修改。

## 3. 测试门禁

| 门禁 | 结果 |
| --- | --- |
| 本地 Lite MLA focused | `5 passed, 7 skipped` |
| exact-source NPU focused | `12 passed` |
| page64 NPU leaf finite | 通过 |
| page64 leaf 对 FP32 oracle | output rel-L2 `0.00186938`，LSE max-abs `1.43e-6` |

测试覆盖 page64 capability 查询、跨页 latent cache 读取、Lite cached Extend 选择，以及当前 latent 在
attention 前写入 live cache 的顺序。page128 能力保持不变。

## 4. 长请求服务准入

eager 与 graph 两个 exact-source 服务分别发送由 tokenizer 构造的长请求。服务实际计入 1102 个
prompt tokens，Prefill 日志均显示两个 batch：`1024 + 78`。两种模式都返回 HTTP 200并生成4个
completion tokens；API usage 为`cached_tokens=0`，服务日志为`prefix_replay_tokens=0`。

这证明 page64 absorbed Extend 覆盖的是同一请求的 chunk continuation。它不代表、也没有重新启用
跨请求 prefix reuse。

## 5. GSM8K first100 消融

两轮正式评测使用完全相同的 EvalScope 1.9.1 配置：

- GSM8K `main/test`前100条，4-shot，seed 42；
- API batch 2，greedy，temperature 0，max tokens 512；
- stop sequence 为真实换行的`"}\n\n"`；
- `ignore_eos=true`、`no_stop_trim=true`。

结果如下：

| 指标 | Decode eager | Decode graph BS1/BS2 |
| --- | ---: | ---: |
| 完成请求 | 100/100 | 100/100 |
| API error / 空输出 | 0 / 0 | 0 / 0 |
| GSM8K mean accuracy | 14/100 | 16/100 |
| 平均输入 tokens | 652.53 | 652.53 |
| 平均输出 tokens | 193.16 | 207.79 |
| max-token 输出 | 18 | 21 |
| 平均延迟 | 26.939 s | 41.6887 s |
| 平均吞吐 | 7.17 tok/s | 4.98 tok/s |

同一100条按样本索引比较，59条完整输出逐字符相同，71条抽取答案相同，98条最终判分相同；剩余
2条仅 graph 正确，没有仅 eager 正确的样本。因此本次100条消融未观察到 Decode graph 导致的精度
退化，但不把两条路径声明为 bitwise 一致。

本阶段没有性能目标。graph 的延迟和吞吐明显差于 eager，只作为后续性能分析输入，不影响本轮功能与
精度准入，也不能被准确率结果掩盖。

## 6. 排除的无效评测

第一次 eager first100 的 CLI 多转义一层，把 stop 写成字面量`"}\\n\\n"`。该轮虽然100/100且
无 API error，但全部输出跑满512 tokens，准确率2/100。该结果目录只保留为负证据，不进入上表、
不与 graph 比较，也未被删除或复用。

## 7. 日志、artifact 与清理

最终 eager/graph 日志均已拉回本地。逐文件 SHA256 清单的聚合摘要分别为：

```text
eager: 06b615c53468357bc060ae2b1fd928944dac9df1b65e6a2ef1a9c0cd1a36946d
graph: 019105bdee831b43969f04ad570310425ac23b133c87e9efc469be1847198ae2
```

两个摘要均与停服后的远端目录逐文件 exact。日志未命中 Traceback、RuntimeError、
NotImplementedError、NaN、Inf 或 non-finite 模式。

两个服务均按各自精确 PGID 发送 TERM，未按名称宽泛清理，也未使用 KILL。最终 graph PGID进程、
固定 listener、source/run marker 和16张卡的 device fd holder 均为0，本地 SSH tunnel 已停止。

## 8. 结论与边界

阶段 10A.4 准入通过。Ascend page64 capability 修正恢复了 Lite 请求内 MLA Prefill continuation；
eager 与 graph 均完成长请求和同口径 GSM8K first100，未观察到 graph 引入的精度退化。

跨请求 prefix cache 仍明确不支持并保持关闭。已知 Prefill collective 低位波动和 eager/graph 输出非
bitwise 一致不在本修复中伪装消除；后续 Decode overlap、graph 性能优化和分布式 cache 容量分别按
独立阶段验证。
