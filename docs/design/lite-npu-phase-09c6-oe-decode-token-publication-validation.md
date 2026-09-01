# Lite NPU 阶段 9C.6：OE Decode Token Publication 验证记录

## 1. 验证对象

本记录验证 Lite host-resident OE 在 Decode token 由设备侧持有时，能够消费 executor 已解析的实际模型
输入，而不再读取带 `-1`占位符的原始控制面字段。

提交链为：

| 类型 | 提交 |
| --- | --- |
| 设计 | `dc19a793338bc0e2df88260d84f767571b1e1a9c` |
| 实现与测试 | `659e0b0157cae8b464a64fcefbe80f66da952b9f` |

实现提交只修改 executor 的 external-input hook、Lite OE runtime 和对应 focused test。scheduler 的
device-token ownership、Mooncake cache wire、KDA/MLA、Grouped MoE、采样和服务拓扑均未修改。

## 2. Exact source 与测试门禁

实现提交生成 tracked-only archive，并在验证节点复算相同摘要：

```text
commit:  659e0b0157cae8b464a64fcefbe80f66da952b9f
archive: 7ba6471b509d30783ff26c31314193aee417946df300b5f227c29e9b5c74c4b9
```

验证使用单节点 16 张 Ascend 910B、CANN 9.0、P8+D8 bounded eager 拓扑。P/D 均为 TP8、EP8，P/D
通过 AscendDirect 传输 cache；目标 checkpoint 与依赖均从节点已有只读目录加载，系统环境未修改。

| 门禁 | 结果 |
| --- | --- |
| Lite OE 本地 focused | `8 passed, 1 skipped` |
| Lite 累计本地 focused | `69 passed, 37 skipped, 2 deselected` |
| exact-source NPU OE + executor cache-state | `13 passed` |
| 实现提交前 `pre-commit run --all-files` | 全部通过 |

本地 deselect 的两项分别依赖真实 accelerator platform 和既有 probe fixture；executor 路径已在 exact
NPU 集合补测，不用 CPU 假平台掩盖。

## 3. 普通服务准入

普通模式先完成 P/D/gateway readiness。首个 4-token 请求返回 HTTP 200、4 个 token，证明 resolved
token 已越过 Prefill、P→D cache transfer、Decode 和 host OE lookup；八个 Decode rank 均未再出现
device-only token 错误。

随后完整 7-case admission 通过：

- 三次 A fresh/slot-reuse 输出摘要完全一致；
- B 的 standalone、late-admission 和 BS2 后 replay 输出摘要完全一致；
- 32-token A stream 产生首 token 后才提交 B，B 在 A 完成前结束，证明实际并发而非串行回放；
- 全部请求均为 HTTP 200、token 数完整、finish reason 非空；
- 请求后日志未命中 HCCL、Mooncake、runtime、OOM、NaN 或 Inf fatal 模式。

匿名 admission summary SHA-256 为
`ebe936933ffe7fb87c00d7c4fcb999a26c0f687783268f2eaf16bcbe6b9d973b`。

## 4. 全链路 finite probe

同一 exact source 另起服务，只增加 `TOKENSPEED_LITE_FINITE_PROBE=1`。probe 是同步诊断，不用于性能
计时。4-token smoke 与一次真实 late-admission BS2 均通过；后者包含 32-token A 和 4-token B，B 的
提交时刻严格位于 A 的首 token 与完成时刻之间。

最终解析结果如下：

| 项目 | Prefill | Decode |
| --- | ---: | ---: |
| rank | 8/8 | 8/8 |
| forward/rank | 18 | 79 |
| tensor 记录 | 66,384 | 291,352 |
| 实际最大 batch | 1 | 2 |
| begin/finish | 全部配平 | 全部配平 |
| `finite=false` | 0 | 0 |
| nonfinite 元素 | 0 | 0 |

覆盖 28 层，其中 KDA 为 21 层、MLA 为 7 层；覆盖 embedding/OE、layer input、KDA projection/core/
conv state/recurrent state、MLA query/latent/cache/attention、attention collective 后输出、Grouped MoE、
final norm 和 logits，共 30 个 stage。所有 rank 的 layer/stage 集合一致，未缺少 required stage。

finite summary SHA-256 为
`30f51ced3039e712f707530a43d62e7c1df470df1b5c389137b846ca84589f15`；BS2 summary SHA-256 为
`860d71ecb5fdd1012d1a562d34fda4bc6955506678991829dca96de4391d4a0d`。

## 5. 同步 probe 下的 bitwise 负结果

直接在同步 probe 模式复用普通 admission 的“相同请求输出摘要必须完全相同”门禁时，前三次 A 请求的
摘要不一致。另行重复 6 次相同 prompt，得到 3 种摘要，出现次数为 4/1/1；请求均成功且全链路 finite。

逐 stage 对比显示：layer 0 的 OE、embedding、KDA projection/core、conv/recurrent state 和本地
`kda.output`聚合统计跨请求一致；首个可见差异位于 linear-attention TP8 all-reduce 后的
`attention.output`。同一次 forward 内，八个 rank 的 reduce 结果彼此一致。当前证据把差异缩到
collective boundary，但没有逐元素证明它来自 HCCL 的 BF16 归约顺序，还是同步 probe 改变时序后放大了
未记录的低位输入差异，因此不把该现象归因于 OE 或 KDA state。

该负结果不被删除，也不表述成“probe 下完整 exact admission 通过”。它与两个正交门禁分开解释：普通
模式负责输出重复性和 slot/BS2 admission，probe 模式负责全链路 finite 与 BS2 stage 覆盖。负结果摘要
SHA-256 为`3a99638ff012ae22b800433c577791c14fefcb6349956a9b88f9e43beec1224c`；若后续需要
bitwise NPU determinism，应单独设计 collective 输入逐元素 digest 和 FP32/HCCL A/B，不能修改 OE
publication 绕过。

## 6. Cache、HBM 与清理

P/D 每个 rank 均记录一只 97,026,048-byte（92.53 MiB）cache arena，46 个 LCM block；scheduler
报告 47 个 device page、prefix granularity 128、最大 batch 2。该数值只属于 bounded admission 配置，
不外推为最终容量规划。

普通请求后 16 张卡的整卡 HBM 为 18,844--19,434 MB；启用同步 probe 并完成累计请求后为
18,882--19,474 MB。整卡值包含 rank 0 的额外进程上下文，不能等同于单 scheduler 的 Torch allocation。

两个服务均按各自记录的精确 PGID 发送 TERM，并在超时前退出，未使用 KILL。最终 PGID 进程、目标
listener、16 张卡的 device fd owner 和服务 process marker 均为 0。cleanup summary SHA-256 为
`7c8154b536d39340d1269309eed051b4f5cd203ec5eb62180d0fc3e55cfd48fb`。

## 7. 结论

阶段 9C.6 准入通过。resolved model input 已成为 host OE 的唯一 Decode token publication boundary；
真实 8P8D 请求、slot reuse、late-admission BS2、AscendDirect transfer 和全链路 finite 检查均闭环，且
没有引入第二份 CPU future-token map 或 vendor-specific executor 分支。

本结论只覆盖 bounded eager 服务。Decode graph、overlap、最终容量、GSM8K first100 与剩余融合优化
继续按后续独立阶段验证，不能由本记录替代。
