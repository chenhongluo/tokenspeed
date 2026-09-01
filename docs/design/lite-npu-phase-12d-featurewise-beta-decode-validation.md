# Lite NPU 阶段 12D：Featurewise-beta Decode 融合验证记录

## 1. 结论

阶段 12D 通过准入。Lite KDA 的 Ascend featurewise-beta Decode 路径保留新融合
kernel，Prefill 继续使用阶段 12A 路径。

- BS1/BS2 NPUGraph replay 分别提升 `83.14%/86.51%`，每层节省
  `116.504/158.712 us`，同时越过 `>=5%` 和 `>=5 us` 的冻结门禁。
- BF16 output 逐元素 exact，FP32 state max abs `5.96e-8`，低于 `1e-6` 门禁。
- exact 8P8D 通过短请求、slot reuse、BS2 late admission、1100-token continuation 和
  GSM8K first100；未观察到 fatal、NaN/Inf、HTTP 5xx 或 graph fallback。

当前 Megatron 修复和 golden 尚未冻结。本记录使用已验证的 TokenSpeed 服务结果作
临时模拟基线，只用于发现新增回归；它不是 Megatron golden，也不代表已完成
Megatron 精度对齐。

## 2. Exact source 与 artifact

```text
implementation: 53ebfb53b1765c0393c08c9d694a72bfceaffc1c
source archive: d701b7458d7559f95b342df755eb09a6452016d0b95b2ee55d20e5164b62bc3d
run manifest:   8d3906d429921306aa3fc0ac1b8a7d1c27f9fdd56b5c61f67f917ef994e79a84
```

远端部署来自上述 source archive，并使用 package-local 的已验证公开 KDA artifact。
原始服务日志、评测输出、环境和连接信息留在仓外；仓库只提交本公开安全的验证
记录。

## 3. 单算子精度与性能

exact-source NPU focused 测试为 `25 passed`，覆盖 BS1/BS2 eager/graph、changed-input
replay、padding、page 0、未选邻页、finite 和连续 128 步轨迹。标准 Triton 验证器
Decode `2/2` 通过，标准 profiler 几何平均为 `5.5901x`。

同一进程内动态加载阶段 12A baseline，与 candidate 交替运行的 NPUGraph 结果为：

| Batch | Baseline | Candidate | 节省 | 提升 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 140.128 us | 23.624 us | 116.504 us | 83.14% |
| 2 | 183.456 us | 24.744 us | 158.712 us | 86.51% |

两个 batch 的 BF16 output 都是逐元素 exact，FP32 state max abs 为 `5.96e-8`。
新路径没有增加 state transpose、host sync、第二套 executor 或 graph fallback。

## 4. Exact 8P8D 服务

最终累积配置为：

- Prefill：8 rank，eager，overlap 关闭；
- Decode：8 rank，NPUGraph BS1/BS2，overlap 开启，Weight-NZ 开启；
- P/D 均保持 KDA TP8、MoE EP8 和已准入的 cumulative Lite 实现。

最终服务前有两次环境级失败，均未修改 12D kernel：

1. 首次启动替换了 CANN 注入的 Python 路径，导致 Weight-NZ 首次编译找不到
   `tbe`。最小修正是保留 CANN 路径并在其前追加 exact source 路径。
2. 第二次启动的 metrics 端口与 Decode rendezvous 派生的 tokenizer IPC 端口冲突。
   最终验收使用物理分离的端口区间；通用 derived-port preflight 不夹带到本 kernel
   阶段。

最终实例中 P/D 各8个 rank 和 Gateway 全部 ready；Decode 日志确认 BS2、BS1 依次
capture 完成，activations + device graphs 约占 `0.34 GiB`。

### 4.1 功能准入

复用现有的 Lite service admission client，7 个 case 全部通过：model list、4-token
fresh/slot reuse、standalone B、32-token live A、真实 late-admission B 和 post-BS2 B。

以目标 tokenizer 构造的精确 1100-token prompt 返回 HTTP 200 并完成 4-token
Decode。固定 64-token 请求重复6次的中位数为：

| 延迟 | TTFT | TPOT | 输出速率 |
| ---: | ---: | ---: | ---: |
| 11.9220 s | 0.5485 s | 180.861 ms | 5.368 tok/s |

6次请求产生3个 digest，与前续已记录的 BF16/collective 低位不确定性一致；
本阶段不将其误判为 state 污染。

### 4.2 GSM8K first100

评测口径为 EvalScope 1.9.1、GSM8K main/test first100、4-shot、seed 42、batch 2、
greedy、`max_tokens=512`、真实 `"}\n\n"` stop、`ignore_eos=true`、
`no_stop_trim=true`；使用全新目录，不读取 prediction cache。

| 完成 | Accuracy | API error | 空输出 | 平均输入 | 平均输出 | 平均延迟 | 输出吞吐 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100/100 | 18/100 | 0 | 0 | 652.53 tokens | 188.33 tokens | 35.77596 s | 5.26 tok/s |

与 Phase 12A 当前服务结果的临时模拟对比为：

| 指标 | 结果 |
| --- | ---: |
| Phase 12A / Phase 12D accuracy | 16/100 / 18/100 |
| Prompt exact | 100/100 |
| Full output exact | 47/100 |
| Extracted answer exact | 66/100 |
| Judgment exact | 96/100 |
| Phase 12A only correct | 1 |
| Phase 12D only correct | 3 |

差异是双向的，且同一服务已观察到低位不确定性。结论限定为“未观察到可归因于
12D 的 first100 回归”，不宣称质量提升或 Megatron 对齐。

## 5. 内存、日志与清理

评测后 P 侧每设备 HBM 为 `19294--19739 MiB`，D 侧为 `19120--19546 MiB`。
这与前一累积服务处在同一级别，没有观察到第二份 recurrent state 或新增常驻副本。

严格扫描结果为：

- Prefill/Decode/Gateway fatal、NaN/Inf、non-finite：0；
- 实际 HTTP 5xx：0；
- graph fallback：0；
- EvalScope API error/空输出：0/0。

原始 artifact 拉回并生成 SHA-256 manifest 后，按顺序停止本地隧道和远端 launcher。
最终 launcher PID/PGID、source/run marker、服务进程、目标及派生监听端口、
16 卡 TokenSpeed device holder 全部为0，相关端口全部可绑定。

## 6. 决策与回退

- 保留 `53ebfb53` 的 Decode-only kernel 和目标 shape dispatch；
- Prefill 保持阶段 12A implementation，不恢复 12C 组合候选；
- 非目标 shape、scalar-beta、CPU 和非 NPU 路径继续使用原有 fallback；
- 如后续在更大 Decode graph batch 或 Megatron golden 上发现稳定回归，回退到
  `a4adc838` 恢复的阶段 12A executable。
