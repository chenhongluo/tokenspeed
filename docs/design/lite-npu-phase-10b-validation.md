# Lite NPU 阶段 10B：Decode Overlap 验证记录

## 1. 验证范围

本记录在 Phase 10A 已准入拓扑上只增加 Decode overlap：Prefill 继续 eager、overlap-off，Decode
继续 graph BS1/BS2并把 scheduler pipeline depth从0改为1。模型数学、checkpoint、PD wire、cache
group、graph bucket和采样参数均未改变。

提交链为：

| 类型 | 提交 |
| --- | --- |
| Decode overlap设计 | `a0fe1132` |
| launcher role-local flag与测试 | `d9daf4c6` |

实现只把`--disable-overlap-schedule`从共享参数移动到 Prefill命令；没有新增 Lite scheduler、cache副本、
同步或 fallback。

## 2. Exact source 与累计测试

服务使用 exact tracked source：

```text
commit:  d9daf4c6034e147b83bc05c6155e70c4b9eb4875
archive: f423013e03181a6191b2c4f70b19ec492cd2cbe935cb5512ab9338b8b80d5f55
```

验证节点为单节点16张 Ascend 910B、CANN 9.0。目标 checkpoint和依赖从节点已有只读目录加载，
系统环境未修改。

| 门禁 | 结果 |
| --- | --- |
| 本地 launcher focused | `13 passed` |
| exact-source NPU overlap/cache/physical-context | `17 passed` |
| 实现提交前 `pre-commit run --all-files` | 全部通过 |

本地缺少`compressed_tensors`和 accelerator platform的组合 collection在测试断言前 fail closed；对应
runtime测试在 exact NPU source运行，未安装依赖或增加 CPU假平台。

## 3. Role policy、capture 与 cache

最终日志解析结果如下：

| 项目 | Prefill | Decode |
| --- | --- | --- |
| `enforce_eager` | `true` | `false` |
| `disable_overlap_schedule` | `true` | `false` |
| scheduler depth | 0 | 1 |
| capture sizes | 无 | `[1, 2]` |
| capture启动次数 | 0 | 1 |
| scheduler device pages | 47 | 48 |

Decode depth1按既有 recipe多保护一个 page。Decode只报告一次`Capturing batches: [1, 2]`，8/8 rank
完成；服务 ready 后未再次 capture或回退 eager。capture的 available memory仍从46.12 GB降至
44.92 GB，与 Phase 10A相同，两个 graph约占1.20 GB HBM。

## 4. 服务、长请求与 BS2

16/16 worker strict load、P/D gRPC、Gateway model list均通过。1100-token请求在 Prefill形成
`1024 + 76`两个 chunk，随后 HTTP 200并完成4-token Decode；日志未出现非有限值或 state/page错误。

稳定 B prompt的6次独立4-token请求 digest exact。late-admission实验中，A首 token在0.5083秒返回，
B在0.5084秒提交并于2.2150秒完成，A于6.1821秒完成；late B和 BS2结束后的 B都与6次 standalone
digest exact。该稳定哨兵未观察到 row串扰或 slot残留。

短 A重复8次存在3种输出。两个同时提交的 B32也产生2种 digest，但首 token时间不同，实际是
staggered admission；随后3次串行 B32复现了完全相同的2种 digest。因此长输出差异不是 BS2特有，
与 Phase 10A已定位到 Prefill collective之后的低位不确定性一致。

## 5. GSM8K first100 消融

EvalScope 1.9.1配置与 Phase 10A完全相同：GSM8K `main/test`前100条、4-shot、seed 42、API batch 2、
greedy、max tokens 512、真实换行 stop、`ignore_eos=true`和`no_stop_trim=true`。

| 指标 | overlap off | overlap on |
| --- | ---: | ---: |
| 完成请求 | 100/100 | 100/100 |
| API error / 空输出 | 0 / 0 | 0 / 0 |
| GSM8K accuracy | 16/100 | 13/100 |
| 平均输入 tokens | 652.53 | 652.53 |
| 平均输出 tokens | 207.79 | 192.85 |
| 平均延迟 | 41.6887 s | 39.2276 s |
| 平均吞吐 | 4.98 tok/s | 4.92 tok/s |
| 总耗时 | 2150.05 s | 2025.69 s |

off/on有51条完整输出 exact、69条抽取答案 exact、97条判分一致；仅 off正确的3条为 index
9、49、59。on和 Phase 10A eager则有53条完整输出 exact、67条抽取答案 exact、99条判分一致，唯一
判分差异是 index49。

在同一 overlap-on服务上把 index49原请求重放5次，抽取结果依次为`30`、无答案（跑满512）、`0`、
`0`、`0`；目标答案是`30`。同一实现既能走正确路径也能走错误/长循环路径，证明单次13比16的总分
差不能作为 overlap确定性精度回归。该结果也不声明 bitwise等价：既有 strict admission client的 A
三次 digest门禁仍然失败。

## 6. 性能观测

只统计正式 EvalScope时间窗内的 Decode周期日志：

| 模式 | BS1中位数 | BS2中位数 |
| --- | ---: | ---: |
| overlap off | 9.07 tok/s | 10.56 tok/s |
| overlap on | 9.10 tok/s（61点） | 10.52 tok/s（207点） |

两组周期吞吐基本持平。overlap-on总耗时缩短约5.8%，但平均输出长度同时下降约7.2%，平均 token吞吐
还略降约1.2%；因此本轮没有证据证明 overlap带来稳定性能收益，也没有观察到明显性能退化。

## 7. Artifact 与清理

EvalScope聚合 SHA256为：

```text
29692704c008d189b354c2e759d754465c2bce7045867d67f75de19ccd1fbc92
```

停服后的服务日志和准入证据已经拉回并通过15/15文件逐项校验，manifest SHA256为：

```text
edf9786ccb9ad530c967c21099250268b9b0592a921db7426ab5898fa740f242
```

最终日志的 Traceback、RuntimeError、NotImplementedError、NaN、Inf和 non-finite命中数均为0。
服务按精确 PGID发送 TERM；最终 process group、固定 listeners、source/run markers、本地 tunnel和
16张卡的 device fd holder均为0。

## 8. 结论与边界

阶段 10B功能与隔离准入通过：P depth0、D graph BS1/BS2 + depth1的 exact 8P8D服务完成长请求、
真实 late-BS2和 GSM8K first100，未观察到可归因于 overlap的 state污染、row串扰、graph fallback、
非有限值或服务错误。

bitwise确定性门禁仍未通过，但它在 Phase 10A overlap-off已存在，并能在同一 overlap-on实例内自行
切换正确/错误路径；本阶段不以修改采样或放宽请求完整性掩盖该问题。Decode overlap也尚未显示稳定
性能收益。Phase 11只能在保留以上口径的前提下继续逐项准入已有融合优化。
