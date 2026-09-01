# Lite NPU 阶段 6C：EP8 Grouped MoE Placement Board 验证记录

## 1. 结论

阶段 6C 的唯一 production winner 是 **A：四个独立 EP8 local leaf**。

B/C 虽把四次 local leaf 合并为一次，并在多数 shape 上显著降低延迟，但都未通过冻结的准入门槛：

- B 的真实 checkpoint P1024 与 A 不满足逐元素 `atol=2e-2,rtol=2e-2`；
- B 的 peak allocated HBM 比 A 增加 `35.73%`，超过 `5%` 上限；
- C 继承相同数值/HBM问题，并在 P128、D1、D2 相对 B 回退超过 `3%`；
- C 的真实 P1024 rank load CV 为 `0.099433`，高于 B/A 的 `0.062982`。

因此阶段 6D 只接入 A：P 在一次 token AG/RS 之间执行四个 leaf，D 在一次 AllReduce 前执行四个
leaf；B/C 只保留在离线 board，不进入 production runtime。

## 2. 提交与 exact source

| 类型 | 提交 |
| --- | --- |
| 6C 设计 | `e66e91a1f898688d2e28cc9a9c20165b7781b2cb` |
| 6C harness | `ea91793762128731366f6118e6ed5a2048c56c1c` |
| A oracle 门槛修正 | `a3d6ac62838d3344b3e42cd9d090913901fa2dd6` |

最终验证从已推送的 `a3d6ac62` 生成全新 archive；没有复用 staged source：

```text
commit:  a3d6ac62838d3344b3e42cd9d090913901fa2dd6
archive: 23d7663aab5943035cd389699ad8ee3fa7e4ca59019160aa384297474ab2da13
board:   645ff055b94672f76b7ce60a145a34dbcee1515484d9b89ee418d0b98658a947
```

本地、tracking 与远端分支 SHA 一致。目标机只暴露 NPU0--7，加载 CANN 9.0.0，并从源码
`PYTHONPATH` 使用隔离测试环境。真实单层 checkpoint 目录仅由环境变量注入，未写入源码、文档或日志。

## 3. 验证口径

- 三种布局使用相同 production-scale grouped hidden：unit-weight RMSNorm 后乘 `2`；
- 四组 router 只计算一次 logical route，B/C 只做 physical expert ID remap；
- P 覆盖不等长 split 的 T32/T128/T1024，一次 AG 与一次 RS；
- D 覆盖 BS1/BS2，一次 AllReduce，并 capture 完整 leaf + HCCL + identity 链；
- 每个 case 预热 20 次、计时 50 次，记录八 rank 最大同步 wall latency；
- identity contribution 在 collective 后只加一次；空 local route 的 rank 不跳过 collective；
- A 额外与独立 FP32 route-accumulation oracle 对齐。

本地 placement 用例为 `3 passed, 1 hardware-only skipped`。NPU8 的 synthetic 和 checkpoint 两轮中，
每个 rank 均完成三项 placement 用例与一个 distributed board 用例；P/D、graph、empty route 和 finite
hard assertion 全部通过。

## 4. Synthetic 20/50

A 对独立 FP32 oracle 的 rel-L2 为 `0.00238204`，低于 `0.005` 门槛。B/C 对 A 的最大 rel-L2 为
`0.000133967`，所有逐元素数值 gate 通过。

| case | A / us | B / us | C / us |
| --- | ---: | ---: | ---: |
| P32 | 1863.394 | 895.911 | 1104.948 |
| P128 | 1890.112 | 1006.207 | 1203.205 |
| P1024 | 2985.087 | 2789.921 | 2750.557 |
| D1 | 1582.478 | 571.578 | 560.485 |
| D2 | 2765.878 | 622.286 | 719.488 |

Decode graph replay 最大 rel-L2 为 `9.6067e-5`；两次 replay 之间 hidden、route weight 和 route ID 均
原地更新，输出随输入变化。C 的 synthetic D1/D2 有两个 zero-route rank，仍完成同序 HCCL 与 graph。

| layout | peak allocated / MiB | peak reserved / MiB |
| --- | ---: | ---: |
| A | 541.322 | 616.000 |
| B | 736.262 | 950.000 |
| C | 736.262 | 950.000 |

Synthetic 本身证明 flattened 数学和 graph 可执行，但其 HBM 已不满足 B 相对 A 增长不超过 `5%` 的
门槛，不能单凭延迟把 B/C 接入生产。

## 5. Checkpoint 20/50

| case | A / us | B / us | C / us | B 数值 gate | C 数值 gate |
| --- | ---: | ---: | ---: | --- | --- |
| P32 | 2766.625 | 1023.742 | 1010.247 | pass | pass |
| P128 | 2074.790 | 963.845 | 1085.681 | pass | pass |
| P1024 | 2882.761 | 2695.012 | 2685.924 | **fail** | **fail** |
| D1 | 1373.194 | 624.379 | 716.272 | pass | pass |
| D2 | 1876.169 | 666.554 | 768.982 | pass | pass |

P1024 的详细数值为：

| layout | max abs vs A | rel-L2 vs A | finite |
| --- | ---: | ---: | --- |
| B | 0.250000 | 0.00220568 | yes |
| C | 0.250001 | 0.00268344 | yes |

整体 rel-L2 仍小于 `0.01`，但部分接近零的元素不满足冻结的逐元素 `atol/rtol`，所以这是明确的候选
拒绝结果，不能放宽测试阈值。所有同布局 Decode graph self-replay 继续通过，最大 rel-L2 为
`0.00276618`。

| layout | peak allocated / MiB | peak reserved / MiB |
| --- | ---: | ---: |
| A | 546.195 | 610.000 |
| B | 741.370 | 950.000 |
| C | 741.370 | 950.000 |

B/C 相对 A 增加 `195.175 MiB` allocated（`+35.73%`）和 `340 MiB` reserved（`+55.74%`）。根因是
单个 192-local-expert leaf 的 fixed-capacity routing/GMM workspace；静态 expert 权重总量相同。

## 6. Placement 与负载

A/B 的 owner 都是每 rank 每组 48 个 expert，所以相同 logical route 下 rank/pair load exact 相同。
C 把每组固定到相邻 rank pair。真实 P1024 统计为：

| layout | rank CV | pair real routes |
| --- | ---: | --- |
| A/B | 0.062982 | `[9935,8399,9151,8882]` |
| C | 0.099433 | `[9741,7830,9357,9439]` |

因此 group-major pair placement 只保证每组有两个 owner rank，不保证 checkpoint router 下的跨 pair 或
pair 内 route 均衡。C 相对 B 的 P128、D1、D2 延迟分别回退 `12.64%`、`14.72%`、`15.37%`，也超过
`3%` 门槛。

## 7. Winner gate

| 门槛 | A | B | C |
| --- | --- | --- | --- |
| placement 1536 expert 双射 | pass | pass | pass |
| A FP32 oracle rel-L2 <= 0.005 | pass | N/A | N/A |
| checkpoint 逐元素数值 | baseline | **fail** | **fail** |
| P/D graph 与 empty route | pass | pass | pass |
| local leaf 次数 | 4 | 1 | 1 |
| HBM 相对前一候选 | baseline | **fail** | pass vs B |
| 稳态延迟相对前一候选 | baseline | pass vs A | **fail vs B** |
| route/pair 负载 | baseline | same as A | worse on checkpoint |

按设计的单调选择规则，B 任一硬门槛失败即选择 A；C 不再有越级成为 production winner 的资格。

## 8. 清理与下一步

两轮 exact-source board 结束后：

```text
board_processes_after=0
npu0_7_processes_after=0
```

NPU8 测试没有启动完整模型服务，也没有占用第二组八张卡。阶段 6C 至此闭环；阶段 6D 只实现 A 的
production 接线，并复用现有 `Mapping.moe.ep_group`、token AG/RS、AllReduce 与阶段 6B leaf，不增加
layout 配置或第二套 executor。
