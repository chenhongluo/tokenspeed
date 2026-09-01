# Lite NPU 阶段 6D：Grouped MoE Production 验证记录

## 1. 结论

阶段 6D 已通过：winner A（四个独立 EP8 local leaf）完成 P/D production 接线，synthetic 与用户提供的
最终 HF checkpoint 均在 8 张 910B 上通过 P32/P128/P1024、D BS1/BS2 eager 和 Decode graph。

- P：CP/SP8 + EP8 + dense TP1，三项二维 token AG 后执行四个 leaf，再做一次 RS；
- D：KVP8 + EP8 + dense TP8，feature AG、expert AR、dense AR 三个代数边界均已覆盖；
- 每 rank 只有一份 `[192,...]` packed expert parent storage，四个 `[48,...]` leaf view 的 storage
  pointer 均与 parent 一致；
- 所有输出 finite；真实 checkpoint 的 leaf、collective、wrapper、完整输出和 graph rel-L2 均通过
  分层门禁。

本阶段没有接入完整 decoder、OE、PD 服务或 GSM8K；这些仍属于后续阶段。

## 2. 提交与 exact source

| 内容 | 提交 |
| --- | --- |
| 6D production 设计 | `637fa49711d851909b53568a69efacef9bc1540d` |
| production 实现与测试 | `8abafb89a6f1bdd3e2264b14a2a10c652aa36744` |
| HCCL 二维 token collective 修复 | `042a9a9af9680a3023f04e2499400f780fab3b0d` |
| 分层精度门禁设计 | `76e56f67`、`c7034d94` |
| 分层精度测试 | `ae4f8a3f`、`f34eefe9` |

最终验证从已推送的 `f34eefe93e2e9bc348662c693c791136d1d0bc0d` 生成全新 archive，没有复用
诊断目录：

```text
archive SHA-256: c07f3b7aac2d199d69324f87853b16133ba93a60102efccdad5cc078e7147b08
lite.py:         9c1dd7c039a8a5cf26cfeac089543618041b5bb98c845ec85b589800c099cb47
board test:      58208543641f465fa6c3f31adae29b2f56c0cf3a44d2a3d0213c62883d37f337
gate design:     88dcef27180b487f7e687cff3d2fbd05fb964cf414880ee5f46b94d05918367d
```

本地 HEAD、tracking 和远端 `lite` 均为同一提交。目标机上的三项文件 SHA 与本地一致；连接信息、
checkpoint 路径和凭据均未写入提交。

## 3. 本地回归

最终测试文件通过 AST 编译；Lite config/loader/cache/KDA/MLA/Grouped MoE/public-kernel 选择集为：

```text
37 passed, 28 skipped
```

skip 均为需显式 NPU/torchrun 的硬件用例。每次提交前均执行仓库完整 `pre-commit run --all-files`，
全部 hook 通过。

## 4. Synthetic exact-source NPU8

下表为 rank0 记录；八个 rank 各自完成 `1 passed, 4 deselected`，且所有 rank 都执行相同硬门禁。

| case | leaf rel-L2 | routed rel-L2 | wrapper rel-L2 | final rel-L2 | graph rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P32 | 0.003481 | 0.000356 | 0.000409 | 0.000622 | N/A |
| P128 | 0.003482 | 0.000246 | 0.000633 | 0.000722 | N/A |
| P1024 | 0.003509 | 0.000270 | 0.000702 | 0.000765 | N/A |
| D1 | 0.003803 | 0.000159 | 0.004151 | 0.003833 | 0.002513 |
| D2 | 0.003096 | 0.000320 | 0.003191 | 0.003158 | 0.002992 |

synthetic 的 leaf max-abs 不超过 `1.5259e-5`，collective 后不超过 `0.001953125`；graph replay 在
更新输入后输出发生变化，且 BS1/BS2 均低于 `0.005` rel-L2 门槛。

## 5. Checkpoint exact-source NPU8

| case | leaf rel / max | routed rel / max | wrapper rel / max | final rel / max | graph rel / max |
| --- | --- | --- | --- | --- | --- |
| P32 | 0.003896 / 0.001953 | 0.001416 / 0.003906 | 0.000299 / 0.03125 | 0.000525 / 0.0625 | N/A |
| P128 | 0.003559 / 0.007812 | 0.003055 / 0.015625 | 0.000591 / 0.0625 | 0.000715 / 0.0625 | N/A |
| P1024 | 0.003676 / 0.007812 | 0.003191 / 0.03125 | 0.000755 / 0.5 | 0.001095 / 1.0 | N/A |
| D1 | 0.003642 / 0.000244 | 0.001378 / 0.003906 | 0.002691 / 0.125 | 0.003439 / 0.125 | 0.003052 / 0.125 |
| D2 | 0.004155 / 0.001953 | 0.003184 / 0.007812 | 0.002105 / 0.0625 | 0.003229 / 0.0625 | 0.003067 / 0.0625 |

每个 rank 均为 `1 passed, 4 deselected`。最坏 leaf rel-L2/max-abs 为 `0.004155/0.0078125`，最坏
collective 后为 `0.003191/0.03125`，都通过根因附近的硬门禁。完整输出 max-abs 只记录，不作硬门禁：
BF16 GMM 的少量舍入在 Wout 后会放大，但 final rel-L2 最大仅 `0.003439`，低于 `0.01`。

## 6. 验证期间发现并修复的问题

1. 初版 P 路径把 `[T,G,H]` 和 `[G,T,K]` 三维 tensor 直接传给 HCCL token AG。该 backend 的 padding
   seam 只接受二维 `[tokens,features]`，导致 NCL/ND format 冲突和 max-token rank 容量错误。
   `042a9a9a` 在通信前分别折成 `[T,G*H]`、`[T,G*K]`，AG 后恢复逻辑 shape；synthetic 全矩阵通过。
2. 原 checkpoint gate 在最终 Wout 后使用统一逐元素容差，无法区分通信错误与 BF16 稀疏抵消误差。
   最终测试把 FP32 oracle 比较前移到 leaf 和 collective 边界，并另行验证 wrapper、final 与 graph；
   阶段 6C 的 A/B/C 逐元素门槛不变。

internal format 关闭时出现的 ND warning 与阶段 6B 的明确选择一致，不影响数值、storage sharing 或
graph 验收。

## 7. 清理

两轮 exact-source 测试完成后：

```text
exact Python processes: 0
NPU0--7 running processes: 0
```

测试未启动完整服务，也未占用 NPU8--15。阶段 6D 至此闭环，下一阶段进入 OE 正确性和 host residency。
