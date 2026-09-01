# Lite NPU 阶段 8C.1：Decode 官方 Dispatch/Combine 准入记录

## 1. 结论

华为公开的非 V2 `npu_moe_distribute_dispatch/combine` **不准入** Lite Decode
EP8/local48 production 路径。目标环境的两个官方运行模式都没有通过第一个 eager
Dispatch：

- 默认通信模式在 host tiling 阶段以 `561002` 明确拒绝每 rank 48 个 local expert；
- 按官方文档启用 A2 PCIe/RoCE 推荐模式后，host 限制被放开，但第一次 Dispatch 的
  device synchronize 在八个 rank 上一致触发 `507057`，AIVector 报告 MTE DDR 地址越界。

因此 Combine、数值 oracle、BS2 和 graph 均没有进入，不可由接口存在或 host 检查通过外推
为可用。本阶段不实现 production adapter，不尝试 V2，也不扩大 Decode rank。继续保留已验证的
四个 local Grouped MoE leaf 加 EP8 AllReduce 路径。

## 2. 验证对象

| 项目 | 值 |
| --- | --- |
| exact source commit | `5d93468b7f80e79780a6608e6a7a208c9d80f901` |
| source archive SHA-256 | `83dab2d97a463ab72963ae797a2f4a5a48705e68dbefe751faad0dffcc910210` |
| harness SHA-256 | `79dfdcfc5f093594a77544a310c0b081a260db7cfcfaae9f60e9fcf5597215f9` |
| CANN | `9.0.0` |
| PyTorch | `2.9.0` |
| Torch-NPU | `2.9.0.post2+git912882d` |
| hardware | 单机 8 张 Ascend 910B2C，仅使用 device 0--7 |
| topology | EP8，每组 `E=384, localE=48, H=768, K=12` |
| API | 仅公开非 V2 Dispatch/Combine；V2 未调用 |

Harness 将默认 process group 固定为 Gloo 控制面，只创建一个专供官方 MoE pair 使用的 HCCL
communicator。这个修正消除了早期测试中同一组 rank 创建两个 HCCL communicator 导致的端口
碰撞；最终两组准入日志均来自该单-HCCL拓扑。

Lite 的四组分别调用官方 pair。测试计划覆盖 BS1/BS2 的 real-only、mixed identity、
identity-only 和 hot-expert，以及 BS1/BS2 graph。identity route 被替换为不重复的真实 expert
ID 和零权重，最后由 Torch oracle 只累加一次 identity contribution。准入采用 fail-closed：任一
rank 的第一个 eager case 失败即停止，后续 case 不再运行。

## 3. 上板结果

| 模式 | 第一个 eager Dispatch | Combine | Graph | 决策 |
| --- | --- | --- | --- | --- |
| 默认 A2 通信环境 | 八个 rank 均失败，ACLNN `561002` | 未到达 | 未到达 | 拒绝 |
| `HCCL_INTRA_PCIE_ENABLE=1`、`HCCL_INTRA_ROCE_ENABLE=0` | 八个 rank 均失败，NPU `507057` | 未到达 | 未到达 | 拒绝 |

### 3.1 默认模式：local48 被 tiling 拒绝

八个 rank 在第一组、BS1、real-only 的 Dispatch host tiling 阶段返回相同首错：

```text
call aclnnMoeDistributeDispatch failed, error code is 561002
moeExpertNum is 48, in case of unlayered, it must no more than 24
MoeDistributeDispatchA2CheckAttrAndSetTiling
```

这里错误文本中的 `48` 是 `384 / EP8` 得到的 local expert 数，而 API 传入的全局
`moe_expert_num` 仍是 `384`。结果与公开 A2 文档的普通模式 `local_expert_num <= 24`
约束一致。

### 3.2 官方推荐模式：放行 tiling 后设备侧 MTE 越界

启用官方建议的 PCIe/RoCE 环境后，`local48` host 检查不再报错，Dispatch 成功 enqueue；但第一组
第一次 `torch.npu.synchronize()` 在八个 rank 上一致失败：

```text
error code is 507057, SUSPECT REMOTE ERROR
The DDR address of the MTE instruction is out of range
```

最终日志没有 `Communication_Error_Bind_IP_Port`，HCCL 初始化和唯一算子 communicator 均已完成，
所以该结果不能归因于此前已修复的双 communicator 端口冲突。公开 A2 文档没有把 EP8 列入非 V2
支持的 EP world size；本次上板进一步证明，取消 local-expert host guard 不等于目标 EP8/local48
shape 可在设备侧安全执行。

## 4. 门禁与后续动作

| 门禁 | 结果 |
| --- | --- |
| 非 V2 binding/schema 存在 | 通过 |
| EP8/local48 BS1 eager Dispatch | 失败 |
| BS1 eager Combine 与 Torch oracle | 未到达 |
| BS2 eager | 未到达 |
| BS1/BS2 graph capture/replay | 未到达 |
| production adapter | 不实施 |

后续仅在以下任一条件出现后重开准入：

1. CANN/Torch-NPU 的公开支持矩阵明确新增 A2 EP8/local48；
2. 华为提供针对该 shape 的修复版本，并能用本 harness 在同样四组边界通过 eager、oracle 和 graph。

V2 不作为 fallback。阶段 8C.2/8C.3 的 kernel adapter 和 8P8D service admission 均终止，当前
local leaf + AllReduce 保持不变。

## 5. 证据与清理

仓库外 evidence 同时保存两组完整 launcher log、八个 rank 的 error JSON、运行前后
`npu-smi` 和逐文件 SHA-256。`artifact-files.sha256` 的 SHA-256 为
`0f8c9ba75679c2d0bebf55403ecefbdf0dcb7f865d95e0ea1036eaa968f0f025`，拉回后全部校验通过。

两组 launcher 的退出码均为 1，符合负准入预期。最终进程 marker 为 0，device 0--15 均为
`Health=OK` 且无本任务残留进程；device 8--15 全程没有用于测试。
