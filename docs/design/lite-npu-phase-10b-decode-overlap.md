# Lite NPU 阶段 10B：Decode Overlap

## 1. 目标与边界

本阶段在已准入的 Phase 10A 拓扑上只启用 Decode overlap：

- Prefill 保持 eager、overlap-off；
- Decode 保持 graph BS1/BS2，仅把 overlap 从 off 切到 on；
- 不修改 Lite 数学、checkpoint、PD wire、cache group、Grouped MoE、graph bucket 或采样语义；
- 不把 Phase 10A 已观察到的 graph 性能缺口归因于 overlap，也不以 overlap 掩盖该基线。

阶段 10B 只回答既有一深度 scheduler pipeline 在 Lite 8P8D Decode graph 上是否正确、是否带来性能
变化。Prefill overlap 不在范围内，并继续对齐当前3B/Pro启动策略保持关闭。

## 2. 已有 runtime 契约

TokenSpeed 已有唯一 overlap 实现，不需要 Lite 专用 scheduler：

1. `should_use_overlap_schedule`在 flag 未禁用且 role 为 Decode 时返回 true；Prefill/Encode role 即使
   未带 flag 也强制返回 false。
2. `EventLoop`把`in_flight_depth`和`overlap_schedule_depth`设为1：先 dispatch step N，再 commit
   step N-1，使 CPU post-process 与设备计算重叠。
3. 任一下一次 dispatch 依赖待 commit 状态时，通过统一 dependency registry drain；不在 Lite caller
   增加旁路。
4. scheduler/cache recipe 按 depth 1保留一步 Decode 写入。MLA full-history 组为每个 live request
   增加 protected page；KDA recurrent/conv 与 OE context 的 state 组已有 input/output 双页。
5. graph wrapper 使用 physical context length覆盖 overlap 的一步逻辑 overshoot，并把 depth 传给
   KDA/MLA metadata 初始化。

设备 forward 仍在同一 execution stream 上有序入队。Lite OE external staging 的下一次写入排在上一
graph 读取之后，不需要第二份 staging；graph output 与 result copy继续使用现有 wrapper 的 pending
result 契约。

## 3. Launcher 最小改动

当前`--disable-overlap-schedule`位于 P/D共享参数。实现只做 role-local 移动：

- 从共享参数删除该 flag；
- 在 Prefill 命令显式加入该 flag；
- Decode 命令不带该 flag，继续保留`--cudagraph-capture-sizes 1 2`和
  `--max-cudagraph-capture-size 2`。

launcher test 必须同时断言：

- P 含`--enforce-eager`和`--disable-overlap-schedule`，不含 graph bucket；
- D 不含`--enforce-eager`和`--disable-overlap-schedule`，只含 BS1/BS2 graph bucket；
- 其余8P8D、端口、checkpoint 和 bounded capacity 门禁不变。

不增加新的环境变量或 tri-state。需要回退时只把 disable flag 放回 Decode。

## 4. 测试计划

实现提交前后运行：

- launcher 命令与全部拓扑/容量负例；
- overlap role truth table；
- Lite hybrid cache depth0/depth1 page accounting；
- graph physical context 与 capture table overlap overshoot；
- executor graph external staging、KDA/MLA/OE cache-state累计 focused；
- `pre-commit run --all-files`。

本机无法加载 accelerator-only executor 测试时，保持 fail-closed，在 exact NPU source运行；不新增
CPU 假平台。

## 5. Exact NPU16 准入

验证使用单节点16张 Ascend 910B与 exact tracked source，不改系统环境：

1. P 日志必须为`disable_overlap_schedule=true`、scheduler depth 0；D 必须为 false、depth 1。
2. D 仍只 capture `[1, 2]`一次，ready 后不新增 capture或 eager fallback。
3. 先运行4-token、1100+ token continuation、sequential BS1、steady BS2、late-admission BS2和 slot
   reuse；请求必须完整且日志无 NaN/Inf、OOM、HCCL、transfer或 state/page错误。
4. 用与 Phase 10A完全相同的 EvalScope 配置运行 GSM8K first100，核对100/100、0 error/空输出、
   accuracy，并逐样本比较 overlap-off结果。
5. 记录 graph HBM、BS1/BS2周期吞吐与 EvalScope latency/throughput。性能没有硬门槛，但必须如实
   报告相对 Phase 10A是改善、持平或退化。
6. 拉回日志后按精确 PGID TERM，确认 listeners、markers、tunnel和16卡device holder归零。

任一 live slot污染、BS2 row串扰、graph fallback、非有限值或服务错误均为准入失败。准确率若有变化，
先按样本判断是否来自已知 Prefill低位波动；不能通过放宽门槛或改采样参数使结果假通过。

## 6. 提交顺序

1. 本设计文档独立 signed-off提交并推送；
2. launcher role-local flag与测试独立提交并推送；
3. 只有出现可复现的共享 runtime根因时，才另写设计并做最小修复；
4. exact NPU16结果以独立验证记录提交并推送。

本阶段不提前接入 Phase 11融合或最终分布式 cache/KVP8工作。
