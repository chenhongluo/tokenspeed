# Lite NPU 阶段 10A.1：外部输入 Graph Token 宽度

## 1. 现象与根因

阶段 10A 的 Decode BS1/BS2 graph 均能完成 capture，P/D 服务也能进入 ready；首个真实请求却在
模型 forward 前的 Lite OE staging 门禁失败。八个 Decode rank 的共同首错都是 graph token 数超过
固定 staging 范围，Prefill 侧没有对应错误。

这不是 OE、PD handoff 或 graph capture shape 的模型特例。非投机服务仍保留命令行的默认
`speculative_num_draft_tokens`，但 `CudaGraphWrapper` 在 `spec_algo is None` 时正确地将每请求捕获宽度
固定为 1。`ModelExecutor` 初始化 OE staging 时也使用相同的有效宽度 1；只有 replay 前调用
`prepare_external_inputs` 时错误地再次读取原始 draft 配置。因此 BS1 被发布为 4 token staging，超过
按 BS1/BS2 capture 初始化的最大 2 token 范围。

有效 graph token 宽度的唯一权威是 `CudaGraphWrapper.max_tokens_per_req`：

```text
graph_tokens = padded_bs * forward_step.max_tokens_per_req
```

原始 speculative 配置不是权威，因为它在未启用 speculative algorithm 时仍有默认值。

## 2. 修复边界

实现只修改共享 `ModelExecutor` 向模型外部输入 preparer 发布的 graph token 数：

- 保留现有 `can_run` 与 `padded_bs` 选择；
- 将原始 `config.spec_num_tokens` 替换为 graph runner 已解析的 `max_tokens_per_req`；
- 不扩大 Lite staging，不放宽 OE 越界门禁；
- 不增加 eager fallback，不修改 graph eligibility；
- 不修改 PD、KDA、MLA、Grouped MoE、采样或 checkpoint 逻辑。

这同时保持投机路径语义：启用 speculative algorithm 时，runner 的有效宽度仍等于 capture 使用的
verify width；未启用时固定为 1。

## 3. 回归测试

新增 CPU 行为测试直接执行 `ModelExecutor.execute_forward_op` 到 `prepare_external_inputs` 边界，使用：

- 原始 `config.spec_num_tokens=4`；
- graph runner `max_tokens_per_req=1`；
- Decode BS1、padded BS1。

测试必须观察到外部 preparer 收到 `graph_tokens=1`。它不能通过读取源码或命令字符串判断。现有 OE
测试继续覆盖 BS1/BS2 固定 staging 的地址复用、padding 清零和越界 fail-closed。

累计门禁包括相关 executor/OE/graph focused tests 和完整 `pre-commit run --all-files`。

## 4. NPU16 验收

使用 exact tracked source 重跑阶段 10A 的 8P8D 服务：

1. P 保持 eager/overlap-off，D 只 capture BS1/BS2 且 overlap-off；
2. 首个 sequential BS1 请求越过 OE staging，并记录 `graph=true,padded_bs=1`；
3. steady/late BS2 记录 `graph=true,padded_bs=2`；
4. 完成 slot reuse、同 prompt repeat 和 post-BS2 replay；
5. 日志无 OE staging 越界、NaN/Inf、OOM、HCCL 或 PD transfer 错误；
6. 精确停止服务，确认 listener、source marker 和 NPU context 清零。

如果修复后出现新的共同首错，将以新的根因另建阶段；不得扩大本提交范围或关闭 graph 使请求假通过。

## 5. 提交边界

1. 本设计文档单独 signed-off 提交并推送；
2. 一处共享 executor 修复和一项行为测试作为独立 signed-off 提交并推送；
3. exact NPU16 结果写入独立验证记录并提交推送。
