# Lite NPU 阶段 9C.6：OE Decode Token Publication

## 1. 目标与实测根因

阶段 9C.5 的 exact 8P8D 已完成 P/D readiness，并让首个真实请求越过旧的 AscendDirect channel
prepare 错误。新的共同首错发生在八个 Decode rank 的 Lite OE 外部输入准备阶段：scheduler 用 `-1`
表示本轮 Decode token 由设备侧 `future_input_map` 持有，而 host-resident OE 仍只读取原始
`ForwardOp.decode_input_ids`，因此按设计 fail closed。

这不是 token 缺失。`InputBuffers.fill_input_buffers` 已先完成以下统一解析：

1. 将 scheduler 的显式 Decode override 写入 `future_input_map`；
2. 对 device-owned 行读取 `future_input_map`；
3. 按 Prefill/Mixed/Decode 的真实 token 顺序写入 active `input_ids_buf`；
4. 在模型 embedding 前完成 vocabulary range clamp。

OE 在该步骤之后却重新读取控制面的原始字段，绕过了统一解析结果。本阶段把 resolved model input 作为
external-input hook 的唯一 token 数据源，不修改 OE 数学、cache wire、scheduler ownership 或
AscendDirect。

## 2. Publication 契约

`ModelExecutor` 在 `fill_input_buffers` 完成后、graph/model forward 开始前，将
`input_ids_buf[:total_tokens]`传给现有 `prepare_external_inputs` hook。Lite 必须遵守：

- active tensor 的元素顺序由 `forward_op.input_lengths`定义，元素总数必须等于各行长度之和；
- host OE lookup 只从这份 resolved tensor 做一次显式 D2H，不再拼接`forward_op.input_ids`和
  `decode_input_ids`；
- D2H 是 host table 的 publication boundary，必须在 n-gram ID 计算和 OE context state 写入前完成；
- Prefill、Mixed 和 Decode 共享同一条路径；Decode 仍只准入每请求一个 token，speculative width 保持
  fail closed；
- request slot、page ownership、ragged length、graph staging 和 finite probe 的既有门禁全部保留；
- resolved tensor shape 或长度错误必须在任何 OE context/page mutation 前失败。

当前 correctness 阶段允许这次有界 D2H 同步，且没有性能目标。后续 graph/overlap 优化若证明它成为
瓶颈，再利用已经存在的 sampler D2H 结果建立异步 CPU shadow；本阶段不预建第二份
`future_input_map`，避免双写和跨 stream ownership。

## 3. 最小实现边界

实现只触及三个既有边界：

- `ModelExecutor`：调用 external-input hook 时透传 active resolved tensor；
- Lite model/OE runtime：接收该 tensor、校验数量、转换为 CPU `int64`并执行现有 prepare；
- Lite OE focused test：把原“device-only token 必须失败”用例改为“原始 ID 为`-1`但 resolved token
  可成功”，并增加数量不一致的 mutation-before-publication 负例。

不新增 runtime token store、model-specific executor 分支、环境开关或第三方依赖。vendor-neutral
runtime 只传递模型已经要消费的 tensor；是否 D2H 由 Lite host OE 实现自行决定。

## 4. 验证

实现提交前：

1. Lite OE focused 覆盖 Prefill、device-owned Decode、slot reuse、graph staging、错误长度和非法
   slot/page；
2. Lite decoder/cache/KDA/MLA 累计 focused tests 通过；
3. 精确执行`pre-commit run --all-files`。

实现提交推送后生成新的 exact archive，并在 16-card 节点验证：

1. P/D/gateway ready，首个 4-token 请求不再出现 device-only OE token 错误；
2. 请求至少完成一次 Prefill、P→D transfer 和 Decode；
3. 日志无 HCCL、Mooncake、runtime、OOM、NaN/Inf 首错；
4. 若出现更后的新首错，保留独立负结果并继续按真实边界拆分；
5. 结束后按精确 PGID 清理，确认所有 listener 和 NPU context 释放。

完整 fresh/slot-reuse/late-admission BS2、finite/cache/HBM 仍属于阶段 9C 的累计验证，不由本设计的
单个4-token门禁替代。

## 5. 回退

若 resolved tensor 与模型实际 embedding token 不一致，停止服务准入并同时记录两侧匿名 digest，审计
`fill_input_buffers`的顺序和 stream fence；不回退到读取`ForwardOp.decode_input_ids`、不把`-1`
替换为固定 token，也不把 OE 主表复制到设备规避 host publication。
