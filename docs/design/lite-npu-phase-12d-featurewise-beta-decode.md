# Lite NPU 阶段 12D：Featurewise-beta Decode 融合

## 1. 目标与边界

本阶段只优化 Ascend 上 Lite KDA 的单 token Decode，不修改 Prefill。

固定目标 shape 为 TP8 rank-local `H=4, K=V=128`、featurewise beta、K-major FP32
recurrent state。候选在一个 Triton-Ascend kernel 内完成：

1. Q/K FP32 L2Norm；
2. `sqrt(sigmoid(beta_logits) + 1e-10)` 和 K/V featurewise scale；
3. forget gate；
4. state decay、delta/read、outer-product update；
5. BF16 output 与 live write-slot 发布。

不改变 projection、causal-conv、output gate、state/cache 布局、PD one-copy、graph executor、
registry 或第三方依赖。Prefill 继续使用已验证的 Torch prepare + public gate/chunk 路径。

## 2. 动机与独立准入

阶段 12C 组合候选因 Prefill 完整链只提升 `1.44%/0.98%/1.37%` 而已完整撤回。
同一 exact A/B 中，Decode 是独立代码路径，BS1/BS2 graph 分别从
`139.899/174.917 us` 降至 `31.920/32.213 us`，提升 `77.18%/81.58%`。

因此本阶段建立新的 Decode-only 准入边界，不修改 12C 的冻结规则，也不用 Decode
收益掩盖 Prefill 负结果。

## 3. 最小实现

只在已有 Ascend `ops/kda.py` 中增加一个 kernel 和一个轻量 launch wrapper。

production dispatch 必须同时满足：

- device 为 NPU；
- beta 为与 Q 同 shape 的 featurewise 模式；
- `H=4, K=V=128`；
- Q/K/V/g/beta/state 最内层 stride 为 1；
- `A_log/dt_bias/read_indices/write_indices/cu_seqlens` contiguous。

任意条件不满足时继续调用现有 tensor 路径。CPU、scalar-beta、非目标 shape 和异常 stride
不增加新分支或复制实现。

kernel 以 `batch * head * value_tile` 为 grid，每个 program 处理 128 个 key 通道和
32 个 value 通道。read/write page 由现有 graph-stable metadata 动态读取；padding row 输出清零且
不发布 state。

## 4. 数值与状态门禁

延用阶段 12C.1 的冻结标准：

- BS1/BS2 eager 和 graph output max abs `<=1e-4`、rel-L2 `<=1e-3`；
- FP32 state max abs `<=1e-6`、rel-L2 `<=1e-6`；
- changed-input replay 满足同一门禁；
- padding output 为零，padding/page 0/未选邻页 bitwise unchanged；
- 至少 4 个 seed，每个 BS2 连续 128 步，同时满足 output/state 门禁；
- output/state 全部 finite。

标准 Triton 验证器必须同时返回 BF16 output 和更新后的 FP32 state。

## 5. 性能门禁

在同一张 910B、同一进程、同一 public artifact 中动态加载阶段 12A baseline，与 candidate
交替执行至少 11 轮。BS1 和 BS2 必须同时：

- graph replay 提升 `>=5%`；
- 每层节省 `>=5 us`；
- 不新增 graph fallback、state transpose、host sync 或第二套 executor。

通过 leaf 门禁后仍不宣称端到端加速，最终以 8P8D 服务 A/B 记录 TPOT 和吞吐。

## 6. 测试与交付顺序

1. 独立提交本设计；
2. 只接入 Decode kernel 和 Decode focused tests，不改 Prefill 代码/测试；
3. exact-source NPU 运行 focused、标准 verifier 和 graph A/B；
4. 累计执行源在 8P8D 上完成 4-token、64-token、1100-token continuation、BS2 late-admission、
   finite/日志门禁和 GSM8K first100；
5. 精确停服，确认监听端口、PID/marker 和 16 卡 device holder 全部清零；
6. 独立提交验证记录。

回退点为 `a4adc838` 恢复的阶段 12A executable。任一数值、graph 或性能门禁失败时，
不保留 Decode kernel 或实验开关。
