# Lite NPU 阶段 4D：Prefill KDA 验证记录

## 1. 验证对象

本记录验证阶段 4D 的实现提交 `b8ea1ae2` 和性能门槛修正 `fca65e1b`。范围限定为 Lite
featurewise-beta Prefill 的公开 `KdaGateCumsum + ChunkKdaFwd` 链、package-local artifact 加载、
空请求 compact、统一 kernel selection、数值和单卡性能。

本阶段没有修改模型 runtime、cache/PD、CP8/KVP8、Decode、causal-conv 或完整 8P8D 服务。目标环境
为单张 Ascend 910B、CANN 9.0.0、PyTorch 2.9.0 和 `torch_npu` 2.9.0.post2；全部实卡验证只使用
NPU0。

## 2. Exact-source 与 artifact

最终验证从完整提交 `fca65e1b58a5262158ce79b2fa68a93ea83278ed` 生成全新 `git archive`。archive
SHA-256 为：

```text
c7f753cb9c8b09c271c86c5e0ce9ab4b50e8310cf590ed644423c6e2759686b2
```

传输前后 SHA-256 一致，`git get-tar-commit-id` 在两端均返回上述完整提交。远端在独立目录解包，只
补入由未修改公开源码 `d543ccee0a1ff677165777e3defafd42b35e83ef` clean-build 的 package-local
artifact，没有覆盖 exact archive 中的 Python、C++、测试或文档。

artifact manifest 使用 schema v2，同时固定并校验 binding、vendor op-api 和 vendor op-tiling 三个
文件的相对路径与 SHA-256。loader 在任何环境修改或 `dlopen` 前完成 manifest、digest 和 symlink
containment 校验；随后先以 `RTLD_GLOBAL` 加载 tiling library，再加载 Torch binding，并保持 tiling
handle 到进程结束。

最终 discovery 在启动前显式清除了外部 `ASCEND_CUSTOM_OPP_PATH` 和手工 artifact/tiling override。
四个公开 schema 均由 package loader 自动发现：

```text
causal_conv1d, chunk_kda_fwd, kda_gate_cumsum, recurrent_kda
```

## 3. Artifact 注册时序

首次统一 API 上板暴露了一个真实生命周期问题：若先分配 NPU tensor，再首次 lazy-load 自定义 OPP，
`KdaGateCumsum` 会异步返回空 executor；即使随后显式 `dlopen` vendor op-api 也不能补注册。相同
artifact 在首次 device 初始化前加载则稳定通过。

最终实现让 Ascend KDA 模块在 kernel registry 导入时预加载可选 artifact。该模块导入发生在模型 tensor
分配前；artifact 缺失或校验失败仍只缓存 unavailable 状态，不会阻止 reference 路径启动。loader 单测
使用隔离 handle list，不再释放进程中已注册的真实 tiling handle。

NPU 数值测试先计算 Torch oracle，再调用公开链，并在公开调用后立即同步。这样异步 NPU 错误会归因
到真实 public launch，而不会错误地出现在后续 `isfinite`、`arange` 或 reference 算子上。

## 4. Lite 数据流与 state

公开链前的 Lite 适配为：

```text
q = L2Norm(q)
s = sqrt(sigmoid(beta_logits) + 1e-10)
k = L2Norm(k) * s
v = v * s
beta = ones per head
```

`KdaGateCumsum` 执行 raw gate、safe lower bound 和 chunk64 的局部 prefix sum；`ChunkKdaFwd` 消费
K-major FP32 initial state，并返回 BF16 output 与 K-major FP32 final state。输入 state 不原地修改。

公开 op 对包含零长度 sequence 的直接调用不会报错，但会静默产生错误 final state。生产 adapter 从
host `cu_seqlens_cpu` 建立 compact boundary/chunk plan，仅 gather 非空 state；kernel 返回后把 final
state scatter 回原 request 槽。无空请求时直接传连续 initial state，并原样返回 kernel final state，
没有额外 gather、clone 或 scatter；全部为空时不启动两个公开 op。

## 5. 数值结果

最终 exact-source discovery 对量化后逐 token oracle 的结果为：

| 场景 | empty compact | output rel-L2 | state rel-L2 | finite |
| --- | --- | ---: | ---: | --- |
| T64 | 否 | `0.004116` | `0.001578` | 是 |
| `[1,3]` | 否 | `0.003622` | `0.001307` | 是 |
| `[65,127]` | 否 | `0.004120` | `0.001633` | 是 |
| `[2,0,2]` 原始 op 负对照 | 否 | `0.040193` | `1.131449` | 是但错误 |
| `[2,0,2]` 生产 adapter | 是 | `0.003694` | `0.001461` | 是 |

真实统一 API 另外覆盖 T4、`[1,63]`、`[32,0,32]` 和 `[65,127]`。全部 output rel-L2 `<0.01`、state
rel-L2 `<0.005`，input state 保持 bitwise，零长度 request 的 final state 与输入 bitwise 相同，没有
NaN/Inf。

## 6. Selection 与回退

Ascend registry 只为 `beta_mode=featurewise`、`recurrent_layout=k_major` 注册 `public_kda`
specialized solution。scalar beta 继续选择 Torch reference；不支持的 dtype、shape、state、gate 参数或
artifact unavailable 在自动模式回退 reference，显式 `solution="public_kda"` 则 fail loud。

公开链有约 2.5 ms 固定启动成本。最终 exact-source 的短序列边界 A/B 为：

| tokens | 公开链 | reference |
| ---: | ---: | ---: |
| 8 | `2.511 ms` | `1.533 ms` |
| 16 | `2.413 ms` | `2.487 ms` |
| 32 | `2.535 ms` | `3.812 ms` |

T16 位于波动交叉区，因此默认模式保守地在总 token `<32` 时使用 reference；显式 public 不受该阈值
影响，仍用于完整 kernel board。阈值测试同时冻结 2-token 自动回退和 32-token 自动命中公开链。

较长序列的最终 exact-source A/B 为：

| tokens | 公开链 | reference | 公开链/reference |
| ---: | ---: | ---: | ---: |
| 4 | `2.678 ms` | `1.321 ms` | `2.03x` |
| 64 | `2.557 ms` | `6.175 ms` | `0.41x` |
| 256 | `2.607 ms` | `21.953 ms` | `0.12x` |
| 1024 | `2.576 ms` | `85.140 ms` | `0.03x` |

T4 是显式 public 的记录值；生产自动路径会按阈值选择 reference。

## 7. 测试汇总

| 门禁 | 结果 |
| --- | --- |
| final exact-source loader/build/Prefill/facade focused | `33 passed` |
| 真实 NPU shape | T4、`[1,63]`、`[32,0,32]`、`[65,127]` 全部通过 |
| artifact v2 / digest / symlink / load-order / failure restore | 全部通过 |
| unavailable fallback / explicit failure / scalar selection | 全部通过 |
| 全仓 `pre-commit run --all-files` | 全部通过 |

测试产生的两个 warning 分别来自目标 wheel 未编译 TorchAir，以及 NPU 对 base-format tensor 的提示；
都不影响算子执行、数值或 selection。

## 8. 结论与回退

阶段 4D 准入通过。Lite featurewise-beta Prefill 已通过统一 API 复用公开 Gate/Chunk KDA，artifact 可在
无启动脚本手工 OPP 配置的进程内正确注册；空请求、短序列、变长、跨 chunk、输入 state 不变和
fallback/fail-loud 均有测试覆盖。

若后续完整模型发现问题，只需移除 Ascend `public_kda` specialized registration；阶段 3 reference、
artifact 完整性校验、runtime、cache 和模型结构均无需修改。下一阶段按既定顺序验证 KDA output
epilogue 的现有融合实现，不在本阶段继续增加 projection、conv 或 recurrence kernel。
