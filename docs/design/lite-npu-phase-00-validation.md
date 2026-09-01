# Lite NPU 阶段 0：基线验证记录

## 1. 结论

阶段 0 的仓库、checkpoint、参考输出、公开算子来源和 NPU 基线门禁均已完成。Lite 可以进入阶段 1
的 config、模型骨架和权重映射实现；这不代表任何候选融合算子已经进入 TokenSpeed 生产路径。

本记录只包含不可逆摘要。checkpoint、tokenizer、原始 tensor、生成文本、连接信息和设备日志继续
保存在仓库外。

## 2. TokenSpeed 与 NPU 基线

验证基线为 TokenSpeed `7955b4f5e359ce7b5fb5f21f27d1903a29ed2c14`，目标环境包含 16 张
Ascend 910B 系列 NPU、CANN 9.0、Python 3.11、PyTorch 2.9 和 torch-npu 2.9。

| 检查 | 结果 |
| --- | --- |
| PR #1281 LayerNorm、RoPE、MHA NPU 测试 | `24/24` 通过 |
| Decode NPUGraph capture/update/replay | 通过 |
| 16 rank 设备可见性 | 通过 |
| 两个独立 8-rank HCCL group | 通过；all-reduce 结果分别为 `36`、`100` |
| 单卡 eager allocation、执行和 synchronize | 通过 |

这些结果只证明公共 NPU 基座可用，不把 KDA、MLA、Grouped MoE、OE、CP/KVP 或 P/D cache 误记为
已有能力。

## 3. Checkpoint manifest

| 项目 | 结果 |
| --- | ---: |
| config 字段 | `47` |
| 未分类 config 字段 | `0` |
| transformer 层数 | `28` |
| KDA / MLA 层数 | `21 / 7`，严格为 `3:1` |
| checkpoint shard | `10` |
| tensor key | `129,870` |
| 归一化 key pattern | `39` |
| 未解释 key / 缺失 shard / metadata 不一致 | `0 / 0 / 0` |
| safetensors header 计算的 tensor payload | `138,634,018,688` bytes（`129.113 GiB`） |

完整 shard header 校验共 `4/4` 项通过：索引中的每个 tensor 都与实际 shard 的 dtype、shape、offset
和 payload 大小一致，且没有索引外 tensor。

冻结摘要：

- config SHA-256：`d2676f137594dea8d3bcdd5a26703e7bf45f000845f9613de51fde6b7321dac6`；
- checkpoint index SHA-256：`2900b080361629254f560d90fe6333d9cfb8b8fdbb990df73c670c41b8ba96b7`。

## 4. 端到端参考输出

参考实现以固定 tokenizer、固定 token 输入和 greedy 解码生成 4 步参考输出。Prefill 使用 eager，
Decode graph、overlap 和候选融合开关均关闭，避免把优化差异混入模型基线。

每步包含 `115` 个语义 tap：embedding、28 层 attention、attention residual、MoE、layer output、
final norm 和 logits。总计 `460` 个 tensor、`8,224,768` bytes，所有值均为 finite；Prefill 保存
第 0 步的 `115` 个 tensor，Decode 保存后 3 步的 `345` 个 tensor。

| Artifact | SHA-256 |
| --- | --- |
| request 摘要 | `8f523350ef66e21809e9a2e9c28a7e8e86732b532741a1efdc5b7f17e67b835c` |
| response 摘要 | `e6ec505c09f276ee134b9ee7a729f4d7bfd8de1199122661a5dd7f8ef9b1ecd1` |
| token contract | `a09f33dca8f7e92f5e9ce70fd35a8ae2885a8665ef0bf7b97ca04fe6c76e8110` |
| Prefill tensor manifest | `fc5da48a9752f7396b0747817d59596bc0dcea6c6cfa77a57b781faf5da2ed2c` |
| Decode tensor manifest | `0df3c750c2f2e1eede64ff430b67bbc3a2ee9dd2a874a6652a05188358125645` |
| 聚合 tensor fingerprint | `0461eb0ce258e588cb2b5d3b06eb4c24d77b749498f10f1d7908af4ac7206eb4` |
| reference summary | `43d02a4465e0793d93577a9289beb4ca7d392a6804cda8d912ab9739f0d971ae` |
| tensor inventory | `a0afaa95db63cdd1253400507cffd350cffb95f0dcd67f7046b5e4749db39f39` |

原始请求、响应、token id、tensor 和 tokenizer 文件没有提交到 Git。

## 5. KDA 逐 token state

使用同一份冻结 core-ready 输入在 NPU 上执行 post-preparation FP32 recurrence。输入已经完成 Q/K
归一化和 featurewise-beta 对 K/V 的吸收，因此不会重复应用前处理。

- 输入为 `B=1, T=64, H=16, D=128`；
- 每个 token 保存一份 `[1,16,128,128]` FP32 state fingerprint，共 `64` 份；
- 64 份 state 全部 finite；
- 最后一步 state SHA-256 为
  `02b9b8aa0524f9a7f7f93849d1b3a6ba35f88bf28093c21bdd4268c1e18445df`，与既有 NPU capture
  完全一致；
- 逐 token fingerprint manifest SHA-256 为
  `9145e77298304dd755afeffea154bd6844991906020763926bf4cd994c0e6e75`。

后续 KDA 实现必须按阶段 0 的 FP32 state 门槛逐 token 对比，不能只比较最终 state。

## 6. 子模块参考证据

仓库外还冻结了下列真实 shape 或生产 shape 的 NPU leaf 结果。这里的“通过”表示 correctness/finite
证据可用；性能未通过的候选仍保持关闭。

| 数据流 | 结果 | Result SHA-256 |
| --- | --- | --- |
| width-4 causal-conv 与 state update | 通过 | `a157fa605b02cab801c1f7124d82d5ea96f13e57972aca90b40081a160bb93ce` |
| Decode recurrent KDA | 通过 | `ed3b7ca5d706541ac549310dc573b6cbab8f29757a09c0ebfd4e2d76ccd90eba` |
| Prefill chunk KDA | 通过 | `d1ddc9390aef1056b8318e69d0cba6dce8d7068e896f419686283c7e4caf62c6` |
| KDA gated RMSNorm / output gate | 数值 finite；融合候选因性能拒绝 | `ef552b4d9f8dd6711f34234d23e17924eec01f203f2e2c280ef8846de2eb46ef` |
| Grouped MoE 外围 projection/merge | 通过 | `305d750021503ec4e6f0fe8f6c901f65a12ea40cf80ccf5bdfc4279141d0aaf5` |
| MLA output gate epilogue | 通过 | `c413cd6b97fcd0a4b7f350ce0faf45088725fb64282da2258f7992196beffa4c` |
| OE block projection/normalize/select | 通过 | `96d474fbb780d5e46128dadcb9d74d45222e130dfdbc0a7afff0d70a26ee2cb8` |

端到端 tap 负责定位层级漂移，leaf artifact 负责定位算子内部漂移。阶段 1 的 loader 只验证结构和
权重映射；projection、routing、MLA latent 和 OE id 等内部 tensor 会在各自实现阶段生成 TokenSpeed
侧对照，不以现有 layer-boundary tap 代替算子准入。

## 7. 公开 kernel 清单

`lite-npu-public-kernels.json` 的 SHA-256 为
`df284335dd49eadd894abad2b4f50b79734d4856b7bac8e0cb43c42688fa46c8`。18 项能力的状态为：

| 状态 | 数量 |
| --- | ---: |
| `existing` | `4` |
| `candidate` | `9` |
| `adapt` | `1` |
| `missing` | `4` |

四项明确缺失的能力是 Lite featurewise-beta prepare、NoPE MLA attention LSE 返回、attention
partial-state merge 和 OE block projection/host lookup。它们不阻塞模型骨架；在对应阶段先使用
Torch/reference baseline，再通过独立设计、上板和性能门禁决定是否增加融合实现。

## 8. 已知非阻塞环境现象

- 图编译可选组件会打印未编译提示，但 PR #1281 的 Decode graph 测试实际通过；
- 参考服务未启用可选的 Responses API，不影响本次使用的 generate 路径；
- 受控停止服务时日志含非模型的编译线程 EOF 信息，参考请求已在停止前成功完成；
- 参考生成结束后，listener、PID 记录和模型进程均已清理。

这些现象没有被归因于模型精度，也不会通过修改 Lite 模型代码规避。

## 9. 阶段 1 准入

阶段 0 的 config 未分类数、checkpoint 未解释数、公开来源缺失数均为 0；端到端和 KDA 逐 token
参考摘要齐全，NPU 公共基线通过。因此阶段 1 准入。

阶段 1 仍需 fail closed：未知数学字段、未知 checkpoint key、重复 target load 或目标参数未覆盖，
任一出现都必须阻止模型构造，不能依赖静默默认值。
