# Lite NPU 阶段 0：基线与公开算子清单

## 1. 目的

本阶段在模型代码进入仓库前冻结可复现的输入、来源和数值门槛，避免后续把 checkpoint 差异、
算子差异和并行差异混在一起调试。

阶段 0 只建立验证契约，不实现 Lite 模型或 NPU kernel。它的完成条件是：

- TokenSpeed NPU 基线和现有能力边界可追溯；
- Lite config 字段和 checkpoint key 均已分类，没有未解释项；
- 参考请求和逐级 tensor 已在仓库外冻结并有摘要；
- 每个公开 kernel 候选均有固定源码 revision 和 license 结论；
- 各类输出使用明确、可运行的精度门槛。

## 2. 基线边界

TokenSpeed 的 NPU 公共基线为
[PR #1281](https://github.com/lightseekorg/tokenspeed/pull/1281) 引入的提交
`c006baf2dc07337a572794eed2c71c5d0e5143ee`。该基线提供 device、HCCL、NPU graph、paged MHA、
RMSNorm、RoPE、embedding dispatch 和 NPU 测试框架，但不代表 Lite 所需的 KDA、NoPE MLA、
Grouped MoE、OE、CP/KVP 或混合 cache P/D 传输已经支持。

Lite 实现必须继续遵守以下边界：

- runtime 只调用 `tokenspeed-kernel` 的厂商无关 API；
- 直接 `torch_npu` 和第三方 kernel 调用只存在于 `tokenspeed-kernel-npu` 后端；
- 融合 kernel 与 Torch reference 使用同一个 API，选择实现不改变模型数据流；
- Prefill 从 eager 开始，Decode graph 在 eager 精度通过后独立准入；
- 完整模型只验收 8P8D，共 16 rank。

## 3. 仓库内外的数据边界

### 3.1 允许提交

- 公开源码 URL、不可变 revision、license 标识和适配状态；
- config/key 的字段名、shape 规则、分类计数和不可逆摘要；
- 合成输入生成规则、精度阈值和验证命令；
- 不含内部环境信息的通过/失败摘要。

### 3.2 只能保存在仓库外

- checkpoint、tokenizer 和服务的实际路径；
- 远端地址、端口、账号、凭据、任务标识和容器信息；
- 原始 prompt、业务数据、完整 config、checkpoint index 和权重 key 列表；
- token、hidden、state、routing weight、logits 和生成文本的原始 artifact；
- 带内部路径或环境信息的日志。

仓库内的验证记录只能引用 SHA-256、dtype、shape、元素数量、分类计数和测试结论，不能引用仓库外
artifact 的物理路径。这样公开提交可复核契约，又不会把私有模型或运行环境写入 Git 历史。

## 4. 冻结产物

### 4.1 Lite config manifest

仓库外 manifest 必须逐项记录：

- 字段名、原始类型、是否必填、默认值策略；
- 消费该字段的模型模块；
- 它影响数学、shape、并行、cache、runtime 还是仅影响 tokenizer；
- TokenSpeed 对应字段或显式转换规则；
- 未消费时是允许忽略还是必须拒绝。

影响数学、shape、cache 或并行的字段禁止使用静默默认值。未知字段可以保留以兼容通用模型配置，
但必须被分类为“不影响 Lite 推理”；无法分类的字段阻塞阶段 1。

### 4.2 Checkpoint key manifest

仓库外 manifest 按 key 记录 shape、dtype、元素数、目标参数和 shard 规则，并归入以下唯一类别：

1. 直接加载；
2. rename 后加载；
3. packed/merged 后加载；
4. TP/EP shard 后加载；
5. 明确忽略且有理由；
6. 未解释。

同一个源 key 只能进入一个类别；所有目标参数必须恰好由一个源或一个有序 source slice 集合产生。
类别 6 非空、目标参数重复装载或应装载 shard 缺失，均阻塞阶段 1。

### 4.3 Reference artifact

仓库外 reference 至少包含：

- 固定 tokenizer、固定 greedy 参数和固定请求的 input/output token id；
- KDA/MLA 层选择序列；
- 第一层 KDA 的 projection、conv、gate、recurrent state 和输出；
- 第一层 MLA 的 latent projection、attention、output gate 和输出；
- 第一层 Grouped MoE 的每组 expert id、routing weight、expert 输出和合并输出；
- OE 的 n-gram id、排除 mask、lookup/聚合输出；
- 末层 hidden、logits 和最终生成 token。

每个 tensor 同时保存 dtype、shape、有限值统计、SHA-256 和生成端源码 revision。FP32 recurrent state
按每个 token 保存 fingerprint，不能只保存最终 state，以便定位首个漂移 token。

## 5. 公开 kernel 清单

阶段 0B 新增 `docs/design/lite-npu-public-kernels.json`。使用 JSON 是为了直接用 Python 标准库校验，
不为静态清单增加 YAML/schema 依赖。

每个候选条目必须包含：

| 字段 | 含义 |
| --- | --- |
| `capability` | 稳定的能力名，例如 KDA Decode recurrent |
| `source_url` | 官方或上游公开源码地址 |
| `revision` | 完整不可变 commit SHA |
| `license` | SPDX license 标识 |
| `upstream_path` | revision 下的源码路径 |
| `status` | `existing`、`candidate`、`adapt` 或 `missing` |
| `roles` | `prefill`、`decode` 或两者 |
| `covered_steps` | 它覆盖的 Lite 数据流步骤 |
| `shape_constraints` | 已知 dtype、rank、head_dim、state_dim 和对齐限制 |
| `integration` | 复用、薄适配、增量开发或重新实现 |
| `validation` | 进入生产路径前必须通过的测试阶段 |

清单至少覆盖下列能力；没有公开候选时也必须以 `missing` 显式记录，不能留空：

- packed Q/K/V width-4 causal convolution 和 state update；
- KDA Prefill chunk scan；
- KDA Decode recurrent update；
- Lite featurewise-beta prepare；
- KDA RMSNorm、output gate 和 output projection epilogue；
- NoPE MLA Prefill/Decode、LSE 返回和 partial-state merge；
- Grouped MoE fused top-k、dispatch、GMM、SwiGLU、combine/finalize；
- OE block projection、host lookup 和聚合；
- fused-add-RMSNorm、RoPE、embedding 等已存在 NPU 能力。

`candidate` 只表示值得上板，不表示已经可用。只有同时满足 license、API、shape、精度、graph 和性能
门槛后才能改为 `existing`。引入第三方源码时还必须更新相应 `THIRDPARTYNOTICES`。

## 6. 精度契约

所有浮点比较先要求 shape/dtype 契约正确并且输出无 NaN/Inf，再把两侧转换为 FP32 计算误差。
门槛以逐项全量比较为主，cosine 只作为额外诊断，不能掩盖局部大误差。

| 对象 | 必须满足 |
| --- | --- |
| BF16 projection、conv、norm、gate、MLA/MoE/OE 中间值 | `atol=2e-2`、`rtol=2e-2`，cosine >= 0.999 |
| FP32 KDA recurrent state | 每 token `atol=5e-4`、`rtol=5e-3`，cosine >= 0.9999 |
| Routing expert id | 非 tie 输入下逐元素完全一致 |
| BF16 routing weight | id 对齐后 `atol=2e-2`、`rtol=2e-2`，每组权重和同样通过 |
| BF16 layer/final hidden | `atol=3e-2`、`rtol=3e-2`，cosine >= 0.999 |
| FP32 logits | `atol=2e-2`、`rtol=2e-2`，top-1 id 完全一致 |
| Greedy token/output | 每步 token id 完全一致 |

测试输入必须避开精确 top-k tie；另设 tie case 只验证选中集合合法和权重有限，不把未规定的 tie-break
顺序写成模型契约。若公开融合 kernel 的稳定舍入导致上述阈值无法满足，必须先留下逐级误差证据，
再在对应算子设计提交中修改该单项阈值，不能通过放宽全局门槛准入。

## 7. 最小校验实现

阶段 0B 只增加公开 kernel JSON 和一个标准库测试。测试需要验证：

- 顶层 schema 版本和 capability 不重复；
- 所有非 `missing` 条目都有 HTTPS 公开 URL、40 位 revision、SPDX license 和源码路径；
- `roles`、`status`、`integration` 只能取约定值；
- 必需 capability 齐全；
- 文本中不含本地绝对路径、凭据字段或非公开 URL；
- 每个候选都有可执行的 validation 阶段。

不新增运行时 manifest loader、配置类或通用 schema 框架：该清单只在开发和 CI 中使用。

## 8. NPU 上板检查

本阶段只执行不加载完整模型的环境和子模块检查：

1. 记录设备型号、CANN、Python、PyTorch、torch-npu 和 HCCL 版本的仓库外摘要；
2. 验证 16 rank 可见性和两个独立 8-rank HCCL group；
3. 运行 PR #1281 已有 NPU kernel 测试，确认基线环境没有退化；
4. 对清单中已有公开候选运行 import/API 探测；
5. 生成 config/key/reference manifest 及其 SHA-256 摘要。

远端探测失败时保留日志并停止，不为绕过环境问题修改模型代码。涉及真实 checkpoint 的读取始终
只发生在受控 NPU 环境，提交中只记录摘要。

## 9. 提交与准入顺序

阶段 0 拆成三个独立 signed-off commit：

1. 本文：冻结数据边界、manifest schema、精度门槛和验证流程；
2. 公开 kernel JSON 与最小 CI 校验测试；
3. 不含私有信息的验证记录，列出 baseline、manifest 摘要、测试结果和剩余 `missing` 能力。

每个提交前运行完整 `pre-commit run --all-files` 并立即推送。阶段 0 完成后，只有在以下门槛全部满足
时才能进入阶段 1：

- config 未分类字段数为 0；
- checkpoint 未解释 key 数为 0，目标参数未覆盖数为 0；
- 公开 kernel 清单没有 license/revision 空缺；
- reference artifact 摘要齐全且可重新生成；
- NPU 公共基线测试通过。

公开 kernel 清单允许保留 `missing`：它表示后续阶段必须先用 Torch baseline，并在独立 kernel 阶段
开发或适配；`missing` 本身不阻塞模型结构阶段。
