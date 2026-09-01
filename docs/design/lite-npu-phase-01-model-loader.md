# Lite NPU 阶段 1：模型结构与权重加载

## 1. 目的

本阶段让 TokenSpeed 能够在不修改原始 checkpoint 的前提下识别 Lite 文本模型、构造与权重一致的
28 层混合 KDA/MLA 骨架，并对每一个 checkpoint key 做确定且可验证的加载。NPU 算子、cache、
Grouped MoE 执行器和 OE 数学仍由后续阶段实现；本阶段只建立这些模块共同依赖的结构与参数契约。

阶段 1 分成三个独立提交：

1. 本文：冻结 config、模型注册、参数命名、权重映射和测试边界；
2. config、模型骨架、严格 loader 及合成测试；
3. 完整 checkpoint index replay 和公开安全的验证记录。

## 2. 复用边界

TokenSpeed 已有两个接近但不等价的模型实现：

| 现有实现 | 可以复用 | 不能直接复用 |
| --- | --- | --- |
| Kimi K3 | 混合层协议、KDA backend seam、NoPE MLA、output gate、hybrid cache 接口 | AttnRes、标量 beta、SiTU MoE、Kimi checkpoint packed layout |
| LongCat Flash | 普通 RMS residual、embedding/norm/lm-head、shared/zero expert 的参数惯例 | 双 MLA 层、CUDA 专用 router/finalize、单组 MoE、宽松 loader |

Lite 不继承上述任一完整模型。最小实现是新增一个真实的 `FLASHLocalForCausalLM` entry class，组合
现有通用 layer，并只为 Lite 独有的 KDA、Grouped MoE 和 OE 参数定义薄模块。这样不会把 CUDA import
带入 NPU，也不会误用 Kimi 的 AttnRes 或 scalar-beta 数学。

## 3. Config 识别和标准化

### 3.1 缺少 `model_type` 的处理

目标 checkpoint 声明 `architectures=["FLASHLocalForCausalLM"]`，但没有 `model_type`。不允许修改
checkpoint 的 `config.json`，也不允许依赖 `trust_remote_code`。

在现有 `get_config()` 中增加一个统一的 config-class 解析步骤：

1. 若第一个 architecture 是 `FLASHLocalForCausalLM`，选择 `LiteConfig`；
2. 否则按已有 `model_type -> config class` registry 解析；
3. 两者都未命中时保持现有 `AutoConfig` 路径。

该判断同时用于本地目录和 revision-pinned 远端 snapshot，避免两条加载路径漂移。architecture 只做
这一处兼容，不将缺失的 `model_type` 静默解释为 Llama。

### 3.2 Canonical 字段

`LiteConfig` 保留 checkpoint 原字段，同时提供 TokenSpeed 已有 layer/cache 所需的 canonical alias：

| Checkpoint 字段 | Canonical 字段或派生值 |
| --- | --- |
| `num_layers` | `num_hidden_layers` |
| `ffn_hidden_size` | `intermediate_size` |
| `expert_ffn_hidden_size` | `moe_intermediate_size` |
| `moe_topk` | `num_experts_per_tok` |
| `linear_num_heads` | KDA `num_heads` |
| `linear_head_dim` | KDA `head_dim` |
| `linear_conv_size` | KDA `short_conv_kernel_size` |
| `fa_interval` | 混合层分布 |
| `kda_nope` | MLA `mla_use_nope` |

对 0-based layer `i`：

```text
is_mla(i) = (i + 1) % fa_interval == 0
is_kda(i) = not is_mla(i)
```

28 层、`fa_interval=4` 必须得到 KDA 21 层、MLA 7 层；MLA 索引必须为
`[3, 7, 11, 15, 19, 23, 27]`。`layers_block_type`、`layer_types`、
`linear_layer_ids` 和 `full_attention_layer_ids` 沿用 Kimi config 已定义的混合层协议。

### 3.3 Fail-closed 校验

下列字段影响数学或 shape，缺失或不满足约束时直接报错：

- architecture 必须包含且首项为 `FLASHLocalForCausalLM`；
- `hidden_size`、attention/KDA head 数和维度必须为正，KDA TP8 与 MoE EP8 切分维度必须整除；
- `linear_method` 必须为 featurewise-gated KDA，且 full-rank output gate 必须开启；
- MLA 必须启用 output gate 和 NoPE，Q/KV LoRA、NoPE、RoPE、V 维度必须齐全；
- `moe_group_size`、每组 real/zero expert 数、每组 top-k 和 expert hidden size 必须齐全；
- EveryLayer MoE、identity zero expert、router activation 和 routing normalization 必须显式可识别；
- OE 的表数量、neighbor/split、special-token 排除和归一化字段必须齐全且内部一致；
- `num_hidden_layers % fa_interval == 0`，避免最后一层类型由隐式补齐决定。

未知字段由 `PretrainedConfig` 保留，但本阶段已冻结的数学字段不使用默认值补齐。模型构造不接受
4P4D 或单卡完整服务；N=1 只用于小尺寸 CPU/meta 结构测试。

## 4. 模型骨架

### 4.1 顶层结构

```text
FLASHLocalForCausalLM
├── model.embed_tokens
├── model.layers[0..27]
│   ├── input_layernorm
│   ├── self_attn: LiteKDAParameters | LiteMLAParameters
│   ├── post_attention_layernorm
│   └── mlp: LiteGroupedMoEParameters
├── model.norm
├── model.ngram_embeddings
└── lm_head
```

每层保持普通的 attention residual 和 MoE residual，不引入 Kimi AttnRes。阶段 1 的 attention、MoE、
OE 模块负责准确持有参数并暴露后续 backend 所需接口；它们的数值 forward 在对应算子阶段接通。
若在阶段 1 被误用于完整推理，必须抛出明确的未实现错误，不能返回占位 tensor。

### 4.2 KDA 参数

21 个 KDA 层保留 checkpoint 的未融合逻辑参数：

- Q/K/V projection 和三路 depthwise width-4 convolution；
- 两层 featurewise-beta projection `b_proj.0/1`；
- 两层 forget projection `f_proj.0/1`、`A_log` 和 `dt_bias`；
- full-rank output gate `g_proj.0`；
- output RMSNorm 和 output projection。

本阶段不提前创建 packed projection 或 packed-conv 参数。后续融合阶段由同一 loader 将这些 source
slice 装入 fused parameter，并用现有 unfused 形状作为 oracle。

### 4.3 MLA 参数

7 个 MLA 层保留独立的 `q_a_proj`、`q_b_proj`、`kv_a_proj_with_mqa`、`kv_a_layernorm`、
`kv_b_proj`、`g_proj` 和 `o_proj`。后续 MLA 阶段可复用 Kimi NoPE MLA 并在 load 时合并 a-projection；
阶段 1 不为了未来融合改变 checkpoint 语义。

### 4.4 Grouped MoE 和 OE 参数

每层保存四个独立 router 及 correction bias、1536 个 group-major real experts、一个 shared expert、
输入/输出 projection 和 MoE norm。identity zero experts 没有可训练 expert GEMM 权重，只存在 router
logit 与逻辑 ID。

OE 保存 12 个 embedding table 和 12 个 post projection。主表参数从创建开始位于 CPU；loader 不先把
完整表放到 device 再搬回 host。具体 n-gram 生成、staging 和聚合在 OE 阶段实现。

## 5. 权重映射

### 5.1 唯一分类

loader 对每个 source key 只允许一次匹配，并记录 source、目标参数、slice/shard 和处理类别。类别固定为：

1. direct；
2. rename；
3. packed slice；
4. TP/EP shard；
5. host-resident OE。

目标 checkpoint 没有允许忽略的训练权重。未知 source、重复 source、重复覆盖同一完整目标、缺少必需
source、错误 shape 或 shard 越界都在 loader finalize 时失败。不能沿用 LongCat/Kimi loader 的
“找不到参数就 warning/continue”行为。

### 5.2 KDA rename

外部 `self_attn.linear_core.*` 归一化到层内 KDA 参数。主要规则为：

| Source suffix | Logical target |
| --- | --- |
| `linear_core.q_proj.weight` | `q_proj.weight` |
| `linear_core.k_proj.weight` | `k_proj.weight` |
| `linear_core.v_proj.weight` | `v_proj.weight` |
| `linear_core.{q,k,v}_conv1d.weight` | 对应 conv weight |
| `linear_core.b_proj.{0,1}.weight` | featurewise-beta 两层 projection |
| `linear_core.f_proj.{0,1}.weight` | forget 两层 projection |
| `linear_core.g_proj.0.weight` | full-rank output gate |
| `linear_core.A_log` / `dt_bias` | recurrent gate 参数 |
| `linear_core.o_norm.weight` / `o_proj.weight` | output epilogue 参数 |

MLA、norm、embedding、lm-head 和未融合 shared expert 直接加载。MLA output gate 保持独立 source，
不与 KDA 的 `g_proj.0` 混用。

### 5.3 Grouped MoE 映射

checkpoint 的 router 保持四组独立。real expert 的 source ID 已是 flattened group-major ID：

```text
global_expert_id = group_id * experts_per_group + expert_in_group
ep_rank = global_expert_id // local_experts
local_expert_id = global_expert_id % local_experts
```

在 EP8 下 `local_experts=1536/8=192`，每个 group 恰好覆盖两个 rank。阶段 1 loader 按该公式切分
gate/up/down 三个 expert weight；阶段 6 只比较执行布局，不重新解释 checkpoint ID。

router、correction bias、shared expert 和 Grouped MoE 外围 dense 权重按既有 dense policy 切分或复制，
不跟随 routed expert 的 EP owner。

### 5.4 角色内并行与 OE

KDA projection 使用 linear-attention TP8；loader 通过现有参数对象的 `weight_loader` 完成 shard，
禁止在 Lite loader 中复制一套 row/column slicing 实现。loader 只负责选择正确目标和 shard ID，并
额外验证 global/local shape。

MLA 不做权重 TP：Prefill 的 CP8 和 Decode 的 KVP/attention-DP8 只分片 token 或历史 cache，MLA
projection 在角色内八个 rank 上保持完整复制。不能因为 MLA cache 被八卡分片，就把 checkpoint 的
head/projection 权重再切八份。

OE table 是 host-resident direct load；12 个 post projection 使用 dense TP 规则。P/D 两个角色加载同一
静态权重语义，CP8/KVP8 只切 cache，不改变本阶段的 checkpoint key 归属。

## 6. 实现范围

阶段 1B 只增加以下最小文件：

- 一个 Lite config 文件及现有 config export/registry 的少量改动；
- 一个 Lite model 文件，包含 entry class、参数骨架和严格 loader；
- 在已有 attention architecture 集合中注册 Lite；
- 一个 focused test 文件。

不新增通用 loader framework、模型专用 parallelism framework、NPU kernel、cache manager 或启动脚本。
若 strict coverage 逻辑能在 Lite model 内用一个集合完成，就不抽象成全仓公共组件。

## 7. 测试与准入

### 7.1 CPU/meta 测试

最小合成 config 使用与真实 key 相同的字段名，但缩小维度和 expert 数。测试必须覆盖：

- 无 `model_type` 时由 architecture 选择 `LiteConfig`；
- registry 解析到真实 `FLASHLocalForCausalLM` class；
- 28 层真实 layer pattern 精确为 21 KDA + 7 MLA；
- 数学字段缺失、非法 interval、不可整除 shape 和不支持语义均 fail closed；
- N=1 小尺寸 CPU/meta 构造，以及不分配真实权重的 N=8 mapping 构造；
- 39 类真实 key pattern 全部命中唯一 mapping；
- missing、unexpected、duplicate 和错误 shape 分别失败；
- KDA rename、四组 router、EP8 group-major expert owner、shared expert 和 12 组 OE 映射正确。

### 7.2 完整 checkpoint replay

阶段 1C 在受控环境只读取 config/index/header，并对全部 key 执行同一个 mapping planner：

- source key 数与冻结 manifest 一致；
- 未解释 source、重复 source、缺失 target 均为 0；
- 28 层和 39 类 pattern 计数与阶段 0 基线一致；
- 每个 EP rank 恰好拥有 192 个 real experts；
- 所有 TP/EP local shape 与目标参数一致；
- OE 表始终标记为 host-resident。

完整 replay 的原始路径和 key 列表保存在仓库外，提交中只记录不可逆摘要和计数。

### 7.3 提交门槛

每个提交前运行 focused test 和完整 `pre-commit run --all-files`。只有 Phase 1C 的 exact coverage 全部
为零差异后才进入 KDA Torch baseline；任何宽松忽略、checkpoint 改写或真实权重未分类都会阻塞后续阶段。
