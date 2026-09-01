# Lite NPU 阶段 5：MLA Baseline 与 Output Gate

## 1. 目的

本阶段把 Lite 的 7 个 NoPE MLA 层接入 TokenSpeed 已有 MLA backend、压缩 latent page 和统一 kernel
API。范围仅包含单 role rank 的模型子模块与 Ascend leaf：projection、Prefill/Prefix Extend、Decode、
partial-state merge、value projection 和 `o_proj` 前 output gate。CP8、KVP8、PD one-copy、Decode graph
服务和完整 8P8D 接线仍由后续阶段完成。

阶段 5 分成三个独立提交：

1. 本文：冻结模型数学、cache ownership、Ascend ABI、上板事实、实现和回退边界；
2. 最小模型/kernel 接线与 focused tests；
3. exact-source NPU0 数值、graph replay、性能和回退验证记录。

不新增 cache manager，不复制 FluentLLM backend，不引入第三方 kernel，也不为未证明收益的 HW prolog
增加开关。

## 2. 模型数学与真实 shape

Lite MLA 固定：

| 符号 | 值 |
|---|---:|
| hidden `He` | 3072 |
| heads `H` | 32 |
| q LoRA rank `Hcq` | 1536 |
| kv LoRA rank `Hckv` | 512 |
| NoPE head dim `Dnope` | 128 |
| auxiliary head dim `Dr` | 64 |
| value head dim `Dv` | 128 |

`mla_scale_q_lora=true` 和 `mla_scale_kv_lora=true` 不是布尔乘数。它们分别要求 RMSNorm 权重在
post-load 后乘：

```text
q_scale  = sqrt(He / Hcq)  = sqrt(2)
kv_scale = sqrt(He / Hckv) = sqrt(6)
```

未应用这两个 scale 的 forward 不具备目标模型语义。post-load 必须幂等，并同时把
`kv_b_proj[H*(Dnope+Dv),Hckv]` 准备成：

```text
W_KC [H,Dnope,Hckv]
W_VC [H,Hckv,Dv]
```

### 2.1 公共 projection

```text
q_a        = q_a_proj(hidden)                         [T,1536]
latent     = kv_a_proj_with_mqa(hidden)               [T,576]
q_norm     = RMSNorm(q_a, q_weight*sqrt(2))
latent_kv  = RMSNorm(latent[:,:512], kv_weight*sqrt(6))
q          = q_b_proj(q_norm)                         [T,32,192]
gate       = g_proj(hidden)                           [T,4096]
```

NoPE 不删除最后 64 维；它表示关闭旋转而不是删除 checkpoint 参数。Q/K 的最后 64 维保持 projection
原值，等价于 identity RoPE。softmax scale 仍为 `192**-0.5`。

### 2.2 Prefill

```text
kv_b(latent_kv) -> k_nope[128] + v[128]
k = concat(k_nope, latent[:,512:576])
context = causal variable-length attention(q, k, v)
context *= sigmoid(gate)
output = o_proj(context)
```

模型先把压缩的 `[latent_kv512, auxiliary64]` 写入已有 MLA page，再调用 explicit `mla_prefill`。缓存
prefix 存在时优先走 absorbed `mla_extend_with_kvcache`；不复制历史 latent，也不在模型中重建 page
table。

### 2.3 Decode

```text
q_absorb = q_nope @ W_KC                         [B,32,512]
q_cache  = concat(q_absorb, q_auxiliary)         [B,1,32,576]
latent   = paged_attention(q_cache, latent_page) [B,1,32,512]
context  = latent @ W_VC                         [B,32,128]
context *= sigmoid(gate)
output   = o_proj(context)
```

当前 token 的压缩 latent 在 attention 前写入同一 live page；history 和 current token 不使用第二份
cache。`slot_mapping` 只负责写，`page_table + cache_seqlens` 只负责读，两者必须解析到同一 page。

## 3. 复用边界

保留并直接复用：

- `MLAAttnBackend` 的 Prefill/Decode/Prefix dispatch、graph metadata 和 cache-group table；
- `MLATokenToKVPool` 的 `[page,128,1,576]` BF16 latent page；
- `MlaCacheGroupMixin` 的 write-location 计算和 page-0/null-page 契约；
- 统一 `mla_*`、`attn_merge_state` facade 及 kernel registry；
- `PagedAttention` 的模型/backend seam。

新增代码只放在两个现有边界：

1. `tokenspeed-kernel-npu`：TorchNPU attention/merge/value leaf；
2. Lite MLA 子模块：独立 checkpoint 参数的 projection、NoPE、scale、cache write 和 output gate。

不继承或顶层导入 `DeepseekV3AttentionMLA`。该模块会在无加速器 CPU test 的 import 阶段触发统一 kernel
平台 fail-closed；为复用其方法而放宽平台检测会扩大变更范围。Lite 只实现目标 P/D 两条短数据流，
backend/cache 状态机仍完全复用现有实现。

## 4. Ascend kernel 设计

### 4.1 Normalize/project 与 value projection

`mla_normalize_project_query` 注册一个 Ascend Torch baseline：FP32 RMS reduction、BF16 materialization、
KV 原地更新和 `torch.mm`。只声明普通连续输出；不为 Decode 建立先 materialize 再 split 的假融合。

`mla_project_value` 使用 `torch.bmm` 完成 per-head `[512,128]` projection，并在 caller 提供 gate 时对刚
产生、只供 `o_proj` 消费的 context 执行：

```python
context.mul_(torch.sigmoid(gate).to(context.dtype))
```

该 alias 与 FluentLLM 已验收路径一致；不接入已证实在 910B 上慢两倍以上的 gate-only Triton kernel。

### 4.2 Explicit Prefill

`mla_prefill` 直接调用 `torch_npu.npu_fused_infer_attention_score`：

- layout `TND`；causal 时 bool upper-triangle mask 与 `sparse_mode=2`；
- `actual_seq_lengths=cu_seqlens_q[1:]`，KV 同理；
- `softmax_lse_flag=return_lse`；
- 不支持 logit cap 时 fail loud；
- native LSE `[T,H,1]` 规范为统一 API 的自然对数 `[T,H]`。

### 4.3 Absorbed Extend/Decode

统一 API 的 Q/cache 最后一维均为 `[latent512,auxiliary64]`。Ascend leaf 只建立 view，分别传为
`query/key` 和 `query_rope/key_rope`，不 concat/copy：

- Extend：`TND`、paged cache、`sparse_mode=3`、causal mask；
- Decode：`BSH`、page size 128、q_len=1、`sparse_mode=0`；
- value 直接复用 latent key plane；
- native Decode LSE `[B,H,1,1]` 规范为自然对数 `[B,H]`。

Decode graph 沿用现有 MHA 的 `NPUGraph.update(cpu_update_input=...)` 更新
`actual_seq_lengths_kv`，不把 live length 固化为 capture 常量。

### 4.4 Partial-state merge

两段 attention 不能平均。Ascend `attn_merge_state` 使用 native `npu_attention_update` 合并 output，输入
LSE 为自然对数；native 第二返回值为 `None`，因此统一 API 的 merged LSE 由
`torch.logaddexp(lse_a,lse_b)` 产生，以支持三个及以上 prefix chunk 的迭代合并。

`inplace=true` 时结果写回 `out_a/lse_a`；空段的 `-inf` LSE 必须保持另一段结果，不产生 NaN。非默认
LSE scale 明确拒绝，不静默混用 log2 与自然对数。

## 5. NPU0 discovery

目标资源的 NPU0、CANN 9.0 / TorchNPU 2.9 上，真实 Lite shape 得到：

| case | output max abs | output rel-L2 | LSE max abs | finite |
|---|---:|---:|---:|---|
| causal Prefill T4/H32/D192/V128 | 0.015625 | 0.001121 | 4.77e-7 | 是 |
| paged Decode BS1/H32/R512 | 0.015625 | 0.001202 | 4.77e-7 | 是 |
| paged Decode BS2/H32/R512 | 0.015625 | 0.001378 | 1.43e-6 | 是 |
| 两段 output merge | 0.007683 | 0.002508 | native 不返回 | 是 |

Prefill LSE 原始 shape 为 `[T,H,1]`，Decode 为 `[B,H,1,1]`；两者逐元素匹配自然对数 oracle，若按
log2 解释则 rel-L2 为 0.30685，因此对数底没有歧义。

`npu_mla_prolog_v3` 在打开内部 NZ 格式后可以执行 `He=3072,Hcq=1536,Hckv=512,H=32` 并返回 finite
结果。但 FluentLLM 既有独立 env-off/on trace 已证明两侧 kernel/task/tiling 相同，且没有稳定收益；
公开 HW 契约也未声明 `He=3072`。本阶段继续 fail closed，不注册 HW prolog、不增加 launcher env。

## 6. 测试矩阵

### 6.1 CPU/meta

- Lite post-load 的 `sqrt(2)`/`sqrt(6)` scale 只应用一次，`W_KC/W_VC` shape 与元素映射正确；
- NoPE 不旋转也不删除 auxiliary 64 维；
- Prefill/Decode projection、value projection、gate on/off 对独立 Torch oracle；
- current token 在 attention 前写入，history-only 与空 history 不越界；
- page boundary 127/128/129、page0/null page 和 neighbor page 不被误写；
- 非目标 mapping、logit cap、错误 dtype/shape 和未准备权重 fail loud。

### 6.2 NPU0 exact source

- normalize/project T1/2/32，校验 KV 原地更新；
- explicit Prefill T1/4/32 和变长 `[1,3]`，causal output+LSE；
- absorbed prefix Extend，prefix 0/1/127/128/129；
- Decode BS1/2，current/history/page boundary，output+LSE；
- value projection gate on/off、context alias 和 gate 不变；
- merge 普通两段、三段迭代、第一/第二段为空、`inplace` on/off；
- Decode BS1/2 NPUGraph capture/replay，更新 Q/cache/page table/sequence length 后结果随输入变化；
- 所有结果 finite，并回归 Phase 2 cache 与 Phase 3/4 KDA focused suites。

## 7. 准入与回退

实现提交要求本地 focused、全量 `pre-commit run --all-files` 和 NPU0 exact-source focused 全部通过。
BF16 attention/value output 对 FP32 oracle使用 `atol=0.02, rtol=0.02`；LSE使用
`atol=2e-5, rtol=2e-5`；merge output使用 `atol=0.02, rtol=0.02`。所有 cache/state/page 邻接测试必须
exact，任何 NaN/Inf、page0 写入或 graph 冻结都拒绝准入。

若 Ascend attention leaf 失败，删除对应 registration 并保留统一 API 的 fail-loud；不复制另一套
backend。若 output gate 性能或 alias 不成立，退回 out-of-place Torch gate；不接入 Triton gate-only。
HW prolog 继续关闭，直到目标 wheel 明确声明 `He=3072` 且独立 trace 证明不同实现和稳定收益。

阶段 5 明确不包含：

- CP8/KVP8 collective、head/history 重排和跨 rank merge；
- P/D cache transfer、Decode graph 服务或 overlap；
- Prefill graph；
- Grouped MoE、OE、decoder/residual 或完整模型 forward；
- projection packing、Weight-NZ 或新 AscendC MLA kernel。
