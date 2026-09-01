# Lite NPU 阶段 6B：Grouped MoE Ascend Leaf 设计

## 1. 目标与边界

本子阶段只实现并验证 Lite Grouped MoE 的两个单卡 Ascend leaf：

1. `softmax + selection-only correction bias + TopK`；
2. precomputed-TopK local expert：routing、两次 GMM、SwiGLU 和 finalize。

目标形状为 `H=768,I=512,K=12`，覆盖独立 group 的 48 local experts 和 flattened executor 的
192 local experts。Decode BS1/BS2 必须支持固定 shape NPUGraph capture/replay。

本阶段不增加 EP collective，不接 Lite 完整 forward，不选择 A/B/C placement，也不启动服务。P 的
AllGather/ReduceScatter、D 的 AllReduce、NPU8 board 和 production wiring 分别留在 6C/6D。

## 2. 最小复用边界

### 2.1 直接复用

| 能力 | 复用对象 | 决策 |
| --- | --- | --- |
| expert 公共入口 | `moe_plan/moe_process_weights/moe_apply` | 注册 Ascend solution，不增加 Lite 专用 facade |
| canonical 权重 | `w13=[E,2I,H]`、`w2=[E,H,I]` | checkpoint loader 不变，post-load 只转换 GMM 视图 |
| kernel 选择 | 既有 registry、format signature 和 traits | NPU runtime 仍只依赖统一 kernel package |
| graph | 既有 `torch.npu.NPUGraph` execution wrapper | leaf 不创建第二套 graph runner |
| oracle | Phase 6A Torch route/expert 数学 | NPU leaf 与同一 FP32 accumulation oracle 比较 |

### 2.2 唯一新增的公共语义

现有 `moe_softmax_topk` 没有 correction bias，`moe_sigmoid_bias_topk` 又使用错误的 sigmoid 归一化。
因此在通用 MoE API 中新增一个窄入口：

```text
moe_softmax_bias_topk(
  router_logits, correction_bias, topk,
  routed_scaling_factor, solution=None
) -> (topk_weights, topk_ids)
```

它固定执行：

```text
prob    = softmax(router_logits.float(), dim=-1)
ids     = topk(prob + correction_bias, K)
weights = gather(prob, ids) * routed_scaling_factor
```

correction bias 只参与 ID 选择；输出权重不做 TopK 内 renorm。Torch reference 和 Ascend leaf 共用该
ABI，已有普通 softmax/sigmoid TopK 不改签名。

## 3. 目标 wheel API 审计

所有候选均已用目标 torch-npu wheel 的本地文档和实际运行时核验。

| API | 本阶段用途 | 关键约束与结论 |
| --- | --- | --- |
| `npu_moe_gating_top_k` | fused biased softmax TopK | 2D ND，expert `<=2048`，支持 FP16/BF16/FP32、bias、graph；目标 `E416/K12` 合法 |
| `npu_moe_gating_top_k_softmax` | 不采用 | 支持普通 softmax TopK，但没有 correction-bias 输入 |
| `npu_moe_init_routing_v2` | local route permutation | `expert_num<=10240`，count mode、active range、graph 均支持；只有有效前缀有定义 |
| `npu_grouped_matmul` | W13/W2 | tensor group list、count mode、单 tensor 最多 1024 groups；目标 48/192 groups 合法 |
| `npu_swiglu` | fused activation | BF16 与最后维 split 合法；目标输出 `[routes,512]` |
| `npu_moe_finalize_routing` | token-major combine | exact CANN9 的 row-index type 0 必须走下述 mode3 扩展契约 |
| `npu_format_cast` | Weight-NZ 候选 | 当前环境请求 format29 后仍返回 base format2，不能构成真实 NZ A/B |

`npu_moe_distribute_dispatch_v2/combine_v2` 不属于本阶段，也不适配 192 experts/rank 的目标约束。

## 4. Ascend TopK Leaf

Ascend registration 调用：

```text
npu_moe_gating_top_k(
  x=router_logits,
  k=topk,
  bias=correction_bias,
  k_group=1,
  group_count=1,
  group_select_mode=0,
  renorm=0,
  norm_type=0,
  out_flag=False,
  routed_scaling_factor=scale,
  eps=1e-20,
)
```

只返回前两个结果，并将 ID 统一为 INT32、weight 统一为 FP32。T0 在公共 facade 返回空 tensor，不调用
vendor op。Lite 仍对四组各调用一次；本阶段不开发跨四组 TopK kernel。

目标环境已验证 T1/T2/T32 非 tie 输入 ID 与 Torch exact，weight max abs `<=2.98e-8`；BS1/BS2 graph
replay 更新 logits/bias 后仍保持同一门槛。

## 5. Local Expert Leaf

### 5.1 Plan 与 metadata

Ascend `moe.apply` registration 固定 traits：

- `weight_dtype=unquant`；
- `activation in {silu,swiglu}`；
- `routing_mode=precomputed_topk`；
- 支持 EP-owned local weights，但不拥有 all-to-all collective；
- 不支持 deferred finalize、expert bias 或 joint shared projection；
- `I` 至少满足 GMM/SwiGLU 的 32 对齐。

leaf 从标准 MoE owner 读取 `num_experts/num_local_experts/ep_rank`，按 contiguous ownership 得到：

```text
lo = ep_rank * num_local_experts
hi = lo + num_local_experts
```

identity、非本 rank 和其它越界 ID 不进入 local GMM，由 routing metadata 表示为无贡献 slot。

### 5.2 Post-load 权重

checkpoint 加载继续写 canonical 参数。Ascend preprocessor 在 load 完成后一次性转换：

```text
w13: [E,2I,H] -> contiguous [E,H,2I]
w2:  [E,H,I]  -> contiguous [E,I,H]
```

本阶段不调用 `npu_format_cast(...,29)`。当前目标环境 cast 前后实际 format 都为 2，加入该调用只会产生
warning，不会获得 NZ 收益。未来只有在目标环境真实返回 format29，且独立数值/性能 A/B 通过时，才在
同一个 preprocessor 增加一行 format cast。

### 5.3 Fixed-capacity 数据流

设 `R=T*K`。eager 与 graph 使用同一条固定容量路径：

```text
expanded_x, row_idx, counts = init_routing_v2(..., active_range=[lo,hi])
valid = arange(R, device) < counts.sum()
expanded_x = masked_fill(expanded_x, ~valid, 0)

gmm1 = grouped_matmul(expanded_x, w13, counts)
gmm1 = masked_fill(gmm1, ~valid, 0)
act  = swiglu(gmm1)
act  = masked_fill(act, ~valid, 0)
gmm2 = grouped_matmul(act, w2, counts)
gmm2 = masked_fill(gmm2, ~valid, 0)

out = finalize(
  expanded_rows=gmm2.unsqueeze(1),
  row_idx=row_idx,
  scales=topk_weights.to(BF16),
  expert_ids=topk_ids,
  drop_pad_mode=3,
)
```

`counts.sum()` 始终留在设备上；没有 `.item()`、D2H active slice 或按 route 数重新分配 tensor。mask 在
routing、GMM1、SwiGLU、GMM2 四个 consumer seam 上都执行，避免任何算子读取未初始化尾行。

### 5.4 Finalize 的 exact 契约

目标 wheel 的公开文档只描述 2D `drop_pad_mode=0`，但 exact CANN9 实际提供另一套 row-index type 0
契约：`expanded_rows=[R,1,H] + drop_pad_mode=3`。同一份真实 metadata 的正交上板得到：

- Torch token-major gather/mask/FP32 sum 对逐 expert oracle正确；
- mode3 与 Torch combine bitwise 一致；
- 2D mode0 即使全部 row index 非负，仍有约 `1.285` rel-L2；
- mode0/2 传 3D 输入会被 tiling 明确拒绝。

因此 mode3 是本阶段的 exact-wheel ABI，不做 silent fallback，也不把负 sentinel clamp 成另一个布局。

### 5.5 空 route 与 T0

- T0 在 leaf 入口直接返回 `empty_like(x)`；
- T>0 且本 rank active route 为 0 时仍执行相同 native op 序列，masked buffer 和输出严格为零；
- 6C 分布式路径仍必须执行 collective，不能根据 local active count early return。

## 6. Graph 与数值门槛

NPU0 discovery 已证明：

- 48 local experts 的 all/partial/zero route 均 finite，zero route 严格为零；
- BS1/BS2 capture/replay 更新 input、ID 和 weight 后输出同步变化；
- 192 local experts、全局 1536 experts、T8/K12 可执行且 finite；
- GMM/SwiGLU/GMM 对 FP32-accumulation oracle 沿用总体设计门槛：`atol=2e-2,rtol=2e-2`、
  rel-L2 `<=5e-3`；finalize 必须与独立 token-major combine 一致。

正式 oracle 必须以 FP32 累加 routes 后再转回 BF16。逐 route BF16 `index_add` 会因累计顺序制造约
`5e-3` 的伪差异，不能用来判断 fused leaf 精度。

## 7. 实现文件边界

预计最小改动为：

1. 通用 MoE softmax TopK 文件增加 biased facade/reference；
2. 通用 MoE package 导出新 facade，并 side-effect import Ascend registration；
3. 新增一个通用 Ascend MoE registration 文件；
4. `tokenspeed-kernel-npu` 新增一个 MoE leaf 文件；
5. 增加最小 CPU/registry test 和 NPU0 operator test。

Lite model、checkpoint schema、parallel mapping、cache、PD 和 launcher 本阶段不改。

## 8. 验证矩阵

### 8.1 CPU/registry

- correction bias 只改变 ID，不进入输出 weight；
- routed scale、不 renorm、T0、dtype/shape/trust-boundary validation；
- Ascend TopK registration 的参数完整转发；
- Ascend `moe_plan` 选择、precomputed TopK traits 和非法 deferred/bias 拒绝；
- post-load canonical-to-GMM 转换与单次执行；
- fixed-capacity leaf 对 0/partial/all route 的 mock metadata 契约。

### 8.2 Exact-source NPU0

- TopK：T1/T2/T32/T1024，ID exact、weight `atol=1e-6,rtol=1e-5`；
- local leaf：48/192 experts，0/partial/all local route、duplicate ID、boundary ID；
- GMM stages、SwiGLU、finalize 与完整输出全部 finite；
- BS1/BS2 NPUGraph capture/update/replay，无新 specialization；
- ND 数值和 warm median；NZ 仅记录为当前环境不可形成候选，不伪造性能比较；
- 累计回归 Phase 3--6A focused tests。

## 9. 提交顺序

1. 本设计文档独立提交并推送；
2. facade、Ascend leaf、测试独立提交并推送；
3. exact-source NPU0 结果写入独立验证记录，再提交并推送。

任何 leaf 数值、graph 或 empty-route 门槛失败都阻止进入 6C；不以 Torch fallback 掩盖失败。
