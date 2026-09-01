# Lite NPU 阶段 5：MLA Baseline 与 Output Gate 验证记录

## 1. 验证范围

本记录验证阶段 5 的 Lite NoPE MLA 实现：模型 post-load scale、Q/KV projection、显式 Prefill、
absorbed Prefix Extend/Decode、单份 paged latent cache、value projection、原地 output gate、自然对数
LSE 与 partial-state merge。完整 decoder、CP8/KVP8、PD、服务 graph 和 8P8D 不在本阶段范围内。

提交链为：

| 类型 | 提交 |
| --- | --- |
| 设计 | `1b85539d262e25b6a26a9641be8478471dfa142a` |
| 实现与测试 | `1f9398e7a97c01ef179c6eb196c2087079f3d6dc` |

生产改动限于 Lite MLA 子模块、Ascend attention registry 和一个 TorchNPU MLA leaf 文件。既有
`MLAAttnBackend`、`MLATokenToKVPool`、cache-group metadata、runtime、PD 与 KDA 均未修改。

## 2. Exact-source 与环境

已推送的实现提交通过 `git archive` 生成只含 tracked source 的验证包：

```text
commit:  1f9398e7a97c01ef179c6eb196c2087079f3d6dc
archive: dce6bd09043aa56e5199f60d3649687b43ded6c053b0866ef486b5bfc8705df7
```

目标机收到 archive 后复算相同 SHA-256，再解包到全新目录。验证只使用一张 Ascend 910B NPU，源码
经 `PYTHONPATH` 加载，复用既有隔离测试环境；基础镜像、系统 Python 和 package 均未修改。

## 3. 模型与 cache 接线

实现核对结果：

- `mla_scale_q_lora/kv_lora` 分别把 norm 权重乘 `sqrt(2)`/`sqrt(6)`，重复 post-load 不再缩放；
- `kv_b_proj` 只派生一份 `W_KC[32,128,512]` 和 `W_VC[32,512,128]`，不保留第二套模型权重；
- NoPE 的最后 64 维保持 checkpoint projection 值，不做旋转、不删除；scale 仍为 `192^-0.5`；
- explicit 和 absorbed 两条路径复用现有两只 `PagedAttention`，都绑定 `full_attention` cache group；
- 模型先经 backend 选择 write location，把 `[latent512,aux64]` 写入 live page，再启动 attention；
- cache write 与 attention read 使用同一 page，不创建 history/current 或 transfer staging 副本；
- Prefill 显式产生 K/V；cached Prefix Extend 和 Decode 只 materialize absorbed Q，V 在 attention 后投影；
- output gate 位于 value/context 产生后、`o_proj` 前，并原地乘到唯一 consumer 的 BF16 context；
- CPU/meta 保留独立 Torch 数学路径，NPU 经统一 kernel facade 选择 Ascend registration。

阶段 5 只准入 pure Prefill 和 pure Decode；`MIXED/IDLE` 在完整 decoder 接线前 fail loud，避免把尚未
实现的 role 混合调度误当成可用能力。

## 4. 数值验证

### 4.1 Projection 与 scale

真实 Lite `q_lora=1536,kv_lora=512,q_b_out=6144` shape 的 T1/2/32 均通过。Ascend leaf 的 Q projection
和原地 KV RMSNorm 对独立 FP32-reduction/BF16-materialization oracle 均为 bitwise 一致，所有元素
finite。模型 focused 另行验证 `sqrt(2)`/`sqrt(6)` 只应用一次以及 WKC/WVC 元素映射。

### 4.2 Attention 与 LSE

| case | output max abs | output rel-L2 | LSE max abs | finite |
| --- | ---: | ---: | ---: | --- |
| 变长 Prefill `[1,3]` | `0.015625` | `8.46e-4` | `4.77e-7` | 是 |
| Decode length `1/127/128/129` | `0.0078125` | `8.35e-4` | `9.54e-7` | 是 |
| discovery Prefill T4 | `0.015625` | `1.12e-3` | `4.77e-7` | 是 |
| discovery Decode BS1 | `0.015625` | `1.20e-3` | `4.77e-7` | 是 |
| discovery Decode BS2 | `0.015625` | `1.38e-3` | `1.43e-6` | 是 |

Prefix Extend 另以 `[2,3]` query、`5/129` visible KV 验证跨 128-token page，output/LSE 均满足
`rtol/atol=0.02/0.02` 与 `2e-5/2e-5` 门槛。原始 Prefill LSE 为 `[T,H,1]`，Decode 为
`[B,H,1,1]`；规范化后逐元素匹配自然对数 oracle。

### 4.3 Partial merge

`npu_attention_update` 负责两段 output 合并，统一 leaf 用 `torch.logaddexp` 补回 native 不返回的 merged
LSE。验证覆盖第一段为空、第二段为空且 `inplace=true`、第三段迭代合并：空段 output/LSE 均 exact；
三段中的第二次合并 output 最大差 `0.00781`、rel-L2 `0.00161`，LSE exact，全部 finite。

## 5. Decode NPUGraph

BS2 测试执行 eager warmup、NPUGraph capture、两组 `actual_seq_lengths_kv` 更新和 replay。每次 replay
都对同一 paged-cache oracle，更新 `3/4 -> 5/7` 后 output 随长度变化且误差在 BF16 attention 门槛内。
既有 Ascend MHA BS2 graph length-update 回归同时通过，证明本阶段沿用相同 CPU input update 协议，
没有把 capture 时的 `[1,1]` 固化为真实长度。

cache isolation 覆盖 page boundary 127/128/129、page table 跨页读取和 attention 前写入顺序；未观察到
page0、相邻 page 或 gate tensor被修改。

## 6. 性能记录

exact-source NPU0 使用 10 次 warmup、7 轮 NPU Event 同步计时。性能仅作 leaf 准入证据，不设服务目标。

| leaf | shape | median |
| --- | --- | ---: |
| explicit Prefill | T4/H32/D192/V128 | `111.44 us` |
| explicit Prefill | T32/H32/D192/V128 | `115.47 us` |
| absorbed Decode | BS1/H32/R512 | `143.89 us` |
| absorbed Decode | BS2/H32/R512 | `152.01 us` |

Value projection + output gate 的 A/B 中，baseline 是原统一 fallback 的 FP32 sigmoid/multiply 临时链；
Ascend leaf 保持 Fluent 已验收的 BF16 原地 gate 语义：

| shape | Ascend leaf | baseline | 加速比 |
| --- | ---: | ---: | ---: |
| BS1/H32/R512/V128 | `32.99 us` | `46.74 us` | `1.42x` |
| BS2/H32/R512/V128 | `31.88 us` | `57.37 us` | `1.80x` |

原地 gate 不修改 gate tensor。相对旧 FP32 multiply 后再落 BF16 的路径，max abs 为 `0.25`、rel-L2 为
`0.00284/0.00280`；该差异来自 gate 的既定 BF16 舍入边界，模型 oracle按 Fluent 相同路径验证，而非
把两种舍入顺序误判为 bitwise 等价。

## 7. 回退与门禁

以下 fail-closed 行为均已验证：

- logit cap 非零抛出 `NotImplementedError`；
- 非自然对数 LSE scale 的 merge 抛出 `NotImplementedError`；
- split-query 请求不误选普通 Ascend projection，返回统一 fallback 的 `absorbed_query=None`；
- 六个 Ascend MLA/merge registration 均可发现；
- `npu_mla_prolog_v3` 没有注册到生产 kernel registry，也没有新增 launcher 开关；
- CPU import 不顶层加载 `torch_npu` 或要求加速器的平台 package。

测试汇总：

| 门禁 | 结果 |
| --- | --- |
| 本地 Lite MLA/loader/cache/KDA focused | `23 passed, 27 skipped`（NPU-only skip） |
| exact-source NPU0 + 既有 Ascend MHA/KDA 回归 | `54 passed` |
| 全仓 `pre-commit run --all-files` | 全部通过 |

目标 wheel 未编译 TorchAir和 internal-format 关闭的两条 warning 为既有环境提示；所有相关算子真实执行、
同步、数值与 graph 均已通过。

## 8. 结论

阶段 5 准入通过。Lite MLA 的单-rank NPU baseline 已覆盖真实 projection shape、显式/absorbed P/D、跨页
cache、自然对数 LSE、partial merge、value projection、原地 output gate 和 Decode graph；既有
backend/cache ownership 保持不变，HW prolog 继续 fail closed。

下一阶段进入 Grouped MoE。CP8/KVP8 collective、PD one-copy、完整 decoder、8P8D 服务与 GSM8K 仍按
总体计划在后续独立阶段实现和验证。
