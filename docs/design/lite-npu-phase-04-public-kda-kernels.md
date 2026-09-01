# Lite NPU 阶段 4：公开 KDA 融合算子

## 1. 目的

本阶段在阶段 3 的 FP32 recurrent-state eager 基线上接入 910B 可用的公开 KDA 融合算子，减少
packed causal-conv、Decode recurrence 和 Prefill chunk scan 的 Python/Torch 小算子数量。模型层、
hybrid cache、page ownership 和统一 attention API 保持不变；任一融合项不可用时，只回退该项，
其余已准入项继续生效。

本阶段只做单卡、rank-local 算子接入和 A/B。Decode graph、speculative replay、PD one-copy、
CP/KVP、8P8D 服务和 projection packing 仍由后续阶段负责。

## 2. 审计结论

目标软件栈为 PyTorch 2.9、torch-npu 2.9 和 CANN 9.0。审计同时核对了当前 wheel 的本地
docstring、实际可调用行为和固定公开源码 `d543ccee0a1ff677165777e3defafd42b35e83ef`。

| 能力 | 当前 wheel / 仓内事实 | 910B 实测或源码约束 | 决策 |
| --- | --- | --- | --- |
| packed width-4 causal-conv | `npu_fused_causal_conv1d` 文档固定 width 3 | width 4 调用缺少对应 `libopapi` 实现 | 使用公开 AscendC `CausalConv1d` |
| Decode recurrence | `npu_recurrent_gated_delta_rule` 接受 featurewise `gk` | `gk` 衰减生效，但 state 只接受 BF16；FP32 明确失败 | 不改变 FP32 cache，使用公开 `RecurrentKda` |
| Prefill chunk scan | wheel 无 KDA chunk API | 公开 `KdaGateCumsum` + `ChunkKdaFwd` 支持 910B 和 FP32 state | 使用两项公开 AscendC 算子 |
| output epilogue | 仓内已有 `rmsnorm_gated_sigmoid` | `[T,4,128]` BF16 在 Triton-Ascend 上 finite，误差小于阶段 3 阈值 | 直接复用，不复制公开 FLA kernel |
| 普通 RMSNorm | wheel 有 `npu_rms_norm` | 不包含 sigmoid output gate | 仅保留为非融合 fallback |

wheel 文档仍写明 recurrent `gk` 暂不支持，但同一环境的数值消融证明它会执行
`state *= exp(gk)` 的逐 key-channel 衰减。该能力仍不准入生产，因为它只支持 BF16 state，而阶段 3
已经冻结 FP32 recurrent state。不能为复用一个现成 API 改变模型数值和 cache 容量口径。

公开算子许可按实际源文件而不是仓库顶层许可记录：`CausalConv1d` 使用 CANN Open Software License
Agreement Version 2.0，`RecurrentKda` 使用 Apache-2.0，`KdaGateCumsum` 和 `ChunkKdaFwd` 使用
BSD-3-Clause。阶段 0 清单中 causal-conv 的路径及两项 Prefill 算子的许可已随本文纠正。

## 3. 复用边界

### 3.1 不新增的部分

- 不新增 Lite 专用 backend、cache pool、request-to-slot map 或第二份 persistent state；
- 不复制模型 projection、safe-gate、page planning 或 Prefill metadata；
- 不移植公开 fused norm-gate，因为仓内已有同数学、同为单 launch 的 Triton 实现；
- 不依赖 vLLM-Ascend Python runtime，公开仓只作为固定的 build-time operator source；
- 不把环境变量、绝对 artifact 路径或机器信息写入模型配置。

### 3.2 新增的最小边界

`tokenspeed-kernel-npu` 增加一个可选 KDA OPP artifact 和薄 Python adapter；`tokenspeed-kernel` 只增加
统一 kernel 注册。runtime 继续调用阶段 3 已有的 causal-conv override、`kda_paged_prefill` 和
`kda_paged_decode`。

```text
Lite model
    -> KdaAttnBackend
        -> tokenspeed-kernel registry
            -> public Ascend adapter (artifact present and shape supported)
            -> Torch reference       (otherwise)
```

artifact 不存在、SHA 不匹配、schema 缺失或 shape/dtype 不支持时，注册层不暴露该实现。运行中真正的
算子错误必须抛出，不能捕获后静默重算，以免重复修改 state。

## 4. 算子数据流

### 4.1 Packed causal-conv

阶段 3 已把 Q/K/V 投影拼成 `[T,3*H_local*D]`，目标 TP8 rank-local shape 为 `[T,1536]`。公开算子
一次完成：

```text
packed projection
    -> width-4 depthwise causal convolution
    -> SiLU
    -> packed output

raw projection window
    -> conv state 原地发布
```

公开 ABI 的权重是 `[4,1536]`，checkpoint 参数仍保持 `[1536,4]`。权重在加载完成后只转换一次，
不得在 forward 中反复 transpose/contiguous。输入、权重和 state 均为 BF16；channel 对齐满足公开
kernel 要求。

当前 backend 同时提供 `state_in` 和 `state_out`，公开算子只原地更新一个 slot。adapter 在调用前把
有效的 `state_in` 复制到 `state_out`，把 `state_out` 作为公开算子的 cache index；fresh request 通过
`initial_state_mode` 忽略旧内容。该 copy 是保持阶段 3 page 语义所需的临时兼容边界，后续 live-state
阶段使两个 index 相同后自然消失，不引入另一份 cache。

### 4.2 Lite featurewise-beta prepare

该部分尚无公开融合实现，继续使用最小 Torch/Triton 前处理：

```text
beta_scale = sqrt(sigmoid(beta_logits) + 1e-10)
q_hat = l2_normalize(q)
k_hat = l2_normalize(k) * beta_scale
v_hat = v * beta_scale
beta_for_core = ones([T,H_local])
```

顺序不能改成“先缩放 K，再让 recurrence kernel 做 L2Norm”，否则 L2Norm 会消掉 beta scale。因此
Lite Decode 调用公开 `RecurrentKda` 时固定 `use_qk_l2norm_in_kernel=false`；Q/K normalization 保持在
kernel 外。projection packing 和将 beta prepare 合入新 kernel 属于后续额外优化，不夹带进本阶段。

### 4.3 Decode recurrent KDA

公开 `RecurrentKda` 在一个 AICore kernel 内执行：

```text
raw forget gate + A_log + dt_bias
    -> safe gate
    -> featurewise state decay
    -> delta update
    -> state readout
    -> FP32 state 原地发布
```

Lite adapter 传入已准备的 `q_hat/k_hat/v_hat` 和全 1 scalar beta，并固定：

```text
use_qk_l2norm_in_kernel = false
use_gate_in_kernel = true
use_beta_sigmoid_in_kernel = false
safe_gate = true
state_v_first = false
```

`state_v_first=false` 使公开 kernel 直接消费 TokenSpeed NPU 的 key-major
`[slots,H_local,D,D]`，不创建 transpose state。输入 Q/K/V 为 BF16，state 保持 FP32。

公开 recurrent ABI 只有一个原地 state index。与 causal-conv 相同，当前 page planner 的
`read_indices != write_indices` 时先复制目标 row，再以 `write_indices` 调用。padding row 不能被读写；
adapter 必须使用现有 pad mask。后续 live-state/graph 阶段再移除这次兼容 copy，本阶段不提前修改
cache ownership。

### 4.4 Prefill gate cumsum 与 chunk KDA

Prefill 继续使用阶段 3 已提供的 host `cu_seqlens_cpu`，不从 device tensor 做同步 D2H：

```text
q/k L2Norm + Lite beta prepare
    -> KdaGateCumsum(raw_gate, A_log, dt_bias, safe_gate)
    -> ChunkKdaFwd(q_hat, k_hat, v_hat, beta=1, initial_state)
    -> packed output + FP32 final state
```

首个准入配置固定 chunk size 64、BSND、`H_local=4`、`D=128`。`cu_seqlens` 和 chunk index 在 host
metadata 中一次构造；公开 chunk kernel 的 initial/final state 都是 key-major FP32。backend 继续负责
把 final state 发布到既有 `state_out` page。

### 4.5 Output epilogue

仓内 `rmsnorm_gated_sigmoid` 已一次完成 per-head RMSNorm、weight、sigmoid gate 和 multiply：

```text
out[T,H,D] -> RMSNorm(D) * weight[D] * sigmoid(output_gate[T,H,D])
```

阶段 4 只补 NPU focused test、A/B 和后续 graph 回归，不再维护第二份来源不同但数学相同的 kernel。
模型侧阶段 3 的 Torch 写法仍是独立 oracle。

## 5. Build 与加载

### 5.1 固定来源

构建脚本只接受 lock 中的完整 Git SHA 和 `ascend910b` SoC。公开 OPP roots 为：

```text
causal_conv1d
recurrent_kda
kda_gate_cumsum
chunk_kda_fwd
```

`ChunkKdaFwd` 的传递依赖由 lock 显式列出。CATLASS 也固定完整 SHA。构建结果包含 custom OPP 和一个
TokenSpeed 命名空间的 Torch binding；不导入上游 Python package。

### 5.2 Source checkout 使用方式

现有 Ascend 安装脚本增加可重复的 KDA build 步骤。它在 build cache 中取得固定公开源码，编译后把
artifact 安装到 `tokenspeed-kernel-npu` 可解析的相对目录。用户只需 TokenSpeed checkout 和该安装
命令，不需单独安装 vLLM-Ascend。

Python loader 在首次注册前：

1. 校验 artifact、公开 SHA/SoC metadata 和所需 Torch schemas；
2. 把 package-local OPP 目录前置到 `ASCEND_CUSTOM_OPP_PATH`；
3. 通过 `torch.ops.load_library` 加载 binding；
4. 分别暴露四个 op 的可用状态。

缺一个 schema 只禁用依赖它的注册。例如没有 chunk op 时，Decode recurrent 仍可使用；不能用一个
全局布尔把全部优化一起关闭。

### 5.3 许可与发布

`THIRDPARTYNOTICES` 记录 CANN OSL 2.0、Apache-2.0、BSD-3-Clause 和 CATLASS 对应来源。仓库不提交
目标机生成的 `.so`、OPP installer 或 build directory；二进制由安装脚本在目标 CANN/PyTorch ABI 下
生成。

## 6. Kernel selection 与回退

| 注册项 | 必需 trait | 不满足时 |
| --- | --- | --- |
| packed causal-conv | NPU、BF16、width 4、channel 1536、indexed state | 阶段 3 Torch conv |
| Decode KDA | featurewise beta、FP32 key-major state、D=128、single-token segments | 阶段 3 Torch recurrence |
| Prefill KDA | featurewise beta、BF16 activation、FP32 state、chunk 64 | 阶段 3 Torch Prefill |
| output epilogue | BF16、H=4、D=128、contiguous | 模型侧 RMSNorm + sigmoid + multiply |

公开实现优先级为 `PERFORMANT`，Torch reference 保持 `REFERENCE`。显式 `solution="torch"` 必须始终
可用，供 A/B、精度诊断和单项回退。无 artifact 时 registry 导入成功且只包含 reference，不能让
TokenSpeed 的非 NPU 环境依赖 CANN 文件。

## 7. 实现与提交顺序

每一步都从本文拆出独立实现、测试和验证记录，并在前一步已推送后开始：

1. **4A Build/ABI**：lock、构建脚本、package-local loader、Torch schemas、许可和 fail-closed UT；
2. **4B Causal-conv**：packed weight preprocess、read/write page adapter、4-token NPU board 和 A/B；
3. **4C Decode recurrent**：Lite prepare、FP32 key-major in-place kernel、变长 Decode 和 A/B；
4. **4D Prefill chunk**：gate cumsum、chunk64 scan、packed variable-length Prefill 和 A/B；
5. **4E Epilogue reuse**：只补现有 Triton kernel 的 NPU production-shape/graph-ready 回归；
6. **4F Validation**：exact-source 累积 focused suite、逐项 off/on 结果和阶段验证记录。

一个提交不得同时启用后续项。4B 失败不阻塞 4C/4D；4D 失败也不能关闭已通过的 Decode。

## 8. 验证矩阵

### 8.1 无 NPU测试

- lock schema、完整 SHA、依赖闭包和许可字段；
- artifact 缺失、错误 SoC、错误 SHA、缺 schema 时逐项 fail closed；
- registry 对 scalar/featurewise beta、FP32/BF16 state、正确/错误 shape 的选择；
- fake op 验证参数顺序、beta 只准备一次、key-major layout 和 page 发布；
- CPU oracle 继续覆盖 4-token、`[1,3]`、non-zero history、`state_in != state_out` 和 neighbor isolation。

### 8.2 910B focused

只使用一张 NPU，固定 TP8 rank-local `H=4,D=128,packed_channels=1536`：

- causal-conv：4-token、`[1,3]`、fresh/resumed、read/write 不同、邻页 sentinel；
- Decode：BS1/2/8、连续 4 token、raw safe gate、featurewise beta、FP32 state；
- Prefill：`[4]`、`[1,3]`、`[2,0,2]`、non-zero initial state；
- Prefill 与逐 token Decode 的 output/final-state 对齐；
- output epilogue：`[4,4,128]` 与独立 FP32 oracle 对齐；
- 每项均检查 finite、目标 page 更新和非目标 page 不变。

### 8.3 A/B 与准入

每项先 warmup，再交替测 reference/fused，记录 median、P90、workspace 和峰值 HBM。精度阈值沿用
阶段 3：BF16 output `atol=rtol=3e-2`，FP32 final state `atol=rtol=2e-2`；Prefill/Decode state
fingerprint 必须在同一实现内一致。

融合项必须同时满足：

- exact-source focused tests 和全仓 pre-commit 通过；
- 公开来源、许可、编译 ABI 和目标 SoC 可追溯；
- target shape 无 NaN/Inf、无 page 污染、无重复 beta/gate 变换；
- target P/D shape 的 median 不慢于该项 reference；
- 禁用该项后输出回到阶段 3 路径，其余项不受影响。

NPUGraph capture/replay 是后续 graph 阶段的最终门禁；4A binding 先提供 Meta schema，4B--4E 只做
最小 capture smoke，不在本阶段改变服务 graph 配置。

## 9. 非目标

- 不把 FP32 KDA state 改成 BF16 来使用 wheel 原生 recurrent；
- 不在 forward 中 JIT 构建 OPP 或下载源码；
- 不复制已有 output epilogue 或写 Lite 专用 cache manager；
- 不在本阶段实现 projection merge、beta-prepare 新 kernel、Decode graph 或 PD live-state；
- 不把单卡算子结果外推成 8P8D 服务吞吐结论。
