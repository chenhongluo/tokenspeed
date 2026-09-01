# Lite NPU 阶段 4E：KDA Output Epilogue 验证记录

## 1. 验证范围

本记录验证阶段 4E 的生产实现：Lite KDA 在 backend 返回 core output 后，对 CUDA/NPU 复用仓库已有的
`rmsnorm_gated_sigmoid` Triton kernel；CPU/meta 保留 Torch reference；Prefill 与 Decode 继续共享
模型侧 epilogue，随后执行原有 `o_proj`。

本阶段提交链为：

| 类型 | 提交 |
| --- | --- |
| 设计 | `edac5b1847dc5e56616334552617d72d83e5b485` |
| 实现与测试 | `b8f824b715666ff0c65d4ffef3028e19c3424e37` |

没有修改共享 Triton kernel、KDA backend、kernel registry、Phase 4A artifact、cache、PD 或 runtime。

## 2. Exact-source 与环境

实现提交通过 `git archive` 生成只含 tracked source 的验证包：

```text
commit:  b8f824b715666ff0c65d4ffef3028e19c3424e37
archive: d312e3bc05ccb358c914dff9c3e02392c934f5e4b96bc35ddfdb463b4f4e409d
```

目标机收到 archive 后先复算 SHA-256，再解包到全新目录。Phase 4B--4D 回归所需的 package-local
公开 KDA artifact 从上一阶段 exact artifact 复制；其 manifest v2 会在首次 device 初始化前重新校验
binding、op-api 与 op-tiling digest。阶段 4E epilogue 本身不依赖该 artifact。

验证只使用一张 Ascend 910B NPU。源码通过 `PYTHONPATH` 加载，基础镜像和系统 Python 未修改；测试
使用既有隔离环境。外部 `ASCEND_CUSTOM_OPP_PATH` 和 artifact override 均未设置。

## 3. 实现核对

生产 diff 保持一个调用点：

```text
LiteKDAParameters.forward
  -> ctx.attn_backend.forward(..., output_gate=None)
  -> accelerator?
       yes: flatten core -> rmsnorm_gated_sigmoid
       no:  existing Torch per-head RMSNorm + sigmoid gate
  -> o_proj
```

核对结果：

- NPU/CUDA 只执行融合路径，不再生成 norm 和 sigmoid 的两个 BF16 中间 tensor；
- CPU/meta import Lite model 时不导入要求加速器的平台 package，reference forward 可继续运行；
- accelerator 分支内按需导入函数，模块加载后由 Python import cache 复用；
- `core_output` 在 kernel 前变为连续 `[T,H*D]`；
- gate 保留原 stride，支持 future packed projection 的行跨步 view；
- output gate 仍不传入 backend，避免 Prefill/Decode 形成两套 epilogue；
- `o_proj` 参数、shape、dtype 和调用顺序未变化。

## 4. 数值与模型接线

NPU focused 覆盖 Lite TP8 rank-local `H=4,D=128`，T1/2/8/32/256/1024。输入 core 为连续 BF16，
gate 是 `[T,704]` packed tensor 中 `[64:576]` 的 view，其 stride 为 `[704,1]`。

每个 shape 同时检查：

- output shape/dtype 不变；
- 所有元素 finite；
- 对一次 FP32 RMSNorm/weight/sigmoid 语义，`max_abs <= 0.004`、`rel_L2 <= 1e-4`；
- 行跨步 gate 被正确读取，没有无条件 contiguous copy；
- NPU model-forward 使用 identity gate/o-proj 构造，最终输出与同一独立 oracle 对齐；
- backend 收到的 `output_gate` 仍为 `None`，证明融合位于统一模型侧 seam。

全部用例通过。阶段设计中的独立 discovery 还确认：相对旧的两次 BF16 中间舍入路径，最大绝对差不
超过 `0.015625`、rel-L2 为 `0.00315--0.00338`；相对一次 FP32 数学语义，rel-L2 不超过
`2.10e-5`。该差异来自删除旧路径的中间 BF16 舍入，不是 norm 轴、gate 或权重语义变化。

## 5. Decode NPUGraph

BS1/2 分别执行：

1. eager warmup；
2. NPUGraph capture；
3. 第一次 replay 并对 oracle；
4. 原地更新 core、packed gate 和 norm weight；
5. 第二次 replay 并重新对 oracle；
6. 断言两次 graph output 不同且全部 finite。

两种 batch 均通过 `max_abs <= 0.004`、`rtol <= 1e-4`，证明 kernel 可被当前 Decode graph 捕获，且
replay 没有冻结 capture 时的输入值。device 类型分支在 capture 外由固定 tensor device 决定，不产生
host sync 或新 specialization。

## 6. 性能 A/B

exact-source NPU0 使用相同 BF16 core/gate/weight，10 次 warmup 后同步计时。baseline 是阶段 4E 前的
手工链：FP32 RMS、weight 后转 BF16、sigmoid 后转 BF16、BF16 逐元素乘。

| tokens | fused | baseline | 加速比 |
| ---: | ---: | ---: | ---: |
| 1 | `44.49 us` | `111.40 us` | `2.50x` |
| 2 | `41.77 us` | `111.82 us` | `2.68x` |
| 8 | `41.69 us` | `107.61 us` | `2.58x` |
| 32 | `40.26 us` | `101.60 us` | `2.52x` |
| 256 | `40.78 us` | `103.44 us` | `2.54x` |
| 1024 | `77.37 us` | `110.09 us` | `1.42x` |

所有验证 shape 都更快，因此 accelerator 路径不需要 token 阈值或运行时开关。该结果只衡量单层
epilogue，不外推完整模型 TPOT/Prefill throughput；完整 8P8D 性能仍按后续服务阶段统一验收。

## 7. 测试汇总

| 门禁 | 结果 |
| --- | --- |
| 本地 Lite KDA/cache/loader focused | `20 passed, 20 skipped`（NPU-only skip） |
| exact-source Lite model/epilogue/graph NPU0 | `21 passed` |
| exact-source Phase 4B--4D KDA NPU0 回归 | `25 passed` |
| 全仓 `pre-commit run --all-files` | 全部通过 |

三个 `tokenspeed-kernel-npu` KDA 回归文件需要先加载统一 `tokenspeed_kernel` package，再交给 pytest；
直接以 NPU adapter test 为第一个 import 会触发既有的 attention/NPU adapter 循环导入。按生产入口的
package 初始化顺序预加载后 25/25 通过。本阶段没有修改该包初始化协议，也没有把测试启动顺序问题混入
output epilogue 实现。

两类 warning 均为目标环境既有提示：torch-npu wheel 未编译 TorchAir，以及 `empty_like` 在当前
`allow_internal_format=False` 下创建 base-format tensor。它们不改变 kernel 执行、数值或 graph replay。

## 8. 结论与回退

阶段 4E 准入通过。Lite KDA 的 Prefill/Decode output epilogue 已复用仓库现有融合 kernel，真实 TP8
shape、行跨步 gate、模型接线、Decode graph、输入更新、finite、数值和性能门槛全部通过；CPU/reference
行为保持不变。

若后续完整模型出现未覆盖问题，回退只需恢复 `LiteKDAParameters.forward` 的旧手工 epilogue；共享
kernel、KDA backend、Phase 4A artifact、cache 与 PD 均无需变化。

阶段 4 至此完成。下一阶段进入 MLA NPU baseline 与 output gate，先独立审计 TokenSpeed 现有 latent
attention、LSE merge、in-place gate 和目标 `He=3072` 的 HW prolog 支持边界，再提交阶段 5 设计。
