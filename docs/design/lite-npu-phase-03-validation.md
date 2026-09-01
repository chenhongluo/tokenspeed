# Lite NPU 阶段 3：KDA Eager 验证记录

## 1. 验证对象

本记录验证阶段 3 设计中的 Lite featurewise-beta KDA eager baseline。实现提交为 `fb6f38a7`，
生产形状测试补充提交为 `6f562697`。NPU 验证使用后者的 clean exact-source worktree，测试期间只
暴露一张 Ascend NPU，不加载完整 checkpoint，不启动服务。

验证范围包括：

- Lite 两层无 bias beta projection、rank-local backend 参数和 output gate epilogue；
- packed causal-conv 的独立输入/输出 state page；
- scalar 与 featurewise beta 的 kernel registry 隔离；
- packed Prefill、连续 Decode、non-zero history 和 state reuse；
- NaN/Inf 早失败及现有 Lite loader/cache、Kimi 接口回归。

本阶段不验证 graph、speculative verify、融合算子、CP/KVP、PD、8P8D 服务或性能。

## 2. 固定矩阵

CPU 独立 oracle 不调用 production recurrence helper，直接按设计公式逐 token 计算。NPU 主 case
使用 TP8 rank-local 形状：

| 项目 | 值 |
| --- | ---: |
| token 数 | 4 |
| packed request 长度 | `[1,3]` |
| local heads | 4 |
| head dim | 128 |
| packed QKV channel | 1536 |
| conv width/history | 4 / 3 |
| recurrent state | FP32 `[slot,4,128,128]` |
| activation | BF16 |
| initial state | non-zero |

Prefill 使用两个目标 page；连续 Decode 从相同初态开始，分别执行长度 1 和长度 3 的 request。另设
一个带固定 sentinel 的邻接 conv/recurrent page，验证目标 page 更新不污染邻接 page。

## 3. 命令

本地 focused 命令：

```bash
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-kernel-npu/python \
python -m pytest -q \
  test/runtime/test_lite_model_loader.py \
  test/runtime/test_lite_hybrid_cache.py \
  test/runtime/test_lite_kda_eager.py \
  test/runtime/test_kimi_k3_kda_eager_commit.py \
  test/runtime/test_kda_fp8_w8a8.py
```

NPU exact-source 命令额外固定设备：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-kernel-npu/python \
python -m pytest -q test/runtime/test_lite_kda_eager.py
```

提交前门禁：

```bash
pre-commit run --all-files
```

## 4. 结果

| 验证项 | 结果 |
| --- | --- |
| 本地 Phase 3 + 相关回归 | `19 passed, 19 skipped` |
| NPU Phase 3 focused | `12 passed` |
| NPU Lite/Kimi 相关回归 | `19 passed, 7 skipped` |
| 全仓 pre-commit | 全部通过 |
| exact-source SHA / clean 状态 | `6f562697` / clean |

本地 skip 均为仓库既有的 accelerator-only case；NPU 相关回归的 7 个 skip 为 CUDA/FP8 专属 case，
没有跳过本阶段新增的三个 NPU case。

数值结果：

- scalar 和 featurewise beta 均通过 Ascend registry 的 Torch reference；
- 生产形状 packed Prefill 的 conv/output/final-state 对独立 FP32 oracle 在 `atol=rtol=3e-2` 内；
- 同一输入连续 Decode 与 packed Prefill 的逐 token 输出、最终 conv state 和 recurrent state 在
  `atol=rtol=0` 下逐元素一致；
- sentinel 邻接 page 未变化，所有输出和目标 state 均 finite；
- 错误 beta width 与 Q/beta/gate 参数 NaN/Inf 在 reference 边界直接失败。

目标环境提示 TorchAir 未编译，并提示 reference 输出采用基础 tensor format；两项均未改变 eager
数值结果。前者不作为 graph 结论，后者也不作为性能结论。

## 5. 准入结论

阶段 3 的 eager 正确性门禁通过。Lite featurewise-beta 沿用现有 KDA cache/backend，没有新增第二份
state 或 Lite 专用 cache；Kimi scalar ABI 保持可选且独立。下一阶段可在相同 attention ABI 下用公开
causal-conv、Prefill KDA 和 recurrent KDA 融合实现替换 Torch reference，并保留本记录的 oracle 与
Prefill/Decode state 等价测试作为回归门禁。
