# Lite NPU 阶段 9C.3：Ascend KDA Backend Policy 设计

## 1. 目标与根因

阶段 9C.2 的 exact 8P8D 服务已完整 ready，但第一个真实 Prefill forward 在八个 P rank 上一致请求
`solution="triton"`，随后因 Ascend 没有该 solution 抛出 `NoKernelFoundError`。公开 KDA artifact 已在
package-local 路径加载；失败发生在 capability selector 之前，因而不是 artifact、shape 或算子执行错误。

根因位于共享 `_resolve_kda_backend`：它只把 AMD 识别为 registry-driven 平台，其余平台全部走 NVIDIA
的 `cutedsl_kda -> flashkda -> fla` 策略。Ascend 的 `auto` 因没有两个 CUDA package 被解析成 `fla`，
而公共 KDA API 又把 `fla` 映射成 `triton`，绕开已经注册的 Ascend `public_kda` 和 Torch reference。

本阶段只修正平台策略，不改变 KDA 数学、kernel ABI、模型调用、cache state 或 launcher 参数。

## 2. 唯一选择策略

| 平台 | `--kda-backend` 解析结果 | 后续选择 |
| --- | --- | --- |
| NVIDIA | 保持既有 auto/显式 CUDA policy | CuteDSL、FlashKDA 或 FLA/Triton |
| AMD | `auto` | registry capability/priority |
| Ascend | `auto` | registry capability/priority |

Ascend 与 AMD 一样忽略 NVIDIA 专属的命名 policy，不导入 CUDA-only availability probes。对 Lite
featurewise-beta Prefill，Ascend registry 优先选择 `public_ascend_kda_paged_prefill`；若公开 artifact
缺失、序列太短或 shape 不满足，现有 public adapter 按设计回退 Torch reference。scalar beta 继续由
`torch_ascend_kda_paged_prefill` 处理。

显式 `public_kda` 不新增为用户 CLI 值：production 默认应由 capability/priority 自动选择，不能要求
启动脚本知道 vendor kernel 名称。

## 3. 最小实现与测试

实现只改 `_resolve_kda_backend` 的平台 guard 和说明文字：`current_platform().is_npu` 与 AMD 一同直接
返回 `auto`，其余 NVIDIA 逻辑原样保留。

测试分两层：

1. NPU policy 单测证明 Ascend 的 `auto/fla/flashkda/cutedsl_kda` 都不会触发 CUDA probe，而统一交回
   registry；
2. 既有 Ascend registry board 继续证明 featurewise beta 选 public KDA、scalar beta 选 Torch；随后
   exact 8P8D 服务必须通过首个 Prefill forward、PD transfer 和完整请求矩阵。

提交前运行相关 focused tests 和完整 `pre-commit run --all-files`。实现推送后生成新的 exact archive，
不得通过把 launcher 改成 `--kda-backend` 私有值或移除公开 artifact 来绕过。

## 4. 非目标

- 不修改 NVIDIA/AMD kernel priority；
- 不实现或重编译 KDA kernel；
- 不把所有非 NVIDIA 平台笼统映射为 Ascend；
- 不在本阶段启用 Decode graph、overlap 或 CP/KVP 分布式 page。
