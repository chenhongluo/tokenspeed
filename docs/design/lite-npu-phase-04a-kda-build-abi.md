# Lite NPU 阶段 4A：公开 KDA 算子构建与 ABI

## 1. 目标

本步骤只建立公开 KDA AscendC 算子的可复现构建、package-local artifact、Torch schema 和逐项
fail-closed 探测。它不注册高优先级 kernel，不改变阶段 3 的 Torch reference，也不执行真实 KDA
数值路径；causal-conv、Decode、Prefill 和 epilogue 分别在后续步骤准入。

完成后，安装者可从 TokenSpeed checkout 运行一条独立安装命令生成目标 CANN/PyTorch ABI 下的
artifact。无 artifact、版本不匹配或只缺部分 schema 时，非 NPU import 和阶段 3 reference 仍正常。

## 2. 为什么需要独立步骤

公开仓提供 AscendC OPP 和面向 Kimi K3 的 Torch adapter，但不能直接作为 TokenSpeed runtime 依赖：

- TokenSpeed 需要固定公开源码和 CATLASS revision，而不是跟随可变分支；
- 公开 Kimi adapter 固定 recurrent state 为 V-major `[slot,HV,V,K]`，TokenSpeed 阶段 3 已冻结为
  K-major `[slot,HV,K,V]`；
- CANN custom OPP 必须先安装 vendor tree，再由 Torch binding 调用对应 aclnn API；
- Qwen 等普通 NPU 模型不需要这些算子，不能让基础 NPU 安装强制下载和编译额外源码；
- runtime 不应导入 vLLM-Ascend Python package。

因此复用公开 operator source 和 build system，仅维护 TokenSpeed 自己的薄 ABI 与 loader。

## 3. 固定来源与依赖闭包

版本锁放在 `tokenspeed-kernel-npu/thirdparty/public_kda_ops.lock.json`，只包含公开信息：

```text
vLLM-Ascend revision: d543ccee0a1ff677165777e3defafd42b35e83ef
CATLASS revision:     41bf90da655bba3c66d0acd7e00abe33960ecfd6
SoC:                  ascend910b
```

root operator 与传递依赖固定为：

```text
causal_conv1d
recurrent_kda
kda_gate_cumsum
chunk_kda_fwd
    -> chunk_gated_delta_rule_fwd_h
    -> kda_layout_swap12
```

构建器拒绝短 SHA、未知 root、缺 dependency entry、依赖环、非 `ascend910b` SoC、source HEAD 或
CATLASS HEAD 不一致。它调用公开仓已有的 `csrc/build.sh --pkg`，不复制其算子实现或维护第二套
AscendC build system。

## 4. 文件与职责

| 文件 | 唯一职责 |
| --- | --- |
| `thirdparty/public_kda_ops.lock.json` | 固定公开来源、SoC、root op 和依赖闭包 |
| `tools/build_public_kda_ops.py` | 校验源码、构建/安装 OPP、编译 binding、写 artifact manifest |
| `csrc/public_kda_ops_binding.cpp` | 注册四项 Torch schema、PrivateUse1 调用和 Meta 实现 |
| `tokenspeed_kernel_npu/public_kda_ops.py` | 校验并惰性加载 package-local artifact，返回逐项能力 |
| `test/ci_system/install_public_kda_ops.sh` | 独立开发安装入口；取得固定源码或使用已有 exact checkout |

不增加通用插件框架、动态 operator discovery、运行时 JIT 或新的配置类。后续 adapter 只调用
`public_kda_ops.is_available(name)` 和 `torch.ops.tokenspeed_npu_public_kda.<name>`。

## 5. Artifact 设计

### 5.1 位置

生成文件固定放在 Python package 下的忽略目录：

```text
tokenspeed_kernel_npu/_public_kda_ops/
    manifest.json
    binding/libtokenspeed_npu_public_kda_ops.so
    opp/vendors/custom_transformer/...
```

源码仓不提交 `.so`、OPP installer、vendor tree 或 build directory。editable install 和 source checkout
共享同一 package-local 位置，不需要绝对 artifact 环境变量。普通 wheel 也不携带与本机 ABI 绑定的
二进制；安装后需要显式运行 KDA 安装命令。

### 5.2 Manifest

manifest 只记录可搬运信息：schema version、完整 source/CATLASS SHA、SoC、依赖闭包、相对 binding/
vendor 路径和关键动态库 SHA-256。不得记录 source checkout、build cache、用户名或机器绝对路径。

loader 同时校验仓内 lock 与 artifact manifest。任何绝对路径、`..`、SHA/SoC 不一致、文件缺失或
digest 不一致都使全部公开 KDA 能力不可用，不修改进程环境。

## 6. Build 与安装流程

独立安装入口沿用现有 `install_triton_ascend.sh` 的 CANN root 和 Python 解释器，不修改它的默认行为：

```text
source target CANN
    -> 解析 lock
    -> 使用显式 exact source，或在忽略的 build cache 取得固定 revision
    -> 初始化并校验固定 CATLASS
    -> 公开 build.sh 构建依赖闭包
    -> installer 安装到临时 artifact
    -> 编译 TokenSpeed Torch binding
    -> schema/Meta 自检
    -> 写 manifest
    -> 原子发布到 package-local 目录
```

构建失败不能覆盖已有可用 artifact；新结果先写同文件系统临时目录，全部自检通过后再替换目标目录。
若环境无外网，可传入已准备的 exact public source；构建器仍执行相同 SHA 与 submodule 校验。

## 7. Torch ABI

namespace 固定为 `tokenspeed_npu_public_kda`，暴露：

```text
causal_conv1d(...) -> Tensor
recurrent_kda(...) -> Tensor
kda_gate_cumsum(...) -> Tensor
chunk_kda_fwd(...) -> (Tensor output, Tensor final_state)
```

schema 保留 mutation annotation：causal-conv state 与 recurrent state 都是原地更新。四项都提供 Meta
实现，使后续 NPUGraph/compile 可在不执行 NPU kernel 时推导输出 shape。

### 7.1 Recurrent layout 修正

公开 Kimi Torch adapter 将 `state_v_first=true` 固定在 wrapper 内，不能直接 include。TokenSpeed binding
复用相同参数校验和底层 aclnn API，但显式传：

```text
output_final_state = false
inplace_final_state = true
state_v_first = false
```

因此输入 pool 直接保持 `[capacity,HV,K,V]` FP32，不生成 transpose cache，也不改变阶段 3 的容量口径。
这属于 ABI 适配，不修改公开 AscendC kernel。

### 7.2 错误边界

binding 在调用前校验 tensor rank、dtype、device、K/V 维度、layout、chunk size 和 index shape。实际
aclnn 调用失败直接抛出，Python adapter 不捕获后重算，因为 state 可能已经被部分修改。

## 8. Loader 与逐项能力

loader 惰性且幂等，只在 NPU registration 请求能力时运行：

1. 定位 package-local manifest；
2. 校验 lock、SoC、相对路径和 digest；
3. 将 artifact vendor root 去重后前置到 `ASCEND_CUSTOM_OPP_PATH`；
4. `torch.ops.load_library` 加载 binding；
5. 分别检查四个 schema，缓存不可变结果。

缺 artifact 是正常的“能力不可用”，不打印 warning、不影响 reference。manifest 已存在但损坏、binding
加载失败也在能力探测阶段 fail closed，并保留可查询原因。真正选择并执行已加载算子的错误不属于
能力探测，必须传播。

四项能力彼此独立：例如缺 `chunk_kda_fwd` 只阻止 Prefill fused 注册，不影响 Decode recurrent 或
causal-conv。4A 本身尚不注册这些 kernel，因此不改变当前自动选择结果。

## 9. 验证

### 9.1 CPU/无 artifact

- lock schema、完整 SHA、依赖顺序和 cycle 拒绝；
- package-local artifact 缺失时四项均不可用，import 不触碰 `ASCEND_CUSTOM_OPP_PATH`；
- 错误 SoC/SHA、绝对/越界路径、缺文件和 digest 不匹配均 fail closed；
- fake `load_library` 后按 schema 独立报告能力，loader 多次调用只加载一次；
- vendor path 只前置一次，保留用户已有 custom OPP 路径；
- manifest 不含私有路径或构建机信息。

### 9.2 910B Build/ABI

- 从固定源码构建完整六项 OPP dependency closure；
- artifact manifest、`libcust_opapi.so` 和 Torch binding digest 复验；
- 加载后四项 schema 均存在，PrivateUse1 与 Meta dispatch key 均有实现；
- Meta 输入覆盖 causal-conv `[4,1536]`、Decode `[2,1,4,128]` 和 Prefill
  `[1,64,4,128]`，只校验 shape/dtype/mutation contract；
- 删除 artifact 的隔离进程中，阶段 3 reference focused suite 保持通过。

真实输出、state 更新、page sentinel、A/B 性能和 graph capture 分别由 4B--4E 验证，不在 ABI 步骤
重复。

## 10. 提交边界

1. 本文单独提交并推送；
2. lock、builder、binding、loader、许可、CPU tests 和安装说明作为 4A 实现提交；
3. exact-source 910B build/schema/Meta 结果作为 4A 验证记录单独提交。

任何一步都不夹带 4B causal-conv adapter 或更改 KDA kernel selection。
