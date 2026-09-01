# Lite NPU 阶段 4A：公开 KDA Build/ABI 验证记录

## 1. 验证对象

本记录验证 TokenSpeed 提交 `9cce79e2` 的阶段 4A 实现。验证范围仅为固定公开源码的可复现构建、
package-local artifact、Torch ABI、Meta contract、逐项能力探测和阶段 3 reference 回退；尚未把任何
公开算子注册为生产 KDA 高优先级实现，也不包含真实算子数值、graph 或性能准入。

公开输入固定为：

| 输入 | 固定版本 |
| --- | --- |
| vLLM-Ascend | `d543ccee0a1ff677165777e3defafd42b35e83ef` |
| CATLASS | `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |
| SoC | `ascend910b` |
| JSON headers | v3.11.3，lock 内 SHA-256 |
| Abseil | 20230802.1，lock 内 SHA-256 |
| Protocol Buffers | 25.1，lock 内 SHA-256 |

目标环境为单张 Ascend 910B、CANN 9.0.0、PyTorch 2.9.0 和 `torch_npu` 2.9.0.post2。验证只使用
NPU0；未启动模型服务或占用其余设备。

## 2. Exact-source 构建

从 `9cce79e2` 生成不含工作区改动的 `git archive`，在独立目录执行：

```bash
TOKENSPEED_PUBLIC_KDA_SOURCE_DIR=<exact-public-source> \
    test/ci_system/install_public_kda_ops.sh
```

构建结果：

- lock 中三个传递包全部通过 SHA-256 校验，JSON headers 在 CMake 配置前安全展开；
- root op 与依赖闭包六项全部为 910B 生成完成：`causal_conv1d`、`recurrent_kda`、
  `kda_gate_cumsum`、`chunk_gated_delta_rule_fwd_h`、`kda_layout_swap12`、`chunk_kda_fwd`；
- OPP installer 成功安装 `custom_transformer` vendor tree；
- TokenSpeed binding 成功链接，四项 schema 均有 PrivateUse1 和 Meta implementation；
- manifest、binding 和 `libcust_opapi.so` 仅在全部自检完成后原子发布。

同一公开 source 上连续执行安装入口也通过。builder 使用上游原生 `--make_clean`，不会复用不支持
重复配置的 CMake build tree；已校验的 package cache 与 JSON headers 保留。

## 3. Artifact 与 ABI

独立进程重新读取 manifest，并逐项复算 binding 与 `libcust_opapi.so` SHA-256，结果一致。惰性加载后
能力集合严格为：

```text
causal_conv1d
recurrent_kda
kda_gate_cumsum
chunk_kda_fwd
```

逐项检查的 `PrivateUse1` 和 `Meta` dispatch key 均存在。Meta 生产 shape 结果为：

| Schema | 输入口径 | 输出口径 |
| --- | --- | --- |
| causal-conv | `[4,1536]`，width 4 | `[4,1536]` BF16 |
| recurrent Decode | `[2,1,4,128]` | `[2,1,4,128]` BF16 |
| gate cumsum | `[1,64,4,128]` | `[1,64,4,128]` FP32 |
| chunk Prefill | `[1,64,4,128]`，chunk 64 | output 同输入；state `[1,4,128,128]` FP32 |

Recurrent binding 直接调用底层 ACLNN，并固定 `state_v_first=false`；因此 TokenSpeed 的
`[capacity,HV,K,V]` FP32 state 不需要转置或第二份 cache。

## 4. 回退与回归

将 package-local artifact 暂时移出 loader 可见路径后：

- loader 返回四项均不可用且不修改 custom OPP 环境；
- 阶段 3 Lite KDA reference suite 为 `12 passed`；
- artifact 恢复后 manifest 仍可加载，digest 和四项 schema 均不变。

最终测试矩阵：

| 环境 | 结果 |
| --- | --- |
| 本地 CPU/meta focused | `20 passed, 3 skipped`；skip 均为 NPU-only |
| 910B exact-source focused | `23 passed` |
| 全仓 pre-commit | 全部通过 |

NPU focused 覆盖 scalar/featurewise-beta reference、生产 shape Prefill/连续 Decode state reuse、
NaN/Inf 拒绝、loader 失败隔离、manifest/digest、依赖闭包、原子发布和 Meta contract。

## 5. 负结果与修复

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| 首次构建无法取得三个 release 包 | 上游下载端点在目标网络不稳定 | lock 固定 URL/SHA；支持已校验的本地预置包 |
| 缓存 `include.zip` 后 JSON 仍重下 | 上游 JSON CMake 不读取该 package cache | 在调用上游 build 前用标准库安全展开 headers |
| SSH 中断后重试出现两个 protobuf builder | 中断前端连接未终止远端子进程 | 停止两棵 candidate session、隔离竞态 build tree、单进程重建 |
| 成功后直接复用同一 build tree 重跑失败 | 上游 CMake 重复注册 compile options | 复用上游 `--make_clean`，exact-source 重跑通过 |

这些失败均发生在临时 staging 或上游生成目录；package-local 原子发布目标没有出现半成品。

## 6. 结论与下一步

阶段 4A Build/ABI 准入通过。提交 `9cce79e2` 可以从固定公开输入生成可校验 artifact，并在 artifact
缺失或损坏时保持阶段 3 reference 可用。

下一步按既定顺序单独准入 causal-conv、Decode recurrent、Prefill gate/chunk 和既有 epilogue。每项
必须补真实 NPU output/state/sentinel A/B 与独立回退证据；本记录不能替代这些数值和性能门禁。
