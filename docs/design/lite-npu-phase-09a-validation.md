# Lite NPU 阶段 9A：Replicated MLA 服务拓扑验证

## 1. 验证对象

本记录验证阶段 9A 的有界 8-rank lockstep 拓扑，不启动完整服务。验证源码为提交
`cc3c08d4c156dcd56673f1601466f4e799349134`，其 tracked-source archive SHA-256 为：

```text
664008301cd51708863b013a2f5288ba9fb754a301270cfe8b9582578e2a4054
```

验证在 CANN 9 镜像的一张 910B NPU 上执行，运行时只使用该归档中的 TokenSpeed 源码和既有
phase-local 依赖。测试工具安装在独立的 system-site-packages virtual environment 中，没有修改基础
Python、CANN 或系统 site-packages。

## 2. 覆盖范围

focused 集合覆盖以下边界：

- CLI 将 `attn TP8 / linear-attn TP8 / dense TP8 / MoE EP8` 解析成一个 8-rank lockstep world；
- mapping 初始化 rank 后，attention、KDA、dense 和 MoE 的 rank group 均为 `0..7`；
- `PDParallelTopology` 将该组合解析为 `TP8/CP1/DP1`，且通过 CachePD 的 CP1 准入；
- Lite `MLAConfig._spec_kwargs` 在该组合下将 MLA component 固定为 TP1，保留完整 32-head geometry；
- 非 Lite architecture 在相同 attention mapping 下仍保持 MLA TP8，不扩大 override 范围；
- Lite model loader、MLA eager 数值路径和 hybrid cache recipe 的既有回归保持通过。

执行集合为：

```text
test/runtime/test_linear_attn_mapping.py
test/runtime/test_lite_model_loader.py
test/runtime/test_lite_mla_eager.py
test/runtime/test_lite_hybrid_cache.py
```

## 3. 结果

最终 exact-source 结果：

```text
51 passed, 3 warnings in 5.52s
```

三条 warning 分别来自 Torch-NPU dynamo 的 eager backend 注册、Transformers 的 RoPE API 弃用提示和
NPU tensor base-format 回退；均不改变断言结果，也没有 NaN/Inf、设备错误或 cache layout 失败。

首次对实现提交 `ffeee70f` 的上板运行得到 `49 passed, 1 failed`。唯一失败是测试在 deferred-rank
mapping 初始化前读取 `tp_group`，触发既有的 `rank is not initialized` 保护。修正测试生命周期，并加入
真实 MLA config 与 CachePD 断言后，产品代码无需改动，最终 51 项全部通过。

验证完成后没有残留 pytest 进程，NPU 进程表为空。原始测试日志 SHA-256 为：

```text
ccb62e7f60c315b8ece58f045ca6d397b5a2b427210edda9cbdc993d4d1b58b7
```

## 4. 结论与边界

阶段 9A 通过。Lite 可以在不实现分布式 KV page ownership 的前提下，用一个 TP8 lockstep execution
group 承载 dense/KDA/EP collective，同时让 MLA 参数、完整 latent cache 和 32-head compute geometry
保持 replicated。CachePD 所需的 CP1 拓扑也已经由真实 mapping 验证。

本记录不证明 16-rank P/D 服务、checkpoint 全量加载、真实 cache 容量、请求联通性或输出精度；这些
属于阶段 9B launcher 和阶段 9C exact-source 服务准入。CP/SP、KVP、分布式 KV ownership 和 2M 容量
继续保留在最终 TODO。
