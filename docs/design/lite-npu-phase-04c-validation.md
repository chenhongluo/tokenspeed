# Lite NPU 阶段 4C：Recurrent Decode 验证记录

## 1. 验证对象

本记录验证阶段 4C 实现提交 `9ff2899e`。范围限定为 Ascend K-major FP32 state 的单 token recurrent
Decode：scalar/featurewise beta、独立读写 page、固定 batch padding、NPUGraph replay、kernel
selection 和性能准入。

本阶段未修改 runtime/backend、cache、PD 协议或 checkpoint；不验证 Prefill chunk KDA、output
epilogue、完整模型服务、CP8/KVP8 或 8P8D。这些能力仍按总计划在后续阶段独立准入。

目标环境为单张 Ascend 910B、CANN 9.0.0、PyTorch 2.9.0 和 `torch_npu` 2.9.0.post2。所有实卡
验证只使用 NPU0，未启动模型服务或占用其余设备。

## 2. Exact-source

从完整提交 `9ff2899ea1b0fda91b1a68b6b97eff0c934c9aa2` 生成全新 `git archive`。archive 的
SHA-256 为：

```text
93f5e354c5ba913247c55cdc85272fad89dcb39f55626d8a93dbf607637802bc
```

传输前后 SHA-256 一致，`git get-tar-commit-id` 在两端均返回上述完整提交。远端在全新目录解包，
仅补入阶段 4A 已校验的 package-local 公开算子 artifact；没有复制候选 Python/runtime 源码。

公开 artifact 在场时，`kda_paged_decode(solution="public_kda")` 仍按设计 fail loud。原因是本阶段没有
注册公开 recurrent，而不是 artifact 缺失触发的偶然回退。

## 3. 实现与选择

真实 Ascend registry 的默认 Decode kernel 为：

```text
torch_ascend_kda_paged_decode
```

它保留既有统一 API、kernel 名和 `solution="torch"`，只把实现从会读取 device metadata 到 host 的
逐请求 reference 切换为 `Priority.PORTABLE` 的 batch tensor 实现。runtime 仍不导入 `torch_npu`，也
没有新增 TorchAir、公开 artifact loader、第二套 state layout 或 graph executor。

NPU 执行只包含 device tensor 的 normalize、gate、gather、batched matmul、state update 和
`index_copy_`，没有 `.item()`、`.tolist()`、D2H 或 request Python 循环。shape/dtype/device 等静态
契约在调用时检查；device metadata 的 active-page 唯一性由 runtime 构造契约保证，不能为了重复检查
而在每层 Decode 引入 D2H。CPU focused 路径另外验证边界单调、尾部 padding、双负 index、page 范围
和 active write 唯一性。

## 4. 数值与 state

### 4.1 Scalar 与 featurewise beta

CPU 独立 oracle 和 NPU0 生产形状均覆盖两种 beta：

- scalar：`beta=[1,B,H]`，`sigmoid(beta)` 乘在 delta 上；
- featurewise：`beta=[1,B,H,K]`，`sqrt(sigmoid(beta)+1e-10)` 分别缩放 K/V。

生产 TP8 rank-local shape 为 `H=4,K=V=128`。BS1、BS2、BS8（5 active + 3 padding）对 FP32 tensor
oracle 的 output/state `max_abs=0`、`rel_l2=0`，page 0 bitwise 不变，全部张量 finite。scalar BS2 对
逐请求 reference 同样为 output/state 零差。

### 4.2 连续状态更新

BS2 连续执行 128 步，每步更新 Q/K/V/gate/beta，并在两组独立 read/write pages 间交替发布 state。
最终结果为：

| 项目 | max abs | rel-L2 | finite |
| --- | ---: | ---: | --- |
| 最后一步 output | `0` | `0` | 是 |
| 完整 state pool | `0` | `0` | 是 |

没有 NaN/Inf、漂移放大、page 0 污染或未选邻页改写。

### 4.3 NPUGraph replay

BS2 graph 首轮使用 1 active + 1 padding，随后在不重新 capture 的条件下同时更新 Q/K/V、gate、beta、
read/write indices 和 `cu_seqlens`，第二轮变为 2 active。两次 graph output/state 均与同一实现的 eager
结果 bitwise 一致，证明输入值、page index 和 padding 边界没有被冻结。

## 5. 性能 A/B

最终 exact-source 代码在同一 NPU0 上做 30 次 warmup、300 次 NPUGraph replay，记录完整
featurewise prepare、recurrent 和双页 state 发布的平均墙钟时间：

| BS | graph-safe tensor | 公开 K-major 审计值 | tensor 相对加速 |
| ---: | ---: | ---: | ---: |
| 1 | `127.8 us` | `2049.7 us` | `16.0x` |
| 2 | `166.4 us` | `2095.8 us` | `12.6x` |
| 8 | `231.3 us` | `2187.3 us` | `9.5x` |
| 64 | `832.3 us` | `12685.4 us` | `15.2x` |
| 256 | `3143.6 us` | `46342.8 us` | `14.7x` |

本阶段性能是记录项，没有设吞吐门槛。结果证明 tensor 路径没有公开 K-major 每次约 2 ms 的固定成本；
BS1/2 也优于已审计的 V-major TorchAir 完整 adapter `269.1/257.8 us`。

## 6. 测试汇总

| 环境 | 结果 |
| --- | --- |
| 本地 recurrent/causal-conv focused | `11 passed, 3 skipped`；skip 均为 NPU-only |
| 910B exact-source recurrent/causal-conv/Lite runtime | `26 passed` |
| 910B 生产形状 128-step/性能 probe | 全部通过 |
| 全仓 `pre-commit run --all-files` | 全部通过 |

额外尝试的 Kimi cache 扩展回归中，15 项通过、3 项硬件无关 skip；剩余 2 项在收集后因隔离环境没有
编译 `tokenspeed_scheduler` 扩展而不可运行，未进入 recurrent 调用链。本阶段没有为此修改目标环境或
扩大实现范围；统一 kernel API 的 scalar-beta NPU 回归已由上述 focused 和生产形状 probe 覆盖。

## 7. 公开方案负结果

公开 recurrent 的数值适配成立，但生产准入仍拒绝：

- K-major eager/NPUGraph 均约 2 ms 起，慢于 tensor 路径；
- V-major + TorchAir 虽能降到 BS1/2 `269.1/257.8 us`，仍慢于本阶段默认路径；
- TorchAir compiled graph 在稳定 pool/index/stream 和同 stream warmup 后仍不能嵌套 TokenSpeed
  NPUGraph capture，capture 内 `LoadGraph` 会触发禁止的 H2D memcpy。

因此没有为公开 op 增加 converter、双 graph executor、state transpose 或第二份 cache。阶段 4A
artifact 保留，供 Prefill 和未来架构变化后重新评估。

## 8. 结论与回退

阶段 4C 准入通过。Ascend Decode 已从 host-sync reference 切换到 graph-safe batch tensor recurrent，
保持 scalar/featurewise beta、K-major FP32 state、独立读写 page 和固定 batch padding 语义。

若后续完整服务发现问题，只需把 Ascend registration 恢复到 portable reference；runtime、cache 和模型
均无需修改。下一阶段按既定顺序准入 Prefill gate-cumsum/chunk KDA。
