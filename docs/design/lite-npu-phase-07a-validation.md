# Lite NPU 阶段 7A：OE Host Storage 与数学基线验证记录

## 1. 结论

阶段 7A 已通过。本阶段在不引入私有算子或额外 host cache 抽象的前提下完成：

- 12 张 OE 主表直接采用 safetensors CPU mmap Storage，不生成同尺寸 anonymous copy；
- ragged 2/3/4-gram local ID、special boundary 和 chunk continuation 与独立手算 oracle exact 一致；
- 12 个 projection source 转置装入一份 `[12,256,3072]` Parameter；
- packed projection、`/sqrt(13)` 和 special base-only 在真实 NPU0 上通过目标维度数值门禁；
- 既有 Lite loader、KDA、MLA、Grouped MoE 与 hybrid cache 累计回归通过。

最近 3 token 的 cache/PD/slot 生命周期属于阶段 7B；完整 checkpoint 的 16 进程 PSS、HBM、graph
staging 和服务集成属于阶段 7C。本记录不提前宣称这两部分已完成。

## 2. 提交与 exact source

| 内容 | 提交 |
| --- | --- |
| Phase 7 总体设计 | `a3dfcd81` |
| Phase 7A 设计 | `7698901b` |
| Phase 7A 实现与测试 | `30859633292ee2459bd54da8b90b10becc8925b4` |

上板源码直接由已推送的 `30859633` 生成 tracked archive，并解包到新的独立目录：

```text
archive SHA-256: 7986824faca6d4f1441df6501d31103fe8b1984f654ec5759a1e12ab92a6133d
lite.py:         ff51cc467dc81b375af8444f7d43de0aa9d512facdad7b121ba41d3383383c81
loader test:     0f762c7064afa3092bb74d8a62ecf21e0a4d60a470f97cf992ea90882715a1a8
OE test:         e77f6b539d249a6a4ca00b3211e7d8d2a89192555eddeb29db9998ee220c2358
```

上述三个文件的目标机 SHA 与本地提交内容逐项一致；验证前本地 HEAD、tracking 与远端 `lite` 也均为
`30859633292ee2459bd54da8b90b10becc8925b4`。动态资源、凭据和 checkpoint 路径未进入提交。

## 3. 本地门禁

格式器改写后重新执行同一累计选择集：

```text
35 passed, 27 skipped
```

其中 OE 新用例覆盖：

1. hand-computed ID、ragged request、零长度 request 与两段 continuation；
2. 23 个 special token 的 boundary 与当前 token base-only；
3. 12 路 lookup exact gather、packed matmul 对独立逐路 FP32 oracle；
4. 真实小 safetensors 的 storage pointer alias；
5. loader 临时结果释放并 GC 后仍可读，且 pointer 仍位于文件 mapping；
6. empty-token shape 与所有结果 finite。

实现提交前完整 `pre-commit run --all-files` 的全部 hook 通过。

## 4. Exact-source NPU focused

目标 CANN 9 环境只暴露 NPU0，运行 loader、OE、Grouped MoE、KDA、MLA 与 hybrid cache 累计集合：

```text
62 passed, 2 warnings
```

两个 warning 分别是镜像未编译 TorchAir 的既知提示，以及关闭 internal format 时创建 base-format tensor
的既知提示；没有 skip、NaN/Inf、异常退出或回退到另一份源码。

## 5. 目标维度 projection/merge 上板

独立微探针直接读取目标 config 几何，在 NPU0 执行：

```text
raw:        [8,12,256] BF16
projection: [12,256,3072] BF16
word/out:   [8,3072] BF16
```

CPU 侧使用 FP32 的单次 packed matmul、归一化和 special oracle；NPU 侧调用生产
`project_and_merge`：

| 指标 | 结果 |
| --- | ---: |
| finite | true |
| special 行与原 word bitwise exact | true |
| max absolute error | 0.00632656 |
| relative L2 | 0.00096589 |

结果通过 `atol=0.05, rtol=0.03` 门禁。该探针证明目标 `[3072,3072]` packed GEMM 和 epilogue 可在
910B eager 路径执行；graph 固定 staging 尚未接入，按设计留给 7B/7C。

## 6. Host residency 证据与边界

小型真实 safetensors 测试证明 Parameter 与 source 的 Storage pointer 完全相同，临时 loader 对象释放后
文件 mapping 仍存活。因此 7A 已验证“采用 mmap storage”这一根内存契约，而不是只检查 tensor 内容。

本阶段没有触碰完整 27 GiB 表页，也没有用单进程 RSS 推断 16 进程共享效果。完整 checkpoint 的
`Shared_Clean/Private_Clean/PSS` 和逐 rank HBM ledger 必须在 7C 以实际 8P8D 进程测量，避免将懒映射
地址空间或 page-cache 首次触页误报为 resident 内存。

## 7. 清理

测试未启动服务。结束后：

```text
exact-source Python processes: 0
NPU0 running processes:        0
```

NPU1--15 未被本阶段使用；裸 `npu-smi info` 也显示所有设备均无运行进程。阶段 7A 至此闭环，后续从
独立的阶段 7B 设计开始接入最近 3 token state、PD snapshot 与 Decode graph 外 staging。
