# Lite NPU 阶段 8B.1：8P8D Checkpoint 账本验证记录

## 1. 结论与边界

阶段 8B 的完整 checkpoint 加载、P8/D8 ownership 和静态内存账本已通过独立验收：

- 两个独立 HCCL world 同时在 16 张 NPU 上完成生产 `DefaultModelLoader` 严格加载，16/16 worker
  全部通过；
- P 使用 MLA CP8、dense TP1、KDA TP8、MoE EP8，D 使用 MLA DP8、dense TP8、KDA TP8、
  MoE EP8，resolved mapping 与设计一致；
- P/D 每 rank 的参数 storage、post-load derived buffer 和实际 HBM 分配均一致，无 rank 倾斜；
- KDA TP8 和 routed expert EP8 在各 role 内均得到 8 个唯一 fingerprint，P/D 相同 local rank 的目标
  shard fingerprint 一致；MLA 和 OE projection 的 replicated fingerprint 在 8 rank 内唯一；
- 12 张完整 OE 表仅保持 file-backed host mapping，未复制到 HBM；16 进程 mapping PSS 总和只有
  66,168 KiB；
- `--check-finite` 以有界 chunk 扫描了全部 NPU 参数，16 rank 均未出现 NaN/Inf；
- 同一 exact source 的 104 项累计回归全部通过，验证后 probe 进程、NPU Python 进程和临时源码均已
  清理。

本记录只准入 loader、静态 ownership、fingerprint、参数 finite 和 HBM/PSS 账本。阶段 8B 设计中的
冻结 reference 分段数值 replay 尚未执行，因此这里不宣称 role-local eager 数值闭环，也不宣称 MLA
CP8/KVP8 cache、PD、连续 28 层请求或 HTTP 服务可用。

## 2. 提交与 exact source

| 内容 | 提交 |
| --- | --- |
| Phase 8B 总体设计 | `94edb15c7613d73f033ef07ced2d7c0229e88cc7` |
| linear-attention TP CLI 接线 | `d105bb1b25a9d09ba6e36c8722c0050a5acec150` |
| post-load derived 账本修正 | `2dc1418c632638ccefc2e94fb9fd095c8a760a1d` |
| checkpoint/controller probe | `1bad8dfaa113bd59214a6f968f6a29e06df4765f` |
| routed expert 独立 fingerprint 门禁 | `445fc14e35bb81094d2b932bf3f327c4dc7a3066` |

最终验证源码由已推送的 `445fc14e` 生成全新 tracked archive：

```text
archive SHA-256: c096bac0fa43d72e57d888ac2b43ebb7fdc8c422290fd5c08c35c6734b9151c0
```

拉回的匿名化 16-rank summary 及完整文件清单分别为：

```text
summary SHA-256:  1b44c3779e5563525cd11b90de5df9bc507e5886b6db913c9510d86fbca1de37
manifest SHA-256: a5785d90f15f472f88797d75644d8b10019da6f0758011fd2d3559d5fdd81a06
```

动态资源、凭据、主机地址、checkpoint 路径和原始 tensor 值均未进入提交。

## 3. 8P8D topology 与 loader

P/D 分别创建独立 world-size 8 的 HCCL group，并在全局 barrier 上保持同时驻留：

| mapping | Prefill | Decode |
| --- | ---: | ---: |
| world / PP | 8 / 1 | 8 / 1 |
| MLA attention | TP1 + CP8 + DP1 | TP1 + CP1 + DP8 |
| dense/shared/vocab | TP1 | TP8 |
| KDA | linear TP8 | linear TP8 |
| routed expert | TP1 + EP8 | TP1 + EP8 |

每个 worker 都经生产模型构造、strict safetensors load、KDA/MLA/Grouped MoE post-load hook 和
`eval()` 完成加载。controller 的六个全局硬检查全部为 true：worker 数与 worker 本地检查、role 参数
均衡、cross-role fingerprint、OE 不进入 HBM、OE PSS one-copy。

## 4. 参数与 HBM 账本

### 4.1 参数分类

| 模块 | P bytes/rank | D bytes/rank | ownership |
| --- | ---: | ---: | --- |
| KDA | 369,143,376 | 369,143,376 | TP8 |
| MLA | 634,023,936 | 634,023,936 | replicated |
| routed experts | 12,683,575,296 | 12,683,575,296 | EP8 |
| Grouped MoE 其余参数 | 2,785,900,544 | 473,790,464 | P dense TP1 / D dense TP8 |
| OE projection | 18,874,368 | 18,874,368 | replicated |
| embedding、head 与 outer norm | 2,013,616,128 | 252,008,448 | P dense TP1 / D dense TP8 |
| **Parameter storage** | **18,505,133,648** | **14,431,415,888** | - |

完整 OE payload 为 28,991,102,976 bytes/rank 的 host mmap，不计入上表 NPU parameter storage。
post-load derived buffer 为：

```text
21 * 3 * 512 * 4 * 2
+ 7 * 2 * 32 * 128 * 512 * 2
= 58,978,304 bytes/rank
```

第一项是 21 个 KDA layer 的三路 packed width-4 conv，第二项是 7 个 MLA layer 的 `w_kc/w_vc`。

### 4.2 实测 HBM

| 指标 | P/rank | D/rank |
| --- | ---: | ---: |
| parameter storage | 18,505,133,648 bytes（17.234249 GiB） | 14,431,415,888 bytes（13.440303 GiB） |
| post-load derived | 58,978,304 bytes（0.054928 GiB） | 58,978,304 bytes（0.054928 GiB） |
| **logical resident static** | **18,564,111,952 bytes（17.289177 GiB）** | **14,490,394,192 bytes（13.495231 GiB）** |
| checkpoint-loaded allocated | 18,651,303,424 bytes（17.370380 GiB） | 14,576,073,728 bytes（13.575027 GiB） |
| allocator above logical | 87,191,472 bytes（83.1523 MiB） | 85,679,536 bytes（81.7104 MiB） |
| peak allocated | 19,598,288,384 bytes（18.252328 GiB） | 14,878,066,176 bytes（13.856279 GiB） |
| reserved | 19,904,069,632 bytes（18.537109 GiB） | 15,290,335,232 bytes（14.240234 GiB） |

同 role 八个 rank 的上述值逐 byte 相同。`checkpoint-loaded allocated` 与逻辑 storage 的小差值保留为
allocator/runtime 开销，没有回填到权重公式中。本探针没有分配最终 token KV pool，表中 peak/reserved
也不是服务级 cache/runtime 预算。

## 5. Fingerprint 与 finite 门禁

fingerprint 从每个 category 的参数名、shape、dtype 和稀疏数值样本生成，不记录 checkpoint 值。最终
role 内唯一数为：

| category | P unique | D unique | 解释 |
| --- | ---: | ---: | --- |
| KDA | 8 | 8 | linear TP8 shard |
| routed experts | 8 | 8 | EP8 local expert shard |
| MLA | 1 | 1 | replicated |
| OE projection | 1 | 1 | replicated |
| Grouped MoE 其余参数 | 1 | 8 | P dense TP1；D dense/shared TP8 |
| outer | 1 | 8 | P dense TP1；D embedding/head TP8 |

最初的 probe 把 routed experts 与 dense/shared/router 放在同一 `grouped_moe` category；这会让稀疏样本
落在复制参数上而掩盖 expert shard。最终提交把 `.mlp.experts.` 独立为 `moe_experts`，并将其纳入 P/D
相同 local rank 的硬比较，所以结果明确覆盖了 192 个 local real expert 的实际值，而不只覆盖参数数目。

当前 hard gate 是 P/D cross-role fingerprint 和 production strict-loader coverage；它不是由 safetensors
另行计算的独立 expected-value manifest。独立 source fingerprint 与冻结 tensor reference 数值 replay
仍留在下一项 8B 工作中，不能用本次 cross-role 一致性替代。

`--check-finite` 对每个 NPU parameter 执行有界 chunk 检查，16/16 worker 的
`parameters_finite=true`。本次没有执行 activation replay，所以该结果只证明静态参数没有 NaN/Inf。

## 6. OE host residency

每个 worker 的 12 张 OE 表仍由五个 file-backed safetensors mapping 承载：

| 指标 | 结果 |
| --- | ---: |
| OE payload/rank | 28,991,102,976 bytes |
| mapping count/rank | 5 |
| mapping `Anonymous` | 16/16 为 0 |
| 16-rank mapping PSS 总和 | 66,168 KiB（64.617 MiB） |
| 单 rank mapping RSS 最大值 | 66,176 KiB（64.625 MiB） |
| OE table HBM copy | 0 |

该测量没有主动触摸完整表，因此只解释 loader 阶段实际 resident page set；它证明没有 16 份匿名 host
copy 或任一 rank 的约 27 GiB HBM copy，不代表完整 OE table 已被预热。

## 7. 测试与清理

exact source 的 probe focused test：

```text
7 passed, 1 warning, 0 skipped
```

同一 exact source 的 loader、hybrid cache、KDA、MLA、Grouped MoE、OE、decoder、role probe 和
linear-attention mapping 累计回归：

```text
104 passed, 3 warnings, 0 skipped
```

warning 分别来自目标环境未编译 TorchAir、关闭 internal format 时使用 base-format tensor，以及
Transformers 的 RoPE API 迁移提示；没有失败、NaN/Inf 或源码回退。

最终 exact 8P8D 结果为：

```text
workers:             16/16 pass
global checks:       6/6 true
P expert fingerprints: 8 unique
D expert fingerprints: 8 unique
```

结果拉回后只删除了两个明确命名的远程临时源码/运行目录；两者不再存在，probe process 为 0，
`npu-smi` 中 NPU0--15 均无 Python 进程。源码可由已推送提交和 archive hash 重新构造，远程临时目录
本身不可恢复。

## 8. 后续

Phase 8B 下一步只补冻结 reference 的分段 replay：对 P 一步、D 三步的语义 tap 做 checkpoint-backed
逐层数值、collective 和 activation finite 对齐，并补独立 source expected fingerprint。该步骤通过后才
关闭 8B；MLA cache owner、CP/KVP partial attention、PD 和连续服务仍按计划从阶段 9 开始。
