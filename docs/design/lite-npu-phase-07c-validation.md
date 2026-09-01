# Lite NPU 阶段 7C：Exact OE Checkpoint 与 8P8D 内存账本验证记录

## 1. 结论

阶段 7C 的独立边界已通过：

- exact checkpoint 的 12 张 OE 表 payload 精确为 28,991,102,976 bytes，Parameter 与 safetensors
  mapping 保持 storage alias；
- 两进程及 8P8D 共 16 个独立 rank 的表 mapping 全部 file-backed，mapping `Anonymous=0`；
- 16 rank 同时触摸相同 256 MiB 文件页时，总 PSS 仍约为一份 256 MiB resident set，而不是 16 份；
- 16 张 NPU 的 host-table HBM 增量均为 0，每 rank 只驻留 18 MiB packed projection 和 KB 级
  staging/context；
- exact checkpoint 的 Decode-like、Prefill、special/EOS、两请求 continuation 和 BS2 graph 全部通过
  FP32 oracle、finite 与 replay 门禁；
- `lite_oe` 的统一 `latest_snapshot` manifest 只复制最终 12-byte context，独立 D arena restore 后续算
  与不拆分路径 exact；
- Lite loader、hybrid cache、KDA、MLA、Grouped MoE 和 OE 的最终 exact-source 累计回归全部通过。

本阶段验证的是 OE 子模块、统一 cache contract 和 16-rank 内存行为，不是完整 decoder 或 HTTP 服务。
普通 Decode 的 device-only token 仍需后续 sampler 到 model preparer 的 CPU effective-token publication；
这里没有通过每 token D2H 或伪造 token 隐藏该边界。

## 2. 提交与 exact source

| 内容 | 提交 |
| --- | --- |
| Phase 7C 设计 | `cc26187dbde853ebcd67db14bd6e8579d383819d` |
| checkpoint/PSS/HBM probe 与 PD restore 测试 | `1af2235077fbf686643707552fd293348eab9f79` |
| 4 KiB 确定性触页修正 | `087997b4b196ad056b0177f979df543ddd6ad525` |
| file-mapping PTE normalization | `887358c65687e735c5d0929ff36e6cb313507a02` |

最终上板源码由已推送的 `887358c6` 生成全新 tracked archive：

```text
archive SHA-256: 088d59abc8736b606528e7c4a3fdb67adc5bb186e148814cd2bef60ccb0a8c51
```

动态资源、凭据、主机地址和 checkpoint 路径没有进入提交。最终匿名化 artifact 的 SHA-256 为：

```text
two-worker ledger: 1dfd33bfae3381a4ac019c78e5e95c07786ff53d37af2304099ccbe6e898ff85
8P8D ledger:        b9d465d7d53c1aff51a5aaa138fa8637b446eae927589f13f1e729a27e54c2b7
NPU0 exact:         164abd7b15594fc4ef7e3532b348b63118e9cb7dc623169f90759312033302b4
cumulative log:     813c2e32b4ce978a314ccdd24774a35f7be7f6932488d4462b58a00e40705deb
NPU live ledger:    5426c15861802c761765884e1af9e68e9363854a37f92c57b2cddc25ecb6e83b
NPU cleanup ledger: d63b5f3c0801be46ccfe316f177146c27c2a53657762bf0a6b47cbd9e9b7f404
```

## 3. Payload、mapping 与测量口径

| 项目 | 结果 |
| --- | ---: |
| OE tensor payload | 28,991,102,976 bytes（27.00007 GiB） |
| packed projection/rank | 18,874,368 bytes（18 MiB） |
| OE key 所在 shard 数 | 5 |
| 五个 shard 的 mapping `Size`/rank | 75,340,884 KiB |
| mapping file-backed | 16/16 true |
| mapping `Anonymous` | 16/16 为 0 |

`Size` 是五个 safetensors shard 的完整虚拟映射，不是 OE payload 或 resident memory。payload 来自 12 个
目标 tensor 的 shape/dtype 总和；物理驻留只由 RSS/PSS 解释。

初始 probe 把触点散到完整表空间，文件系统预读使两个进程分别形成不同的 file-backed clean PTE，虽
`Anonymous=0`，但 PSS/单进程 RSS 为 1.270，未通过 1.25 门槛。最终 probe 不放宽门槛，而是在 adoption
完成后对进程自己的五个 mapping 执行 `MADV_DONTNEED`，再以 `MADV_RANDOM` 禁止预读，逐 4 KiB 触摸
固定 256 MiB。该操作不修改 checkpoint，也不执行系统级 page-cache 清理。

最终两进程结果：

| 指标 | rank 0 | rank 1 | 总计 |
| --- | ---: | ---: | ---: |
| normalization 前 mapping RSS | 3,573,024 KiB | 3,572,900 KiB | - |
| normalization 后 mapping RSS | 0 | 0 | 0 |
| barrier mapping RSS | 262,976 KiB | 262,720 KiB | 525,696 KiB |
| `Shared_Clean` | 262,480 KiB | 262,480 KiB | - |
| `Private_Clean` | 496 KiB | 240 KiB | 736 KiB |
| mapping PSS | 131,736 KiB | 131,480 KiB | 263,216 KiB |
| anonymous 增量 | 45,788 KiB | 45,900 KiB | - |

PSS 总和约等于一份实际触页集合，且显著小于两份 RSS 之和。

## 4. 8P8D host residency

16 个独立 worker 中 rank 0--7 标记为 P，rank 8--15 标记为 D；所有 worker 映射同一 checkpoint 并触摸
同一页集合：

| 指标 | 每 rank | 16 rank 总计 |
| --- | ---: | ---: |
| mapping RSS | 262,784 KiB | 4,204,544 KiB |
| `Shared_Clean` | 262,784 KiB | RSS 口径重复计数 |
| `Private_Clean` | 0 | 0 |
| mapping PSS | 16,424 KiB | 262,784 KiB |
| mapping `Anonymous` | 0 | 0 |

`262,784 / 16 = 16,424 KiB`，证明同一 file-backed resident set 被 16 个进程按 PSS 均分。每 rank 的
全进程 anonymous 约 983--984 MiB，其中绝大部分在 NPU runtime baseline 已存在；加载 OE 后增量仅：

- P：51,904--52,912 KiB；
- D：52,004--53,032 KiB。

没有 rank 形成 27 GiB anonymous table copy。

## 5. 逐 rank HBM 账本

16 rank 的 TorchNPU 结果完全一致：

| publication point | P/rank allocated 增量 | D/rank allocated 增量 |
| --- | ---: | ---: |
| host table adoption | 0 | 0 |
| packed projection | 18,875,392 bytes | 18,875,392 bytes |
| role staging/context | 13,824 bytes | 13,312 bytes |

projection 的逻辑 payload 为 18,874,368 bytes；实际 allocated 多 1,024 bytes allocator 对齐。P 的 eager
probe 同时保留一份 one-token prepared tensor，D 使用 BS2 fixed staging，因此两者 KB 级增量略有差异。
统一 cache arena 的 parent/plane 数不因 12-byte OE child view 增加，本账本未重复归因 parent bytes。

barrier 期间独立 `npu-smi` 显示 16 张卡各有一个 Python worker，process memory 均为 129 MiB。该数字
含 NPU context/runtime，不能当作 projection payload；它与 TorchNPU allocated 分列记录。退出后 16 张
卡均无 Python 进程。

## 6. Exact checkpoint 数值与 graph

CPU 使用 FP32 packed projection/归一化 oracle，NPU 使用生产 `project_and_merge`；门槛为
`atol=0.02, rtol=0.02`：

| 路径 | tokens | max-abs | relative-L2 | 结果 |
| --- | ---: | ---: | ---: | --- |
| Decode-like | 1 | 0.004620 | 0.002445 | pass |
| Decode-like | 2 | 0.003989 | 0.000713 | pass |
| Decode-like | 4 | 0.004233 | 0.000440 | pass |
| Decode-like | 32 | 0.006608 | 0.001688 | pass |
| Prefill | 128 | 0.008360 | 0.002204 | pass |
| Prefill | 1,024 | 0.010075 | 0.002434 | pass |

所有结果 finite；EOS/special 行与 word exact。两请求分块 continuation 的 ID/tail 与 one-shot exact。
BS2 graph 在 staging/word/token 更新后与 eager bitwise exact，输出相对首轮发生变化且无 NaN/Inf。

NPU0 单 worker 的 HBM 采样为：

| point | allocated | reserved | max allocated |
| --- | ---: | ---: | ---: |
| baseline | 0 | 0 | 0 |
| host table adopted | 0 | 0 | 0 |
| projection loaded | 18,875,392 | 23,068,672 | 18,875,392 |
| exact + staging published | 18,889,216 | 85,983,232 | 59,519,488 |

最后一行的 reserved/peak 包含 exact T1024 和 graph capture workspace；常驻 allocated 仍只有 projection、
staging 和小型 context。

## 7. PD manifest 与累计回归

PD focused test 使用完整 Lite cache plan 生成 prompt length 129 的 manifest，确认 `lite_oe` 只选择第二个
也是最终 snapshot page，field payload 为 12 bytes。测试将该 page 复制到独立 D context arena；D owner
miss 恢复一次后继续显式 CPU token，与 P 不拆分 continuation 输出 exact，未创建第二份 transfer cache。

最终 exact-source NPU0 累计集合覆盖 loader、hybrid cache、KDA、MLA、Grouped MoE 和 OE：

```text
73 passed, 2 warnings, 0 skipped
```

两个 warning 是镜像未编译 TorchAir，以及关闭 internal format 时采用 base-format tensor 的既知提示；
没有失败、NaN/Inf、异常退出或源码回退。

## 8. 清理与后续边界

最终采样后：

```text
Lite OE probe processes: 0
NPU Python processes:    0
```

Phase 7A--7C 至此完成 OE 子模块闭环。后续进入完整 decoder/8P8D role execution 时，仍必须先解决正常
Decode effective token 的 CPU publication，然后才能连接服务并做 GSM8K first100；本记录不提前宣称
该后续集成已经完成。
