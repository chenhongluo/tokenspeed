# Lite NPU 阶段 7B：OE State、PD Snapshot 与 Graph Staging 验证记录

## 1. 结论

阶段 7B 的独立边界已通过：

- Lite 专属 cache recipe 在保留 `kimi_k3` family 的同时增加一组 12-byte OE context state；
- OE state 复用既有 7 个 plane，TP8 的 2,064,384-byte parent 和原 KDA/MLA field layout 不变；
- KDA backend 只消费 layer-owned state group，不把模型自有 `lite_oe` 当作 KDA state；
- request-slot mirror 可在 steady continuation 命中，并在 prefix/PD admission 或 slot reuse 时从
  authoritative page 恢复；
- Decode fixed staging 在 NPU graph capture/replay 中保持地址不变，更新真实行、清零 padding 后输出
  正确更新且 finite；
- Lite/KDA/MLA/Grouped MoE 累计 exact-source NPU0 回归全部通过。

普通 Decode 的 token 当前只存在于 device `future_input_map`，scheduler 的 CPU 字段以 `-1` 表示无
override。7B 对这种情况明确 fail closed，没有引入每步 D2H 或伪造 token。完整 decoder 接线必须先
补齐 sampler 到 model preparer 的 CPU effective-token publication；本记录不宣称正常 Decode 服务已
经 OE 路径跑通。完整 checkpoint 的 16 进程 host PSS/HBM 也仍属于 7C。

## 2. 提交与 exact source

| 内容 | 提交 |
| --- | --- |
| Phase 7B 设计 | `7c35ce9d56ebe5251808e89b361dfa536a35c229` |
| cache、mirror、staging 实现与测试 | `aaa65feabd9374fa5c9a5c6cf066e4ff9fde00ee` |
| recipe dispatch 测试输入修正 | `4f5fc10106846ba5e801fa1c024ecf8e7ea96f52` |
| NPU graph staging 回归 | `f1eb9bc148b3fa761a097887e8f625a92687ba62` |
| output page 原子校验与边界回归 | `8d1384bf941b1e0bed76b0afd9bf79a2b0d1eec4` |

最终上板源码由已推送的 `8d1384bf` 生成全新 tracked archive：

```text
archive SHA-256: 79db5341b3a2bac636064e9e540ac6652cfb711c0aa9c4cc7cc208ee40053e59
```

上板前对 archive 中全部 tracked 文件执行逐项 SHA-256 比对，结果全部一致。关键文件 SHA-256 为：

```text
model_executor.py:        0bedbd1589d879d3d9c4881e12842f04ee2b3c9ffe0b687788f58459e5e112a6
hybrid_linear_attn.py:    ba11c1f129ad776d5ee68da01da645ce84bb6e61de3ffd34e201e3e19a7acb59
recipes/lite.py:          9ce5116e49405b1da945ce3fffd9bccfff3a54d2f8e67de0a5d58f4e87f704bc
recipes/setup.py:         76022dc5a45669a1af6537c829d66b9ff4f677fff41e0cc1917fb419ae7f4939
models/lite.py:           d53b79cefc9d4706d8c592f92658876c770fffc35ba1f8989f197b6927d6f494
hybrid-cache test:        533a4af685d22e8fa87ed8fbe4fa557eecedcb5193d2ec531d9b73a43a2fa9f6
OE test:                  088b25763eefb999c212a34d671a38effcb7537f19d8ef500d146c1f8fedcab7
```

动态资源、凭据、主机地址和私有 checkpoint 路径没有进入提交。

## 3. Cache geometry 与 KDA 隔离

TP8 layout 结果：

| 项目 | 结果 |
| --- | ---: |
| OE field | `[3] int32`，12 bytes/page |
| OE packing | 24,576 child pages/parent |
| plane 数 | 7 |
| plane bytes | 294,912 |
| LCM parent bytes | 2,064,384 |
| 2M token、D256、overlap depth 1 parent demand | 9,985 |

测试分别构造原 `KimiK3Recipe` 与 `LiteRecipe`，移除新增 OE field 后对全部既有
`CacheFieldLayout` 和 `group_packing` 做 dataclass exact 比较，结果相同。graph metadata 仍只建立三个
KDA state group；`lite_oe` 不出现在 KDA dual-index table 中。

## 4. Mirror、snapshot 与 fail-closed

CPU/meta 测试覆盖：

1. 新请求以三个 EOS token 初始化；
2. 同 request、同 slot、连续长度直接复用 mirror；
3. slot 改 owner 后从 page snapshot 恢复 12 bytes，不泄漏旧 context；
4. 128/129 token 边界切换 output page，continuation 与不分块 oracle 一致；
5. graph bucket 从两行缩到一行后固定 staging pointer 不变，尾行严格为零；
6. request slot、page 0、page 越界、重复 owner 和 ragged metadata 均 fail loud；
7. 普通 Decode 只有 `-1` device-token sentinel 时 fail closed。

OE state 的 transfer policy 为统一 cache contract 的 `latest_snapshot`，没有增加模型私有 sender、receiver
或第二份 transfer cache。真实 P→D 服务 admission 将在 decoder 和 CPU token publication 接通后做
端到端验证；本阶段只准入 state schema、page publication、restore 和统一 transfer contract。

## 5. 本地与 exact-source NPU

最终本地累计选择集：

```text
28 passed, 28 skipped
```

skip 均来自本地无 NPU 或完整 attention runtime；完整
`pre-commit run --all-files` 的全部 hook 通过。

最终 exact-source 在 NPU0 执行同一 Lite OE、hybrid cache、KDA、MLA 和 Grouped MoE 集合：

```text
56 passed, 2 warnings, 0 skipped
```

warning 是镜像未编译 TorchAir 的既知提示和关闭 internal format 后采用 base-format tensor 的既知提示；
没有失败、NaN/Inf、异常退出或源码回退。

## 6. NPU graph staging

独立 graph 用例先在 graph 外执行 host hash/lookup 和 H2D publication，再 capture
`[2,12,8] @ [12,8,96]` packed projection/merge。验证步骤为：

1. BS2 输入 warmup 后 capture，并记录 staging data pointer；
2. replay 与同输入 eager 输出 bitwise exact；
3. continuation 只写一行真实 OE，第二行 staging 清零；
4. 更新 word、token ID 和 OE staging 后再次 replay；
5. replay 与更新后的 eager 输出 bitwise exact、与首轮输出不同且全 finite；
6. capture 前后 staging data pointer 相同。

该证据验证的是 7B 定义的固定地址/固定 bucket seam；目标 `[12,256,3072]` packed projection 的 NPU
数值证据已由 7A 给出。完整 Decode graph 仍需等待 decoder 接线，不在这里重复声明。

## 7. 清理与后续边界

测试没有启动服务。结束后 candidate Python 进程为 0，NPU0--15 均显示无运行进程。

阶段 7C 继续完成真实 checkpoint 的 OE mmap/PSS、逐 rank HBM、目标维度 staging ledger，以及在完整
decoder 接线前可独立验证的 PD manifest。普通 Decode CPU effective-token publication、8P8D 服务和
GSM8K first100 仍按总体计划后续完成。
