# Lite NPU 阶段 1：完整 Checkpoint Replay 验证

## 1. 目的

本记录验证阶段 1 的 config、模型骨架和严格权重映射是否覆盖完整 Lite checkpoint。验证只读取
`config.json`、safetensors index 和各 shard header，不读取 tensor payload，也不记录 checkpoint
路径、原始 key 清单或设备连接信息。

## 2. Replay 方法

在阶段 1 实现提交的 exact source tree 上执行以下步骤：

1. 由 `LiteConfig.from_pretrained()` 解析原始 config，禁止补写 `model_type`；
2. 从 safetensors index 读取完整 `weight_map`，检查 key 和 shard 文件集合；
3. 读取每个 shard 的八字节 header 长度和 JSON header，校验 header 边界；
4. 对每个 tensor 比较 index shard、header shard、dtype、全局 shape 和
   `LiteCheckpointLayout.spec()`；
5. 比较 checkpoint key 集合与 `iter_source_names()` 的双向差集，并检查 source 和 normalized target
   的唯一性；
6. 对 Prefill 和 Decode 的八个 role rank 重放 dense TP8、KDA TP8、MoE EP8、MLA replicated 和
   host OE placement，校验每个 local shape、expert owner 和 OE residency；
7. 汇总不可逆计数与摘要，原始路径、key、header 和日志留在仓库外。

任何未知 key、缺失 key、重复 key、index/header shard 不一致、dtype/shape 不一致、无法整除的 shard、
缺失 local target 或错误 placement 都使本阶段失败。

## 3. 准入门槛

| 检查项 | 门槛 |
| --- | ---: |
| config 未分类数学字段 | 0 |
| index/header 未解释或缺失 tensor | 0 |
| dtype/shape 不一致 | 0 |
| source/target 重复覆盖 | 0 |
| TP8/EP8 local shape 不一致 | 0 |
| 每个 EP rank real expert 数 | 192 |
| MLA 权重错误切分 | 0 |
| OE 非 host-resident table | 0 |

此外，CPU/meta focused tests、NPU runtime focused tests 和全仓 pre-commit 必须全部通过。

## 4. 验证结果

完整 replay 在实现提交 `c30c58ceb5da0269d531ee23e14177459767e77e` 的 clean source tree 上
通过。config 和 index 与阶段 0 冻结摘要一致，10 个 shard header 全部完成边界、offset、dtype、shape
和 index owner 交叉校验。

| 项目 | 结果 |
| --- | ---: |
| config 字段 / 未分类 | `47 / 0` |
| 层数 / KDA / MLA | `28 / 21 / 7` |
| shard / tensor / normalized pattern | `10 / 129,870 / 39` |
| tensor payload | `138,634,018,688` bytes |
| unknown / missing source | `0 / 0` |
| index/header / dtype/shape error | `0 / 0` |
| duplicate source / target | `0 / 0` |
| local shape / placement error | `0` |
| MLA weight sharding error | `0` |
| OE residency error | `0` |

Loader 分类计数为：

| 类别 | Tensor 数 |
| --- | ---: |
| expert EP | `129,024` |
| rename | `315` |
| router | `224` |
| dense shard | `142` |
| replicated | `141` |
| host OE | `24` |

Prefill 和 Decode 的八个 role rank 得到完全相同的静态权重映射。每个 rank 拥有 `16,974` 个 source
和同数 target，其中 routed-expert 参数为 `16,128` 个；每层恰好拥有 `192` 个 real expert。全部
7 个 MLA 层的 projection 权重在八个 rank 上保持完整复制；每个 rank 的 12 张 OE 主表均为 host
resident。P/D 的差异只留给后续 cache 和执行并行阶段。

冻结摘要：

- config SHA-256：`d2676f137594dea8d3bcdd5a26703e7bf45f000845f9613de51fde6b7321dac6`；
- checkpoint index SHA-256：`2900b080361629254f560d90fe6333d9cfb8b8fdbb990df73c670c41b8ba96b7`；
- checkpoint manifest SHA-256：`c63fa13edc92dab81b9ea30ee22aa8750b9b9a47ddfb391119a731f60fa612eb`；
- loader layout SHA-256：`e01e80d466a15a7e8b5121e9201af008ddb88f7345a98f45fb315c68d5e3014f`；
- shard summary SHA-256：`e5aa9908fce93d945f0db881137018e82655d3d97ebfc7e6a2e1715773944dc9`；
- 仓库外 replay result SHA-256：`ab2ef0ef12fc63ad655c2445058b0a3e9f2cdd0c43bd6490776b8fa0850808c5`。

同一 exact source tree 的 NPU runtime focused tests 为 `12/12`；提交实现前的本地结果为
`10 passed / 2 platform-skipped`，全仓 pre-commit 通过。阶段 1 的 config、结构和加载契约因此准入；
数值 forward、cache 和算子仍按后续阶段分别实现，不能把本次 header replay 视为端到端推理通过。
