# Lite NPU 阶段 2：混合 Cache 验证记录

## 1. 目的

本记录验证 Lite 是否通过最小 architecture 注册复用现有 hybrid MLA+KDA cache，并确认阶段 2 设计
冻结的字段形状、packing、容量、arena view、复用清零和 graph metadata 地址契约。验证不加载完整
checkpoint，不运行 attention 数值 forward，也不把 metadata 地址测试解释为真实 NPU graph 已准入。

## 2. 验证对象

设计提交为 `201ea554a21b8d30a7b2e4523ff69d07c81431d3`，实现提交为
`5aadb92cccd660cd6b9c76962a53bbf5cf10dd7d`；实现测试的机械格式修正提交为
`264a37bc5b475a1c4cc0b012c063f8d668d481be`，也是本记录最终复验的 exact source。实现只包含：

- 将 `FLASHLocalForCausalLM` 加入已有 hybrid MLA+KDA architecture set；
- 一个 focused test 文件，直接验证现有 `KimiK3Recipe`、`CacheArena`、
  `HybridKDATokenToKVPool` 和 linear-attention graph metadata。

没有新增 Lite recipe、cache manager、allocator、metadata class、kernel 或依赖。验证在实现提交的
clean exact source tree 上进行；本地、tracking branch 和 NPU exact source SHA 一致。

## 3. 验证方法

### 3.1 Registry 与 layout

1. 在完整 NPU runtime 中导入 attention registry，检查 Lite 同时进入 hybrid MLA+KDA architecture
   set 和 `LinearAttnConfig` registry；
2. 用真实 Lite 默认 config 构造 N=8 和 N=1 的 `KimiK3Recipe`；
3. 检查 21 KDA/7 MLA、三个 7-layer state group、PD transfer policy、field dtype/shape/bytes；
4. 对 packing、plane bytes、LCM parent bytes 做整数精确比较；
5. 对 2 × 1024 × 1024 token、256 live request、Decode width 1、overlap depth 1 做 parent 和
   token capacity 反算。

### 3.2 Arena、复用和 metadata

1. 绑定小容量真实 `CacheArena`，检查 allocation bytes 与 plan 相等；
2. 检查 21 层 conv/recurrent view 均存在、同一层组件共享 arena storage、不同 layer 的可写起始地址
   不同；
3. 为三个 state group capture 固定的 `state_in/state_out` buffer，replay 新 operation-bound table，
   检查 data pointer 不变、group 值独立、padding row 为 `-1`；
4. 在单张 Ascend NPU 上给同一 state block 的 null/target/neighbor 三行写入不同值，调用真实
   `zero_new_blocks()`，同步后检查只清零 target block。

metadata 测试复用 backend 的 operation-bound `require_table` seam，并使用最小 freshness-checked
stub；scheduler 的 abort/retract/readmission 状态机已有通用测试，本阶段没有复制 scheduler fixture。

## 4. 精确结果

### 4.1 Layout

| 项目 | N=8 / KDA TP8 | N=1 / KDA TP1 |
| --- | ---: | ---: |
| KDA conv shape | `[1536, 3]` | `[12288, 3]` |
| KDA recurrent shape | `[4, 128, 128]` | `[32, 128, 128]` |
| State bytes/layer/snapshot | 271,360 | 2,170,880 |
| MLA latent shape | `[128, 1, 576]` | `[128, 1, 576]` |
| MLA latent bytes/layer/block | 147,456 | 147,456 |
| History/state packing | `2 / 1` | `15 / 1` |
| Plane bytes | 294,912 | 2,211,840 |
| LCM parent bytes | 2,064,384 | 15,482,880 |

2M/256-request 场景的 `parents_needed` 为 `9,984`，反算 token capacity 精确为 `2,097,152`。
该结果是阶段 2 未应用 CP8/KVP8 的逻辑 layout，不是最终 8P8D 每 rank cache 分配量。

### 4.2 Contract 与生命周期

| 检查项 | 结果 |
| --- | ---: |
| Architecture/linear-attention registry error | 0 |
| Layer/group count error | 0 |
| Field dtype/shape/byte error | 0 |
| Packing/parent byte error | 0 |
| Arena allocation byte error | 0 |
| 缺失 state layer view | 0 |
| 跨 layer 可写起始地址别名 | 0 |
| Graph metadata pointer 变化 | 0 |
| 跨 group block ID 混用 | 0 |
| Padding slot error | 0 |
| NPU block 清零范围错误 | 0 |

NPU block 复用测试中，目标 conv/recurrent block 全零；null block 和 neighbor block 分别保持原写入值，
证明 `HybridKDATokenToKVPool.zero_new_blocks()` 通过共享 arena 的 group-aware byte segments 工作。

### 4.3 测试门禁

| 门禁 | 结果 |
| --- | --- |
| 本地 Lite Phase 1+2 focused | `11 passed / 8 environment-skipped` |
| exact source NPU Lite Phase 1+2 focused | `19 passed` |
| NPU exact source worktree | clean |
| 完整 `pre-commit run --all-files` | passed |
| 本地 / tracking / NPU source SHA | exact |

本地 skip 来自无 accelerator platform 和未安装完整 attention runtime 的可选量化依赖；相同测试在
NPU exact source 上没有 skip。NPU runtime 发出未编译 torchair 的既有 warning，不影响本阶段 eager
allocation、Triton block zeroing 或 CPU metadata ABI 测试。

仓库外 exact-source bundle 的 SHA-256 为
`17cdc921fc6627e38f9b04779875ecdef47f427e7bca7b53e223ae190f0d9044`。

## 5. 结论与边界

阶段 2 准入：Lite 已复用现有 `kimi_k3` storage family，cache layout、容量、arena 绑定、state group
选择、复用清零和 graph-stable metadata 均满足设计契约。生产改动保持为一行 registry 注册。

本记录不证明 KDA/MLA 数值 forward、真实 NPU Decode graph、CP8/KVP8 或 PD one-copy；这些能力分别
由后续算子、并行和 transfer 阶段验证。下一阶段可以在该固定 cache ABI 上实现 Lite featurewise-beta
KDA eager baseline。
