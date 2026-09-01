# Lite NPU 阶段 12C：Featurewise-beta 融合验证记录

## 1. 结论

阶段 12C 组合候选负准入，执行源已回退到阶段 12A。

- Prefill 在 T32/T256/T1024 的完整 public KDA 链提升仅为
  `1.44%/0.98%/1.37%`，没有满足“至少两个 shape `>=3%`”的冻结门禁。
- Decode BS1/BS2 NPUGraph 分别提升 `77.18%/81.58%`，数值和 state 门禁通过；
  这份证据不能改变阶段 12C 的组合准入规则，只作为后续 Decode-only 独立候选的
  输入。
- 候选实现曾以 `a95b6fc0` 提交用于 exact-source 复验；发现 Prefill 门禁失败后，
  以 `a4adc838` 完整撤回。最终分支不保留 Prefill/Decode 候选 kernel、测试或不可达开关。

## 2. 验证环境与方法

- 设备：Ascend 910B，固定单卡。
- baseline：阶段 12A executable `5da19b9b`。
- candidate：`a95b6fc0`，只改动 Ascend KDA adapter 及 focused tests。
- public KDA artifact：两侧复用同一份已验 manifest，不替换 binary。
- 计时：同一进程动态加载 baseline `kda.py`，与 candidate 交替执行 11 轮，每轮 warmup
  后取中位数。Prefill 两侧都调用完整 `public_kda_paged_prefill`，Decode 两侧都采集
  graph replay。
- Prefill 输入使用生产 `fused_qkv_split_gdn_prefill` 输出的 contiguous Q/K/V 布局；
  Decode 使用 H4/K128/V128、BS1/BS2 和 K-major FP32 live state。

## 3. 精度与安全门禁

exact-source focused 测试为 `25 passed`，覆盖：

- Prefill T32/T256/T1024、空段 compact、final-state scatter 和 strided 能力；
- Decode eager/NPUGraph BS1/BS2、changed-input replay、padding、page 0 与未选邻页；
- 4 个 seed、每个 BS2 连续 128 步轨迹。

标准 Triton 验证器为 Prefill `3/3` 和 Decode `2/2`，BF16 output 与 FP32 state 均通过。
最终 exact A/B 中 Prefill output/state 全部逐元素 exact；Decode output exact，state max abs
为 `5.96e-8`，rel-L2 不超过 `4.76e-8`，低于 `1e-6` 门禁。

## 4. 性能结果

### 4.1 Prefill 完整 public KDA 链

| tokens | baseline us | candidate us | 节省 us | 提升 |
|---:|---:|---:|---:|---:|
| 32 | 2277.754 | 2244.987 | 32.767 | 1.44% |
| 256 | 2260.070 | 2237.904 | 22.167 | 0.98% |
| 1024 | 2273.255 | 2242.141 | 31.114 | 1.37% |

三个 shape 都低于 `3%`，Prefill 候选负准入。

### 4.2 Decode NPUGraph

| batch | baseline us | candidate us | 节省 us | 提升 |
|---:|---:|---:|---:|---:|
| 1 | 139.899 | 31.920 | 107.979 | 77.18% |
| 2 | 174.917 | 32.213 | 142.704 | 81.58% |

Decode 同时跨过 `>=5%` 和每层 `>=5 us` 两个门禁。

## 5. 原型结果修正

早期原型曾记录 Prefill `12%--15%` 提升。复核发现原型 candidate 把
`_prefill_chunk_plan` 和部分 wrapper 工作预先移出计时，而 baseline 仍计入完整
production wrapper；该数据不是 prepare kernel 的可得收益。

修正后的 exact A/B 使用相同 wrapper、host plan、公开 core、输入布局和 artifact，只替换
featurewise-beta 实现，因此以本记录为准。

## 6. 回退与后续

- 当前 executable 已恢复为阶段 12A 路径；不需要为一个最终无 source 改动的负准入阶段
  重复 8P8D/GSM8K。
- Decode 证据可供后续独立设计，但不在本验证记录中改变阶段 12C 门禁或恢复代码。
