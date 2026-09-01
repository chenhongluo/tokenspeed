# Lite NPU 阶段 10A.2：Decode Graph State Publication

## 1. 现象与已排除边界

阶段 10A.1 已让 Decode BS1/BS2 graph 完成 capture 和真实 replay。首个 BS1 请求可以完整返回，
但同一 prompt 的连续 fresh/slot-reuse 请求不能保持 greedy token digest 一致。Prefill bootstrap token
与第一个 Decode token 可一致，分叉从下一 Decode step 开始；服务仍健康，未出现 NaN/Inf、OOM、
collective 或 transfer 错误。

以下叶子边界已在单卡 NPU graph 中排除：

- KDA causal-conv 与 recurrent state 从 capture dummy page 切换到 live page，并连续更新两步；
- MLA latent cache 的动态 write location、page table、sequence length 及下一步读取；
- capture stream 与调用方 execution stream 的 replay 顺序：调用方更新输入、replay、立即消费输出的
  64 轮压力板为逐轮 exact，增加 device-wide synchronize 不改变结果。

因此本阶段不修改 KDA/MLA 数学或叶子 kernel，也不在 replay 后盲目增加全局同步。

## 2. 诊断顺序

### 2.1 同批相同行判别

先在未改代码的 exact 8P8D graph 服务中同时提交两个相同 greedy 请求：

- 若 BS2 两行相同，而跨请求重放不同，优先检查 request slot/page 生命周期及 remote-prefill landing；
- 若同一 BS2 内两行已经不同，优先检查 graph 内跨 rank collective、padding/row metadata 或共享 workspace；
- 同时保留 sequential repeat，防止只证明 BS2 行内一致而漏掉 slot reuse。

请求只比较 token IDs、length 和 finish reason；不以文本字符串替代 token 比较。

### 2.2 Remote-prefill completion 可见性对照

只有 2.1 指向跨请求 admission 后，才在 `RemotePrefillDoneEvent` 已聚合全部 Prefill rank completion、
但 scheduler 尚未执行首个本地 Decode 的位置做一次诊断 A/B：

1. A 保持现有 direct-to-live completion；
2. B 在相同位置增加一次 device-wide synchronize；
3. 两轮使用相同请求序列、graph size 和 checkpoint，并比较 exact digest。

全局同步只用于定位。若 B 修复而 A 失败，继续寻找 transfer/runtime 提供的最窄 publication primitive；
最终实现不得保留每请求全局同步。若 A/B 同样失败，立即删除诊断改动，不扩大 PD 路径。

### 2.3 Executor slot/page/state 指纹

若 2.2 未确认 transfer publication 问题，在 graph 外、首个 Decode 前后采集最小 device-side 指纹：

- request pool slot 与每个 cache group 的实际 page IDs；
- KDA conv/recurrent live page；
- MLA prompt 尾页与首个 Decode write location；
- graph replay 后的 next-token state publication。

指纹必须绑定 request、step、layer/group 和 rank，并在请求结束后校验 slot reuse。探针不得执行 host
scalar 读取后再回写模型、不得改变 graph 输入或 collective 次序；若 host snapshot 会扰动 bitwise，改为
单独诊断运行，不能把该运行作为最终准入证据。

## 3. 根修复门槛

生产修复只允许落在所有相关 caller 共用的真实根边界，并满足：

- eager 路径不变，Decode graph BS1/BS2 不回退；
- 不增加第二份 KDA/MLA transfer cache；
- 不增加每请求 device-wide synchronize；
- 不放宽 exact repeat、slot reuse、BS2 或 finite 门槛；
- abort、late admission 和 request-pool slot reuse 不得读取旧页；
- runtime 保持 vendor-neutral，NPU 特有 primitive 只能经过既有 device/kernel 边界。

如果问题来自 graph collective 或 workspace，修复相应共享 graph wrapper/kernel adapter；如果来自
cache admission，只修复统一 landing/publication seam，不给 Lite 增加模型专用 scheduler 分支。

## 4. 测试与提交边界

1. 本设计文档独立 signed-off 提交并推送；
2. 诊断实现若只用于 A/B，不进入 production commit；负结果写独立验证记录；
3. 确认根因后，最小共享修复和一个能复现旧错误的行为测试独立提交并推送；
4. exact NPU focused、8P8D sequential/BS2/late-admission/slot-reuse、finite、HBM 和清理结果写入独立
   验证记录并提交推送；
5. 阶段 10A.2 通过前不启用 Decode overlap。
