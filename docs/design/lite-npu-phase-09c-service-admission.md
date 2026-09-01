# Lite NPU 阶段 9C：Exact 8P8D Eager 服务准入设计

## 1. 目标与边界

本阶段在单节点 16 张昇腾 NPU 上，以阶段 9B 的唯一 launcher 启动完整 Lite PD 服务：Prefill 使用
8 rank，Decode 使用另外 8 rank，两个 role 都保持 eager、overlap off 和有界 4K/BS2 配置。验证覆盖
exact checkpoint 加载、SMG/Mooncake 接线、4-token 确定性生成、BS2 late admission、全链路 finite、
cache/HBM 账本以及精确清理。

本阶段是最终 CP8/KVP8 之前的 correctness admission。MLA cache 仍按阶段 9A 的 replicated-MLA
lockstep 拓扑保存，P/D 通过 1:1 local-rank 传输完整页；它不证明分布式 KV page ownership、PD
fragment transfer、KDA one-copy、2M token 容量、Decode graph 或 overlap。上述能力继续按总体计划
独立设计和准入，不能由本阶段结果外推。

## 2. Exact source 与依赖闭包

每次运行使用新的阶段目录，至少包含：

1. 从已推送 `hz/lite` commit 生成的 tracked-source archive；
2. archive SHA-256、commit SHA 和逐文件 manifest；
3. `python/pyproject.toml` 固定版本的三项 `tokenspeed-smg*`、Mooncake 及运行时缺失依赖；
4. 与 source 同一 revision 的 `tokenspeed_kernel` 和 `tokenspeed_kernel_npu` Python/二进制产物；
5. 运行命令、环境版本和证据文件 SHA-256。

依赖只安装到阶段目录的 target/venv，不写系统 site-packages，不覆盖基础 Torch、Torch-NPU 或 CANN。
运行时使用显式 `PYTHONPATH` 连接阶段依赖、exact source 和两个 kernel package。开始启动前执行一个
独立 import/version probe；缺模块、版本不符、导入来源不在阶段目录或 exact source 时直接停止。

checkpoint、目标主机、凭据、代理和阶段目录的私有绝对路径只出现在仓库外运行记录中，不写入脚本、
报告或提交。

## 3. 启动前门禁

正常模式调用 `test/ci_system/serve_lite_npu_pd_8p8d.sh`，不复制第二份 launcher。创建进程前必须证明：

- source commit、archive SHA 和远端解压后 tracked manifest 一致；
- checkpoint architecture、tokenizer metadata、safetensors index 可读；
- 16 张目标 NPU health 正常、无本任务或其它进程，P/D device 集合恰好为 0--7/8--15；
- engine、bootstrap、rendezvous、gateway 和 metrics 端口全部不同且可绑定；
- CANN、Torch、Torch-NPU、Mooncake、TokenSpeed、两个 kernel package 和三项 SMG 包导入成功；
- launcher `--check` 在同一环境仍精确输出两条 8-rank engine 命令和一条 gateway 命令；
- 运行根目录不存在或为空，PID 文件和日志不会覆盖旧证据。

启动前保存 NPU process/HBM baseline。任一门禁失败时不得创建 engine/gateway，也不得通过降低 world
size、关闭 strict loader、切换 aggregate serving 或修改 checkpoint 来绕过。

## 4. 服务生命周期

Launcher 同时启动 P/D engine，分别形成一个且仅一个 8-rank HCCL world。准入顺序固定为：

1. 16/16 worker 完成 strict checkpoint load；
2. Prefill 和 Decode gRPC health 都进入 `SERVING`；
3. gateway `/v1/models` 返回包含目标 served name 的 `data`；
4. 保存 ready 后的 NPU process/HBM、cache geometry 和 engine 配置；
5. 依次执行第 5 节请求矩阵；
6. 保存请求后证据并向 launcher 发送一次 SIGTERM，统一 trap 完成清理。

若任一 engine 在 ready 前退出、gateway 未 ready、请求失败或诊断门禁失败，保留首错和完整日志后走
相同 cleanup；禁止直接杀机器上未记录的 Python 进程。服务成功也必须主动停止，不能把长期运行当作
验证结果。

## 5. 请求矩阵

请求使用 OpenAI-compatible `/v1/completions`、greedy、`temperature=0` 和 `ignore_eos=true`，避免 EOS
提前结束掩盖 Decode 步数。仓库只保存匿名 prompt 编号和 token/text digest，不保存业务内容。

| Case | 执行方式 | 硬门禁 |
| --- | --- | --- |
| model list | ready 后查询一次 | served name 唯一且 endpoint 无错误 |
| fresh 4-token | 同一冻结 prompt 独立执行两次，`max_tokens=4` | 两次均恰好 4 token、非空、finish 正常，输出 digest exact |
| slot reuse | fresh case 完成后第三次执行同一 prompt | 与前两次 digest exact，证明释放后没有残留 state |
| BS2 late admission | A 以 streaming 生成 32 token；收到 A 首 token 后才提交 B 的 4-token 请求 | A/B 均完成；B 确实晚于 A 首 token admission；B 与其 standalone digest exact |
| post-BS2 replay | BS2 完成后再次执行 B | 输出与 standalone exact，slot ownership 未串扰 |

每个 request summary 记录匿名 request id、提交/首 token/完成时间、HTTP 状态、completion token 数、
finish reason 和输出 SHA-256。任一请求超时、空输出、token 数不足、API error、输出不确定或 B 在 A
首 token 之前被提交都使本阶段失败。

## 6. Finite 与 state 隔离诊断

阶段 8B 的 checkpoint probe 已覆盖静态参数 finite，但不能替代真实服务 forward。本阶段增加一个
Lite-only、默认关闭的 eager 诊断开关；它不进入非 Lite 模型，也不在后续 graph capture 内启用。
实现复用现有 Lite forward 和 `CacheArena` 字段视图，不注册通用框架级 hook。

每个实际 forward 检查以下边界：

- token embedding 与 OE projection/merge 后 hidden；
- 每层 attention 输入/输出、attention residual、MoE 输入/输出和层输出；
- KDA QKV/core/output、目标 conv state 和目标 recurrent state；
- MLA query/latent/output 和本次写入的 latent cache entry；
- OE 本次 staging 及写入的三 token context；
- final norm hidden 和 logits。

cache 只扫描本轮 `out_cache_loc`、request slot 或 page table 指向的已写区域，不扫描整个预分配 arena，
也不把 tensor 内容复制到文件。每个 role/rank 只输出 stage、layer、shape、dtype、检查元素数、finite
布尔值和累积 digest；发现第一个 NaN/Inf 时同步失败并记录 non-finite count。诊断开关打开时允许 eager
同步成本，本阶段没有性能目标；默认关闭时 forward 不增加检查和文件 I/O。

针对 late-admission case，诊断摘要还必须证明 A/B 使用不同 request slot，B 的初始 KDA/MLA/OE state
来自自己的 PD manifest，A 完成或释放后未改写 B 的 live slot。该证明只使用 slot/page id 和 digest，
不导出真实 cache 内容。

## 7. Cache 与 HBM 账本

从 engine 日志中的 `CacheArena` plan/runtime contract 和 ready 后的设备采样生成逐 role/rank 账本：

- logical token capacity、prefix granularity、LCM parent 数；
- 每个 cache group 的 page count、packing、payload bytes、transfer policy；
- arena allocated bytes，以及 MLA、三组 KDA state、OE context 的逻辑归因；
- checkpoint load 前、load 后、cache allocation 后、ready 后和请求后的 NPU process HBM；
- P/D 每 rank min/max/median，最大 rank 与最小 rank 的差值；
- OE 主表继续为 host file-backed，任何约 27 GiB/rank 的 HBM 副本立即失败。

实际 token capacity 必须不少于 launcher 声明的 8,192；P/D cache contract 的 field order、dtype、page
geometry 和 wire digest 必须一致。HBM 只设置安全门禁：16 rank 均不得 OOM，ready 后仍有正余量，
请求后不得出现随请求完成持续增长的 GiB 级泄漏。性能比较留到 graph/overlap 阶段。

## 8. 证据、测试与提交边界

实现提交只允许包含：

- 最小 Lite eager finite/state probe；
- 一个标准库 service admission client/aggregator；
- 对应 CPU 单测和 launcher 环境接线。

CPU 测试覆盖 4-token/BS2 时序聚合、digest 确定性、NaN/Inf 负例、错误 slot/page、匿名化输出、证据
schema 和 cleanup 契约。提交前运行 focused pytest、launcher `bash -n` 和 full pre-commit。实现提交
推送后再生成新的 exact archive，不使用设计 commit 或未推送工作树执行上板。

NPU16 运行完成后把仓库外证据拉回并逐文件校验 SHA-256，独立验证记录只写匿名结果、exact commit、
archive/evidence digest、版本、逐 rank 汇总和清理状态。验证记录使用单独 signed-off commit 并立即
推送 `hz/lite`；原始日志、checkpoint 路径、主机和凭据始终留在仓库外。

## 9. 准入与回退

只有 16/16 ready、全部请求、全部 finite/state、cache/HBM 和清理门禁同时通过，阶段 9C 才准入。
失败时先定位实际 source/runtime 根因；不降低并行度、请求数、strict coverage 或 finite 范围。

本阶段新增诊断和 client 都是可删除边界，production 数值路径仍由阶段 1--9B 的实现提供。若修复需要
改变模型数学、cache wire contract 或 parallel ownership，则停止 9C 并为根因另写独立设计，不把
扩大的修复夹进本阶段验证提交。
