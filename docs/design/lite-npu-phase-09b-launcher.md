# Lite NPU 阶段 9B：有界 Eager 8P8D Launcher 设计

## 1. 目标

本阶段提供一个单节点 16-rank Lite PD launcher：Prefill 使用 NPU 0--7，Decode 使用 NPU 8--15，
SMG gateway 提供一个 OpenAI 兼容入口。Launcher 固定阶段 9A 已准入的 replicated-MLA lockstep
mapping 和有界 eager 配置，并在启动任何进程之前 fail closed。

本阶段只交付启动入口及其无设备 `--check` 测试，不启动真实 checkpoint。完整模型加载、服务联通、
4-token/BS2 请求和资源清理由阶段 9C 验证。

## 2. 复用边界

实现直接复用现有 `serve_qwen35_397b_nvfp4_pd_1p1d.sh` 的三个进程边界：两个统一 TokenSpeed gRPC
engine 加一个 `smg launch --pd-disaggregation` gateway。进程退出继续复用
`worker_cleanup.sh::stop_worker_pids`，readiness 继续以 worker 日志中的 `health status -> SERVING` 和
gateway `/v1/models` 为准。

不增加通用 launcher framework、Python command builder 或第二套进程管理器。Lite 的增量只有 NPU
设备变量、固定并行参数、有界 cache 参数和启动前验证。

## 3. 公开入口

新增：

```text
test/ci_system/serve_lite_npu_pd_8p8d.sh [--check]
```

checkpoint 没有公开默认值，调用方必须通过 `MODEL` 传入本地目录。`SERVED_MODEL_NAME`、端口、日志
目录和 Python executable 可通过环境覆盖；脚本不包含主机、凭据、代理、内部路径或自动下载逻辑。

默认设备与容量：

| 参数 | Prefill | Decode |
| --- | --- | --- |
| visible NPU | `0,1,2,3,4,5,6,7` | `8,9,10,11,12,13,14,15` |
| world size | 8 | 8 |
| attention TP/CP/DP | 8/1/1 | 8/1/1 |
| linear-attention TP | 8 | 8 |
| dense TP | 8 | 8 |
| MoE TP/EP | 1/8 | 1/8 |
| max model length | 4,096 | 4,096 |
| max total tokens | 8,192 | 8,192 |
| max sequences | 2 | 2 |
| chunked prefill | 1,024 | 1,024 |
| prefix granularity | 64 | 64 |

两个 engine 都固定：

```text
--device npu
--dtype bfloat16
--kv-cache-dtype auto
--attention-backend mla
--kda-backend auto
--sampling-backend greedy
--enforce-eager
--disable-prefill-graph
--disable-pdl
--disable-overlap-schedule
--disable-autotune
--disable-kvstore
--disaggregation-transfer-backend mooncake
--disaggregation-layerwise-interval 0
```

`--enforce-eager` 明确关闭 Decode graph；阶段 10 才允许移除。`temperature=0` 是请求参数，不伪装成
engine CLI。`max_total_tokens=8192` 是每个 role 的逻辑池容量；MLA cache replicated，因此不乘 8。

## 4. 启动前门禁

脚本在创建日志目录或进程前依次验证：

1. 只接受空参数或单个 `--check`；
2. `MODEL` 是可读目录，`config.json` 的 architecture 精确为 `FLASHLocalForCausalLM`，tokenizer
   metadata 和 safetensors/index 至少各有一项可读；
3. P/D 各恰好 8 个十进制 device id，单侧不重复、两侧不相交；
4. world size 必须为 8，容量、并发和 chunk 参数为正，且 chunk 不超过总 token 数；
5. engine、bootstrap、rendezvous、gateway 和 metrics 端口均为合法整数、彼此不同且当前可绑定；
6. normal mode 额外检查 `torch`、`torch_npu`、`tokenspeed`、`tokenspeed_kernel`、
   `tokenspeed_kernel_npu`、Mooncake 和三项 `tokenspeed-smg*` runtime import；
7. normal mode 检查 16 张指定 NPU 可见，随后才创建 P/D worker。

Launcher 不安装包。若 SMG 缺失，阶段 9C 的部署步骤按 `python/pyproject.toml` 精确 pin 安装到独立
phase-local target，再把该 target 放入显式 `PYTHONPATH`；基础 Python 和系统 site-packages 保持不变。

## 5. `--check` 契约

`--check` 运行全部静态门禁并打印三条 shell-escaped 命令：Prefill engine、Decode engine 和 gateway；
它不检查 NPU/runtime import、不创建日志目录、不监听端口、不启动或终止任何进程。这样 CPU CI 可以
覆盖最终 command construction，而不会增加测试专用 platform backdoor。

输出必须能证明：

- P/D 都包含完整且相同的 8P8D mapping/cache/eager 参数；
- 两条 engine 命令只在 role、visible devices、gRPC/bootstrap 和 rendezvous 端口上不同；
- gateway 只注册一个 P 和一个 D endpoint；
- 输出不含 credential、proxy 或调用方未提供的私有路径默认值。

## 6. 进程生命周期

normal mode 同时启动 P/D engine，分别导出自己的 `ASCEND_RT_VISIBLE_DEVICES`。任一 worker 在进入
`SERVING` 前退出，脚本打印对应日志尾部并通过统一 trap 停止另一 worker。两侧均 ready 后才启动
gateway；gateway readiness 必须返回带 `data` 字段的 JSON model list，不能只接受任意 HTTP 200。

脚本阻塞在任一子进程退出。正常 SIGINT/SIGTERM、readiness timeout 或子进程异常都走同一个 bounded
cleanup；先 TERM，超时后只 KILL 记录过的进程树，不扫描或停止机器上的其它 TokenSpeed workload。

## 7. 测试

新增一个 CPU pytest，使用临时最小 Lite checkpoint metadata 运行 `--check`，验证：

- `bash -n` 通过；
- 三条命令和全部固定 mapping/eager/cache 参数存在；
- P/D visible-device 集合为 0--7/8--15 且不重叠；
- `MODEL` 缺失、architecture 错误、设备数量错误、设备重叠、端口重复和非法容量均返回非零；
- `--check` 后没有创建日志目录或遗留进程。

同时把新脚本加入既有 `test_worker_launchers_use_bounded_cleanup` 列表，防止后续删除 trap 或绕过
`stop_worker_pids`。提交前执行 focused pytest、`bash -n` 和 full pre-commit。

## 8. 准入与回退

9B 通过条件是静态/命令构造测试全部通过，且 exact source 在目标 CANN 环境执行 `--check` 时输出同一
拓扑。它不等同于服务可用。

若 normal mode 在 9C 失败，只修复实际失败的 preflight、参数或既有 runtime 根因；不放宽 mapping、
容量、strict checkpoint、readiness 或 cleanup 门槛，也不切换到 aggregate serving、CP8、attention-DP8、
V2 MoE 通信或分布式 KV 的未准入路径。删除该单一 launcher 即可回退 9B，不影响阶段 1--9A。
