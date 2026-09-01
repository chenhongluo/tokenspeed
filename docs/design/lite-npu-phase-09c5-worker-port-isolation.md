# Lite NPU 阶段 9C.5：8P8D Worker 端口域隔离

## 1. 目标与根因

8P8D launcher 会同时启动 Prefill 和 Decode 两个 TokenSpeed engine。除显式的 API、bootstrap、
rendezvous、gateway、metrics 和 HCCL 端口外，每个 engine 的 `PortArgs.init_new` 还会从
`engine API port + [100, 1000]` 随机选择一个 distributed 初始化端口。

现有 launcher 只校验显式端口彼此不同。它的 P/D API 端口相邻，P/D 两个随机窗口几乎完全重叠；
bootstrap 和 rendezvous 端口也落在窗口内。真实启动已经复现随机端口与 bootstrap 完全相同，导致
bootstrap 在 worker 加载前以 `EADDRINUSE` 失败。启动前对固定端口做一次 bind probe 不能发现这个
进程内部的未来碰撞。

本阶段让 launcher 在创建 worker 前证明两个动态窗口与所有固定 listener 互斥。它不改变 runtime 的
通用端口分配器、模型、NPU kernel、Mooncake 或 cache 协议。

## 2. 端口契约

设 Prefill 和 Decode API 端口分别为 `P`、`D`，则两个动态窗口为：

- Prefill：闭区间 `[P + 100, P + 1000]`；
- Decode：闭区间 `[D + 100, D + 1000]`。

launcher 必须同时满足：

1. 两个动态窗口不重叠并且上界不超过 65535；
2. 除各自 API 端口外，所有显式固定端口均不落入任一动态窗口；
3. 原有固定端口唯一性、范围和 bind probe 继续保留；
4. `--check` 与正常启动执行相同的几何校验，错误在任何进程创建前报告。

默认值改为三个互斥区：P engine、P bootstrap/rendezvous、D engine/rendezvous。HCCL 仍沿用独立的
低端口区间，不改变阶段 9C.4 的 channel 契约。

## 3. 最小实现

实现仍只修改现有 8P8D launcher 及其测试：

- 调整存在碰撞风险的 P/D API、bootstrap 和 rendezvous 默认值；
- 在现有端口校验后增加两个闭区间的纯 shell 几何检查；
- 复用已构造的固定端口数组，不增加新配置对象或 runtime 特例；
- 为动态窗口重叠、bootstrap 落入窗口、rendezvous 落入窗口增加负例。

不修改 `PortArgs.init_new`：它只知道单个 engine，无法看见另一角色、gateway、metrics 和 HCCL 的
完整端口集合；跨角色隔离属于 launcher 的节点资源职责。

## 4. 验证

1. launcher focused test 和完整 pre-commit 通过；
2. exact-source `--check` 使用真实 checkpoint 通过；
3. 同节点 8P8D 的 P/D 均进入 SERVING，日志没有 `EADDRINUSE`；
4. 首个请求完成 Prefill、P→D cache transfer 和 Decode；
5. 失败或结束后按精确进程组清理，并证明固定端口、动态 worker 和 NPU context 全部释放。

若隔离后的启动仍出现端口占用，必须记录当时的 listener PID/命令和具体端口；不能继续扩大随机窗口、
增加重试或把 bind 错误归因于 NPU 通信。
