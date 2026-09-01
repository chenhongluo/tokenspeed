# Lite NPU 阶段 9C.1：Ascend Mooncake 传输准入设计

## 1. 目标与根因

阶段 9C 已完成模型权重、cache arena 和 eager warmup，PD engine 在创建 Mooncake data-plane 时才
失败。当前共享 `MooncakeTransferEngine` 把 transport 固定为 `rdma`，同时 8P8D launcher 只导入
顶层 `mooncake` package；后者不能证明 native `mooncake.engine` 与当前设备后端匹配。

本阶段只补齐两个通用边界：

1. Ascend 平台初始化 Mooncake 时使用 `ascend_direct`，其它平台继续使用 `rdma`；
2. NPU launcher 在创建任何 worker 前验证 native `TransferEngine` 及 TokenSpeed 实际使用的方法集。

该修复不改变 Lite 模型数学、P/D cache layout、wire contract、rank mapping、buffer 地址、传输长度或
请求协议。PD 与 EPD 都继续复用同一个 transfer-engine wrapper，不增加 Lite 特判或第二套实现。

## 2. Transport 选择

唯一选择表如下：

| 当前平台 | Mooncake transport | 其它初始化参数 |
| --- | --- | --- |
| Ascend NPU | `ascend_direct` | 保持 `P2PHANDSHAKE`、hostname 和 device name |
| 其它平台 | `rdma` | 完全保持现有行为 |

平台信息复用 TokenSpeed 已有的统一 platform abstraction。判断只在 engine 初始化时惰性执行，避免让
CPU 侧 import 或不使用 Mooncake 的进程提前加载 accelerator kernel package。不增加环境变量、用户
CLI 或自动 fallback：Ascend native engine 不支持 `ascend_direct` 时必须在启动前失败，不能静默退回
`rdma` 后继续运行。

`register`、`deregister`、同步/批量/异步写、状态轮询和 session ID 生成均保持原实现。P/D 与 E/P
调用者不各自选择 transport，避免同一进程角色因调用入口不同而产生协议分叉。

## 3. Native ABI 门禁

正常启动模式新增独立的 Mooncake native probe；`--check` 继续只校验命令、拓扑和容量，不加载硬件
依赖。probe 必须成功导入 `mooncake.engine.TransferEngine`，并确认 class 暴露下列全部方法：

- `initialize`；
- `get_rpc_port`；
- `register_memory`、`unregister_memory`；
- `transfer_sync_write`、`batch_transfer_sync_write`；
- `transfer_submit_write`、`transfer_check_status`。

缺少 native extension、动态库或任一方法时，launcher 在创建 P/D engine 和 gateway 之前 fail closed。
顶层 `import mooncake` 不再作为能力证据。

当前目标 Ascend extension 在只做 import 的短进程退出阶段存在 native teardown 问题。probe 成功校验
后可用 `os._exit(0)` 结束该隔离子进程；这只隔离预检生命周期，不进入服务 worker，也不掩盖 import
或 ABI 失败。真正的 engine 初始化、注册和传输仍由服务进程按正常生命周期执行。

## 4. 依赖闭包

Ascend Mooncake native package 必须来自显式的阶段依赖目录，位于通用依赖 target 之前，并记录：

- package/version metadata；
- native extension 和 bundled shared libraries 的 SHA-256；
- import 后实际 `engine` 文件来源；
- 动态依赖中存在 Ascend runtime/transport，且不存在 CUDA runtime 依赖。

阶段目录可复用目标镜像已经提供且通过准入的 Ascend build，但必须复制到 phase-local overlay 后再运行，
不能依赖不透明的全局 import 顺序，也不能覆盖系统或通用依赖目录。公共 Python 依赖下限不在本阶段
修改；低版本 native package 只有在真实方法集、初始化、注册和传输板测全部通过后，才能作为当前
Ascend 环境的验证闭包。

## 5. NPU 板测顺序

在完整 8P8D 服务之前，使用一张空闲 NPU 做最小 direct board：

1. 加载 CANN，并证明实际导入的是阶段 Ascend overlay；
2. 构造 `TransferEngine`，以 `ascend_direct` 初始化并取得非零 RPC port；
3. 分配小型 NPU tensor，按其 device pointer/byte length 调用 `register_memory`；
4. 对同一 pointer 调用 `unregister_memory`；
5. 记录返回码、运行前后 NPU process 状态和 artifact digest；
6. 精确结束 probe，证明没有残留 Python/NPU 进程。

任一步失败都停止 8P8D 重试。不得用 CUDA build、`rdma` fallback、host-only buffer 或跳过注册来绕过。
单卡板测通过后，再由阶段 9C exact-source 8P8D 服务证明 P/D 16 个 engine 均完成初始化、cache arena
注册、真实 cache transfer 和请求生命周期。

## 6. 实现与测试边界

实现提交只允许修改：

- `python/tokenspeed/runtime/pd/base/mooncake_engine.py`：共享 transport 选择；
- `test/ci_system/serve_lite_npu_pd_8p8d.sh`：native ABI 预检；
- 一个 Mooncake engine CPU 单测；
- 既有 launcher 单测的预检契约补强。

CPU 单测使用 fake `TransferEngine` 和 fake platform，分别断言 NPU 初始化参数为 `ascend_direct`、非 NPU
仍为 `rdma`，同时覆盖非零初始化返回码和 session ID。launcher 测试继续执行 `bash -n` 与 `--check`，
并静态约束 normal-mode probe 必须读取 native class 和完整方法集；不在 CPU CI 伪造 NPU shared object。

提交前运行 focused pytest、`bash -n test/ci_system/serve_lite_npu_pd_8p8d.sh` 和完整
`pre-commit run --all-files`。实现提交推送后才生成新的 exact archive 和 overlay，不使用未提交工作树
上板。

## 7. 准入与回退

只有以下证据同时成立才准入：

- NPU/非 NPU transport 单测通过，原 CUDA `rdma` 行为未改变；
- launcher 对错误 native build 或缺失方法在 worker 创建前拒绝；
- NPU0 `ascend_direct` initialize/register/unregister 全部返回成功且清理完成；
- exact-source 8P8D 中 16/16 engine 完成注册，服务请求触发真实 P→D cache transfer；
- 阶段 9C 的4-token、late-admission BS2、finite、cache/HBM和清理门禁全部继续通过。

若目标 native build 缺 `ascend_direct` 或注册失败，回退是撤销本阶段实现并保持阶段 9C 未准入；不得
改动 cache wire、降低 rank 数或切换 aggregate serving。若后续错误要求改变 buffer ownership 或
P/D transfer contract，另开独立设计，不扩大本阶段。

## 8. 非目标

- 不实现 CP8/KVP8、PD fragment mapping 或 KDA/MLA one-copy；
- 不修改 Mooncake native C++/Ascend transport；
- 不降低或改写公共 Mooncake Python 依赖版本；
- 不增加用户可选 transport 开关；
- 不启用 Decode graph、overlap 或新增融合算子。
