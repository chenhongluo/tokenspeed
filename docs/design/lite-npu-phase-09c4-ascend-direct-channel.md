# Lite NPU 阶段 9C.4：AscendDirect Channel 环境设计

## 1. 目标与实测根因

阶段 9C.3 已让 Ascend KDA Prefill 回到 registry 自动选择。exact 8P8D 中 P/D 各八个模型、cache arena、
gRPC 和 gateway 均完成初始化；首个真实请求也已越过旧的 KDA `solution="triton"` 错误。新的共同首错
发生在 Prefill forward 之后、P 向 D 写 cache 之前：八个 P rank 的 AscendDirect 首次建连都在
`HcclCommPrepare` 返回失败，随后 Mooncake 报远端传输失败。

当前 NPU launcher 只设置 P/D 可见设备，没有提供 AscendDirect 建立跨角色 HCCL channel 所需的三类
节点信息：16 个物理 NPU 的 HCCN 地址表、自动建连/intra-RoCE 策略，以及 P/D 互不重叠的 HCCL listener
端口。已有 Ascend 8P8D launcher 同时提供这三类信息并已通过真实 PD 传输。本阶段补齐这一共享启动
契约，不修改模型、cache wire、Mooncake Python API 或 KDA 算术。

## 2. Channel 契约

正常启动时，launcher 在创建 worker 前完成以下工作：

1. 用驱动自带的 `hccn_tool` 读取物理设备 `0..15` 的 HCCN 地址，写入本次运行目录下的临时配置；
2. 严格校验配置恰好包含 `address_0` 到 `address_15`，每行只有一个非空地址；
3. P/D worker 都读取同一份 `HCCN_CONF_FILE`，从而使用相同的物理设备地址空间；
4. 两个角色都固定启用 `ASCEND_AUTO_CONNECT=1`、`HCCL_INTRA_ROCE_ENABLE=1`、
   `HCCL_INTRA_PCIE_ENABLE=0`，并使用有界的 HCCL connect/RDMA timeout；
5. P 使用 `PREFILL_HCCL_BASE_PORT..+7`，D 使用 `DECODE_HCCL_BASE_PORT..+7`，两个区间必须互不重叠，
   且与 launcher 的 API、bootstrap、rendezvous 和 metrics 端口全部不同。

默认 HCCL base 沿用已验证的 `8282` 和 `8382`。两者保留环境变量入口，便于同节点隔离运行；任何派生
端口越界、重复或无法 bind 都在创建 worker 前失败。`--check` 不调用硬件工具，但仍校验完整派生端口
集合并打印两个角色的 HCCL base；正常模式才生成并校验 HCCN 配置。

HCCN 文件属于运行产物，不写入源码、不进入日志正文或提交。launcher 只传路径，避免暴露节点地址。

## 3. 最小实现边界

实现只修改 8P8D NPU launcher 及其现有测试：

- 增加两个 role-local HCCL base 参数和 16 个派生端口的 preflight；
- 增加一个小型 `prepare_hccn_conf`，复用系统 `hccn_tool`，不引入依赖或第二个 launcher；
- 在两个已有 worker 子 shell 内设置相同 channel 策略、不同 base port；
- 测试覆盖默认值、端口冲突/越界、缺失或畸形 HCCN 输出，以及 P/D 环境隔离。

不把 HCCN 生成放入 Python runtime：这是节点启动资源，不属于 vendor-neutral TokenSpeed runtime，也不应
让非 NPU/Mooncake 调用者承担 Ascend 驱动依赖。

## 4. 上板验证

实现提交推送后生成新的 exact archive，并在同一 16-card 节点按以下顺序验证：

1. `--check` 和正常 preflight 均通过，manifest 记录源码 SHA、HCCN 文件 SHA 和两个 HCCL base；
2. 8P8D 三角色 ready，16 个 engine/cache arena 全部注册；
3. 首个 4-token 请求完成 P forward、AscendDirect P→D 写入和 Decode，不再出现
   `HcclCommPrepare`、Mooncake transfer 或 KDA backend 错误；
4. 完整 admission 覆盖 fresh、slot reuse、standalone、late-admission BS2 与 post-BS2 replay；
5. 日志逐 token 检查 NaN/Inf、runtime/HCCL/OOM/5xx，并记录 P/D HBM 与 cache arena；
6. 拉回独立验证记录后精确停服，确认所有 listener 与 16 个 NPU context 清空。

若只设置环境仍不能完成两 rank direct transfer，不继续增加 timeout 或重试；下一步转向 native Mooncake
版本、物理 device mapping 和 channel ABI 的最小双进程 board。

## 5. 非目标

- 不实现 CP8/KVP8、分布式 KV page 或 KDA/MLA one-copy；
- 不切换 Mooncake transport，不修改 native AscendDirect；
- 不在本阶段启用 graph、overlap 或其它融合；
- 不在本提交修复远端 SSH 断开后的进程组清理，该问题单独设计和验证。
