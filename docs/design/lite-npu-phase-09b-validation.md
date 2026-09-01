# Lite NPU 阶段 9B：有界 Eager 8P8D Launcher 验证

## 1. 验证对象

本记录验证提交 `55530d636e90e93fcc05ec787560c6f5907f269b` 新增的
`serve_lite_npu_pd_8p8d.sh`。tracked-source archive SHA-256 为：

```text
1727a6e156dbcb523a2f8a135c9d8dc745dbe68f0cc154b2dc978d86ece0a906
```

验证只覆盖 launcher 静态门禁、命令构造和进程清理契约。真实 16-rank checkpoint 加载与服务请求属于
阶段 9C，本阶段没有占用 NPU 或启动 engine/gateway。

## 2. CPU 门禁

focused 集合为：

```text
test/ci_system/test_lite_npu_pd_launcher.py
test/ci_system/test_worker_cleanup.py
```

结果：

```text
8 passed in 1.56s
```

测试证明：

- `bash -n` 接受 launcher；
- `--check` 精确输出 Prefill、Decode 和 gateway 三条命令；
- P 使用 device 0--7，D 使用 8--15，两个集合不重叠；
- 两侧均包含 attention TP8、linear-attention TP8、dense TP8、MoE EP8；
- 两侧均固定 4,096 model length、8,192 total tokens、BS2、1,024 chunk 和 64-token granularity；
- 两侧均为 NPU BF16、MLA/KDA、greedy、eager、Prefill graph off、PDL off、overlap off；
- gateway 只注册一个 P endpoint 和一个 D endpoint；
- 缺失 model、错误 architecture、设备数量错误、P/D 设备重叠、超过有界容量及重复端口全部
  fail closed；
- `--check` 不创建日志目录，也不启动进程。

新增 launcher 同时进入既有 bounded-cleanup 回归，要求继续 source `worker_cleanup.sh` 并调用
`stop_worker_pids`。全量 `pre-commit run --all-files` 通过。

## 3. Exact-source 目标环境检查

同一 archive 部署到目标 CANN 环境后，以可读的完整 Lite checkpoint metadata 执行 `--check`。结果为
三行且退出码为 0：

```text
Prefill: ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Decode : ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
Gateway: one Prefill endpoint plus one Decode endpoint
```

两条 engine 命令解析出的公共配置完全一致，只在 role、visible devices、gRPC/bootstrap 和
rendezvous 端口上不同。目标 checkpoint 的 architecture、tokenizer metadata 和 safetensors index
静态门禁均通过。

目标检查额外确认：

- `--check` 指定的日志目录没有被创建；
- exact source 相关 Python/engine/gateway 进程数为 0；
- 原始三行命令日志已拉回仓库外，SHA-256 为
  `e7bdf3a2c3c362c06aa26d8c870ea531e301e777a147e0a20cdc8ebee327fad0`。

## 4. 结论与边界

阶段 9B 通过。公开 launcher 已能以单一入口、明确的 16 张卡 ownership 和有界 eager 参数构造
8P8D Lite PD 服务，并在任何进程启动前拒绝越界配置。

本结果不证明 SMG/Mooncake runtime 依赖齐备、16 个 worker 能加载模型、CachePD 能完成真实 transfer，
也不证明生成结果正确。阶段 9C 必须从该 exact launcher 正常模式启动，依次通过 runtime import、
16-rank `SERVING`、4-token、BS2 late admission、finite/cache/HBM 和精确清理门禁。
