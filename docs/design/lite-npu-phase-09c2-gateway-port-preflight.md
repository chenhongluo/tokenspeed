# Lite NPU 阶段 9C.2：Gateway 通配地址端口预检设计

## 1. 目标与根因

阶段 9C.1 的 exact 8P8D eager 启动中，Prefill 和 Decode 均完成模型加载、AscendDirect cache arena
注册并进入 gRPC `SERVING`，但 gateway 在创建 metrics listener 时因端口冲突退出。launcher 已有端口
预检，却把所有端口只绑定到 `127.0.0.1`；gateway 与 metrics 实际绑定通配地址时，另一个进程若只占用
非 loopback 接口，loopback 探针仍可通过，随后的通配地址 bind 才失败。

本阶段只修正 launcher 的地址/端口预检，使探针与真实 listener 的 bind 地址一致。它不改变模型数学、
P/D 拓扑、Mooncake、cache wire、请求协议或端口默认值。

## 2. 唯一预检映射

| listener | 实际绑定地址 | 预检地址 |
| --- | --- | --- |
| Prefill engine/bootstrap/rendezvous | `127.0.0.1` | `127.0.0.1` |
| Decode engine/rendezvous | `127.0.0.1` | `127.0.0.1` |
| gateway API | `LB_HOST` | `LB_HOST` |
| gateway metrics | `0.0.0.0` | `0.0.0.0` |

预检仍在创建任何 P/D/gateway 进程之前完成，并保持所有端口必须为合法、互异正整数。任一精确地址的
bind 失败都 fail closed，不能自动换端口或杀死占用者。

## 3. 最小实现

复用 launcher 现有标准库 `socket.bind` 探针，不新增依赖或通用端口分配抽象。shell 传入按顺序配对的
`host:port` 参数，Python 子进程逐个解析并同时持有全部 socket，防止同一组端口在探针内部被重复使用。

错误信息必须包含监听角色、地址和端口，方便区分 gateway API 与 metrics 冲突；不得打印进程环境、
checkpoint 或远端资源信息。

## 4. 测试与准入

CPU 测试覆盖：

1. 七个空闲端口的 `--check` 仍通过并输出唯一 8P8D 命令；
2. 通配地址占用 metrics 端口时，即使 loopback 绑定语义不同，`--check` 必须在进程创建前失败；
3. 已有重复端口、非法拓扑和容量负例继续通过。

提交前运行 launcher focused pytest、`bash -n` 和完整 `pre-commit run --all-files`。实现提交推送后，
重新生成 exact source archive，并用空闲的显式 metrics 端口重跑阶段 9C 请求、finite、cache/HBM 和
清理门禁。验证成功后再形成独立上板记录。

## 5. 非目标

- 不自动寻找或随机分配 production 端口；
- 不修改 gateway/SMG listener 实现；
- 不把端口冲突解释为模型、Mooncake 或 NPU 算子失败；
- 不在本阶段启用 Decode graph、overlap 或分布式 KV page。
