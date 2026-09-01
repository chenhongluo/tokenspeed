# Lite NPU 阶段 7A：OE Host Storage 与数学基线

## 1. 范围

本提交只实现并验证四件事：

1. 12 张 OE 主表直接接管 checkpoint CPU storage，不做整表 `copy_`；
2. 用最近 3 token 计算 12 路 local n-gram ID；
3. 将 12 个 post projection 装入一份连续 Parameter，并做一次 block matmul；
4. 实现 `/sqrt(13)` 与 special-token base-only merge。

cache recipe、PD snapshot、slot mirror、Decode graph staging、完整 embedding/model forward 均留在 7B
及后续集成阶段。本阶段 API 只接受显式 `initial_context` 和 ragged `lengths`，不私自拥有请求状态。

## 2. 最小参数布局

`LiteNgramParameters` 保留 12 个独立 host table Parameter，名字继续严格对应：

```text
model.ngram_embeddings.embedders.{0..11}.weight
```

每个 Parameter 初始化为空 CPU BF16 storage；加载时直接令其 data 指向已校验的 source tensor。使用空
Parameter 而不是 `UninitializedParameter.materialize(full_shape)`，避免 materialize 短暂创建 27 GiB
anonymous storage。

12 个 checkpoint source：

```text
model.ngram_embeddings.post_projs.{i}.weight [H,K]
```

映射到唯一 target：

```text
model.ngram_embeddings.projection [G,K,H]
```

loader 按 component `i` 写入 `projection[i] = source.T`。source coverage 仍逐个检查，因此共享一个
target 不会掩盖缺失、重复或乱序。projection 是普通 device Parameter，大小固定 18 MiB/rank。

## 3. mmap adoption 门禁

host-OE source 在进入 adoption 前已经通过全局 source name、shape 和 BF16 检查；adoption 分支额外
要求 source 位于 CPU。它只替换 Parameter storage，不改变 source stride、内容或生命周期。

```text
before: empty CPU Parameter
after:  Parameter.data aliases loaded CPU tensor storage
```

普通 replicated/sharded/dense/expert 权重完全不进入此分支。非 safetensors 测试 tensor 也可以被
adopt，但只有 file-backed safetensors 具备跨进程 OS page-cache 共享保证。

本阶段测试用真实小 safetensors 文件证明：

- source 与 Parameter 的 storage data pointer 相同；
- iterator/result 临时对象释放并 GC 后 Parameter 内容仍可读；
- `/proc/self/maps` 中 data pointer 仍落在 safetensors file mapping；
- 不出现与表同尺寸的 anonymous Parameter storage。

## 4. CPU n-gram oracle

输入契约：

```text
input_ids       CPU int32/int64 [T]
initial_context CPU int32/int64 [B,3]
lengths         B 个非负整数，sum(lengths)=T
```

输出：

```text
local_ids       CPU int64 [T,12]
special_mask    CPU bool  [T]
final_context   CPU int64 [B,3]
```

实现不按请求写 Python token 循环。先从 flat ragged batch 一次构造每个 token 的
`[t-3,t-2,t-1,t]` 小窗口；窗口越过本次 chunk 起点时从 `initial_context` 读取。special token 映射为
0，任一候选历史到当前 token 的闭区间出现 0 时，该历史项失效。随后只保留 12 次固定 component
循环，按各自 `rows_i` 和预计算 `pow(V,d,rows_i)` 生成 ID。

`final_context` 保存最近 3 个原始 token，而不是 clean token；下一步计算时重新应用同一 special/boundary
规则。这样 snapshot 不丢失 token identity，且 special 后续仍严格断开历史。

## 5. Lookup、projection、merge API

host lookup：

```text
lookup_host(local_ids) -> BF16 CPU [T,12,256]
```

每路只索引自己的表，不构造 56,623,248 行融合表，也不做跨 rank collective。

device 侧：

```text
raw      [T,12,256]
packed   = raw.reshape(T,3072)
oe       = packed @ projection.reshape(3072,3072)
ordinary = (word + oe) / sqrt(13)
special  = word
```

本阶段把该边界实现为 `project_and_merge(word, raw, input_ids)`。7B 可把 `raw` 替换成 graph 固定 staging，
无需改 projection/merge。空 `T=0` 返回形状一致的空 tensor，不执行 matmul。

special mask 使用已经注册在同 device 的 23-ID non-persistent buffer做广播比较；不依赖 `torch.isin`，
以保持后续 NPUGraph 的简单固定算子链。

## 6. 测试

实现提交至少覆盖：

1. 小配置 loader：12 表 adoption、12 projection slice transpose、strict target/source coverage；
2. 真实小 safetensors mmap 生命周期和 pointer alias；
3. 手算 2/3/4-gram、split 顺序、chunk continuation、ragged/零长度请求；
4. EOS、23 个 special、请求边界与 final raw context；
5. lookup 对显式 row gather exact；
6. packed matmul 对 12 路独立 `F.linear` 求和 oracle；
7. ordinary `/sqrt(13)`、special base-only、`T=0`；
8. 既有 Lite config/loader/KDA/MLA/Grouped MoE focused regression。

CPU 数学使用 exact ID/context/lookup 门禁；BF16 projection/merge 对独立 FP32 oracle采用
`atol=0.02,rtol=0.02` 且必须全部 finite。

## 7. 失败与回退

- host source 非 CPU、shape/dtype 错误或 storage adoption 后 alias 不成立：立即失败；
- projection source 缺失/重复：沿用 strict loader 失败，不以零 slice 继续；
- 输入越界、ragged length 不匹配或 context shape 错误：在 host trust boundary 失败；
- 不提供整表 copy fallback；这会破坏阶段的核心内存契约；
- NPU/native n-gram op不在 7A 接线，当前镜像已证明其 dispatcher 存在但底层 aclnn 不可执行。

## 8. 提交边界

本设计文档独立提交并推送。下一提交只包含最小模型/loader 实现和 focused tests；完成本地与 NPU
focused 后，再以独立验证文档记录 exact source、数值、mmap/PSS 和清理状态。
