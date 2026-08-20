# DeepSeek-V4 DSA-CP 实现原理与异构 TP 适配说明

本文档基于 `hetero_cp` 目录的 vllm / vllm-ascend 代码，说明 DeepSeek-V4
`enable_dsa_cp`（DSA Context Parallel）的原仓实现流程、数据流、通信语义，
以及为支持异构 DP/TP（dp4、tp=[3,4,4,4]，dp0 带 `tp_sharding_ratios=[2,1,1]`）
所做的修改。

> 本文档只覆盖 Python 侧实现。C++/CANN 自定义算子
> （`aclnnInplacePartialRotaryMul`、sparse attention、KV scatter 等）
> 只描述其输入输出约束，不展开内核实现。

---

## 1. 背景与术语

DeepSeek-V4 使用 DSA（DeepSeek Sparse Attention）+ MLA 结构：

- `q_lora_rank=1024`，`o_lora_rank=1024`，`num_attention_heads=64`，
  `head_dim=512`，`qk_rope_head_dim=64`，`num_key_value_heads=1`。
- `wq_a / wq_b` 生成 query；`wkv` 生成唯一的 KV 头；`wo_a / wo_b` 做输出投影。
- 部分层有 compressor（ratio 4/128）和 indexer（ratio 4），KV cache 为
  hybrid 多组结构。
- `enable_dsa_cp` 表示在 TP group 内把 **token 序列**切到各 rank 上做
  context parallelism；query 侧每个 rank 都计算**完整 64 个 head**，
  KV cache 全量写入，最后通过一次 all_to_all 恢复成 TP head 分片。

### 1.1 开关要求

`vllm_ascend/utils.py` 中 `enable_dsa_cp()` 的要求：

1. 模型 HF config 有 `index_topk`，即模型带 indexer（DeepSeek-V3.2/V4 系列）。
2. `additional_config["enable_dsa_cp"] == True`。
3. 必须开启 SP，即 `enable_sp()` 为 True，否则直接报错：

```python
if dsa_cp_enable and not enable_sp():
    raise ValueError(
        "DSA CP requires SP to be enabled. Please enable SP(...) to use DSA CP."
    )
```

---

## 2. 原仓代码总览

### 2.1 关键文件

| 文件 | 职责 |
|---|---|
| `vllm_ascend/utils.py` | `enable_dsa_cp()` / `enable_dsa_cp_with_layer_shard()` / `enable_dsa_cp_with_o_proj_tp()` 开关 |
| `vllm_ascend/attention/dsa_v1.py` | `AscendDSABackend`，根据开关选择 CP 的 metadata builder 与 impl |
| `vllm_ascend/attention/context_parallel/dsa_cp.py` | CP 核心实现：`AscendDSACPMetadataBuilder` + `AscendDSACPImpl` |
| `vllm_ascend/models/deepseek_v4.py` | `DeepseekV4Attention`：构造 q/kv/o 投影与 KV cache 层 |
| `vllm_ascend/ops/dsa.py` | `AscendDeepseekSparseAttention` / `DSAModules` / `dsa_forward` 自定义 op 入口 |
| `vllm_ascend/ops/rope_dsv4.py` | DSA 的 RoPE cos/sin 缓存与 `get_cos_and_sin_dsa()` |
| `vllm_ascend/models/layer/attention/layer.py` | `DSAAttention`，实例化 `AscendDSACPImpl` |
| `vllm_ascend/device/device_op.py` | A3/A5 分平台的 q RMS、KV unpack、metadata op 选择 |
| `vllm/vllm/model_executor/layers/linear.py` | `ColumnParallelLinear` / `RowParallelLinear` 的异构 ratio 分片 |
| `vllm/vllm/distributed/utils.py` | `get_tp_partition_size()` / `get_tp_partition_offset()` |
| `vllm_ascend/distributed/parallel_state.py` | 异构 TP 下 Ascend 并行组初始化与 DSA-CP 允许逻辑 |

### 2.2 Backend 选择

`vllm_ascend/attention/dsa_v1.py`：

```python
class AscendDSABackend(AttentionBackend):
    @staticmethod
    def get_builder_cls():
        if enable_dsa_cp():
            return AscendDSACPMetadataBuilder
        return AscendDSAMetadataBuilder

    @staticmethod
    def get_impl_cls():
        if enable_dsa_cp():
            return AscendDSACPImpl
        return AscendDSAImpl
```

即：`enable_dsa_cp=False` 时走普通 DSA（`AscendDSAMetadataBuilder` +
`AscendDSAImpl`），打开后整条路径切换到
`AscendDSACPMetadataBuilder` + `AscendDSACPImpl`。

---

## 3. 模型侧结构

`vllm_ascend/models/deepseek_v4.py` 的 `DeepseekV4Attention.__init__` 中：

```python
self.n_heads = config.num_attention_heads          # 64
self.n_groups = config.o_groups                    # 8

# 按 TP rank 计算本地 heads/groups
self.n_local_heads = get_tp_partition_size(
    config.num_attention_heads, tp_rank, tp_size, _ratios)
self.n_local_groups = get_tp_partition_size(
    self.n_groups, tp_rank, tp_size, _ratios)

self.enable_dsa_cp = enable_dsa_cp()

# DSA-CP 下 attn_sink 需要全量 heads
attn_sink_heads = self.n_heads if self.enable_dsa_cp else self.n_local_heads

# wq_a / wkv 始终 replicated
self.wq_a = ReplicatedLinear(...)
self.wkv = ReplicatedLinear(...)

# DSA-CP 下 wq_b 复制全量 heads；普通 TP 下按 heads 分片
wq_b_cls = ReplicatedLinear if self.enable_dsa_cp else ColumnParallelLinear
self.wq_b = wq_b_cls(self.q_lora_rank, self.n_heads * self.head_dim, ...)

# 输出投影仍按 TP 分片（异构时按 ratio 分片）
self.wo_a = ColumnParallelLinear(
    self.n_heads * self.head_dim // self.n_groups,
    self.n_groups * config.o_lora_rank, ...)
self.wo_b = RowParallelLinear(
    self.n_groups * config.o_lora_rank, self.dim, ...)
```

要点：

- **q 路径复制全量**：`wq_a`、`wq_b`、`attn_sink` 在每个 TP rank 上都是完整权重，
  所以每个 rank 能对自己负责的 local tokens 计算全部 64 个 head 的 attention。
- **o 路径保持 TP 分片**：`wo_a/wo_b` 是 Column/RowParallel，
  CP 结束时必须通过 all_to_all 把 attention 输出恢复成
  “全局 token × 本 rank 的 head 分片”布局。
- **异构时 `n_local_heads/n_local_groups` 不再是整除结果**，
  由 `get_tp_partition_size()` 按 `tp_sharding_ratios` 计算。

实例化链路：

```text
DeepseekV4Attention
  └─ AscendDeepseekSparseAttention(dsa_modules)
       └─ DSAAttention
            └─ AscendDSABackend.get_impl_cls()
                 └─ AscendDSACPImpl(n_heads=64, n_local_heads=..., ...)
```

---

## 4. 前向入口

`vllm_ascend/ops/dsa.py`：

```python
class AscendDeepseekSparseAttention:
    def forward(self, positions, hidden_states, kv_cache=None, attn_metadata=None):
        need_gather_q_kv = get_forward_context().flash_comm_v1_enabled
        torch.ops.vllm.dsa_forward(
            hidden_states, need_gather_q_kv, output, self.prefix)
```

`dsa_forward` 从 `forward_context.no_compile_layers[layer_name]` 取得当前 layer，
过滤出该层 metadata，拼接 KV cache tuple，然后：

```python
self.dsa_attn.impl.forward(
    layer_name, hidden_states, kv_cache, attn_metadata,
    need_gather_q_kv, output)
```

之后进入 `AscendDSACPImpl.forward`。

---

## 5. Metadata 构建流程

### 5.1 `AscendDSACPMetadataBuilder.build()`

位于 `dsa_cp.py`，主要步骤：

1. 从 `common_attn_metadata` 取 `num_reqs / num_input_tokens / num_actual_tokens`。
2. `split_decodes_and_prefills()` 拆分 decode/prefill。
3. 取 positions，调用 `get_cos_and_sin_dsa(input_positions, use_cache=not has_prefill)`
   生成全局 cos/sin。
   - prefill：`use_cache=False`，按 position 索引 full RoPE cache。
   - decode：`use_cache=True`，使用 runtime buffer，避免重复取数。
4. 调用 `build_req_metadata()`，其中：
   - 调 `_build_local_token_metadata()` 得到本 rank 的 local token 区间；
   - 构造 SAS（sparse attention）metadata、QLI（lightning indexer）metadata；
   - 返回 `AscendDSAReqMetadata`，其中包含 `DSACPMetadata`。

### 5.2 `_build_local_token_metadata()`：CP 的 token 切分

这是 DSA-CP 的**序列维切分**核心：

```text
global token stream:  [0 ...................... num_tokens_pad)
                           ↓ 按 tp_size 均分
rank r:               [r*L, (r+1)*L)   L = num_tokens_pad / tp_size
```

对每个 rank：

```python
local_start = tp_rank * tokens_per_rank
local_end   = local_start + tokens_per_rank

local_query_start = clamp(global_query_start_loc, local_start, local_end)
local_query_end   = clamp(global_query_start_loc[1:], local_start, local_end)
local_query_lens  = local_query_end - local_query_start
local_query_start_loc = cumsum(local_query_lens)

# KV seq lens 也按同样的 local 区间裁剪
local_seq_lens = clamp(seq_lens - (query_start_loc[1:] - local_query_end), 0, ...)
```

RoPE 也切成 local：

```python
input_positions = pad(input_positions, num_tokens_pad)
cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=...)
local_cos = cos[local_start:local_end]
local_sin = sin[local_start:local_end]
```

返回 `DSACPMetadata`：

```python
DSACPMetadata(
    local_query_start_loc=...,
    local_seq_lens=...,
    local_start=local_start,
    local_end=local_end_with_pad,
    tokens_per_rank=tokens_per_rank,
    num_tokens_pad=num_tokens_pad,
    local_sin=local_sin,
    local_cos=local_cos,
)
```

### 5.3 SAS / QLI metadata

- `_build_sas_metadata()`：构造 sparse attention 的 metadata op 输入，
  **`num_heads_q` 传完整 `num_heads=64`**，`num_heads_kv=1`；
  `cu_seqlens_q` 使用 local query，`seqused_kv` 使用 local KV length。
- `_build_qli_metadata()`：ratio=4 层的 indexer metadata，同样使用 local
  query/key length。
- 相同 ratio 的层共享一份 metadata，通过 `common_ratio_to_sas_metadata`
  dict 缓存，避免逐层重复生成。

### 5.4 MTP draft metadata

`build_req_metadata_for_drafting()` 与主流程基本一致，但使用 draft 的
positions/slot_mapping，并为每步 draft 单独维护
`spec_local_query_start_loc / spec_local_seq_lens`。
MTP 层和主模型使用同一套 `DeepseekV4Attention`，因此 DSA-CP 的
local-token 切分逻辑在 draft 路径同样生效。

---

## 6. `AscendDSACPImpl` 前向流程

### 6.1 `forward()`：总入口

`dsa_cp.py: AscendDSACPImpl.forward()`：

```text
forward()
 ├─ wait_for_kv_layer_from_connector(layer_name)      # PD 加载
 ├─ self._forward(...)                                 # q/kv/attention
 ├─ self._restore_tp_head_layout(...)                  # all_to_all 恢复 head 布局
 ├─ o_proj: wo_a / wo_b                                # 输出投影
 └─ maybe_save_kv_layer_to_connector(layer_name)       # PD 保存
```

### 6.2 `_forward()`：q 路径

```python
hidden_states = maybe_all_gather_and_maybe_unpad(
    hidden_states_local, need_gather_q_kv)

hidden_states_cache = hidden_states[:common_attn_metadata.num_actual_tokens]

qr_local = q_norm(wq_a(hidden_states_local))
q = wq_b(qr_local)                    # [T_local, 64*512]，wq_b 复制全量 heads
q = q.unflatten(-1, (64, 512))
q = apply_dsa_q_rms(q)

inplace_partial_rotary_mul(
    q.unsqueeze(1),                   # [T_local, 1, 64, 512]
    local_cos,                        # [T_local, 1, 1, 64]
    local_sin,
    rotary_mode="interleave",
    partial_slice=[448, 512],         # 只旋转 rope_head_dim=64
)
```

说明：

- `hidden_states_local` 是 SP 切分后的本地 token 流。
- `need_gather_q_kv = flash_comm_v1_enabled`：SP 打开时先把 local hidden
  gather 回全量序列，供 KV cache 写入使用。
- q 的 RoPE 使用 **local cos/sin**，与 local token 的原始位置对应。

### 6.3 KV 路径

```python
kv = wkv(hidden_states_cache)         # 全量 token，replicated
kv = kv_norm(kv)
kv = kv.view(-1, 1, nope_head_dim + rope_head_dim)

inplace_partial_rotary_mul(
    kv.unsqueeze(1),                  # [N, 1, 1, 512]
    cos[:kv.shape[0]],                # [N, 1, 1, 64]，全局 cos
    sin[:kv.shape[0]],
    partial_slice=[448, 512],
)

dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)
```

- KV 写入的是**完整 token 序列**，所以用全局 cos/sin。
- 每个 TP rank 都写完整 KV cache（DSA-CP 的 KV 不做跨 rank 分片）。

### 6.4 Attention

```python
attn_output = attn_op(
    q,                                # [T_local, 64, 512]
    ori_kv=swa_kv_cache,              # 全量 KV cache
    ...
    cu_seqlens_q=local_seq_lengths_query,
    seqused_kv=local_seq_lengths_key,
    sinks=self.attn_sink,             # full 64 heads
    ...
)
```

每个 rank 对自己负责的 local tokens、完整 KV cache、完整 64 heads 做
sparse attention。输出：

```text
attn_output: [T_local, 64, 512]
```

### 6.5 `_restore_tp_head_layout()`：CP 最关键的通信

attention 输出目前是：

```text
rank r: [T_local, total_heads=64, head_dim=512]
```

但 `wo_a/wo_b` 是 TP head 分片。原仓通过一次 all_to_all 同时完成两个变换：

1. token 维：把各 rank 的 local token 块拼回全局 token 流；
2. head 维：每个 rank 只保留自己的 head 分片。

**均匀 TP 路径**（原仓逻辑）：

```python
send = local_attn_output.view(
    T_local, tp_size, n_local_heads, head_dim)   # n_local_heads = 64/tp
send = send.permute(1, 0, 2, 3).contiguous()     # [tp, T_local, local_heads, D]
recv = torch.empty_like(send)

dist.all_to_all_single(recv, send, group=tp_group)

return recv.view(-1, n_local_heads, head_dim)
```

语义：

```text
send 的第 j 块 = rank r 的 token 中、属于 destination rank j 的 head 分片
all_to_all 后：
rank r 收到每个 source rank 发来的 head 分片 r，
形状为 [T_local, n_local_heads, D] × tp_size
展平后 = [num_tokens_pad, n_local_heads, D]
```

### 6.6 o_proj

```python
o_proj_groups = n_local_groups
o_proj_input.view(num_tokens, o_proj_groups, -1)
# wo_a: [groups, o_lora_rank] 的 batched matmul
# wo_b: 输出回 hidden_size
```

最后 `output` 写回 `[num_tokens, hidden_size]`，PD producer 侧再执行
`maybe_save_kv_layer_to_connector()`。

---

## 7. 原仓流程图（文字版）

```text
scheduler_output
   │
   ▼
model_runner._build_attention_metadata()
   │  enable_dsa_cp?
   ▼
AscendDSACPMetadataBuilder.build()
   ├─ positions → 全局 cos/sin
   ├─ _build_local_token_metadata(): 序列按 TP 均分
   ├─ local query_start_loc / seq_lens / cos / sin
   └─ SAS / QLI metadata（q heads 传全量 64）
   │
   ▼
model.forward → layer.self_attn → AscendDeepseekSparseAttention.forward
   │
   ▼
torch.ops.vllm.dsa_forward → AscendDSACPImpl.forward
   ├─ _forward()
   │    ├─ SP: local hidden → 全量 hidden（供 KV 使用）
   │    ├─ wq_a/q_norm/wq_b → q[local_tokens, 64, 512]
   │    ├─ q RMS + local RoPE
   │    ├─ wkv/kv_norm/global RoPE → KV scatter（全量）
   │    └─ sparse attention(local q, full KV, full heads)
   ├─ _restore_tp_head_layout()
   │    └─ all_to_all: (local token, full head) → (global token, local head)
   └─ wo_a / wo_b → output
```

---

## 8. 异构 TP 需要改什么

异构 DP/TP 场景的关键差异：

| 维度 | 均匀 TP4 | 异构 dp4 / tp=[3,4,4,4] |
|---|---|---|
| 每 rank head 数 | `64/tp`，整除且各 rank 相等 | dp0 为 `[32,16,16]`，不相等 |
| 每 rank group 数 | `8/tp` | dp0 为 `[4,2,2]` |
| SP token padding | 按本地 tp 对齐即可 | 必须按 `lcm(3,4)=12` 对齐 |
| head 恢复通信 | `all_to_all_single` 等长切分即可 | 需要 per-rank 不等长切分 |
| 模型侧 q 权重 | CP 下全复制，heads 只影响 o_proj | 同样全复制；o_proj 按 ratio 分片 |

### 8.1 移除异构 TP 下的入口拒绝

文件：`vllm_ascend/distributed/parallel_state.py`

原 hetero 分支曾经直接禁止 DSA-CP：

```python
if enable_dsa_cp():
    raise RuntimeError("DSA context parallelism ... not supported under heterogeneous TP")
```

现改为允许进入 `_init_ascend_heterogeneous_fallbacks()`。DSA-CP 只依赖
基础 TP group，不需要 fine-grained Ascend TP group，因此异构 fallback
即可满足。

### 8.2 读取当前 DP rank 的 sharding ratios

文件：`vllm_ascend/attention/context_parallel/dsa_cp.py`
（`AscendDSACPImpl.__init__`）

```python
self._hetero_head_ratios: list[int] | None = None
parallel_config = self.vllm_config.parallel_config
if parallel_config.is_heterogeneous_tp:
    ratios = parallel_config.get_sharding_ratios_for_dp(
        parallel_config.data_parallel_rank)
    if ratios is not None:
        self._hetero_head_ratios = list(ratios)
```

注意：DP1..3 的 config 没有 ratios，`_hetero_head_ratios` 为 `None`，
继续走原仓均匀路径；只有 dp0 进入非均匀路径。

### 8.3 非均匀 all_to_all 恢复 head 布局

文件：`vllm_ascend/attention/context_parallel/dsa_cp.py`
（`AscendDSACPImpl._restore_tp_head_layout`）

均匀路径的 `all_to_all_single` 要求每个 rank 的 send/recv 块大小一致；
dp0 的 heads 是 `32/16/16`，不满足，因此异构路径：

1. 用 `get_tp_partition_offset/size` 计算每个 rank 的 head 区间：
   - rank0: `[0,32)`，rank1: `[32,48)`，rank2: `[48,64)`。
2. 把 `local_attn_output` 按 destination rank 的 head 区间切成
   `tp_size` 个 chunk 后拼接为 1D `send`。
3. 显式指定 per-source / per-destination 的 split sizes：

```python
dist.all_to_all_single(
    recv,
    send,
    output_split_sizes=[
        num_tokens * n_local_heads_self * head_dim
    ] * tp_size,
    input_split_sizes=[
        num_tokens * head_sizes[j] * head_dim
        for j in range(tp_size)
    ],
    group=tp_group,
)
```

4. `recv.view(-1, n_local_heads_self, head_dim)` 恢复成
   `[num_tokens_pad, local_heads, D]`，与均匀路径返回布局一致。

其中：

```text
source rank r → destination rank j 的 chunk 大小：
    num_tokens * heads_j * head_dim

destination rank i 收到的每个 source chunk 大小：
    num_tokens * heads_i * head_dim
```

全局守恒：所有 rank 的 send 总量 = 所有 rank 的 recv 总量 =
`num_tokens_pad * 64 * head_dim`。

### 8.4 按 LCM 对齐 token padding

文件：`vllm_ascend/attention/context_parallel/dsa_cp.py`
（`AscendDSACPMetadataBuilder._build_local_token_metadata`）

异构 SP 下 `ascend_forward_context` 会把整条 token 流 pad 到：

```text
padded_length = align_to_lcm(max_i(ceil(N_i / tp_i) * tp_i))
lcm = lcm(tp_0, tp_1, tp_2, tp_3) = lcm(3,4,4,4) = 12
```

而原仓 DSA-CP metadata 只按本地 `tp_size` pad：

```python
num_tokens_pad = ceil(num_input_tokens / tp_size) * tp_size
```

两者不一致时，`local_cos/local_sin` 的行数会小于 q 的本地 hidden 行数，
导致 `aclnnInplacePartialRotaryMul` 的 dim0 检查失败
（运行时表现为 `EZ9999: Inner Error!`）。

修改后：

```python
if parallel_config.is_heterogeneous_tp:
    align = math.lcm(*[parallel_config.get_tp_size_for_dp(i)
                       for i in range(parallel_config.data_parallel_size)])
else:
    align = tp_size

num_tokens_pad = ceil(num_input_tokens / align) * align
tokens_per_rank = num_tokens_pad // tp_size
```

均匀 TP 下 `align == tp_size`，逻辑不变。

### 8.5 关闭 ratio 分片下的 A5 full-o_proj 路径

A5 专用 `enable_dsa_cp_with_o_proj_tp` 会 all_gather `wo_a/wo_b`
全量权重并按均匀 head 布局处理；ratio 分片下不再适用。异构 ratio 存在时：

```python
self.enable_dsa_cp_with_o_proj_tp = (
    enable_dsa_cp_with_o_proj_tp()
    and get_ascend_device_type() == AscendDeviceType.A5
    and self._hetero_head_ratios is None
)
```

### 8.6 已有的异构基础能力

以下能力来自 `feat/hetero-merge-tp` 基线，DSA-CP 直接复用：

| 能力 | 位置 |
|---|---|
| 非整除 head 数的模型校验跳过 | `vllm/vllm/config/model.py` |
| `get_tp_partition_size/offset` 按 ratio 分片 | `vllm/vllm/distributed/utils.py` |
| Column/RowParallel 权重按 ratio 加载 | `vllm/vllm/model_executor/layers/linear.py` |
| `n_local_heads / n_local_groups` ratio 计算 | `vllm_ascend/models/deepseek_v4.py` |
| MTP sink/head 加载同样按 ratio | `vllm_ascend/models/deepseek_v4_mtp.py` |
| SP / EP group 的异构 padding 与 gather/reduce-scatter | `vllm_ascend/ascend_forward_context.py`、`ops/register_custom_ops.py`、`ops/fused_moe/*` |

---

## 9. 数值示例

### 9.1 均匀 tp=4

```text
num_heads = 64, n_local_heads = 16, n_groups = 8, n_local_groups = 2
num_input_tokens = 100
num_tokens_pad = 100
tokens_per_rank = 25

rank0: tokens [0,25), heads [0,16)
rank1: tokens [25,50), heads [16,32)
...
attention 输出: [25,64,512]
all_to_all 后: [100,16,512]
```

### 9.2 异构 dp0，tp=3，ratios=[2,1,1]

```text
num_heads = 64
head_sizes = [32, 16, 16]
head_offsets = [0, 32, 48]
n_groups = 8 → n_local_groups = [4, 2, 2]

num_input_tokens 已为 DP 间 pad 后的最大值，例如 8197
align = lcm(3,4,4,4) = 12
num_tokens_pad = 8208
tokens_per_rank = 8208 / 3 = 2736

rank0: tokens [0,2736),     heads [0,32)
rank1: tokens [2736,5472),  heads [32,48)
rank2: tokens [5472,8208),  heads [48,64)
```

all_to_all：

```text
rank0 send chunks:
    dest0: tokens×[0,32)   = 2736*32*512
    dest1: tokens×[32,48)  = 2736*16*512
    dest2: tokens×[48,64)  = 2736*16*512

rank0 recv:
    每个 source 发来 2736*32*512
    拼接后 = 8208*32*512
    view = [8208, 32, 512]
```

---

## 10. 关键入口索引

按阅读顺序建议：

1. `vllm_ascend/utils.py: enable_dsa_cp()` — 开关与约束。
2. `vllm_ascend/attention/dsa_v1.py: AscendDSABackend` — CP/非 CP 分发。
3. `vllm_ascend/models/deepseek_v4.py: DeepseekV4Attention` — 模型侧 q/kv/o
   结构，CP 下哪些权重 replicated、哪些按 TP/ratio 分片。
4. `vllm_ascend/ops/dsa.py: dsa_forward` — 前向 custom op 入口。
5. `vllm_ascend/attention/context_parallel/dsa_cp.py`
   - `AscendDSACPMetadataBuilder.build / build_req_metadata`
   - `_build_local_token_metadata`
   - `_build_sas_metadata / _build_qli_metadata`
   - `AscendDSACPImpl.forward / _forward / _restore_tp_head_layout`
6. `vllm_ascend/ops/rope_dsv4.py` — RoPE 缓存与 local/global cos/sin 来源。
7. `vllm_ascend/distributed/parallel_state.py` — 异构入口允许逻辑。
8. `vllm/vllm/distributed/utils.py` — ratio 分片工具函数。

---

## 11. 配置约束与注意事项

- `enable_dsa_cp=true` 必须同时满足：
  - 模型带 indexer（如 DeepSeek-V4）；
  - `enable_sp=true`（目标脚本通过 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 满足）。
- 当前异构支持只针对 **DeepSeek-V4** 模型实现验证，未扩展其他 DSA 模型。
- `enable_dsa_cp_with_o_proj_tp` 仅 A5 且要求均匀 head 分片；
  异构 ratio 分片下已自动禁用。
- 异构 token padding 依赖 `lcm(tp_sizes)`，如果未来出现 tp size 组合变化，
  `_build_local_token_metadata` 会自动跟随 `parallel_config` 计算。
- PD 场景下 `enable_reduce_sample` 按官方文档不支持，DSA-CP 与 PD
  同时开启时 rejection sampler 会走全量 logits fallback，属预期路径。
- prefill/decode 的 dp、tp 可任意搭配（受 DSA-CP 原支持的 TP 范围约束）。
  DeepSeek-V4 的 KV cache 在每个 prefill TP rank 上全量复制，
  `MooncakeHybridConnector` 不再要求 `prefill_tp >= decode_tp`；
  `prefill_tp < decode_tp` 时每个 decode rank 按请求哈希选择一个 prefill
  TP rank。
- A3 目标脚本使用 `--enforce-eager`，非均匀 `all_to_all_single` 使用
  eager 动态 shape；如需图模式，需要额外确认 ACL graph 对
  `output_split_sizes/input_split_sizes` 的静态化支持。

---

## 12. 验证与调试

### 12.1 静态检查

```bash
python -m py_compile vllm_ascend/attention/context_parallel/dsa_cp.py
git -C hetero_cp/vllm-ascend diff --check
```

### 12.2 运行时重点日志

- DSA-CP 应选择：
  `AscendDSACPMetadataBuilder / AscendDSACPImpl`。
- dp0 tp=3 应看到 `n_local_heads=32/16/16`，无
  `aclnnInplacePartialRotaryMul ... EZ9999`。
- 非均匀 all_to_all 无 shape 报错。

### 12.3 精度对比

分别对比：

1. `hetero_sp`（关闭 DSA-CP 基线）；
2. `hetero_cp` 打开 DSA-CP；
3. dp0 tp=3 与 dp1 tp=4 输出 logits/token 一致性。

---

## 13. 关联提交

在 `hetero_cp/vllm-ascend` 分支 `feat/hetero-cp-dsa-cp` 上：

| 提交 | 内容 |
|---|---|
| `22a683612` | 允许异构 TP 使用 DSA-CP；非均匀 head all_to_all；A5 full-o_proj 禁用 |
| `55d39b925` | 修复异构 LCM 与 local token metadata 不一致导致的 `inplace_partial_rotary_mul EZ9999` |
