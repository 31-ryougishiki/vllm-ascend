# cp_balance 精度问题 —— 交接说明

> **怎么读**：本文回答"现在确定知道什么 / 已经排除什么 / 下一步做什么 / 哪些坑会让你得出错误结论"。
> 工具怎么用、判据细节见 [`README.md`](./README.md)。`log.log` 是临时粘贴的判读草稿（`.gitignore` 覆盖 `*.log`），可随手覆盖。
>
> **术语**：`B` = DSA-CP 开 + `VLLM_ASCEND_CP_BALANCE=0`（连续切片，参照）；`C` = 同前但 `=1`（zigzag，被测）；
> `B2` = B 的重复跑（量噪声地板）。**两条路径 kernel/权重完全相同，只有 token 排布不同**，所以 `C−B` 就是 cp_balance 的账。

## 0. 一句话现状

`cp_balance`（zigzag）与连续切片在 prefill 上**确实不等**（p99 ≈ 0.39~0.58 nats，top-1 翻转 17~30%）。
逐层定位已经做到**单层内部**：**layer 0 的 embedding/attention 输出/MLP 输入全部逐字节相同，
而 layer 0 的 MLP 输出从 token 256 起就不同**（1664/2048 行，`max|d|=9.77e-04`，`rel=7.5e-3`）。

```
in ──attention──▶ out ──pre-MLP norm──▶ mlp_in ══gate_up══▶ gu_out ──silu──▶ dn_in ══down_proj══▶ mlp_out
   ✅ 相同            ✅ 相同             ✅ 相同      ❓未测             ❓未测              ❌ 从 token 256 起不同
```

⇒ 分歧**诞生在 layer 0 的 dense MLP 内部**（layer 0/1/2 是 dense MLP，`first_k_dense_replace=3`，
**没有专家、没有路由**），再逐层放大（到 layer 77 时未量化的 rope 段已差到 7e-1）。
**下一步**：把 MLP 内部那三步分开 —— `gu_out`（gate_up_proj 输出）与 `dn_in`（down_proj 输入）已在
`probe` 轮里打点，见 §5。

已排除：indexer 选点错、KV 重排/写错、元数据契约错、**attention 数值错**、pre-MLP norm/残差/归约错、
合并 2B 调用（`2call`）。

## 1. 目标与判据口径

- **目标**：判定 zigzag 切分是否引入精度差异、差在哪一步，并给出可复现的证据链（不是"疑似"）。
- **判据**：`C−B` 的 `p99|d|` 与阈值 `max(0.05, 5×noise_p99)` 比（噪声地板实测 **0.0**，所以阈值就是 0.05）；
  目前 `C−B` 超阈 8~11 倍 ⇒ 差异是实打实的。
- **定位口径**：全精度（bf16 原样）逐 token 比，绝不按行号比（两种排布下同一行号是不同 token）；
  比较键统一用写 KV 的同一个 `slot_mapping_cp`，判读侧再按公共 base 归一化成 token 序号。

## 2. 环境 / 模型 / 代码地图（接手先对齐）

### 2.1 站点与版本

| 项 | 值 |
| --- | --- |
| 站点 | `LOCAL_IP=141.61.133.104`、`VLLM_ASCEND_REPO=/home/z30055003/vllm-ascend`、`MODEL_PATH=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4` |
| TP / cp_size | **8**（zigzag 的 `cp_size` 就是 TP；SP padding 到 `2*tp`） |
| 代码 | 本仓库 `glm52_cp_balance_v3`；改动在 `tools/cp_balance_compare/`、`vllm_ascend/attention/sfa_v1.py`、`vllm_ascend/worker/model_runner_v1.py`、`vllm_ascend/layers/cp_zigzag.py`、`vllm_ascend/ascend_forward_context.py` |
| vLLM | **装好的包**（`site-packages/vllm`，0.26.0），不是源码 checkout；模型层代码在包里改不到（→ 打点走 hook，见 §2.4） |
| 现场数据 | `/root/cp_dump`（KV 全层 + 早期 act）、`/root/cp_probe`（probe 轮：act/mlp/topk/kv）、`/dev/shm/cp_ab*/<round>/`（summary/logs） |

### 2.2 模型结构里与本题相关的字段（`config.json`）

| 字段 | 值 | 为什么重要 |
| --- | --- | --- |
| `model_type` / `architectures` | `glm_moe_dsa` / `GlmMoeDsaForCausalLM` | 代码在 **vLLM 的 `vllm/models/deepseek_v32/nvidia/model.py`**（DSA 架构，DeepSeek-V3.2/GLM-5.2 共用） |
| `first_k_dense_replace` / `mlp_layer_types` | **3** / `[dense,dense,dense,sparse,…]` | **layer 0/1/2 是 dense MLP**（无专家、无路由），layer ≥3 才是 MoE |
| `scoring_func` / `topk_method` | `sigmoid` / `noaux_tc` | **不走** `sqrtsoftplus`+`tid2eid` 的 hash 路由 ⇒ 路由不依赖 `input_ids` |
| `n_routed_experts` / `num_experts_per_tok` | 256 / 8 | 后续若查 MoE 层，用 `--enable-return-routed-experts` 抓路由 |
| `hidden_size` / `kv_lora_rank` / `qk_rope_head_dim` | 6144 / 512 / 64 | **packed KV 一行 = 512(fp8) + 128(64×bf16) + 16(4×fp32) = 656 字节** —— 整行按 fp8 解码是错的 |
| `index_topk` | 2048 | 2048-token prompt 下 indexer 走"全选"路径（identity），故 topk 断言在长 prompt 下信息量低 |

### 2.3 `CP_BALANCE=1`（zigzag）触发条件与它到底改了什么

唯一判据：`vllm_ascend/layers/cp_zigzag.py::can_enable_zigzag_for_batch`（模型侧 `_pad_for_sequence_parallelism` 用同一谓词）：

`CP_BALANCE=1` · `cp_size=TP>1` · 非 speculative / 非 v2 runner / `dp_size==1` / 非 `dcp_replicated` ·
`full_o_proj`（`enable_dsa_cp_with_o_proj_tp`）· 纯 prefill（`PrefillNoCache|PrefillCacheHit|ChunkedPrefill`）·
每序列 `query_len ≥ 2*cp_size` · `num_actual_tokens ≥ MIN_TOKENS`（**源码默认 8192，driver 钉成 2048 并写进指纹**）·
`num_tokens_pad % (2*cp_size) == 0`。

每轮验证链：`[cp-ab] CP_BALANCE=1 … MIN_TOKENS=2048` 指纹 + `[CP_BALANCE] metadata zigzag=1 … cp_size=8 … local_tokens=256` +
`forward zigzag_active=1`（driver `--zigzag-check strict --on-zigzag-miss skip`，**没进 zigzag 整轮作废**）。

**它只改"谁算哪些 token、按什么顺序算"**：rank 局部行从 `[local_start,local_end)` 变成 `[block r, block 15−r]`；
KV 写 slot 映射、attention/indexer 调用形状、模型边界 gather/rerange、集合通信顺序随之改变。
**权重与任何"逐 token 数学"不变** ⇒ 凡出现"同样输入、逐 token 数学、结果却不同"的地方，一定是**顺序/分组/按批量化**类机制。

### 2.4 代码地图（打点挂在哪、谁 patch 谁）

| 想知道什么 | 看哪里 |
| --- | --- |
| zigzag 的 token 计划 / gather 顺序 | `vllm_ascend/layers/cp_zigzag.py::build_zigzag_plan` |
| SFA/indexer 的 zigzag 分支、KV dump、act dump | `vllm_ascend/attention/sfa_v1.py`（`zigzag_active`、`_maybe_dump_*`、`_token_positions`） |
| 当前 forward 的 zigzag 状态与 DSA ctx | `vllm_ascend/ascend_forward_context.py`（**注意**：`zigzag_cp_context` 只在 zigzag 生效时才有值） |
| MoE aux 重排（`input_ids`/`mc2_mask`） | `ascend_forward_context.set_ascend_forward_context` + `cp_zigzag.zigzag_reorder_moe_aux` |
| MLP 边界打点（本模型唯一可行的挂钩方式） | `vllm_ascend/worker/model_runner_v1.py::_install_cp_balance_mlp_dumps`（按模块名 `layers.<L>.mlp[.gate_up_proj\|.down_proj]` 挂 forward hook） |
| ⚠️ 误导项 | `vllm_ascend/patch/worker/patch_deepseek_v2.py` 里的 `_zigzag_layer_forward` / `_patched_forward` **对本模型不生效**：它 patch 的是 `DeepseekV2DecoderLayer/Model`，而本模型用 `DeepseekV32DecoderLayer/Model`（无继承关系） |

## 3. 已确认的事实（带数字，可直接引用）

### 3.1 噪声地板 = 0

`B2 − B` 恰好 0.0（两次独立测量）⇒ 连续切片路径完全确定性；阈值稳定在 `max(0.05, 5×0)=0.05`。

### 3.2 logprob 指标（同一套随机 prompt，长度 2048/2049/4096）

| case | TP=16（旧机器） | TP=8（当前） |
| --- | --- | --- |
| L2048 | p99 0.391 / top1 86.96% | p99 **0.575** / top1 82.41% |
| L2049 | 0.388 / 90.19% | 0.560 / 83.15% |
| L4096 | 0.392 / 79.68% | 0.557 / 70.23% |

两条规律：① `first_div` 随 zigzag 块大小（`2048/(2*cp_size)`）成比例后移 ⇒ 起点由块边界决定；
② **rank 数减半、差异反而变大（0.39→0.57）** ⇒ 指向按 rank 局部的粒度效应，而非单纯加法顺序。

### 3.3 全层 KV 剖面（`/root/cp_dump`，1248 个 dump）

| layer | `rows_byte` / `first_byte` | `rows_value` / `first_value` |
| --- | --- | --- |
| **0** | **0** | **0** → 逐字节相同（连 NaN 位置都一致） |
| 1 | 1664 / **256** | 1017 / **256** |
| 2..77 | ≈1000~1800 / 256 | 727~1092 / 256 |

分段比较（nope=fp8 / rope=bf16 / scale=fp32）在 layer 1/2/77 上**三段都真的不同，且逐层放大**：
`scale_fp32`（block-max 代理）6.8e-7 → 1.7e-3、`rope_bf16`（未量化）9.4e-2 → 7.0e-1、`nope_fp8` 在 layer 1 只差
1 个量化步（极值段）。⇒ 分歧真实、持续放大；layer 0 干净 ⇒ 写 KV/rope/布局/量化没错。

### 3.4 op 级剖面（`probe` 轮，全精度、按 token 对齐）

| layer | op | differ | first_pos | max\|d\| | rel |
| --- | --- | --- | --- | --- | --- |
| **0** | in / out / mlp_in | **0 / 0 / 0** | – | – | – |
| **0** | **mlp_out** | **1664** | **256** | 9.766e-04 | 7.52e-03 |
| 1 | in / mlp_in / mlp_out | 1664 / 1792 / 1792 | 256 | 1.562e-02 / 7.8e-03 / 5.7e-03 | ~1e-2 |
| 2 | in / mlp_in / mlp_out | 1792 / 1792 / 1792 | 256 | 1.953e-02 / 8.6e-02 / 4.6e-03 | ~2e-2 |

⇒ **attention 无罪**（layer 0 的 `in`/`out` 连 1 ULP 都不差）、**pre-MLP norm/残差/归约无罪**（`mlp_in` 相同），
分歧诞生在 **layer 0 的 dense MLP 内部**。形态**普遍而平滑**（80% token 都差、幅度同量级），不是"少数 token 跳专家"。

### 3.5 indexer 跨排布：集合与顺序都相同

`[topk/cross] set_diff=0 且 order_diff=0`（`identity` 几乎全是 256/256）⇒ indexer 输出与排布无关，
"索引顺序不同导致 SFA 累加不同"这条线**排除**；连带 **`2call`（T1）无信息量**（它只改 attention 调用形状）。

### 3.6 其他

- 元数据不变量 `[CP_BALANCE][check][*]`（prefix sum / block_table 行数 / `kv_len>=q_len` / 请求对齐）**从未触发**。
- 出现过一次 **C server 崩溃**：全层 dump 把 `/dev/shm` 写满（vLLM 的 IPC/prometheus 也在那里）→ 见 §6.2。

## 4. 已排除 / 仍开放

| 假设 | 状态 |
| --- | --- |
| indexer 选点错 | **已排除**（rank 局部 `mismatched=0`；跨排布集合与顺序都相同） |
| KV 重排 slot 写错 / rope / 布局 / 量化写错 | **已排除**（layer 0 KV 逐字节相同） |
| 元数据契约错（prefix/block_table/kv_len） | **已排除**（`[check]` 告警从未触发） |
| attention 数值（SFA/indexer/o_proj） | **已排除**（layer 0 attention 输出逐字节相同） |
| pre-MLP norm / 残差 / 跨 rank 归约 | **已排除**（layer 0 `mlp_in` 逐字节相同） |
| 合并 2B 调用（`2call`） | **已排除**（只改 attention 形状，而 attention 已逐位相同；别跑） |
| **dense MLP 内部：`gate_up_proj`（含 A 量化）** | **下一刀**（`gu_out` 已打点，等待新一轮判读） |
| **dense MLP 内部：`silu` 或其量化 / `down_proj`** | **候选**（`dn_in` 已打点） |
| MoE 层（≥3）的路由与专家计算 | 未测；等 layer 0 定死后再看是否需要 |
| `MIN_TOKENS` 边界（`l1024`） | 未测；与本问题无关，最低优先级 |

**主导猜想**：MLP 是逐 token 数学、输入又逐位相同，却算出不同结果 ⇒ 最可能是**激活量化（A-quant）的 scale
不是按 token 独立算的**（按批/按 rank 局部行集算），于是"同一 token 在不同排布下拿到不同的量化 scale"，
输出系统性偏移 ~1e-3。这一条用 `gu_out` 一点即可证真/证伪。

## 5. 下一步（一步一条命令 + 判据）

### 5.1 当前这一步：把 layer 0 的 MLP 再切三段

```bash
cd /home/z30055003/vllm-ascend
unset DUMP_DIR                                  # ⚠️ 否则继承的 DUMP_DIR 会把数据吸到别的目录（§6.3）
bash tools/cp_balance_compare/run_cp_diag.sh probe      # B/C 各一次加载 ≈ 20 分钟 → /root/cp_probe
python tools/cp_balance_compare/check_zigzag_dumps.py --dir /root/cp_probe \
    --kind act --summary-only --block-size 128 2>&1 | tee tools/cp_balance_compare/log.log
```

跑起来先确认一行（模型加载完打印，9 个模块/rank）：

```
[CP_BALANCE][dump] mlp trace armed for [(0,'model.layers.0.mlp'), (0,'…mlp.gate_up_proj'), (0,'…mlp.down_proj'), (1,…), (2,…)]
```

判读表每层 6 行，**取最早不等的那一行**：

| 最早不等 | 结论 | 下一步查什么 |
| --- | --- | --- |
| `gu_out` | **gate_up_proj 这一个 GEMM 内部** | 它的 A 量化（per-token MX？per-tensor？flashcomm gather 与 quant 的先后）+ 入参形状 |
| `dn_in` | **silu（或其量化）** | 激活实现与中间量化 |
| `mlp_out` | **down_proj**（含跨 rank 归约） | row-parallel 的 partial/reduce 路径 |

### 5.2 后续候选（按需，别预先跑）

- 若 `gu_out` 就先不等 ⇒ 给该线性层加"量化 scale"打点（打 `npu_dynamic_mx_quant`/`npu_dynamic_quant` 的返回 scale，按 token 位置比），一步定死是不是 scale 粒度问题。
- 若 layer 0 全部相同、差异从 layer ≥1 才出现 ⇒ `export DUMP_SPEC=act:3,4,mlp:3,4` 再跑一轮（层数几乎免费，贵的是模型加载）。
- 若最终落在 MoE 层（≥3）⇒ 用内置 `--enable-return-routed-experts` 抓全层逐 token 路由（**已验证可用**，见 `README.md` §四）。

## 6. 易错点清单（按"会让你得出错误结论"排序）

1. **`max|d|` 的 NaN 陷阱**：`delta.max()` 遇 NaN 就是 NaN，而 Python 的 `max(0.0, nan)` 返回 `0.0` —— 曾把真实分歧印成 `0.000e+00` 并误导了一整轮结论。任何"最大幅度"列都必须先排除 NaN（现版本已修，并有测试兜底）。
2. **`/dev/shm` 写满 = server 一起死**（IPC/prometheus 在那儿）。dump 落真实磁盘；现场特征是 `torch.save … inline_container.cc unexpected pos` + 请求 `Connection refused` + 截断文件。
3. **继承的 `DUMP_DIR` 会静默改道**：`probe` 尊重已导出的 `DUMP_DIR`，sweep 留下的 `export DUMP_DIR=/root/cp_dump` 会让 probe 数据写去那里、`/root/cp_probe` 根本不出现。跑前 `unset DUMP_DIR`，或用 `run_cp_diag.sh probe --dry-run` 看 `dir=`。
4. **`positions` 是 KV cache slot，不是 token 序号**（块表可能从 block 1 起 ⇒ slot = token + 128）。判读会按公共 base 归一化；跨轮次比位置不可靠，只有**同轮内 B vs C** 严格对齐。
5. **`_EXTRA_CTX.zigzag_cp_context` 只在 zigzag 生效时才有值** ⇒ 只读它会让 **B 侧全部静默跳过**（现场：`mlp=48` 只有 C、`act=96` 两侧都有，判读表里干脆没有 mlp 行）。取位置键要回退到 per-layer metadata 的 `dsa_cp_context.slot_mapping_cp`；判读侧对单侧缺失会打 `[act] INCOMPLETE`。
6. **packed KV 行不是纯 fp8**：512(fp8) + 128(bf16 rope) + 16(fp32 scale) = 656 字节。整行按 fp8 解码会把后两段读成垃圾值和 NaN ⇒ 要**按段比较**。
7. **`act`/`mlp` 打点是 one-shot**（每层每进程一次、跳过 profile/warmup）⇒ 只有**第一个 prefill 请求**（driver 发的 2048）的数据；换层要改 `DUMP_SPEC` 再跑一轮，不是改判读。
8. **`kv_fp_nat` 不是更高精度**：它只是"写 cache 之前的同源副本"（绕开不可用的 NPU `index_select`），比 fp8 更细的差异照样看不见 ⇒ 报出的"第一处不同"只是深度上界。
9. **numpy 没有 fp8 类型**：fp8 张量转 numpy 必须先经 torch `.float()`，否则 `Got unsupported ScalarType Float8_e4m3fn`。
10. **`VAR=... cmd | tee` 的前缀会静默丢**（长命令粘贴时）→ 脚本退回默认值（曾因此白跑一轮 B2）。用 `export` 单独一行，或改用模式词（`sweep`/`probe`/`--no-repeat`）。
11. **launcher 静默忽略 env 会让 B≡C 假通过**：driver 的 `--config-check strict` + `[cp-ab]`/`[cp-ab-cfg]` 指纹 + `zigzag_active=1` 检查就是为堵它；缺任何一条则整轮作废。
12. **改 `vllm_ascend/` 模块时防"模块级用了未 import 的名字"**（`sfa_v1.py` 把 stdlib import 放函数内）：只会在**加载模型时**炸 ⇒ `selftest_mock.py` 有 AST 静态检查；改完先跑 `python tools/cp_balance_compare/selftest_mock.py`。
13. **MoE 的 Triton 报错是噪音**（上游 vllm 的 MXFP4 oracle 探测 CUDA 后端），与 Ascend 路径无关，别被带偏。

## 7. 工作流程（本地 ↔ 远端）

```
本机（当前仓库）                          远端离线服务器（用户执行）
  改代码 + 提交 commit  ──自动 push──▶ origin/glm52_cp_balance_v3
                                          git pull / 拷贝文件 → 跑轮次 → 判读
  ◀────────── 把 log.log / 屏幕输出贴回来 ──────────┘
```

**协作约定（务必遵守）**

1. **代码只在本仓库改、改完立刻 commit**：轮次靠 `git rev-parse HEAD` 对齐版本；未提交的改动会让"这轮结果对应哪份代码"无法追溯。每提交一次会被推到 `origin`（本机 reflog 显示 `update by push`），远端只需同步到同一 commit。
2. **每一步都要有判据**：脚本必须打印"期望看到什么"；不给判据的命令视为未完成。
3. **一次只给一步**：上一步结果确认后再给下一步，不预先罗列后续步骤。
4. **远端执行 = 用户的手**：我给命令 + 判据；用户跑完把 `log.log`（或屏幕输出）贴回来。**新的打点代码必须先同步再跑**，同步后建议先 `python tools/cp_balance_compare/selftest_mock.py`（末行 `SELFTEST OK`）。
5. **每轮的产物三件套**：`[cp-ab]` 指纹行（配置对不对）、`dump=… dir=…` + `dump: +N file(s)`（数据在不在、在哪个目录）、判读的 `FIRST DIVERGENCE` 行（结论）。
6. **成本结构**：一轮几乎全花在模型加载（≈10 分钟/次），推理 ~2 秒/条 ⇒ **加长度/加打点层数几乎免费，多跑一个配置就是 +10 分钟**。所以一轮尽量多带点。

## 8. 工具速查

| 目的 | 命令 | 判据 |
| --- | --- | --- |
| CPU 自测（改完代码必跑） | `python tools/cp_balance_compare/selftest_mock.py` | 末行 `SELFTEST OK`（45 项） |
| 首跑/换机器自检 | `python tools/cp_balance_compare/selfcheck.py` | `[verdict] READY`，无 FAIL |
| 不加载模型校验 env/指纹 | `ab_cp_compare.py --preflight` | `[preflight] all configs OK` |
| 省掉每轮 source | `source tools/cp_balance_compare/prepare_env.sh` | 打印两段 source 耗时 + `CP_AB_SKIP_SOURCE=1` |
| **当前这一步**：MLP 内部剖面 | `unset DUMP_DIR; bash tools/cp_balance_compare/run_cp_diag.sh probe` | `mlp trace armed for [...]`、`dump: +N file(s)` |
| 判读剖面 | `check_zigzag_dumps.py --dir /root/cp_probe --kind act --summary-only --block-size 128` | `FIRST DIVERGENCE (act): layer L op=… at token P` + 每层 6 行 |
| 判读索引表（跨排布） | `… --kind topk --summary-only` | `[topk/cross] RESULT: … ORDER / DIFFERENT SET / invariant` |
| 全层 KV 剖面 | `export DUMP_DIR=/root/cp_dump; run_cp_diag.sh sweep` → `… --kind kv --summary-only` | `FIRST DIVERGENCE (fp/value\|fp/bytes): layer L` |
| 单配置手工调试 | `python tools/cp_balance_compare/run_single.py [--config C]` | `[http] <- 200` + `[result]` 行 |
| 收整轮证据 | `python tools/cp_balance_compare/selfcheck.py --collect --out-root /dev/shm/cp_ab_sweep` | 一个文件含 HEAD/dump 清单/指标/日志关键行 |

## 9. 关键提交（按时间倒序，只留里程碑）

| commit | 内容 |
| --- | --- |
| `0877aa35a` | MLP 内部再切三段：`gate_up_proj`/`down_proj` 打点（`gu_out`/`dn_in`） |
| `dc5748126` | 修"MLP 打点只写出 C 侧"（B 侧拿不到位置键被静默跳过）+ 单侧缺失报 `INCOMPLETE` |
| `94baf11ed` | MLP 边界打点（worker 按模块名挂 hook）+ probe 默认带 `mlp:` |
| `b0c105974` | `positions` 是 slot 不是 token 序号（旧版丢尾部、起点偏移一个 base） |
| `d78349625` | probe 判读结论（分歧诞生在 layer 0 的 MLP 段）+ launcher 支持 `EXTRA_SERVE_ARGS` |
| `28d9b40f3` / `298bbff36` | `[kv/fp]` 拆字节/数值；修 `max\|d\|` 的 NaN 陷阱并纠正错误结论 |
| `7b838e5ea` | op 级激活剖面（`act` dump + `--kind act` + `probe` 模式） |
| `2091b4a8d` | 判读支持 fp8 dump；纠正"FP 副本"定性 |
| `26aa438c2` | dump 目录可配置（`DUMP_DIR`）+ 坏 dump 跳过 |
| `e38bb25b7` | dump 不再依赖 NPU `index_select` |
