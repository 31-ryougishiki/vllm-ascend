# cp_balance 精度问题 —— 交接说明

> **接手先做三件事**：① 读 §1（现状与结论）与 §2（已排除什么）；② 跑一次
> `python tools/cp_balance_compare/selftest_mock.py`，末行 `SELFTEST OK`；③ 与对方对齐版本——
> 远端跑 `python tools/cp_balance_compare/selfcheck.py --fingerprint`，末行 `[fp]` 必须与本机一致（见 §6.1）。
>
> **术语**：`B` = DSA-CP 开 + `VLLM_ASCEND_CP_BALANCE=0`（连续切片，参照）；`C` = 同前但 `=1`（zigzag，被测）；
> `B2`/`C2` = B/C 的重复跑（量噪声地板）。**两条路径 kernel/权重完全相同，只有 token 排布不同** ⇒ `C−B` 就是 cp_balance 的账。
> 工具用法与判据细节在 [`README.md`](./README.md)；`log.log` 是随手粘贴判读输出用的草稿（`.gitignore` 覆盖 `*.log`）。

## 0. 背景与目标

**cp_balance 是什么**：DSA-CP 在 prefill 把 token 按 **TP rank 连续切片**分给各 rank（`cp_size == TP`），
KV 全量复制。因果注意力下 rank r 的可见 KV ∝ token 位置 ⇒ 首 rank 轻、末 rank 重，**最大/平均 ≈ 2**；
MLP/MoE 按 token 数均分，所以不均衡集中在 attention + indexer。`cp_balance`（本仓库分支
`glm52_cp_balance_v3`）把切片改成 SGLang 式 **zigzag**：每条序列切 `2*cp_size` 块，rank r 拿第 `r` 与第
`2P-1-r` 块（头尾配对），并让每 rank 的本地 token 数严格相等；**整个模型边界**（入口 shard、出口
all-gather + 反序）都在这个局部序上跑，因此均衡的不只是 attention，而是整网。

**为什么查精度**：开启后 prefill 的 logprob 与连续切片**确实不等**（§3），必须先弄清"差在哪一步、
是否可消除"，否则这个性能特性无法上线。

**目标**：给出**可复现的证据链**（不是"疑似"），落到"哪个算子、什么输入/输出、什么条件下不可复现"，
以及可选的规避方案。

## 1. 现状：结论已闭合到**最后一个算子**

对 layer 0（dense MLP，无专家无路由）逐 token 全精度比对，八行采样点齐全：

```
in ─attn─▶ out ─norm─▶ mlp_in ─[量化]─▶ gu_q ─gate_up─▶ gu_out ─silu─▶ dn_in ─[量化]─▶ dn_q ─down_proj─▶ mlp_out
   ✅0         ✅0          ✅0         ✅0              ✅0             ✅0         ✅0                 ❌ 1664/2048, token≥256
```

* `in/out/mlp_in/gu_q/gu_out/dn_in/dn_q` **全部 0 差异** ⇒ attention、pre-MLP norm、gate_up 的输出、
  silu 的输出、down_proj 的 bf16 输入**与**它实际吃到的 fp8+e8m0 scale **都逐位相同**；
  **不存在"前序差异被量化吸收"这回事**。
* 只有 `mlp_out` 不同：1664/2048、首个分歧 token 256（= 第 2 个 zigzag 块）、`max|d|=9.766e-04`、`rel=7.52e-03`；
  layer ≥1 是它的继承 + 逐层放大。
* **`C2 − C = {'p99': 0.0, 'max': 0.0}`** ⇒ C 路径完全确定 ⇒ 这是**确定性的行序效应**，不是运行间抖动。
* 行序置换（`repro_row_order.py` 数据模式，无需 NPU）：两侧 `rows=2048`、
  `positions_from=gather_natural`(B)/`gather_zigzag`(C)、`q=(2048,1536)`、`s=(2048,24,2)`，
  **1920/2048 个 token 的行号不同**（例：token 128(slot 256) → `row_B=128 row_C=256`）。

⇒ **输入（含权重）逐位相同、每个 token 的值相同、只有行序不同**，而结果不同 ⇒ 只剩两个可能：
① `npu_quant_matmul`（down_proj）与行序相关；② 紧随其后的 `tensor_model_parallel_reduce_scatter`。
（本站 MXFP8 **不走**融合 mm+RS：`linear_op.py` 的 mmrs 分支只认 `AscendW8A8LinearMethod`。）

**2026-09-12 这一步已跑完（offline repro，rank 0 + 随机权重）**：

```
[repro] device: npu:0 …
[repro] 随机权重（同形状，nz=True, device=npu:0）: weight=(1536, 768) … scale=(24, 768, 2)
[repro] 输出比较（逐 token，同一 weight、同一 token 值，只有行序不同）: differing=0/2048 max|d|=0.000e+00
[repro] RESULT: 该 op 在这个调用形状下与行序无关 ⇒ 差异不在 matmul，往紧随其后的 … 归约查
```

⇒ **① 基本排除**（同一调用形状下行序无关），矛头转向 ② 归约。

**2026-09-12 A3 交叉对照轮（`its` 站点 7.246.78.75，W4A8C8，TP=16，vllm py3.12）**——**同一形态复现**：

```
runtime: B{metadata=False forward=False} C{metadata=True forward=True}      ← C 真进 zigzag
case            len   top1%   p99|d|   max|d|  first_div  block@first_div      verdict
single_L2048   2048   86.96    0.391     1.55         92  rank1/prev(b64-128)   DIFF
single_L2049   2049   90.19    0.388      1.2        227  rank3/prev(b195-260)  DIFF
single_L4096   4096   79.68    0.392     1.13        171  rank1/prev(b128-256)  DIFF
act: layer 0 的 in/out/mlp_in/gu_q/gu_out/dn_in/dn_q 全 0；只有 mlp_out 差
     （1920/2048，首个分歧 token 64 = 第 2 个 zigzag 块，max|d|=4.883e-04）；layer 1 是它的继承
dump: 16 个 rank 齐全（act/mlp 各 64、guq/dnq/topk 各 32、kv 64）且 **首次落盘 w dump（2 个）**
```

⇒ 与 A5 完全相同的形状（layer 0 输入逐位相同、输出先不等）⇒ **"A5/mxfp4 特有"被排除**：
两个芯片、两套量化（A5 的 W8A8_MXFP8 与 A3 的 W4A8C8）上，cp_balance 都扰动了 layer 0 的
`mlp_out`，量级同阶（A3 p99≈0.39 nats vs A5≈0.56~0.58）。共性嫌疑仍落在
**down_proj 的行并行 GEMM + 紧随的跨 rank 归约**（A3 上现在可以用真实 `w` dump 跑行序复现，
不必再依赖 `--random-weight`）。

**A3 的量化方案与 A5 不同（复现器已按方案分派）**：A3 层 0 的 `dn_q` 是
`q=(2048,768) int8` + `s=(2048,) float32`（W8A8_DYNAMIC：int8 激活 + **每 token 一个 fp32 scale**
→ `npu_quant_matmul(q, w, w_scale, pertoken_scale=s, output_dtype=bf16)`，见
`w8a8_dynamic.py:114-121`）；A5 是 `q=(2048,1536) fp8` + `s=(2048,24,2) uint8`（MXFP8，需
`scale_dtype`/`group_sizes=[1,1,32]`）。`repro_row_order.py --run-op` 现在按 dump 的 `q` dtype
自动选形状（`--scheme auto|mxfp8|int8`），`--random-weight` 两种方案都支持（int8 需要 `--n`）。

**⚠️ 2026-09-12 两处工具缺陷（都已修，见 §9）**：
① `w` dump 原来按"每层一次"去重 ⇒ 层 0 的 `gate_up_proj` 先跑，占掉了名额，复现器拿到的是**兄弟 op 的权重**
（`[K=6144, N=1536]`），喂给 `dn_q` 的调用就报 `K dimension of x1 and x2 must be equal (768 vs 6144)`。
现在按 `(layer, op)` 各写一份，复现器按 op 选（选错会打印"只有 op=[…] 的 dump"而不是让内核报错）。
② **`--n` 的语义原来写错了**：行并行 down_proj 的权重是 `[K=I/tp, N=hidden]`（N 不切分），
`--n` 必须是 **hidden_size（GLM-5.2 = 6144）**，不是 `hidden/tp`；旧默认 768 让 A5 的那次
"GEMM 与行序无关"跑在了**非模型真实形状**上（N=768），结论需在 N=6144 下复核（免费，一次 `--run-op`）。
A3 侧可由已有 `w` dump 直接读出真形状：`gate_up [6144, 1536] = [hidden, 2I/tp]` ⇒ hidden=6144、
I=12288、`dn_q` 的 K=I/tp=768 ⇒ down_proj 权重应为 `[768, 6144]`。

**2026-09-12 A3：真实形状下的复现（免费，无需加载模型）**——在 `[K=768, N=6144]`、int8、
每 token fp32 scale 的**模型真实形状**上重跑两种行序：

```
[repro] 随机权重（int8, device=npu:0）: weight=(768, 6144) torch.int8 | scale=(6144,) torch.float32
[repro] rank 0: 输出比较（逐 token，同一 weight、同一 token 值，只有行序不同）: differing=0/2048 max|d|=0.000e+00
```

⇒ **GEMM 侧（int8/W4A8C8、真实形状）也排除**；A5 侧同样的复核还欠一次（它之前那次跑在 N=768 上，
用 `--run-op --random-weight --n 6144` 即可免费补做）。⇒ 两个站点、两套量化、真实形状下都排除 GEMM，
**跨 rank 归约成为唯一剩余嫌疑**。（同一轮还验证了 op 过滤生效：`w_cpbal0/1_layer0_rank0` 都只有
`op=['gu_q']`，复现器按预期打印"需要 op=dn_q"而不是让内核报 K 不匹配。）
`differing=0/2048 max|d|=0.000e+00`；rank 0 在此前两轮同命令下同样 0。⚠️ 那轮 dump 目录里
**没有 rank 0 的 `dnq`**，所以工具报的是 `7/7 … 覆盖不完整（缺 rank [0]）`（工具现在会显式告警，
判词不再写成 N/N）。综合 ⇒ **GEMM 侧在全部 8 个 rank 上都排除**（rank≠0 用同形状随机权重），
根因锁定在那次跨 rank 归约。
`mlp_out` 是 **reduce_scatter 之后**的 rank-local 张量（`dn_q` 是 all-gather 后的 2048 行、`gu_q` 是 rank-local），
所以"GEMM 输入逐位相同 + GEMM 行序无关"留下的唯一去处就是那次跨 rank 归约本身。

**rank 覆盖现状**（`differing=0` 目前只覆盖 rank 0，别直接外推）：

| 数据 | 覆盖范围 |
| --- | --- |
| `act`/`mlp`/`qin`/`topk`/`kv` dump | **全 rank**：每个 rank 各写一份（文件名带 `rank<R>`）；判读 `act`/`qin` 时把同一 layout 的所有 rank 按 token 位置拼回整条序列再逐 token 比，`kv` 按 `(layer, rank)` 逐对，`topk` 跨排布比较先按 layout 池化所有 rank（zigzag 下 rank 间 token 集合不相交） |
| `w`（权重 dump） | **只有 rank 0**（`_dump_weights` 里 rank≠0 直接 return） |
| `repro_row_order.py` | 默认**单 rank**（`--rank`，默认 0）；`--all-ranks` 扫全部有 dump 的 rank，非 0 rank 用同形状随机权重（日志打 `w=random`） |

**下一步（仍然不需要加载模型）**：

```bash
DIR=$(ls -dt /root/cp_probe/*/ | head -1)
# 全 rank 扫一遍（8 个 rank；rank 0 若有 w dump 会自动优先用真实权重）
python tools/cp_balance_compare/repro_row_order.py --dir "$DIR" --layer 0 \
    --run-op --random-weight --n 768 --all-ranks
```

**判据**：`RESULT(all-ranks): 8/8 rank … 与行序无关` ⇒ matmul 侧彻底排除，转 ② 归约；
任一 rank `differing>0` ⇒ 该 rank 上最小复现成立（交算子侧）；出现 NaN/全零 ⇒ 权重布局问题（试 `--no-nz`）。

单 rank 手工调试（默认 rank 0，`--rank <R>` 换 rank）：

```bash
DIR=$(ls -dt /root/cp_probe/*/ | head -1)
# 首选用同层 w dump（带 qin:<层> 的一轮里 rank0 会写 weight/weight_scale）
python tools/cp_balance_compare/repro_row_order.py --dir "$DIR" --layer 0 --run-op
# 旧轮次没有 w dump：用同形状随机权重（行序相关是调用形状的性质，与权重取值无关）
python tools/cp_balance_compare/repro_row_order.py --dir "$DIR" --layer 0 \
    --run-op --random-weight --n 768 [--no-nz]     # --n = down_proj 输出维 = hidden/TP
```

⚠️ 2026-09-12 站点三连败（都是**工具侧**问题，不是模型发现；三条都已修，见 §9）：
① `--random-weight` 曾 `RuntimeError: Ascend config is not initialized`（复现器 import 了
`vllm_ascend.utils.maybe_trans_nz`）；② 改用 `npu_format_cast` 后变成
`NotImplementedError: ... 'npu::npu_format_cast' with arguments from the 'CPU' backend`
—— 因为 dump 是 `map_location="cpu"` 加载的，`q.device` 恒为 CPU。
现在设备由 `--device`（默认 `npu:0`）决定，权重与算子输入都会先搬过去；跑起来应先看到
`[repro] device: npu:0（dump 是 CPU 加载的…）`。
③ `qin:` 轮在**模型 forward 内部**崩：`_dump_weights` 里 `weight.detach().to("cpu")` 撞上
NZ 内部格式（`RuntimeError: ... copy_ do not support internal format`）——这才是"有 `dnq` 却没有
`w`"的真因（`w` 打点是 bf1fa116a 加的，而带 `qin` 的轮次都在它之前）。现在权重走 `_weight_to_cpu`：
先试直拷，失败则 `npu_format_cast(..., FRACTAL_ND)` 再拷，并把 `weight_format` 写进 payload；
复现器读到 `ND` 会重放 NZ cast（`w=w->ND` 会打在汇总行里）。**打点永不阻断推理**。

**后续分支**（8/8 rank 都行序无关 ⇒ 归约侧；两步走，先便宜的后贵的）：

**第 1 步 — 开 HCCL 归约确定性（一轮 probe ≈20 分钟；A3 的 CANN 与 A5 不同，值得在这台先试）**。这条归约就是
`tensor_model_parallel_reduce_scatter` → vLLM `base_device_communicator.reduce_scatter` →
`torch.distributed.reduce_scatter_tensor` → **HCCL ReduceScatter**，而 HCCL 本身有归约类算子的
确定性与保序开关（官方说明见 [HCCL_DETERMINISTIC](https://gitcode.com/cann/hccl/blob/master/docs/user_guide/hccl_env/HCCL_DETERMINISTIC.md)：
`false`(默认)/`true`/`strict`，覆盖 AllReduce / ReduceScatter / ReduceScatterV / Reduce；
`LCCL_DETERMINISTIC=1` 在 rankSize ≤ 8 时生效 —— 本站 TP=8 正命中）。

⚠️ **`strict` 已证不可用（2026-09-12 实测）**：B 配置在 profile run 就死 —— MoE dispatch
（`npu_moe_distribute_dispatch_v2`）内部的 **HcclReduceScatter** 报
`E39999 … AICPU … RunAicpuIndOpCommInit get kernel failed (11003)`（`libccl_kernel.so`），
即保序所需的 AICPU kernel 在这套芯片/CANN 上装不出来；而且环境变量是**全局**的，会连 MC2/MoE
内部通信域一起吃。**按通信域配置也走不通**：torch_npu 的 `pg_options.hccl_config` 只支持
`hccl_buffer_size`/`group_name`/`qos_service_level`/`qos_traffic_class`/`hccl_op_expansion_mode`，
没有确定性键。⇒ 候选顺序 **`true` → `atb` → `expand` → 换归约实现**。

```bash
export HCCL_DET=true          # 单独一行：VAR=... 前缀在同一条命令里会静默丢
unset DUMP_DIR
bash tools/cp_balance_compare/run_cp_diag.sh probe      # → r_probe_det，env=HCCL_DETERMINISTIC=true LCCL_DETERMINISTIC=1
```

判据：① 日志有 `[cp-ab-hccl] … HCCL_DETERMINISTIC=true LCCL_DETERMINISTIC=1 …`；
② `compare_cp_rounds.py 基线=/dev/shm/cp_ab_probe/r_probe 确定性=/dev/shm/cp_ab_probe/r_probe_det`
末行 `back to the noise floor`（`p99|d|C-B ≤ 0.05`）⇒ 归约顺序即根因、且这是可用的缓解手段（代价是性能）；
③ 同轮 `[act]` 里 layer 0 `mlp_out` 从 1664/2048 变 0 ⇒ 算子级确认。
`true`/`atb`/`expand` 都不行时，走第 2 步。

**第 2 步（第 1 步无效才做）** — 换归约实现：把 row-parallel 归约换成 `all_reduce + slice` 做 A/B
（或继续查 HCCL 算法/展开模式 `HCCL_ALGO`、`HCCL_OP_EXPANSION_MODE`），再用 `probe2` 看 `C−B` 是否回噪。

## 2. 已确认 / 已排除（可直接引用）

| 事实 | 数字 |
| --- | --- |
| 噪声地板 `B2−B` | **0.0**（两次独立测量）⇒ 阈值恒为 `max(0.05, 5×0) = 0.05` |
| `C−B`（L2048/2049/4096） | p99 **0.575 / 0.560 / 0.557** nats；top-1 一致率 82.4% / 83.2% / 70.2% ⇒ 超阈 8~11 倍 |
| layer 0 KV | **逐字节相同**（连 NaN 位置都一致）⇒ 写 KV / slot 映射 / rope / 布局无误 |
| layer ≥1 KV | 从 token 256 起不同并逐层放大（layer 77 未量化的 rope 段差到 7e-1） |
| indexer 跨排布 | `[topk/cross] set_diff=0 且 order_diff=0` ⇒ 索引集合与顺序都与排布无关 |
| 元数据契约 | `[CP_BALANCE][check][*]`（prefix sum / block_table 行数 / `kv_len>=q_len` / 请求对齐）从未触发 |

**已排除**：indexer 选点错、KV 重排/写错、元数据契约错、**attention 数值错**（layer 0 `in`/`out` 逐字节相同）、
pre-MLP norm/残差/跨 rank 归约错（`mlp_in` 相同）、**激活量化与排布相关**（`gu_q`/`dn_q` 逐位相同；
`w8a8_mxfp8.py` 的 `npu_dynamic_mx_quant` + `group_sizes=[1,1,32]` 是按行、每 32 个 K 元素一个 scale）、
prev/next 两次调用形状（验证用的 `2call` 开关已随结论删除，见 README §七）、
**"这是 A5/mxfp4 特有的算子问题"**（A3+W4A8C8 上同一形态复现，见 §1）。

**仍开放**：**那次跨 rank 归约**（`tensor_model_parallel_reduce_scatter`；GEMM 已在两个站点、
真实形状下排除 —— A3 见 §1 的 `[768,6144]` 复现，A5 待用 `--n 6144` 补一遍）；
MoE 层（≥3，需 `--enable-return-routed-experts`）；`MIN_TOKENS` 边界（与本问题无关）。

## 3. 环境与关键事实

| 项 | 值 |
| --- | --- |
| 站点 | 切换用 `export CP_AB_SITE=its\|share`（档案在 `tools/cp_balance_compare/sites.sh`，**默认 `its`**）：<br>`its` = A3 `7.246.78.75`、repo `/opt/its/z30055003/vllm-ascend`、权重 `/opt/its/model/GLM-5.2-W4A8C8`、**TP=16**、可见卡 0..15、vendor env 跳过<br>`share` = 当前站点 `141.61.133.104`、repo `/home/z30055003/vllm-ascend`、权重 `/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4`、TP=8、可见卡 0..7<br>⚠️ `its` 的权重路径是历史档案（W4A8C8）；那边若挂的是 mxfp4 权重，`export MODEL_PATH=…` 覆盖（档案不会挡） |
| A3 现场实测（2026-09-12，preflight） | `vllm=/usr/local/python3.12.13/bin/vllm`（**与 share 站点的 py3.11.10 不是同一套 vllm 构建**）、`model=/opt/its/model/GLM-5.2-W4A8C8`、`repo=/opt/its/z30055003/vllm-ascend`、`vendor=none`、`tp=16`；`--preflight` 两配置 OK。⚠️ `vendor=none`：若这套 vllm-ascend 依赖 `_cann_ops_custom` 的 vendor set_env，第一轮会在加载早期失败 → 那时 `export VENDOR_SET_ENV=<path>` |
| 量化与芯片的绑定（2026-09-12 用户确认） | **mxfp4 是 A5 特有的** ⇒ `GLM-5.2-w4a4c8-mxfp4` 只能在 A5（`share` 站点 141.61.133.104）上跑，A3（`its` 站点 7.246.78.75）只有 **W4A8C8**。因此 **全套精度证据链（layer 0 MLP = W8A8_MXFP8、`gu_q`/`dn_q`、"GEMM 行序无关"）只属于 A5+mxfp4**；A3 的轮次是另一套硬件+量化，用来回答"cp_balance 在 A3/W4A8C8 上是否也不等"，**结论不能互相外推**（复现器的 `--random-weight` 造的是 fp8+e8m0 权重，只对 mxfp4 成立；A3 上要真实 `w` dump） |
| 通信 | launcher 固定 `HCCL_ALGO=level0:fullmesh`（现可覆盖）、`HCCL_BUFFSIZE=1200`、`HCCL_EXEC_TIMEOUT=204`；**归约确定性默认关**，由 `HCCL_DET=true\|atb\|expand\|strict` 打开（`run_cp_diag.sh` → `--env` → launcher 打印 `[cp-ab-hccl]` 行） |
| 版本对齐 | launcher 每次打印 `[cp-ab-site] CP_AB_SITE=… LOCAL_IP=… TP_SIZE=… VISIBLE=… MODEL=…`：**轮次在哪个节点、多少卡，日志里必须能看出来**（TP 就是 zigzag 的 cp_size，半切换会让整轮结论作废） |
| vLLM | **装好的包**（0.26.0，`site-packages`），模型层代码不在本仓库 ⇒ 本仓库只能靠 **hook/patch** 打点 |
| 站点特性 | **每次 `bash` 启动 ≈22s**（自测里 4 个 launcher 用例会自动 `[skip]`）；`git` 常常不可用 |
| 现场数据 | `/root/cp_probe/<时间戳>/`（probe 轮 dump，每轮一个子目录）、`/dev/shm/cp_ab*/<round>/`（summary + server 日志） |

**量化（实测，必须按现场为准）**：`quant_model_description.json` 全局 `model_quant_type=W4A4_MXFP4`、
`group_size=32`，但 **layer 0 的 MLP 三个投影是 `W8A8_MXFP8`**（attention 的 `q_a_proj`/`kv_a_proj_with_mqa`/
`o_proj`/`indexer.wq_b` 同）；`kv_b_proj` 与 indexer 的 `wk`/`weights_proj`/`k_norm` 是 `FLOAT`。
`config.json` 的 `quantization_config` 为 **null 是正常的**——站点靠 launcher 的 `--quantization ascend` 走
ModelSlim，声明文件就是上面那个 json。
⚠️ 本机 `GLM-5.2-w8a8c8-mxfp8/config.json` 只有**架构类字段**可参考（`first_k_dense_replace=3` ⇒ layer 0/1/2 是
dense、`n_routed_experts=256`、`hidden_size=6144`、`kv_lora_rank=512`、`qk_rope_head_dim=64`、`index_topk=2048`），
**量化字段不同**（mxfp8 vs mxfp4）——凡结论依赖"走了哪条量化路径"的，先在远端核实。

## 4. 机制：`CP_BALANCE=1` 到底改了什么

唯一判据：`vllm_ascend/layers/cp_zigzag.py::can_enable_zigzag_for_batch`（模型侧
`_pad_for_sequence_parallelism` 用**同一个**谓词，保证 padding 与 attention 布局不会不一致）：
`CP_BALANCE=1` · `cp_size=TP>1` · 非 speculative / 非 V2 runner / `dp_size==1` / 非 `dcp_replicated` ·
`full_o_proj` · 纯 prefill · 每序列 `query_len ≥ 2*cp_size` · `num_actual_tokens ≥ MIN_TOKENS`
（源码默认 8192，driver 钉成 2048 并写进指纹）· `num_tokens_pad % (2*cp_size) == 0`。

它只改**"谁算哪些 token、按什么顺序算"**：rank 局部行从 `[local_start, local_end)` 变成
`[block r, block 15−r]`，KV 写 slot、attention/indexer 的调用形状、模型边界 gather/rerange、集合通信顺序随之改变；
**权重与任何"逐 token 数学"不变**。⇒ 凡出现"同样输入、逐 token 数学、结果却不同"的地方，
一定是**顺序/分组/按批量化**类机制（这也正是 §1 结论的形状）。

每轮的验证链（缺一条整轮作废）：`[cp-ab] CP_BALANCE=1 … MIN_TOKENS=2048` 指纹 →
`[CP_BALANCE] metadata zigzag=1 … local_tokens=256` → `forward zigzag_active=1`
（driver 用 `--config-check strict --zigzag-check strict --on-zigzag-miss skip`）。

## 5. 代码地图（打点挂在哪）

| 想知道什么 | 看哪里 |
| --- | --- |
| zigzag 的 token 计划 / gather 顺序 | `vllm_ascend/layers/cp_zigzag.py::build_zigzag_plan` |
| SFA/indexer 的 zigzag 分支、KV/act dump | `vllm_ascend/attention/sfa_v1.py`（`zigzag_active`、`_maybe_dump_*`） |
| 当前 forward 的 zigzag 状态与 DSA ctx | `vllm_ascend/ascend_forward_context.py`（`zigzag_cp_context` **只在 zigzag 生效时才有值**） |
| MLP 边界打点（`mlp:<层>`：`mlp_in`/`mlp_out`/`gu_out`/`dn_in`） | `vllm_ascend/worker/model_runner_v1.py::_install_cp_balance_mlp_dumps`（按模块名挂 forward hook） |
| 量化输入打点（`qin:<层>`：两个 GEMM 实际吃到的 fp8+scale；rank0 另写 `w` = 该层 weight/scale） | `...::_install_cp_balance_quant_dumps`（按层包住 `AscendLinearMethod.apply`） |
| 采样点位置键（两种布局共用） | `...::_cp_balance_dump_positions`：本地行用 `slot_mapping_cp`，all-gather 后的行按 `zigzag_gather_index`(C)/自然序(B) 取自然 slot |
| ⚠️ 误导项 | `patch/worker/patch_deepseek_v2.py` 的 `_zigzag_layer_forward`/`_patched_forward` **对本模型不生效**（patch 的是 `DeepseekV2*`，本模型用 `DeepseekV32*`，无继承关系） |

## 6. 工作流程（本地 ↔ 远端）

```
本机（当前仓库）                                远端离线站点（用户执行）
  改代码 → commit → push  ─────────────▶  同步文件（常常没有 git）
  ◀──── 把 log.log / 屏幕输出贴回来 ─────  跑轮次（每次加载 ≈10 分钟）→ 判读（CPU，秒级）
```

### 6.1 版本对齐（离线站点没有 git 也能做）

* 本机：改完立刻 commit 并确认 `git rev-parse --short HEAD origin/glm52_cp_balance_v3` 两行相同。
* 两边都用 **内容指纹**：`python tools/cp_balance_compare/selfcheck.py --fingerprint`
  → 11 个关键文件的 sha256（**对 LF 归一化**，所以 CRLF 工作区与 Linux 站点可比）+ 9 个"修复标记"（`OK`/`MISSING`）
  + 用例数，末行 `[fp] <16 位>`。**两边的 `[fp]` 必须相同**；marker 直接回答"这台机器有没有某个修复"。
* 每次改完代码，把新的 `[fp]` 一起告知对方。指纹只覆盖上面那 11 个文件 + 标记 ⇒
  **只改文档（README/HANDOVER）不会改变 `[fp]`**，改代码才会。

### 6.2 一轮 = 一条命令 + 三件套证据

```bash
unset DUMP_DIR                                                   # 否则继承值会把 dump 吸到别处
bash tools/cp_balance_compare/run_cp_diag.sh probe2 --dry-run     # 先看 configs/spec/dir
bash tools/cp_balance_compare/run_cp_diag.sh probe2               # C,B,C2（3 次加载 ≈30 分钟）
DIR=$(ls -dt /root/cp_probe/*/ | head -1)                         # probe/probe2 每轮一个时间戳子目录
python tools/cp_balance_compare/check_zigzag_dumps.py --dir "$DIR" \
    --kind act --summary-only --block-size 128 2>&1 | tee tools/cp_balance_compare/log.log
```

三件套：① `[cp-ab]` 指纹行（配置对不对）；② `dump=… dir=…` + `dump: +N file(s)`（数据在不在、在哪个目录）；
③ 判读的 `FIRST DIVERGENCE` 行（结论）。`probe`（B/C，2 次加载）用于看剖面；`probe2`（C,B,C2，3 次加载）
额外给出 `[noise] C2 vs C`（确定性与否）。

**成本结构**：一轮几乎全花在模型加载（≈10 分钟/次），推理 ~2 秒/条 ⇒ **加长度、加打点层数几乎免费，
多跑一个配置就是 +10 分钟**；`qin` 打点让"量化 vs GEMM"一轮可分辨，避免多轮试错。

### 6.3 当前这一步之后的分支

* 落在"GEMM"⇒ 用 §1 的 `repro_row_order.py --run-op`（优先真实 `w` dump；没有就 `--random-weight --n <输出维>`）。
* 落在"归约"⇒ 把 row-parallel 归约换成 `all_reduce + slice` 做 A/B，并查 HCCL 算法/确定性开关。
* 若要换层看剖面（如 layer ≥3）⇒ `DUMP_SPEC=act:3,4,mlp:3,4,qin:3` 再跑一轮（层数几乎免费）。
* 判读器对"缺行/单侧缺失/采样方式不一致"都会显式告警，**先解释告警再读结论**。

## 7. 工具速查

| 目的 | 命令 | 判据 |
| --- | --- | --- |
| CPU 自测（改完代码必跑） | `python tools/cp_balance_compare/selftest_mock.py` | 每项 `[run]`/`[ok] …(Ns)`，末行 `SELFTEST OK`（59 项）；卡住时最后一行 `[run]` 就是卡住的用例，90s 后自动超时并打栈；慢 bash 站点自动 `[skip]` 4 个 launcher 用例（`--only/--skip` 可覆盖） |
| 版本指纹（无 git） | `python tools/cp_balance_compare/selfcheck.py --fingerprint` | 末行 `[fp] <16 位>` + 文件摘要 + marker OK/MISSING |
| 环境体检 | `python tools/cp_balance_compare/selfcheck.py` | `[verdict] READY`、无 FAIL |
| 配置门（不加载模型） | `ab_cp_compare.py --preflight --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}" --cp-size <TP>` | 末行 `[preflight] all configs OK`；查 `[cp-ab]` 指纹 + additional_config + vllm/model/repo/vendor 路径。⚠️ 漏 `--launcher` 会用 `launcher_template.sh` 的占位路径 |
| 命令预演（只预览） | `bash tools/cp_balance_compare/run_cp_diag.sh <mode> --dry-run` | 打印 site/tp/dir/spec 与实际 driver 命令；**不碰 launcher、不查路径** |
| 一轮 A/B（B/C） | `unset DUMP_DIR; bash tools/cp_balance_compare/run_cp_diag.sh probe` | 上述三件套；N≈272（融合路径下 `dn_in` 缺 16 个，N≈256） |
| 一轮 A/B + C 重复 | `… run_cp_diag.sh probe2` | 三件套 + `[noise] C2 vs C` |
| **归约确定性轮** | `export HCCL_DET=strict; unset DUMP_DIR; bash tools/cp_balance_compare/run_cp_diag.sh probe` | 轮次目录 `r_probe_det`；日志有 `[cp-ab-hccl] … HCCL_DETERMINISTIC=strict LCCL_DETERMINISTIC=1`；与 `r_probe` 并排看是否回噪（README §七） |
| 判读激活剖面 | `check_zigzag_dumps.py --dir <dir> --kind act --summary-only --block-size 128` | `FIRST DIVERGENCE (act): layer L op=…` + layer 0 八行 + 判词 |
| 判读索引表 / 全层 KV | `… --kind topk --summary-only` / `export DUMP_DIR=/root/cp_dump; run_cp_diag.sh sweep` → `… --kind kv --summary-only` | `[topk/cross] RESULT: …` / `FIRST DIVERGENCE (fp/value\|fp/bytes): layer L` |
| **行序复现** | `repro_row_order.py --dir <dir> --layer 0 [--run-op [--random-weight --n <hidden_size>] [--no-nz]] [--all-ranks [--tp-size N]]` | 数据模式：token→行号置换；`--run-op`：`differing>0` ⇒ 行序相关，最小复现成立；`--all-ranks` 出逐 rank 汇总（缺 rank 会告警，判词写"覆盖不完整"）；权重来源打在 `w=w\|random`（`w->ND` = 该 dump 走了 ND 兜底、已在复现器里重放 NZ）；⚠️ `--random-weight` 的 `--n` 是**该层输出维**（down_proj = hidden_size = 6144，不是 hidden/tp），不传会直接报错 |

| 单配置手工调试 | `run_single.py [--config C]` | `[http] <- 200` + `[result]` |
| 收整轮证据 | `selfcheck.py --collect --out-root /dev/shm/cp_ab_sweep` | 一个文件含 HEAD/摘要/dump 清单/指标/日志关键行 |

## 8. 易错点（按"会让你得出错误结论"排序）

1. **缺行 ≠ 相同**：表里少一行可能是"该采样点被跳过"或"文件没写"。判读器会打
   `INCOMPLETE`（单侧缺失）、`PROVENANCE`（两侧采样方式不同，`fused` 不一致本身就是根因候选）、
   以及"缺 X 行（mlp 已打点 ⇒ 是被跳过）"——**先解释这些再读结论**。
2. **dump 文件 kind 必须与判读器的名字表一致**：曾因 `kind_prefix + op` 拼成 `gugu_out`/`dndn_in`
   而**静默丢了两行两轮**（文件在、判读器不认）。现已显式携带 kind、兼容旧名，并对任何解析不了的文件名告警一次。
3. **`positions` 是 KV cache slot，不是 token 序号**（块表可能从 block 1 起 ⇒ `slot = token + 128`）。
   判读/复现都按两侧公共 base 归一化；只有**同轮内 B vs C**的位置才严格可比。
4. **判读目录必须用那一轮的时间戳子目录**：glob 不递归，指到父目录会**静默判读上一轮**（判读器会提示
   "更新的一轮在 …"）；另外会打印"选中 dump 时间跨度"，>90 分钟即说明混轮。
5. **继承的 `DUMP_DIR` 会改道**：跑前 `unset DUMP_DIR`，或先用 `--dry-run` 看 `dir=`。
6. **`/dev/shm` 很小**：dump 落真实磁盘（写满会连 server 一起搞死，它的 IPC/prometheus 在那儿）；
   判据是 `torch.save … inline_container.cc unexpected pos` + 请求 `Connection refused` + 截断文件。
7. **打点是 one-shot**：每层每进程只写一次、跳过 profile/warmup ⇒ 只有**第一个 prefill 请求**的数据；
   换层要改 `DUMP_SPEC` 再跑一轮，改判读没用。
8. **权重必须用"运行时的张量"**：checkpoint 里的权重经过 `process_weights_after_loading`（transpose + NZ），
   还原不等价 ⇒ 复现器要么读 `w` dump，要么用 `--random-weight`（同形状随机权重，NZ 由
   `torch_npu.npu_format_cast` 现做）。随机权重下若输出 NaN/全零 ⇒ 布局不对，**别下结论**。
9. **离线脚本不要 import 需要 ascend config 的 vllm_ascend 路径**（`maybe_trans_nz` 会抛
   "Ascend config is not initialized"）；直接用 `torch_npu.npu_format_cast(..., ACL_FORMAT_FRACTAL_NZ)`。
10. **`max|d|` 的 NaN 陷阱**：`delta.max()` 遇 NaN 就是 NaN，而 `max(0.0, nan) == 0.0`
    ——曾把真实分歧印成 `0.000e+00`，误导一整轮。现版本只在有限值上取最大，全 NaN 报 `inf`。
11. **packed KV 行不是纯 fp8**：512(fp8) + 128(bf16 rope) + 16(fp32 scale) = 656 字节 ⇒ 必须按段比较。
12. **`VAR=... cmd | tee` 的前缀会静默丢**（长命令粘贴时）→ 脚本退回默认值（曾因此白跑一轮 B2）；
    用 `export` 单独一行，或用模式词（`probe`/`probe2`/`sweep`）。
13. **launcher 静默忽略 env 会让 B≡C 假通过**：靠 driver 的 `--config-check strict` +
    `[cp-ab]`/`[cp-ab-cfg]` 指纹 + `zigzag_active=1` 三条堵住；缺任何一条整轮作废。
14. **driver 的 `--configs` 只认 B/C 且至少两个**；想要 C 的重复跑只能 `--repeat-a` 加在**第一个**配置上
    （`--configs C,B --repeat-a` ⇒ C,B,C2），这正是 `probe2` 模式做的事。
15. **改 `vllm_ascend/` 模块后先跑自测**：`selftest_mock.py` 含作用域感知的"未定义名"静态检查
    （`py_compile` 看不出来，只在加载模型时炸）；另注意 `sfa_v1.py` 把 stdlib import 放在函数内。
16. **MoE 的 Triton 报错是噪音**（上游 vllm 的 MXFP4 oracle 探测 CUDA 后端），与 Ascend 路径无关。

## 9. 关键提交（里程碑）

| commit | 内容 |
| --- | --- |
| `a02a44635` | 复现器 `--run-op` 显式设备（`--device`，默认 `npu:0`）：dump 是 CPU 加载的，权重与算子输入必须先搬设备，否则 `npu_format_cast` / `npu_quant_matmul` 报 "'CPU' backend"（站点日志 09-12 的第二次失败）+ 对应 CPU 自测与指纹 marker |
| `b7de04fe2` | 复现器离线可用（`npu_format_cast` 直连，不依赖 ascend config）+ 退化保护 + `--no-nz` |
| `e6292ab0e` | 复现器 slot 归一化 + `--random-weight`（不必再跑一轮）；HANDOVER 记录"只差最后一个算子" |
| `18cb6a7c2` | 指纹按 LF 归一化（跨平台可比）+ 慢 bash 站点自动跳过 launcher 用例 |
| `235a21471` | 无 git 的版本指纹 + 自测逐项日志/超时/栈（`--only/--skip/--list`） |
| `bf1fa116a` | 行序复现器 + 权重打点（`w`，rank0） |
| `7355e190d` | 修 `gugu_out`/`dndn_in` 命名 bug（判读器兼容旧名 + 未解析文件名告警） |
| `c3e06a4af` | 采样点位置键共用（gather 行可 dump）+ probe 每轮独立目录 + 通用未定义名静态检查 |
| `51a4383fd` | `qin:<层>` 量化输入打点（一轮区分"量化 vs GEMM"）+ `probe2` 模式（`82f072468`） |
