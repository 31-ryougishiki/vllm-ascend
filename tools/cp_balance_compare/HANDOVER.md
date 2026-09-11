# cp_balance 精度问题 —— 交接说明

> 读者：接手本问题的下一个人。先读本文，再读同目录 [`README.md`](./README.md)（工具怎么用、判据是什么）。
> 本文只讲三件事：**现在确定知道什么**、**已经排除什么**、**下一步该做什么**。

## 0. 一句话现状

`cp_balance`（zigzag 切分）与连续切片（`VLLM_ASCEND_CP_BALANCE=0`）在 prefill 上**确实不等**，差异远超噪声地板（p99 ≈ 0.39~0.58 nats，top-1 翻转 17%~30%），且**差异幅度随 rank 数减少而变大**。TP=8 的全层 KV 剖面已判读（§2.5）：**layer 0 逐字节相同** ⇒ 写 KV / rope / 布局 / 量化没写错；**layer 1 起 KV 的解码值就不同了**（约一半行、从 token 256 开始）⇒ 分歧在 **layer 0 的输出**里进入并被逐层放大。**不是** indexer 选点错、**不是** KV 重排写错、**不是**元数据契约错。下一步：**分段比较 packed KV 行**（nope/rope/scale，§4.1）定量分歧幅度，然后跑 **`probe`（op 级、全精度）** 定位到 attention vs MoE/MLP。

## 1. 环境与版本（接手时先对齐）

| 项 | 值 |
| --- | --- |
| 站点 | 旧站点：`LOCAL_IP=141.61.133.104`、`VLLM_ASCEND_REPO=/home/z30055003/vllm-ascend`、`MODEL_PATH=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4`、vendor `/mnt/share/l00622059/vendors/custom_transformer/bin/set_env.bash`、`PROFILER_DIR=/home/z30055003/profiling_no_pooling` |
| 模型 | **GLM-5.2 w4a4c8-mxfp4**（78 层，`index_topk=2048`，`indexer_types` 前 8 层 = `full,full,full,shared,shared,shared,full,shared`）→ **packed KV 是 fp8 e4m3 + e8m0 scale** |
| TP / cp_size | **8**（launcher `TP_SIZE`、driver `--cp-size` 都是 8；zigzag 的 `cp_size` 就是 TP） |
| 可见卡 | `ASCEND_RT_VISIBLE_DEVICES=0..7` |
| 代码 | 本仓库 `glm52_cp_balance_v3` 分支；工具与诊断改动都在 `tools/cp_balance_compare/` 与 `vllm_ascend/attention/sfa_v1.py`、`vllm_ascend/envs.py` |
| 现场数据 | `/root/cp_dump`（**1248 个 fp8 KV dump**，来自 TP=8 的 sweep 轮）、`/root/run_sweep3|4_*.log`、`/dev/shm/cp_ab_sweep/r_sweep/`（该轮 summary/logs） |
| 说明 | 同目录 `log.log` 只是临时粘贴/判读的草稿（`.gitignore` 覆盖 `*.log`，不入库），可随时覆盖 |

## 2. 已经确认的事实（带数字，可直接引用）

### 2.1 噪声地板 = 0，差异是实打实的

- `B2 − B`（同一配置重复跑）**恰好 0.0**（两次独立测量，逐位相同）→ 连续切片路径完全确定性。
- 因此 driver 的判定阈值稳定在 `max(0.05, 5×0) = 0.05`，`C−B` 超阈 8~11 倍。

### 2.2 指标（同一套 prompt：2048 / 2049 / 4096）

| case | **TP=16（旧机器）** | **TP=8（当前机器）** |
| --- | --- | --- |
| L2048 | p99 **0.391** / max 1.547 / first_div **92** / top1 86.96% | p99 **0.575** / first_div **294** / top1 82.41% |
| L2049 | 0.388 / 1.202 / **227** / 90.19% | 0.560 / **263** / 83.15% |
| L4096 | 0.392 / 1.133 / **171** / 79.68% | 0.557 / **622** / 70.23% |

两条规律（都很关键）：

1. **`first_div` 随块大小成比例后移**。zigzag 把序列切成 `2*cp_size` 个块，块大小 = `2048/(2*cp_size)`：TP=16 → 64，TP=8 → 128。TP=16 的 first_div 是 1.3~1.4 个块，TP=8 是 2.3~2.4 个块，且都落在 `rank2/prev`。→ **分歧起点由块边界决定**，不是固定位置、也不是某个坏层。
2. **rank 数减半，差异反而变大（0.39 → 0.57，1.45×）**。纯浮点归约顺序不该随 rank 数变化（每个输出 token 的加法次数不变）→ 指向**按 rank 局部的候选/聚合粒度效应**：块越大，每个 rank 的局部子集越偏离精确结果。

### 2.3 P1（indexer 选点）：**成立**

`check_zigzag_dumps.py --kind topk` 在 TP=16 那轮：16 个 rank、cpbal0/cpbal1 两种排布，`rows=128 width=2048`，`mismatched=0`（`identity=4 set-only=124` / `identity=0 set-only=128`，即集合正确、只有顺序不同）。
⚠️ 注意它验证的是**rank 局部**选点 == 该 rank 的局部因果窗口；**跨 rank 合并后的全局 top-k 没有被验证**。

### 2.4 P2（per-token KV 内容）：TP=16 那轮在 **layer 6 违反**，layer 0 完好

| layer | 结果 |
| --- | --- |
| layer 0 | 16/16 rank `differing rows=0`（逐字节相同） |
| layer 6 | 16/16 rank `differing rows=1984`（2048 行中），**`first differing token = 64`**，`bytes_differing=267/656`，`max\|int8\| diff=3` |

- **token 64 正是"第一个因果窗口跨 block（=跨 rank）的 token"**（block 0 = token 0..63 属于 rank0/prev，窗口全在块内；token 64 属于 rank1/prev，窗口要跨到 rank0）。
- layer 0 逐位相同 ⇒ **重排 slot 写入/KV 投影本身没写错**（写错就会是别的 token 的 KV，差异会是几十而不是 3）。
- layer 6 的 KV 是"layer 6 输入隐状态"的投影 ⇒ **分歧是在 layer 0..5 里产生的**。
- 差异是**平滑且普遍**的（1984/2048 行都不同，幅度小），不像"少数 token 大幅偏"的选点/路由跳变。

### 2.5 TP=8 全层 KV 剖面（`/root/cp_dump`，1248 个 dump，**已判读**）

命令：`python tools/cp_balance_compare/check_zigzag_dumps.py --dir /root/cp_dump --kind kv --summary-only`
（`[kv/int8]` 全列 `n/a`：这台机器 NPU `index_select` 不可用，没有 `kv_nat` 回读副本，只能看 `[kv/fp]`。）

| layer | `rows_byte` / `first_byte` | `rows_value` / `first_value` |
| --- | --- | --- |
| **layer 0** | **0** / `-` | **0** / `-` → 逐字节相同（连 NaN 的位置都一致） |
| layer 1 | 1664 / **256** | **1017** / **256** |
| layer 2 | 1792 / 256 | 996 / 256（`byte_only=20`） |
| layer 77 | 1792 / 256 | 841 / 256 |
| 其余层 | ≈1000-1800 / 256 | 727~1092 / 256 |

结论（已用 4 层抽样复核，`first_value` 全部 = 256）：

- **layer 0 逐字节相同** ⇒ 写 KV / rope / 布局 / 量化本身**没有写错**（§2.4 的 TP=16 结论在这里复现）。
- **layer 1 起 KV 的"解码值"确实不同**（约一半行、从 token 256 开始）⇒ 不是"只翻零点符号位"，
  之前的"亚量化、数值全同"结论**作废**（见下面的 NaN 陷阱）。
- 分歧**在 layer 0 的输出里进入**（layer 0 的 KV 来自 embedding 且逐位相同；layer 1 的 KV 是 layer 0 输出的投影）。

⚠️ **两个必须记住的坑**：

1. **`max|d|` 的 NaN 陷阱**：旧 checker 打印的 `max|d| = 0.000e+00` 是假的。真实情况是 `delta.max() = NaN`
   （打包行里有 NaN 字节模式），而 Python 的 `max(0.0, nan)` 返回 `0.0`。**任何"最大幅度"列都必须先排除 NaN。**
   新版 checker 已修（有限最大值；只有 NaN 对时给 `inf`；单边 NaN 计数进 `NaN-only pair(s)`），并有
   `test_check_zigzag_kv_reports_nan_difference_instead_of_zero` 兜底。
2. **整行按 fp8 解码是错的**：packed KV 一行 656 字节 =
   `k_nope`(512, 真 fp8 e4m3) + `k_pe`(64×bf16 = 128B) + `knope_scale`(4×fp32 = 16B)
   （`get_sfa_qsfa_packed_head_dim`：512 + 128 + 16 = 656）。后两段是 **bf16/fp32 的字节被"借道" fp8 张量运输**，
   按 fp8 解码只会得到垃圾值和 NaN（0x7F/0xFF）。因此现在这个 `rows_value` **不能**直接当作
   "KV 内容不同"的证据 —— 必须**分段比较**（nope 按 fp8、rope 按 bf16、scale 按 fp32），这是 §4.1 的当前这一步。

### 2.6 其他已确认

- 日志侧元数据不变量 `[CP_BALANCE][check][*]`（prefix sum / block_table 行数 / `kv_len>=q_len` / 请求对齐）**从未触发**。
- `[CP_BALANCE] metadata zigzag=1 … cp_size=8 … local_tokens=256`、`forward zigzag_active=1` 每轮都出现 → C 确实走 zigzag，分片算术自洽（2048/8=256，2049→pad 2064→258）。
- 出现过一次 **C server 崩溃**（请求 `Connection refused`）：根因是**全层 dump 把 `/dev/shm` 写满**（`torch.save … inline_container.cc unexpected pos` 从某层起持续报错 + 截断的 dump 文件）；vLLM 自己的 IPC/prometheus 也在 `/dev/shm`，所以 server 一起死。

## 3. 已排除 / 仍开放

| 假设 | 状态 |
| --- | --- |
| indexer 选点错（P1） | **已排除**（rank 局部 `mismatched=0`；全局合并仍未被工具覆盖） |
| KV 重排 slot 写错 | **已排除**（layer 0 逐位相同） |
| 元数据契约错（prefix/block_table/kv_len） | **已排除**（`[check]` 告警从未触发） |
| 量化掩盖导致的假"相同" | **反转了**：全层剖面里 KV 的解码值确实不同（`rows_value≈1000/2048`，first=256）。但整行按 fp8 解码会把 rope/scale 段读成垃圾+NaN，所以"KV 内容是否真的变了"要等**分段比较**（§4.1）才能定 |
| 合并 2B 单次调用（T1，`MERGED_CALL=0`） | **未测**（`run_cp_diag.sh 2call`，2 次加载） |
| `MIN_TOKENS` 边界（T4） | **未测**（`l1024`） |
| **主假设 A**：跨 block/rank 的归约顺序类差异，被 78 层 MoE 路由（离散选择）放大 | 未证实；"平滑、普遍、起点在块边界"与之一致 |
| **主假设 B**：按 rank 局部的候选/聚合粒度效应（块越大偏得越多） | 未证实；**"p99 随块变大而变大"支持它**，是最值得追的一条 |

## 4. 下一步（按优先级；每步一条命令 + 判据）

### 4.1 分段比较 packed KV 行（数据已在盘上，几秒，不需要 NPU）——**当前这一步**

§2.5 已经知道"整行按 fp8 解码后 layer 1 起有 ~1000 行不同、从 token 256 开始"，但**整行按 fp8 解码是错的**：
一行 656 字节里只有前 512 字节是真 fp8，后 144 字节是 bf16 rope + fp32 scale 借道运输。
所以要把行拆成三段分别比：

```bash
cd /home/z30055003/vllm-ascend
python - <<'PY'
import glob, os
import numpy as np, torch
NOPE, ROPE_END, SCALE_END = 512, 640, 656   # k_nope(fp8) + k_pe(64*bf16) + knope_scale(4*fp32)

def load(layer, cpbal, rank=0):
    hits = glob.glob(f"/root/cp_dump/kv_cpbal{cpbal}_layer{layer}_rank{rank}_pid*_*.pt")
    assert hits, (layer, cpbal)
    return torch.load(max(hits, key=os.path.getmtime), map_location="cpu", weights_only=False)["kv_fp_nat"]

def stats(x, y):
    av, bv = x.numpy().astype(np.float64), y.numpy().astype(np.float64)
    d = np.where(np.isnan(av) & np.isnan(bv), 0.0, np.abs(av - bv))
    rows = (d > 0).any(axis=1)
    fin, rel = d[np.isfinite(d)], (d / np.maximum(np.abs(av), 1e-30))
    rel = rel[np.isfinite(rel)]
    return (int(rows.sum()), int(rows.argmax()) if rows.any() else -1,
            float(fin.max()) if fin.size else 0.0, float(rel.max()) if rel.size else 0.0,
            int(np.isnan(av).sum()), int(np.isnan(bv).sum()))

for layer in (0, 1, 2, 77):
    a, b = load(layer, 0), load(layer, 1)
    print(f"--- layer {layer} shape={tuple(a.shape)} {a.dtype} ---")
    parts = {
        "nope_fp8":   (a[:, :NOPE].float(), b[:, :NOPE].float()),
        "rope_bf16":  (a[:, NOPE:ROPE_END].contiguous().view(torch.bfloat16).float(),
                       b[:, NOPE:ROPE_END].contiguous().view(torch.bfloat16).float()),
        "scale_fp32": (a[:, ROPE_END:SCALE_END].contiguous().view(torch.float32),
                       b[:, ROPE_END:SCALE_END].contiguous().view(torch.float32)),
    }
    for name, (x, y) in parts.items():
        rows, first, mx, mr, na, nb = stats(x, y)
        print(f"  {name:>10}: rows_diff={rows:5d} first={first:5d} "
              f"max|d|={mx:.3e} max_rel={mr:.3e} nan={na}/{nb}")
PY
```

**判据**（layer 0 应当三段全 0；关键看 layer 1/2/77）：

| 看到什么 | 含义 | 下一步 |
| --- | --- | --- |
| `nope_fp8` 行数≈0（或只有个位数）而 `scale_fp32 rows_diff` 大、`max_rel` 极小（~1e-7） | 量化内容相同，只有 fp32 scale 变了 ⇒ **隐状态差异在 fp32-ulp 量级**（纯归约顺序舍入） | 进 4.2，并预期 logprob 差异全部来自 MoE 放大 |
| `nope_fp8 rows_diff` 数百~上千，且 `rope_bf16 max_rel` ≥1e-3 | KV 内容确实变了 ⇒ 隐状态差异已达 1e-3 量级 | 进 4.2，重点看 op=out 是否在 layer 0 就先不等 |
| 三段全 0 | KV 完全一致（与 §2.5 的 layer 0 一样） | 分歧在 KV 写之后 → 仍然进 4.2 |

补充：`first` 期望都是 **256**（§2.5 实测）；`nan` 说明该段里有多少 NaN 字节模式（rope/scale 段本来就可能有，
因为它们是借道 fp8 运输的 bf16/fp32 数据）。

### 4.2 op 级激活剖面（`probe` 模式，**已实现**，2 次模型加载 ≈ 20 分钟）——4.1 之后的下一步

目的：把"哪一步先不等"从"第几层的 KV"细化到 **attention 内部 vs MoE/MLP**，而且是**全精度**（不像 KV 那样被 fp8 量化掩盖）。

```bash
bash tools/cp_balance_compare/run_cp_diag.sh probe          # B/C 无 B2 + act:0,1,2,3,kv:0,1,2,3 → /root/cp_probe
python tools/cp_balance_compare/check_zigzag_dumps.py --dir /root/cp_probe --kind act --summary-only
```

**判据**（每层两行的紧凑表 + 一行结论）：
- `FIRST DIVERGENCE (act): layer L op=out at token P` → 差异**在本层 attention 内部**产生（indexer 选点顺序 / SFA 归约顺序 / o_proj）；
- `... op=in at token P` → 差异是**上一层（L−1）的 MoE/MLP**产生的；
- `none` → 探测的这几层全精度逐字节相同，分歧在更深处 → `export DUMP_SPEC=act:4,5,6,7,8,9` 再来一轮（层数几乎免费，贵的是模型加载）。

要点：位置键取自写 KV 用的同一个 `slot_mapping_cp`，跨布局可比；`slot=-1` 的 padding 行被丢弃；one-shot，所以只有第一个请求（2048）的数据。

### 4.3 两个未做的判别实验（各 2 次模型加载，约 20 分钟）

- `bash tools/cp_balance_compare/run_cp_diag.sh 2call` → 若 `C−B` 回到噪声级，根因锁定"合并 2B 单次调用"；否则排除它。
- `bash tools/cp_balance_compare/run_cp_diag.sh l1024` → `MIN_TOKENS` 边界是否被正确遵守。

### 4.4 环境类（**可能直接影响数值，别跳过**）

1. `df -h /dev/shm` —— 确认它有多小；全层 dump 必须落在真实磁盘（`export DUMP_DIR=/root/cp_dump`）。
2. **mxfp4 的 Triton 内核导入失败**：`ERROR [mxfp4.py:56] Failed to import Triton kernels … cannot import name 'constexpr_function' from 'triton.runtime.jit'`（每次请求都在刷）。这台机器跑的是 **w4a4c8-mxfp4**，量化矩阵乘可能整条走回退实现 → **会改变数值**。要 `pip show triton` 对齐版本，并确认回退路径与正式路径数值等价；否则我们测的可能不是目标路径。
3. `ulimit -n 1024`（日志里有警告）→ 全层 dump 场景建议提高。
4. `torch_npu` 的 `index_select` 在这台机器上不可用（`aclnnIndexSelect 161002`）→ dump 已改为不依赖它（用写 cache 之前的同源副本）。

## 5. 工具速查（细节见 README）

| 目的 | 命令 | 判据 |
| --- | --- | --- |
| 一次自检（首跑/换机器时） | `python tools/cp_balance_compare/selfcheck.py` | `[verdict] READY`，无 FAIL |
| 不加载模型校验站点/env/指纹 | `... selfcheck.py --preflight` 或 `ab_cp_compare.py --preflight` | `[preflight] all configs OK` |
| 省掉每轮 source | `source tools/cp_balance_compare/prepare_env.sh` | 打印两段 source 耗时 + `CP_AB_SKIP_SOURCE=1` |
| 诊断轮（2 次加载 + 全层 KV dump） | `export DUMP_DIR=/root/cp_dump; bash tools/cp_balance_compare/run_cp_diag.sh sweep` | `dump: +N file(s)`、`[runtime] C:{T,T}`、`[case.*] p99≈0.575` |
| 判读 KV dump | `python tools/cp_balance_compare/check_zigzag_dumps.py --dir $DUMP_DIR --kind kv --summary-only` | `FIRST DIVERGENCE (fp): layer L` |
| op 级剖面轮（2 次加载 + 全精度 in/out） | `bash tools/cp_balance_compare/run_cp_diag.sh probe` | `dump: +N file(s)`、`[runtime] C:{T,T}` |
| 判读激活剖面 | `python tools/cp_balance_compare/check_zigzag_dumps.py --dir /root/cp_probe --kind act --summary-only` | `FIRST DIVERGENCE (act): layer L op=in|out at token P` |
| 单配置手工调试 | `python tools/cp_balance_compare/run_single.py [--config C]` | `[http] <- 200` + `[result]` 行；server 默认保留 |
| 收证据 | `python tools/cp_balance_compare/selfcheck.py --collect --out-root /dev/shm/cp_ab_sweep` | 一个文件里含 HEAD/dump 清单/指标/日志关键行 |
| CPU 自测（改了代码就跑） | `python tools/cp_balance_compare/selftest_mock.py` | 末行 `SELFTEST OK`（40 项） |

## 6. 踩过的坑（血泪清单，改代码前先看）

1. **env 前缀会静默丢**：`VAR=... cmd | tee ...` 长命令粘贴后前缀可能被吃掉 → 脚本退回默认值（曾因此白跑一轮 B2）。用 `export` 单独一行，或用模式词（`sweep`/`--no-repeat`）。
2. **`/dev/shm` 写满 = server 一起死**（IPC 在那儿）；全层 dump 必须落真实磁盘。
3. **不要给 `kv:all` 用默认目录**，也不要忘了 `DUMP_DIR` 同时决定 writer 与 checker。
4. **launcher 静默忽略 env 覆盖**会让 B≡C 看起来"完全一致" → driver 的 `--config-check strict` + `[cp-ab]`/`[cp-ab-cfg]` 指纹就是为堵这个；C 没打 `forward zigzag_active=1` 则整轮作废。
5. **模块级用了未 import 的名字**（`sfa_v1.py` 把 stdlib import 放函数内）：会在**加载模型时**才炸 → 已加 AST 静态检查（`selftest_mock.py`）。
6. **numpy 没有 fp8 类型**：fp8 张量转 numpy 必须先经 torch `.float()`（否则 `Got unsupported ScalarType Float8_e4m3fn`）。
7. **跨排布只能按 token 位置比，不能按行号比**（zigzag 下 rank 持有 `[prev,next]` 两块，连续切片下持有 `[local_start,local_end)`，同一个行号是不同 token）。`topk` dump 因此只能做"局部选点 == 因果窗口"的断言；`act` 剖面用写 KV 的同一个 `slot_mapping_cp` 做位置键，才跨布局可比。
8. **截断的 dump** 会让判读崩（已改为跳过 + 告警）；无 dump 的轮次 `run_cp_diag.sh` 会以 rc=3 明确失败。
9. **`act` 是 one-shot**：每层每进程只写一次、跳过 profile/warmup，所以只有第一个 prefill 请求（driver 发的 2048）有数据；想换层要改 `DUMP_SPEC` 再跑一轮，不是改判读。
10. **`probe` 轮默认落 `/root/cp_probe`**（百 MB 级），别指回 `/root/cp_dump`：同一个目录混两轮会互相干扰（判读只按"最新一份"取）。

## 7. 相关提交（最近，按时间倒序）

| commit | 内容 |
| --- | --- |
| `7b838e5ea` | op 级激活剖面：`act` dump（attention in/out，按 `slot_mapping_cp` 定位）+ `--kind act` 判读 + `probe` 模式 |
| `cfb196b55` | 新增本交接说明 HANDOVER.md，并修正 README 中已过期的描述 |
| `2091b4a8d` | 判读支持 fp8 dump；纠正"FP 副本"定性（同源副本，非量化前） |
| `6dfa8013e` | 判读向量化 + 进度输出（全层 1248 文件从分钟级到秒级） |
| `2a9fa0992` | 修 `_dump_dir` 的 `os` NameError；加 stdlib 静态检查用例 |
| `26aa438c2` | dump 目录可配置（`DUMP_DIR` 一钮两用）+ 坏 dump 跳过（/dev/shm 写满现场） |
| `e38bb25b7` | dump 不再依赖 NPU `index_select`（FP/同源副本为主，int8 回读降级） |
| `d6826e8ab` | 跑完却没有 dump 时明确失败（rc=3）+ `kv:all` 不再复制 1M 元素集合 |
| `73d65814f` | `prepare_env.sh` + `CP_AB_SKIP_SOURCE`（省掉每轮两次 source） |
| `0be3dd6bc` | TP/cp_size 默认改回 8 |
| `43a8b336d` | 站点参数改回旧站点（参考 `a5507f5e3` 的反向） |
| `3dd8b6fa2` | `sweep` 模式 + `--no-repeat`（避免 env 前缀静默失效） |
