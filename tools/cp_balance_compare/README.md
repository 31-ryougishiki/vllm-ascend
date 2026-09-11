# cp_balance_compare 测试说明

本目录的工具只做一件事：**确认 DSA-CP 的 `cp_balance`（zigzag 切分）在 prefill 阶段是否引入精度差异，以及差异出在哪一步**。

对比对象只有 `VLLM_ASCEND_CP_BALANCE` 这一个开关，DSA-CP 始终打开：

| 配置 | DSA-CP | `VLLM_ASCEND_CP_BALANCE` | 含义 |
| --- | --- | --- | --- |
| `B` | on | `0` | 连续切片基线（可信参照） |
| `C` | on | `1` | zigzag cp_balance（被测） |
| `B2` | on | `0` | B 的重复跑（`--repeat-a`），只用来量噪声地板 |

因为两条路径的 kernel、模型、权重完全相同，只有 token 排布不同，所以 **`C-B` 的差值就是 cp_balance 的账**；`B2-B` 给出它必须打败的噪声地板。原设计里的 “A（关 DSA-CP）锚点” 已下线：它走的是不同代码路径，只能给粗参照，要 A 会被 driver 直接报错拒绝。

> 本目录所有工具（含后续新增的诊断脚本）必须遵守[第八节](#八工具与脚本的五条硬要求)的五条要求。
>
> **接手这个问题**（或想知道"现在查到哪一步、下一步做什么"）请看 [`HANDOVER.md`](./HANDOVER.md)：里面有已确认的事实与数字、已排除/仍开放的假设、以及按优先级的下一步命令。本文（README）只讲工具怎么用、判据是什么。

## 一、目录里各文件的角色

| 文件 | 角色 | 需要 NPU |
| --- | --- | --- |
| `ab_cp_compare.py` | 主 driver：顺序拉起 B/C/B2、发请求、算指标、写 `summary.json` | 是 |
| `launcher_glm52_w4a4c8_mxfp4.sh` | 站点 launcher（GLM-5.2 w4a4c8-mxfp4，TP=8），支持全部 cp_balance 环境覆盖并打印指纹 | 是 |
| `launcher_template.sh` | launcher 参考实现（说明任何 launcher 必须遵守的两条规则） | 是 |
| `run_cp_diag.sh` | 一轮 A/B 的标准入口，封装 `baseline` / `sweep` / `probe` / `2call` / `l1024` / `check` 模式（加 `--no-repeat` 跳过 B2） | 是（`check` 除外） |
| `check_zigzag_dumps.py` | CPU 侧判读 NPU 落盘的 `topk` / `kv` / `act` dump（对应 P1 / P2 / "哪一步先不等"） | 否 |
| `compare_cp_rounds.py` | CPU 侧把多轮 `summary.json` 并排对比，回答“改一个变量后 C−B 是否回到噪声级” | 否 |
| `mock_vllm_server.py` | 假 vLLM `/v1/completions` 服务（`/health` + echo/logprobs），用 offset 模拟 B/C 差异 | 否 |
| `selftest_mock.py` | driver 的 CPU 自测（38 个 `test_*`：mock 端到端 + 指纹/payload/校验和 + zizzag dump 判读 + launcher 静态检查），不需要 NPU | 否 |
| `selfcheck.py` | 一键自检：现场体检 + driver 自测 + 配置门（launcher `DRY_RUN=1` 指纹）+ 轮次命令预演；`--collect` 收集整轮证据 | 否 |
| `run_single.py` | 单配置手工调试：单独拉起 Base（`B`）或 `C` 的 server + 用与 A/B **完全相同**的请求体发一次推理；失败时打印 HTTP 状态与响应正文，并把 payload/response 落盘供 curl 复现；默认保留 server | 是 |
| `prepare_env.sh` | 一次性准备环境（`source` 站点 rc + vendor `set_env.bash`，并 `export CP_AB_SKIP_SOURCE=1`），省掉每轮 launcher 的两次 source；**必须 source**，不能直接执行 | 否 |

## 二、三层测试

### L1 — CPU 自测（不需要 NPU，最快）

`selftest_mock.py` + `mock_vllm_server.py`：mock 只实现 driver 用到的协议面，logprob 固定为 `-0.5 - 0.001*(token%100) + offset`，不同端口天然像不同配置。覆盖点包括：

- 响应解析：token-id prompt / 文本 prompt / `prompt_logprobs` 回退 / 三种 `top_logprobs` key 风格必须解析一致；
- 对齐与拒绝：`prompt_token_ids` 与发送的 id 不一致时 `verify_response` 必须报错，不能拿不可比的响应当结果；
- 配置校验：`[cp-ab]` 指纹比对、`additional_config` 逐字比对、base config 里不得含 PD-only 开关；
- 激活判定：`zigzag_state_from_log` 从日志识别 zigzag 标记；
- preflight：假 launcher 下 env 覆盖与指纹逻辑；
- 端到端：两/三个 mock 端口 + `--repeat-a` 跑完 `run()`，校验 `summary.json` 的 noise / first_div / p99 / top1% 数值。

本机实测：`SELFTEST OK`（38 项；依赖假 `vllm` 可执行位的 launcher 用例在非 Linux 主机上打印 `[skip]`，fingerprint 格式校验那半边仍会跑）。

### L2 — NPU 上的 A/B 轮次（真正测精度）

`ab_cp_compare.py` 逐个配置**顺序**启停 server（TP=N 独占整机，所以不能并行），每个配置都：

1. 用 launcher 起 server（注入该配置的 env），轮询 `/health` 直到就绪；
2. 校验 `[cp-ab]` 指纹 + `[cp-ab-cfg]` 里的 `additional_config`（`--config-check strict`）；launcher 若静默忽略覆盖，B 和 C 会跑成同一份配置，看起来“完全一致”，这里就是为了堵掉这种假通过；
3. 从 server 日志确认 **C 真的走进了 zigzag**（`--zigzag-check strict`）——没进 zigzag 就说明 C 退化成了 B，后面的对比没有意义，此时 `--on-zigzag-miss skip` 会停掉剩余配置并以非 0 退出；
4. 逐 prompt 请求 `/v1/completions`：`echo=true`、`logprobs=K`、`prompt_logprobs=K`、`max_tokens=1`、`temperature=0`、`add_special_tokens=false`、`return_token_ids=true`；
5. 记录每个位置的 `token_logprobs`（真实 token 的 logprob，所有配置同一个 token，差值是干净的逐位置精度信号）与 `top_logprobs`（top-1 / top-5），写完 `results_<配置>.json` 后停 server。

server 的 stdout/stderr 既写入 `<轮次>/logs/server_<配置>.log`（**原始行**，指纹解析与 `tail()` 读的都是这份），也**实时打屏**，屏幕上的每一行带 `[<配置>]` 前缀，所以模型拉起/加载过程可以直接看到、不会和 driver 自己的输出混淆；不想要屏幕噪声时加 `--no-stream-log`（日志文件照写）。

prompt 默认是确定性随机 token-id（`--seed 1234`，词表 10 万），长度 2048 / 2049 / 4096：2048 正好是 `2*cp_size`(=16) 的整数块，2049 不整块（覆盖 padding/切分边界），4096 是长序列。

### L3 — CPU 侧判读（不需要 NPU）

- `check_zigzag_dumps.py --kind topk` → **P1**：对短于 `sparse_count` 的 prompt，LightningIndexer 每行有效前缀应当恰好是因果窗口 `{0..valid-1}`（identity 或仅顺序不同的同一集合）。有效集合缺项 ⇒ indexer 让 SFA 关注的位置少于因果窗口要求。
- `check_zigzag_dumps.py --kind kv` → **P2**：dump 按自然 token 顺序存 packed KV，所以 cpbal0 的第 p 行与 cpbal1 的第 p 行是同一个 token，可逐字节比较。首个不同的 layer/token 就是两种排布第一次分歧的地方。
  dump 里同时带一份**写 cache 之前的同源副本**（`kv_fp_nat`，同样自然序）：它取自 `fused_kv_no_split`，**与 cache 里是同一批打包数字**（sparse-C8 站点是 fp8 e4m3 + e8m0 scale），所以它**不消除量化掩盖**——比 fp8 更细的差异看不出来。它的价值在于：不再依赖某些站点上不可用的 NPU cache 回读（`aclnnIndexSelect 161002`），dump 因此仍能产出。判读打印两张表（`[kv/int8]` 与 `[kv/fp]`），`FIRST DIVERGENCE` 以同源副本为准。
  `[kv/fp]` 表把**字节**与**数值**拆开报，因为量化让这两件事含义不同（列：`val_ranks` / `rows_val` / `first_val` / `rows_byte` / `first_byte` / `max|d|` / `rel` / `byte_only`）：

  | 读数 | 含义 | 判据 |
  | --- | --- | --- |
  | `rows_val>0`、`max\|d\|` 非 0 | 存储的**数值**不同 ⇒ ≥1 个量化步的真差异 | `FIRST DIVERGENCE (fp/value): layer L`，rc=1 |
  | `rows_val=0`、`rows_byte>0`、`byte_only>0`、`max\|d\|=0` | 数值**全同**、只有字节不同。e4m3 里每个有限值只有一种编码（`±0` 除外）⇒ 只能是**量化零点的符号位**翻转 ⇒ **亚量化（sub-quantization）**分歧：量化前的值确实不同，但小于一个 fp8 步 | `FIRST DIVERGENCE (fp/bytes): layer L`，rc=0 |
  | 全 0 | 逐位相同 | `FIRST DIVERGENCE (fp): none`，rc=0 |

  `byte_only` 是**最灵敏的探测器**：数值完全一样时它照样能看出两种排布不等价。⚠️ 别把 `rows_byte>0` 当成"内容写错了"——内容写错会体现在 `rows_val` 与 `max|d|` 上；反过来也别只看 `rows_val`。

  ⚠️ **两个读数陷阱**（都真实踩过）：
  1. **NaN 不能当 0**：`delta.max()` 遇到 NaN 就是 NaN，而 Python 的 `max(0.0, nan)` 返回 `0.0` —— 一次真实分歧因此被印成 `max|d| = 0.000e+00`。现版本只取有限最大值，只有 NaN 对时给 `inf`，并在行尾标 `[N NaN-only pair(s)]`。
  2. **packed KV 行不是纯 fp8**：一行 = `k_nope`(kv_lora_rank, 真 fp8) + `k_pe`(qk_rope_head_dim×**bf16**) + `knope_scale`(kv_lora_rank/tile×**fp32**)，后两段是借道 fp8 张量运输的字节（本站点 656 = 512 + 128 + 16）。整行按 fp8 解码会把这两段读成垃圾值和 NaN ⇒ **要按段比较**（nope 用 fp8、rope 用 bf16、scale 用 fp32），否则 `rows_val` 的含义会被污染。GLM-5.2 站点分段脚本见 `HANDOVER.md` §4.1。
  做**多层扫描**（`DUMP_DIR=... DUMP_SPEC=kv:all`）时加 `--summary-only`：每层一行、进度打到 stderr（全层 1248 个文件约几秒），直接给出 `FIRST DIVERGENCE: layer L`——第 L 层 KV 是"第 L 层输入隐状态"的投影，分歧实际是在 **L−1 层的输出**里进入的；若落在 layer 0，则只可能出自写 KV / rope / 布局本身。
- `compare_cp_rounds.py baseline=... 2call=...` → 各轮指标并排 + 结论：最后一轮的 `C-B` 是否全部回到噪声级。
- `check_zigzag_dumps.py --kind act` → **O1**：逐 token 比较 attention 的**输入**（`in` = 上一层的输出）与**输出**（`out` = 本层 attention 的贡献）。这是全精度数据（不像 KV 那样被 fp8 量化掩盖），判据只有两条：
  - `in` 相同、`out` 不同 ⇒ 差异是**本层 attention 内部**产生的（indexer 选点顺序 / SFA 归约 / o_proj）；
  - 第 L 层 `out` 相同、第 L+1 层 `in` 不同 ⇒ 差异是**中间那层的 MoE/MLP**（或其间 norm/残差）产生的。

  剖面按 **token 位置**对齐（不是按行号：zigzag 下每个 rank 持有 `[prev,next]` 两块，连续切片下持有 `[local_start, local_end)`，同一个行号是不同 token）。位置取自写 KV 用的同一个 `slot_mapping_cp`，所以不会和写路径漂移；`-1` 的 padding 行被丢弃。⚠️ 它是 **KV cache slot**，不是 token 序号（请求的块表可能从 block 1 起，slot = token + 128）；判读会按 B/C 的公共 base 归一化，`--block-size 128` 还会标出"落在第几个 zigzag 块"。判读输出每层两行（`in`/`out`）的紧凑表 + `FIRST DIVERGENCE (act): layer L op=… at token P`。

dump 由 `vllm_ascend/attention/sfa_v1.py` 写，开关是 `VLLM_ASCEND_CP_BALANCE_DUMP`（语法 `kind:layers[,...]`，如 `topk:6,kv:0,6`、`kv:all`、`act:0,1,2,3`；空值=关），目录默认 `/dev/shm/cp_balance_dump`，可用 `VLLM_ASCEND_CP_BALANCE_DUMP_DIR` 覆盖（`run_cp_diag.sh` 的 `DUMP_DIR` **同时**设置两者，一个旋钮保证 writer 与 checker 一致）；文件名 `<kind>_cpbal<N>_layer<L>_rank<R>_pid<P>_<ts>.pt`——自带 cp_balance 标记，所以**一轮 A/B 的 B 与 C 数据可以同时收**，且多轮共用一个目录不会互相覆盖（判读时按 `(layer, rank, cpbal)` 取最新一份；`act` 按 `(layer, op, rank, cpbal)`）。
⚠️ `kv:all` 一轮是 **GB 级**（78 层 × rank 数 × 2 配置 × 每份约 2.7MB），`act` 是**百 MB 级**（每个 sample ≈ 全 rank 合计 `hidden × 2B × 序列长度`，hidden=7168、2048 token 时约 29MB，一层两个 sample），**都别写在 `/dev/shm` 上**：写满会连 server 一起搞死（vLLM 的 IPC/prometheus 目录就在 `/dev/shm`），表现为 `torch.save ... inline_container.cc unexpected pos`、请求 `Connection refused`、以及截断的 dump 文件。`probe` 模式因此默认把 `DUMP_DIR` 指到 `/root/cp_probe`。

## 三、指标与判定口径

driver 对每个 case 的每个请求算：

| 指标 | 含义 |
| --- | --- |
| `p99\|d\|X-Y` / `max\|d\|X-Y` | 两配置真实 token logprob 差值的 p99 / 最大值 |
| `top1%X~Y` | top-1 token 一致率 |
| `top5_ovl%X~Y` | top-5 集合互相包含率 |
| `first_div_X-Y` | 首次出现“连续 `--run-len`(=8) 个位置 `\|d\|>threshold`”的位置 |
| `block@first_div_X-Y` | 该位置落在哪个 zigzag 分片 `rankX/prev\|next(b起-止)`（需能 import `vllm_ascend.layers.cp_zigzag`） |
| `gen_top1%` / `gen_top5_ovl%` | 生成 token 位置的对应指标 |

判定规则：

- **噪声地板**：`threshold = max(--delta-threshold, 5 × noise_p99)`，其中 `noise_p99` 来自 B2−B；默认 `--delta-threshold 0.05`。
- **回到噪声级**：`p99|d|C-B <= threshold` 记为 `OK`，否则 `DIFF`。
- **T1 的结论句**：`compare_cp_rounds.py` 只看最新一轮——`OK` 则说明“差异只由该轮改动的那个变量造成”，`DIFF` 则说明该轮配置不足以解释异常，继续查 dump 侧假设。
- 退出码：`ab_cp_compare.py` 配置/zigzag 校验失败返回 1；`check_zigzag_dumps.py` 违规返回 1、找不到 dump 返回 2；`compare_cp_rounds.py` 回到噪声级返回 0、仍超阈返回 1、无可比 case 返回 2。

## 四、诊断矩阵（`run_cp_diag.sh` 的模式）

> 编号沿用 `run_cp_diag.sh` 注释与 `vllm_ascend/envs.py` 的引用（原计划文档 `CP_BALANCE_精度问题_下一步行动计划.md` 未随代码入库，这里按代码现状整理）。

| 模式 | 轮次目录 | 做的事 | 回答的问题 |
| --- | --- | --- | --- |
| `baseline` | `r1_baseline` | B/C/B2 三个配置 + `topk:6,kv:0,6` dump（T0+T2+T3） | 现状是否真的有差异、差异多大、落在哪个分片 |
| `sweep` | `r_sweep` | B/C（无 B2）+ `kv:all` 全层 KV dump | 第几层的 KV 先不等（→ 分歧是在上一层输出里进入的） |
| `probe` | `r_probe` | B/C（无 B2）+ `act:0,1,2,3,kv:0,1,2,3`，dump 默认落 `/root/cp_probe` | **O1**：哪一步先不等——attention 内部，还是中间那层的 MoE/MLP |
| `2call` | `r2_2call` | `VLLM_ASCEND_CP_BALANCE_MERGED_CALL=0`，把合并的 2B 调用退回 prev/next 两次调用 | T1：差异是否来自“合并 2B 调用” |
| `l1024` | `r3_l1024` | prompt 长度与 `MIN_TOKENS` 都降到 1024（激活阈值边界） | T4：阈值边界上 `MIN_TOKENS` 是否被正确遵守 |
| `check` | — | 直接 `exec` `check_zigzag_dumps.py --dir <dump> --kind ${KIND:-both}`（`KIND=act` 时自动加 `--summary-only`） | T2/T3/O1 的判读（不需要 NPU） |

默认参数：`PROMPT_LENS=2048,2049,4096`、`MIN_TOKENS=2048`、`OUT_ROOT=/dev/shm/cp_ab`、`BASE_PORT=8034`、`DUMP_SPEC=topk:6,kv:0,6`、`DUMP_DIR=/dev/shm/cp_balance_dump`，`LAUNCHER` 默认指向 `launcher_glm52_w4a4c8_mxfp4.sh {port}`。

## 五、怎么跑

前置：在 vllm-ascend 仓库根目录执行，宿主机已 source 站点环境（CANN / vendor `set_env.bash`），`MODEL_PATH` 指向 GLM-5.2 权重。

```bash
# 0) 一键自检（不需要 NPU）——只在首跑、或换了机器/挂载点/launcher/TP 时跑一次，
#    日常迭代不用重复跑，直接执行下面的目标命令即可
python tools/cp_balance_compare/selfcheck.py
#    跳过最慢的 mock 自测 / 指定站点 launcher：
#    python tools/cp_balance_compare/selfcheck.py --skip-selftest \
#        --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}"

# 0b) 无 NPU 自测（只有改了 driver/工具代码时才需要）
python tools/cp_balance_compare/selftest_mock.py
# 或 pytest tools/cp_balance_compare/selftest_mock.py -q

# 1) 不加载模型，只校验 launcher/env/指纹（DRY_RUN=1）
python tools/cp_balance_compare/ab_cp_compare.py \
    --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}" \
    --configs B,C --repeat-a --preflight

# 2) 一轮基线（T0+T2+T3，2~3 次模型加载）
bash tools/cp_balance_compare/run_cp_diag.sh baseline
# 只看将要执行的命令、不真跑：
bash tools/cp_balance_compare/run_cp_diag.sh baseline --dry-run

# 2b) 多层扫描诊断轮：B/C 两个配置（无 B2，噪声地板已知为 0）+ 全层 KV dump，
#     输出到 /dev/shm/cp_ab_sweep/r_sweep；一条命令、不依赖任何 env 前缀。
#     dump 是 GB 级，先把 DUMP_DIR 指到真实磁盘（单独一行 export，别写成命令前缀）：
export DUMP_DIR=/root/cp_dump
bash tools/cp_balance_compare/run_cp_diag.sh sweep

# 3) T1：回退成 prev/next 两次调用
bash tools/cp_balance_compare/run_cp_diag.sh 2call

# 3b) O1：op 级激活剖面（B/C 无 B2，attention in/out 全精度、按 token 位置对齐）。
#     dump 默认落 /root/cp_probe（百 MB 级，别放 /dev/shm）：
bash tools/cp_balance_compare/run_cp_diag.sh probe
#     要改层数/范围就用 DUMP_SPEC（单独一行 export，别写成命令前缀）：
#     export DUMP_SPEC=act:0,1,2,3,4,5; bash tools/cp_balance_compare/run_cp_diag.sh probe

# 4) CPU 侧判读：dump 表（每层一行 + FIRST DIVERGENCE）与多轮并排对比
python tools/cp_balance_compare/check_zigzag_dumps.py --dir "${DUMP_DIR:-/dev/shm/cp_balance_dump}" \
    --kind kv --summary-only
python tools/cp_balance_compare/check_zigzag_dumps.py --dir /root/cp_probe \
    --kind act --summary-only
python tools/cp_balance_compare/compare_cp_rounds.py \
    sweep=/dev/shm/cp_ab_sweep/r_sweep baseline=/dev/shm/cp_ab/r1_baseline

# 5) 收集整轮证据（HEAD / dump 清单 / summary 指标 / 日志关键行）到一个文件里
python tools/cp_balance_compare/selfcheck.py --collect --out-root /dev/shm/cp_ab_sweep
```

常用覆盖（`run_cp_diag.sh` 的环境变量）：`PROMPT_LENS`、`MIN_TOKENS`、`TP_SIZE`（launcher 的 TP，默认 8）、`CP_SIZE`（driver 的 `--cp-size`，默认取 `TP_SIZE`）、`REPEAT_A`（默认 1；`REPEAT_A=0` 或命令加 `--no-repeat` 跳过 B2 重复跑，**省一次模型加载**，噪声地板已知为 0 时用）、`OUT_ROOT`、`BASE_PORT`、`DUMP_SPEC`（置空=不 dump；支持 `kv:all` / `topk:all` / `act:0,1,2`）、`DUMP_DIR`（**writer 与 checker 共用**，默认 `/dev/shm/cp_balance_dump`，`probe` 模式默认 `/root/cp_probe`）、`KIND`（`check` 模式的判读类型，默认 `both`）、`LAUNCHER`。

launcher 还支持透传额外的 `vllm serve` 参数：`EXTRA_SERVE_ARGS="--enable-return-routed-experts"`（空格分隔；launcher 会 echo 一行 `[cp-ab] EXTRA_SERVE_ARGS=` 便于追溯）。这是给"路由抓取"预检用的——vLLM 自带的 `--enable-return-routed-experts` 会在 `/v1/completions` 响应里多返回 `routed_experts`（base64 的 `.npy`，形状 `(num_tokens-1, num_layers, num_experts_per_tok)`，**一次请求拿到全部层逐 token 的专家选择**），代价是**必须关掉 KV connector**（`VLLM_ASCEND_KV_TRANSFER_CONFIG=""`）且 PP=1。

> 诊断轮优先用 **`sweep` 模式**而不是一堆 env 前缀：脚本模式写在同一行命令里，粘贴时不会像 `VAR=... cmd` 那样被折断后**静默退回默认值**（那样会白跑一轮 B2）。

> **一轮的成本结构**：几乎全在模型加载（约 10 分钟/次；TP=8/16 量级相当），推理只要 ~2 秒/条。所以"加长度/加 dump 层数"几乎免费，而"多跑一个配置"（B2、2call）就是 +10 分钟。诊断轮按这个取舍：`sweep` 模式（= `--no-repeat` + `DUMP_SPEC=kv:all`）是 2 次加载 + 全层数据；`probe` 模式（= `--no-repeat` + `act:0,1,2,3`）也是 2 次加载，多花的是磁盘（几百 MB）而不是时间——所以**一次打点尽量多带几层**。

已经有 server 在跑时，可以跳过拉起，直接驱动两个/三个地址（注意此模式无法读 server 日志，指纹与 zigzag 检查会被跳过）：

```bash
python tools/cp_balance_compare/ab_cp_compare.py --out /dev/shm/cp_ab \
    --urls B=http://n1:8034,C=http://n2:8034,B2=http://n3:8034 --repeat-a
```

### 省掉每次 source 的启动时间（可选，每轮省两次 source）

launcher 每次启动都会 `source /root/.bashrc` + vendor `set_env.bash`，而一轮要起 2~3 次 server；如果这两个脚本在站点上很慢，就白付好几次。因为 driver 起 server 用的是 `env = os.environ.copy()`，所以**在跑轮次的那个 shell 里先 source 一次**与"launcher 自己 source"对 server 完全等价。

```bash
# 每个 shell / tmux 会话做一次（必须 source，会打印两段 source 的耗时）
source tools/cp_balance_compare/prepare_env.sh
# 之后照常跑，launcher 会跳过 source：
bash tools/cp_balance_compare/run_cp_diag.sh sweep
```

等价的显式写法（不想用脚本时）：

```bash
source /root/.bashrc
source /mnt/share/l00622059/vendors/custom_transformer/bin/set_env.bash   # 路径以 launcher DRY_RUN 打印的 vendor= 为准
export CP_AB_SKIP_SOURCE=1
```

判据：日志里应出现 `[cp-ab] CP_AB_SKIP_SOURCE=1: 跳过 source …` 和 `[cp-ab] env check: ASCEND_HOME_PATH=… LD_LIBRARY_PATH=…B PYTHONPATH=…`——**必须确认这两行里的路径/长度是你要的那套环境**，否则 server 会在缺 CANN/custom ops 的环境里起步。必须 `export`（不是仅赋值），且要在同一个 shell 里；恢复原行为：`unset CP_AB_SKIP_SOURCE`。指纹行不受影响（`selfcheck.py`/driver 的配置门照常校验）。

### 单配置手工调试（推理报错时用这个）

只拉起 **Base（`B`：DSA-CP on + `CP_BALANCE=0`）**，用与 A/B 轮完全相同的请求体发一次推理，然后把 server 留着继续调试；模型拉起日志实时打屏（前缀 `[B]`）。

```bash
# 拉起 Base + 发一次推理；server 默认保留，Ctrl-C 停止并释放 NPU
python tools/cp_balance_compare/run_single.py
# 换 zigzag（C）/ 只跑一个长度 / 推理完就停 / 只发请求（server 已在跑）
python tools/cp_balance_compare/run_single.py --config C
python tools/cp_balance_compare/run_single.py --prompt-lens 2048 --no-keep
python tools/cp_balance_compare/run_single.py --url http://127.0.0.1:8034
```

报错时会打印 HTTP 状态码 + **响应正文**（真正的报错通常在那里）和异常 traceback，并给出 server 日志尾部；请求体/响应体落在 `--out`（默认 `/dev/shm/cp_single`）下的 `payload_*.json` / `response_*.txt`，可直接用脚本打印的 curl 命令原样复现。

## 六、产物

```
$OUT_ROOT/<round>/
├── summary.json          # noise / runtime(zigzag 标记) / skipped / 每个 case 的 metrics（compare_cp_rounds.py 的输入）
├── results_all.json      # 各配置全部原始解析结果
├── results_<配置>.json
├── <case>_req<N>.png     # logprob 曲线 / |Δ| 与阈值线 / top-1 滑动一致率（--no-plot 关闭）
├── <case>_req<N>.csv     # 逐位置原始数据（--no-csv 关闭）
└── logs/server_<配置>.log # 指纹行、zigzag 标记、异常栈
/dev/shm/cp_balance_dump/  # topk / kv 的 .pt dump（文件名自带 cpbal0|1）
/root/cp_probe/            # act（attention in/out）的 .pt dump（probe 轮的默认目录）
```

## 七、注意事项

- **顺序执行是设计前提**：TP=N 的 server 独占整机，所以 driver 一个一个拉起/停掉；多机场景请自己起 server 后用 `--urls`。
- **TP 与 zigzag 的 `cp_size` 是同一个数**：`vllm_ascend` 把 `global_tp_size` 当作 zigzag 的 `cp_size`（SP padding 到 `2 * tp_size`，每条序列切成 `2 * cp_size` 块），所以 launcher 的 `TP_SIZE` 必须等于 driver 的 `--cp-size`。站点默认 8（`ASCEND_RT_VISIBLE_DEVICES` 默认 `0..7`，需要 8 张可见 NPU）；`run_cp_diag.sh`/`run_single.py` 用 `CP_SIZE`、`TP_SIZE` 统一取默认值，`selfcheck.py` 会核对 launcher 打印的 `tp=` 与 `--cp-size`，server 日志里的 `[CP_BALANCE] metadata zigzag=1 … cp_size=8` 是最终确认。
- **C 没进 zigzag 就没有结论**：`VLLM_ASCEND_CP_BALANCE_MIN_TOKENS` 被 driver 显式钉住并写入指纹（源码默认是 8192），否则盒子默认值会让 C 悄悄留在连续切片路径上，看起来像“cp_balance 对精度无影响”。`run_cp_diag.sh` 用 `--zigzag-check strict --on-zigzag-miss skip` 兜底。
- **阈值不是绝对精度判据**：`0.05` 只是起步线，真正的判据是它和 5×噪声地板取大者；没有 B2 就没有噪声地板，`--repeat-a` 应默认带上。
- **dump 目录会跨轮累积**：判读只取每个 `(layer, rank, cpbal)` 的最新一份（`act` 是 `(layer, op, rank, cpbal)`），旧文件被忽略（会打印提示），不要求手动清理。但**同一个目录里混两轮不同代码/配置的 dump 会串味**，所以每类诊断轮用独立目录（`sweep`→`/root/cp_dump`、`probe`→`/root/cp_probe`）。
- **`--kind topk` 对短 prompt 才最有信息量**：长 prompt 下 indexer 不做 `validS2Len < topkCount_` 快捷路径，identity 断言不再适用。
- driver 需要 `requests` + `numpy`；`check_zigzag_dumps.py` 需要 `torch`（跑在 vLLM 宿主机上）+ `numpy`；画图需要 `matplotlib`（缺了只警告跳过）。
- 非 Linux 主机上，`selftest_mock.py` 里依赖 bash 的 launcher 用例会打印 `[skip]`（`bash not usable on this host` / 假 `vllm` 无法置可执行位时的 `fake vllm is not executable on this host`），属宿主差异，不是 driver 的问题；fingerprint 格式校验那半边仍会跑。
- **`/dev/shm` 很小是常见坑**（容器默认 64MB 量级）：全层 dump 会把它写满，进而连 server 一起搞死（IPC/prometheus 都在那儿）。跑 sweep 前先 `df -h /dev/shm`，用 `DUMP_DIR=/真实磁盘` 落盘；写满的现场特征是 `torch.save … inline_container.cc unexpected pos`、请求 `Connection refused`、checker 读到截断文件（新版会跳过并告警，不再崩）。
- **环境变量用 `export`，别用命令前缀**：`VAR=... cmd | tee ...` 这类长命令粘贴后前缀可能被吃掉，脚本会静默退回默认值（曾经因此白跑一轮 B2）。`sweep`/`baseline` 这类**模式词**和 `--no-repeat` 就是为此加的。
- **dump 的量化精度**：sparse-C8 站点的 packed KV 是 fp8 (e4m3) + e8m0 scale。`kv_fp_nat` 只是"写 cache 之前的同源副本"（用来绕开不可用的 NPU `index_select`），**不是更高的精度**；要观察比量化更细的差异，只能在 op 级打点（如 attention 输入/输出）。
- **`act` 剖面的两个前提**：① 按 token 位置对齐（位置取自 `slot_mapping_cp`，与写 KV 用的是同一个数组），**不要**按行号比——zigzag 与连续切片下同一个行号是不同 token；② 它是 one-shot（每层每进程一次，跳过 profile/warmup），所以只有**第一个** prefill 请求的数据（driver 发的 2048 那条），后面的 2049/4096 不再 dump。
- **改了 `vllm_ascend/` 的模块要防"模块级用了未 import 的名字"**：`sfa_v1.py` 把 stdlib import 放在函数内，`os`/`time` 在模块级直接用会在**加载模型时**才炸（`selftest_mock.py` 里有 AST 静态检查覆盖这一类）。

## 八、工具与脚本的五条硬要求

> 这五条是对本目录所有工具（含后续新增的诊断脚本）的固定要求，新增或修改代码时都必须满足。

1. **修改代码直接 commit。** 不留未提交的改动：无论改的是 `vllm_ascend/` 里的实现还是本目录的诊断工具，改完就在当前仓库 commit，提交信息写清动机、改动点和验证方式。诊断轮次靠 `git rev-parse HEAD` 对齐版本（`selfcheck.py` 会打印 HEAD 与工作区改动），未提交的改动会让“这一轮结果对应哪份代码”无法追溯。

2. **执行脚本尽量做到一行执行。** 每一个测试/诊断动作都要有一条可直接复制粘贴、不需要手工拼参数的命令；默认值覆盖常见场景，环境差异用参数或环境变量覆盖。步骤多于一步的，用一个脚本把它们串成一条命令（例如 `python tools/cp_balance_compare/selfcheck.py` 一次完成现场体检 + driver 自测 + 配置门 + 轮次预演），不要写成需要人工分多步交互的形式。

3. **必须说明需要得到的结果。** 每个脚本、每一轮测试都要明确给出“期望看到什么”：判据是哪个数字/哪个阈值/哪一行标记，`PASS`/`WARN`/`FAIL`/`[skip]` 各代表什么，以及不满足时下一步去查什么。只打印数据、不给判据的脚本视为未完成——没有判据就无法下结论。范例见 `selfcheck.py`：每个检查项都会打印一行 `期望结果:`。

4. **一次只给一步。** 每次只说明当前要执行的一步：一条命令 + 这一步需要得到的结果；上一步的结果确认后再给下一步，不预先罗列后续步骤。每一步的产出决定下一步走向，一次给多步会诱导跳过判据、在错误的假设上继续跑。

5. **不要每次都跑 `selfcheck.py`。** 自检是入口门槛，只在首跑、或换了机器/挂载点/launcher/TP 配置、或怀疑环境变化时跑一次；日常迭代（改代码、单配置调试、跑轮次）直接执行目标命令，不做重复自检。

