# cp_balance_compare —— 工具说明

本目录只做一件事：**确认 DSA-CP 的 `cp_balance`（zigzag 切分）在 prefill 上是否引入精度差异、差异出在哪一步**。

> **接手问题**先看 [`HANDOVER.md`](./HANDOVER.md)：现在确定知道什么、已排除什么、下一步做什么、有哪些坑。
> 本文只讲：工具怎么用、怎么判读、以及工具侧必须遵守的约定。

## 一、对比对象

| 配置 | DSA-CP | `VLLM_ASCEND_CP_BALANCE` | 含义 |
| --- | --- | --- | --- |
| `B` | on | `0` | 连续切片基线（参照） |
| `C` | on | `1` | zigzag cp_balance（被测） |
| `B2` | on | `0` | B 的重复跑（`--repeat-a`），只用于量噪声地板 |

两条路径 kernel/模型/权重完全相同，**只有 token 排布不同** ⇒ `C−B` 就是 cp_balance 的账；`B2−B`（实测 0.0）给出它必须打败的噪声地板。
"关 DSA-CP 的 A 锚点"已下线：它走不同代码路径，driver 直接拒绝。

## 二、工作流程

```
① 改代码（本仓库，改完立刻 commit）
        ↓ 确认 origin 跟上：git rev-parse --short HEAD origin/glm52_cp_balance_v3（不同则 git push origin glm52_cp_balance_v3）
        ↓ 用户在远端同步到同一 commit（同步后建议先跑 selftest_mock.py）
② 远端跑一轮（NPU 独占整机：B、C 顺序起停，每次约 10 分钟加载）
        ↓ 产物：<round>/summary.json + logs/server_*.log，dump 落在 DUMP_DIR
③ CPU 侧判读（几秒~十几秒，不需要 NPU）
        ↓ 把 log.log / 屏幕输出贴回来 → 决定下一步（回到 ① 或 ②）
```

三条铁律：**改完就 commit**（轮次靠 `git rev-parse HEAD` 对齐）、**每条命令都要有判据**、**一次只走一步**。
每轮的"三件套"证据：`[cp-ab]` 指纹行 → `dump=… dir=…` / `dump: +N file(s)` → 判读的 `FIRST DIVERGENCE` 行。

## 三、目录里各文件的角色

| 文件 | 角色 | 需要 NPU |
| --- | --- | --- |
| `ab_cp_compare.py` | 主 driver：顺序拉起 B/C/B2、发请求、算指标、写 `summary.json` | 是 |
| `launcher_glm52_w4a4c8_mxfp4.sh` | 站点 launcher（TP=8），支持全部 cp_balance 环境覆盖、打印指纹、透传 `EXTRA_SERVE_ARGS` | 是 |
| `launcher_template.sh` | launcher 参考实现（任何 launcher 必须遵守的两条规则：env 覆盖必须生效、必须打印指纹） | 是 |
| `run_cp_diag.sh` | 一轮 A/B 的标准入口（模式见 §四） | 是（`check` 除外） |
| `run_single.py` | 单配置手工调试：起一个 server + 用与 A/B 相同的请求体发一次推理，落盘 payload/response 供 curl 复现 | 是 |
| `check_zigzag_dumps.py` | CPU 侧判读 dump（`kv` / `topk` / `act`，见 §五） | 否 |
| `repro_row_order.py` | 离线复现"同一 token 的行换个位置结果就变"：读一轮的 `dnq`/`guq` dump（+ 同层 `w` 权重 dump）重跑 `npu_quant_matmul`，比较两种行序下同一 token 的输出 | 否（`--run-op` 需要 NPU） |
| `compare_cp_rounds.py` | 把多轮 `summary.json` 并排，回答"改了某个变量后 `C−B` 是否回到噪声级" | 否 |
| `mock_vllm_server.py` / `selftest_mock.py` | 假 server + CPU 自测（51 项：协议解析、指纹/payload、dump 判读、launcher 静态检查、端到端）。逐项打印 `[run]`/`[ok] … (耗时)`，单项 90s 超时（Linux 下 SIGALRM + faulthandler 打印卡住的栈），失败不中止整轮；支持 `--list`、`--only <子串>`、`--skip launcher,preflight`、`--test-timeout N`。**每次 `bash` 启动 >2s 的机器**（重 `BASH_ENV`/慢挂载）会自动跳过 4 个起 launcher 的用例并说明原因（要强制跑用 `--only`） | 否 |
| `selfcheck.py` | 环境体检（解释器/依赖/import 来源/NPU/端口/磁盘/残留进程）+ **不依赖 git 的版本指纹**（`--fingerprint`：关键文件 sha256 + 修复标记 + 用例数，末行 `[fp] …` 贴回来即可对齐版本）+ 跑一遍自测 + `--collect` 收整轮证据。**不做**配置门与命令预演——那两件事由 `ab_cp_compare.py --preflight` 与 `run_cp_diag.sh <mode> --dry-run` 负责（每轮都会跑，不会腐化） | 否 |
| `prepare_env.sh` | 一次性 `source` 站点 rc + vendor 环境并 `export CP_AB_SKIP_SOURCE=1`，省掉每轮两次 source；**必须 source** | 否 |

## 四、一轮 NPU 轮次怎么跑

`run_cp_diag.sh` 的模式（默认参数：`PROMPT_LENS=2048,2049,4096`、`MIN_TOKENS=2048`、`TP_SIZE=CP_SIZE=8`、`BASE_PORT=8034`）：

| 模式 | 轮次目录 | 做什么 | 回答什么 |
| --- | --- | --- | --- |
| **`probe`** | `r_probe`（dump→`/root/cp_probe/<轮次时间戳>`） | B/C（无 B2）+ `act:0,1,mlp:0,1,qin:0,topk:0,kv:0,1` | **当前主用**：哪一步先不等（attention / pre-MLP / MLP 内部的量化 vs GEMM） |
| **`probe2`** | `r_probe_c2`（dump→同 probe，每轮时间戳目录） | 同 probe，但 `--configs C,B --repeat-a` ⇒ **C, B, C2**（3 次加载 ≈30 分钟） | 上面那张表 **+ `[noise] C2 vs C`**：区分"内核与行序相关"（C2−C=0）与"内核不可复现"（C2−C≠0） |
| `sweep` | `r_sweep` | B/C + `kv:all` 全层 KV dump | 第几层的 KV 先不等（→ 上一层输出进入） |
| `baseline` | `r1_baseline` | B/C/B2 + `topk:6,kv:0,6` | 现状差异多大、落在哪个分片 |
| `check` | — | 直接 `exec check_zigzag_dumps.py --dir <dump> --kind ${KIND:-both}`（`KIND=act` 自动加 `--summary-only`） | 判读（不需要 NPU） |

```bash
# 当前这一步（一轮 ≈ 20 分钟，2 次模型加载）
unset DUMP_DIR                                     # ⚠️ 继承的 DUMP_DIR 会让 dump 改道（§八.1）
bash tools/cp_balance_compare/run_cp_diag.sh probe --dry-run    # 先看 dir= 与 spec 对不对
bash tools/cp_balance_compare/run_cp_diag.sh probe
# 判读（--dir 用上一行 dry-run/运行输出里打印的那个，每轮都不同）
python tools/cp_balance_compare/check_zigzag_dumps.py --dir /root/cp_probe/<时间戳> \
    --kind act --summary-only --block-size 128 2>&1 | tee tools/cp_balance_compare/log.log
```
⚠️ `probe` 每轮写进 `/root/cp_probe/<时间戳>` 子目录：共用一个大目录时，判读按
`(layer, op, rank, cpbal)` 取"最新一份"，会把上一轮的旧文件混进来（spec 变过就更是两次
测量拼一张表）。判读开头会打印**选中文件的时间跨度**，超过 90 分钟会告警。

**跑起来先看两行**：模型加载完应打印 `[CP_BALANCE][dump] mlp trace armed for [...]` 与
`[CP_BALANCE][dump] quant trace armed for [...]`（后者只在 spec 带 `qin:` 时出现）；
结束时应有 `dump: +N file(s)`（`probe` 轮 N≈272：layer 0 八个采样点 + layer 1 六个，
× 8 rank × 2 配置 = 224，加 `topk:0` 16 + `kv:0,1` 32）。
⚠️ 若打印的是 `mlp trace requested … but no 'layers.<L>.mlp*' module matched`（或 `quant trace … no '….mlp.gate_up_proj'`）
说明模块名不匹配 → 把该行贴回来改匹配规则。
⚠️ 若出现 `[CP_BALANCE][dump] dnin layer=0 skipped: value is a quantized tuple …`，
说明该层的 norm+quant 已被融合（`fuse_norm_quant`），此时 `dn_in` 少 16 个文件（N≈256），
**该缺口由同一轮的 `dn_q` 补上**——它是同一个数据的量化版本。
⚠️ 若出现 `… skipped: no token positions for rows=2048 …`，说明该采样点既不是 rank 本地行、
也不在补齐后的自然 slot mapping 覆盖范围内 → 该行会缺失（判读会打 `注意: … 缺 … 行`）。

常用覆盖：`PROMPT_LENS`、`MIN_TOKENS`、`TP_SIZE`、`CP_SIZE`、`REPEAT_A=0`/`--no-repeat`（省一次加载）、`OUT_ROOT`、
`DUMP_SPEC`（如 `act:0,1,2,3,mlp:0,1,2,3`、`kv:all`；置空=不 dump）、`DUMP_DIR`（**writer 与 checker 共用一个旋钮**）、`KIND`、`LAUNCHER`。
launcher 还支持 `EXTRA_SERVE_ARGS="--enable-return-routed-experts"`（空格分隔，透传额外 `vllm serve` 参数；
该 flag 要求 **PP=1 且不能用 KV connector**——driver 与 `run_single.py` 默认都已经关掉 KV connector ✓）。

已有 server 在跑时可跳过拉起：`ab_cp_compare.py --urls B=…,C=… --repeat-a`（此模式读不到 server 日志，指纹/zigzag 检查会被跳过）。
每轮都要 `source` 两次很慢时：先 `source tools/cp_balance_compare/prepare_env.sh`（判据：日志出现 `CP_AB_SKIP_SOURCE=1` 与 `env check: ASCEND_HOME_PATH=…`，**必须确认路径是你那套环境**）。

## 五、CPU 侧判读（`check_zigzag_dumps.py`）

| `--kind` | 数据 | 判据 |
| --- | --- | --- |
| `kv` | packed KV（自然 token 序） | `[kv/fp]` 表：`rows_val` / `first_val` / `rows_byte` / `first_byte` / `max\|d\|` / `byte_only`；`FIRST DIVERGENCE (fp/value\|fp/bytes\|fp): layer L` |
| `topk` | indexer 索引表 | ① 每行有效前缀 == 因果窗口（P1 断言）；② `[topk/cross]`：B vs C 逐 token 比**集合**与**顺序** |
| `act` | attention 与 MLP 的逐层采样 | 每层 6 行（数据流顺序），`FIRST DIVERGENCE (act): layer L op=… at token P` |

`--kind act` 的一行 = 一个采样点，顺序即数据流顺序（每层最多 8 行）：

```
in ─attention─▶ out ─pre-MLP norm─▶ mlp_in ─[量化]─▶ gu_q ─gate_up_proj─▶ gu_out
   ─silu─▶ dn_in ─[量化]─▶ dn_q ─down_proj─▶ mlp_out
```

**最早不等的那一行决定下一步查哪里**：`in`=层间（残差/norm）；`out`=本层 attention；`mlp_in`=attention→MLP 之间；
`gu_q`=gate_up 的**量化输入**（fp8 + e8m0 scale）；`gu_out`=gate_up 这个 GEMM；`dn_in`=silu 之后的 bf16；
`dn_q`=down_proj 的量化输入；`mlp_out`=down_proj（MoE 层还含专家路由/分组）。
`gu_q`/`dn_q` 与相邻行合起来把"同一个 bf16 输入却算出不同结果"拆成互斥的两种根因：

| 观察 | 根因 |
| --- | --- |
| `mlp_in` 相同、**`gu_q` 不同** | **激活量化这一步**（同一批 bf16 行量化出不同 fp8/scale） |
| `gu_q` 相同、**`gu_out` 不同** | **`npu_quant_matmul` 这个 GEMM 内核**（逐行数学与输入都相同 ⇒ tiling/workspace/确定性） |
| `gu_out` 相同、**`dn_q` 不同** | silu 之后的**量化** |
| `dn_q` 相同、**`mlp_out` 不同** | **down_proj 这个 GEMM**（含跨 rank 归约） |

若某层没有 `gu_q`/`dn_q` 行（spec 没写 `qin:<层>`），判词会明确写"本层未打点 qin，无法区分"，
**不会**把缺失当成"相同"。表尾会打印 `[act] INCOMPLETE layer=L op=…: only cp_balance=[…]`
—— 表示该采样点只有单侧 dump（**不是"相同"**）。

关键读表约定：

- **`positions` 是 KV cache slot，不是 token 序号**（请求块表可能从 block 1 起 ⇒ slot = token+128）。判读按 B/C 公共 base 归一化成 token 序号；`--block-size 128` 会把首个分歧标到第几个 zigzag 块。
- `kv` 的 `rows_val>0` ⇒ 存储**数值**不同（真差异）；`rows_val=0` 而 `rows_byte>0` ⇒ 只有量化零点符号位翻转（**亚量化**，差异小于一个量化步）；两者都为 0 ⇒ 逐位相同。
- `--summary-only` 用于多层扫描：进度打到 stderr（kv 每层一行、act 每层 6 行；全层 1248 文件约十几秒）。
- 退出码：**1** = 发现真差异/违规（kv 数值不同、topk 集合不同、dump 与因果窗口不符）；**0** = 干净（含"只翻零点符号位"的亚量化）；**2** = 找不到 dump。

其它工具：`compare_cp_rounds.py <轮1>=<目录> <轮2>=<目录>` 并排多轮指标（回到噪声级 rc=0、仍超阈 1、无可比 case 2）；
`selfcheck.py` 首跑/换机器时跑一次即可（**日常迭代不要重复跑**）；配置门与命令预演它不再重复实现：
`python tools/cp_balance_compare/ab_cp_compare.py --preflight`（判据末行 `[preflight] all configs OK`）
与 `bash tools/cp_balance_compare/run_cp_diag.sh probe --dry-run`。

## 六、指标与判定口径（driver）

| 指标 | 含义 |
| --- | --- |
| `p99\|d\|X-Y` / `max\|d\|X-Y` | 两配置真实 token logprob 差值的 p99 / 最大值（同一 token，信号干净） |
| `top1%X~Y` / `top5_ovl%X~Y` | top-1 一致率 / top-5 集合互相包含率 |
| `first_div_X-Y` | 首次出现"连续 `--run-len`(=8) 个位置 `\|d\|>阈值`"的位置 |
| `block@first_div_X-Y` | 该位置落在哪个 zigzag 分片（`rankX/prev\|next`） |
| `gen_top1%` / `gen_top5_ovl%` | 生成 token 位置的对应指标 |

判定：`threshold = max(--delta-threshold(0.05), 5×noise_p99)`；`p99|d|C-B <= threshold` 记 `OK`，否则 `DIFF`。
driver 每个配置都会：校验 `[cp-ab]` 指纹与 `[cp-ab-cfg]` 里的 `additional_config`（`--config-check strict`）、
确认 C 真进了 zigzag（`--zigzag-check strict`，没进就 `--on-zigzag-miss skip` 停掉整轮非 0 退出）。

## 七、已删除的实验（别再去找开关）

| 曾经的模式 | 为什么删掉 |
| --- | --- |
| `2call`（`VLLM_ASCEND_CP_BALANCE_MERGED_CALL=0`，prev/next 两次调用） | A/B 已判定**无信息量**：它只改 attention/indexer 的调用形状，而 layer 0 的 attention 输出在 B/C 下逐字节相同。开关、两条调用分支与 `run_cp_diag.sh 2call` 一并删除（需要时从 git 历史取回） |
| `l1024`（`MIN_TOKENS=1024`） | 只查 `MIN_TOKENS` 边界是否被遵守，与本问题无关。`MIN_TOKENS` 本身仍是产品开关，可用 `MIN_TOKENS=1024 PROMPT_LENS=1024 run_cp_diag.sh probe` 复现 |

仍然有效的两条判读约定：

| 约定 | 说明 |
| --- | --- |
| `--kind topk` 的 P1 断言 | 只对**短 prompt**（`< index_topk`）有信息量；长 prompt 下 indexer 走全选捷径，identity 断言不适用 |
| `baseline` 的 `B2` | 噪声地板已实测为 0；除非换机器/换 TP，可 `--no-repeat` 省一次加载 |

## 八、易错点（工具侧）

1. **`DUMP_DIR` 会被继承**：`probe` 尊重已导出的 `DUMP_DIR`，sweep 留下的 `/root/cp_dump` 会让 probe 数据改道、`/root/cp_probe` 不出现。跑前 `unset DUMP_DIR`，或先 `--dry-run` 看 `dir=`（脚本会对继承值打 WARN）。
2. **`/dev/shm` 很小**（容器默认几十 MB）：`kv:all` 是 GB 级、`act`/`mlp` 是几百 MB 级 ⇒ 一律落真实磁盘；写满会连 server 一起搞死（IPC/prometheus 在那儿）。
3. **dump 目录跨轮累积**：判读按 `(layer[, op], rank, cpbal)` 取最新一份 ⇒ 不同轮次混在同一目录会"串味"。`probe`/`probe2` 已默认写进 `/root/cp_probe/<时间戳>` 子目录；`sweep` 等其他模式仍写固定目录，跨轮复用前先换目录。判读会打印选中文件的时间跨度（>90 分钟告警），并在**指到父目录**（还有更新的子目录）时直接给出该用的 `--dir`；子目录里没有 dump 时也会提示最近一轮在哪。
4. **`VAR=... cmd | tee` 前缀会静默丢**：用 `export` 单独一行，或用模式词（`probe`/`sweep`/`--no-repeat`）。
5. **打点是 one-shot**（每层每进程一次、跳过 profile/warmup）⇒ 只有**第一个 prefill 请求**（driver 发的 2048）有数据；换层要改 `DUMP_SPEC` 再跑一轮。
6. **截断的 dump** 判读会跳过并告警（不再崩）；一轮跑完却没有 dump，`run_cp_diag.sh` 以 rc=3 明确失败。
7. **改 `vllm_ascend/` 的模块**后先跑 `python tools/cp_balance_compare/selftest_mock.py`（含"模块级用了未 import 的名字"的 AST 静态检查）。
8. **非 Linux 主机**上依赖 bash 的 launcher 用例会 `[skip]`，属宿主差异，不是 driver 的问题。

## 九、工具与脚本的五条硬要求

> 对本目录所有工具（含后续新增的诊断脚本）固定生效。

1. **修改代码直接 commit。** 不留未提交的改动；提交信息写清动机、改动点、验证方式 —— 轮次靠 `git rev-parse HEAD` 对齐版本。
2. **执行脚本尽量做到一行执行。** 每个动作都要有一条可直接粘贴、不用手工拼参数的命令；多步骤用一个脚本串起来（如 `selfcheck.py`）。
3. **必须说明需要得到的结果。** 每个脚本/每轮测试都要写清"期望看到什么"：判据是哪个数字/阈值/标记行，以及不满足时下一步查什么。只打印数据、不给判据的脚本视为未完成。
4. **一次只给一步。** 只说明当前要执行的一步（一条命令 + 期望结果），上一步确认后再给下一步。
5. **不要每次都跑 `selfcheck.py`。** 只在首跑、换机器/挂载点/launcher/TP、或怀疑环境变化时跑一次；日常迭代直接执行目标命令。
