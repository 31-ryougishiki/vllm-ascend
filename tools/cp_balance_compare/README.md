# cp_balance_compare 测试说明

本目录的工具只做一件事：**确认 DSA-CP 的 `cp_balance`（zigzag 切分）在 prefill 阶段是否引入精度差异，以及差异出在哪一步**。

对比对象只有 `VLLM_ASCEND_CP_BALANCE` 这一个开关，DSA-CP 始终打开：

| 配置 | DSA-CP | `VLLM_ASCEND_CP_BALANCE` | 含义 |
| --- | --- | --- | --- |
| `B` | on | `0` | 连续切片基线（可信参照） |
| `C` | on | `1` | zigzag cp_balance（被测） |
| `B2` | on | `0` | B 的重复跑（`--repeat-a`），只用来量噪声地板 |

因为两条路径的 kernel、模型、权重完全相同，只有 token 排布不同，所以 **`C-B` 的差值就是 cp_balance 的账**；`B2-B` 给出它必须打败的噪声地板。原设计里的 “A（关 DSA-CP）锚点” 已下线：它走的是不同代码路径，只能给粗参照，要 A 会被 driver 直接报错拒绝。

> 本目录所有工具（含后续新增的诊断脚本）必须遵守[第八节](#八工具与脚本的四条硬要求)的四条要求。

## 一、目录里各文件的角色

| 文件 | 角色 | 需要 NPU |
| --- | --- | --- |
| `ab_cp_compare.py` | 主 driver：顺序拉起 B/C/B2、发请求、算指标、写 `summary.json` | 是 |
| `launcher_glm52_w4a4c8_mxfp4.sh` | 站点 launcher（GLM-5.2 w4a4c8-mxfp4，TP=8），支持全部 cp_balance 环境覆盖并打印指纹 | 是 |
| `launcher_template.sh` | launcher 参考实现（说明任何 launcher 必须遵守的两条规则） | 是 |
| `run_cp_diag.sh` | 一轮 A/B 的标准入口，封装 baseline / 2call / l1024 / check 四种模式 | 是（`check` 除外） |
| `check_zigzag_dumps.py` | CPU 侧判读 NPU 落盘的 `topk` / `kv` dump（对应 P1 / P2 两个假设） | 否 |
| `compare_cp_rounds.py` | CPU 侧把多轮 `summary.json` 并排对比，回答“改一个变量后 C−B 是否回到噪声级” | 否 |
| `mock_vllm_server.py` | 假 vLLM `/v1/completions` 服务（`/health` + echo/logprobs），用 offset 模拟 B/C 差异 | 否 |
| `selftest_mock.py` | driver 的 CPU 自测（26 个 `test_*`），跑通全链路而不需要 NPU | 否 |
| `selfcheck.py` | 一键自检：现场体检 + driver 自测 + 配置门（launcher `DRY_RUN=1` 指纹）+ 轮次命令预演；`--collect` 收集整轮证据 | 否 |
| `run_single.py` | 单配置手工调试：单独拉起 Base（`B`）或 `C` 的 server + 用与 A/B **完全相同**的请求体发一次推理；失败时打印 HTTP 状态与响应正文，并把 payload/response 落盘供 curl 复现；默认保留 server | 是 |

## 二、三层测试

### L1 — CPU 自测（不需要 NPU，最快）

`selftest_mock.py` + `mock_vllm_server.py`：mock 只实现 driver 用到的协议面，logprob 固定为 `-0.5 - 0.001*(token%100) + offset`，不同端口天然像不同配置。覆盖点包括：

- 响应解析：token-id prompt / 文本 prompt / `prompt_logprobs` 回退 / 三种 `top_logprobs` key 风格必须解析一致；
- 对齐与拒绝：`prompt_token_ids` 与发送的 id 不一致时 `verify_response` 必须报错，不能拿不可比的响应当结果；
- 配置校验：`[cp-ab]` 指纹比对、`additional_config` 逐字比对、base config 里不得含 PD-only 开关；
- 激活判定：`zigzag_state_from_log` 从日志识别 zigzag 标记；
- preflight：假 launcher 下 env 覆盖与指纹逻辑；
- 端到端：两/三个 mock 端口 + `--repeat-a` 跑完 `run()`，校验 `summary.json` 的 noise / first_div / p99 / top1% 数值。

本机实测：`SELFTEST OK`（26 项全过，其中 2 项 launcher 用例在非 Linux 主机上打印 `[skip] bash not usable on this host`）。

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
- `compare_cp_rounds.py baseline=... 2call=...` → 各轮指标并排 + 结论：最后一轮的 `C-B` 是否全部回到噪声级。

dump 由 `vllm_ascend/attention/sfa_v1.py` 写，开关是 `VLLM_ASCEND_CP_BALANCE_DUMP`（语法 `kind:layers[,...]`，如 `topk:6,kv:0,6`、`topk:all`；空值=关），落在 `/dev/shm/cp_balance_dump`，文件名 `<kind>_cpbal<N>_layer<L>_rank<R>_pid<P>_<ts>.pt`——自带 cp_balance 标记，所以**一轮 A/B 的 B 与 C 数据可以同时收**，且多轮共用一个目录不会互相覆盖（判读时按 `(layer, rank, cpbal)` 取最新一份）。

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

## 四、诊断矩阵（`run_cp_diag.sh` 的四个模式）

> 编号沿用 `run_cp_diag.sh` 注释与 `vllm_ascend/envs.py` 的引用（原计划文档 `CP_BALANCE_精度问题_下一步行动计划.md` 未随代码入库，这里按代码现状整理）。

| 模式 | 轮次目录 | 做的事 | 回答的问题 |
| --- | --- | --- | --- |
| `baseline` | `r1_baseline` | B/C/B2 三个配置 + `topk:6,kv:0,6` dump（T0+T2+T3） | 现状是否真的有差异、差异多大、落在哪个分片 |
| `2call` | `r2_2call` | `VLLM_ASCEND_CP_BALANCE_MERGED_CALL=0`，把合并的 2B 调用退回 prev/next 两次调用 | T1：差异是否来自“合并 2B 调用” |
| `l1024` | `r3_l1024` | prompt 长度与 `MIN_TOKENS` 都降到 1024（激活阈值边界） | T4：阈值边界上 `MIN_TOKENS` 是否被正确遵守 |
| `check` | — | 直接 `exec` `check_zigzag_dumps.py --dir <dump> --kind both` | T2/T3 的判读（不需要 NPU） |

默认参数：`PROMPT_LENS=2048,2049,4096`、`MIN_TOKENS=2048`、`OUT_ROOT=/dev/shm/cp_ab`、`BASE_PORT=8034`、`DUMP_SPEC=topk:6,kv:0,6`、`DUMP_DIR=/dev/shm/cp_balance_dump`，`LAUNCHER` 默认指向 `launcher_glm52_w4a4c8_mxfp4.sh {port}`。

## 五、怎么跑

前置：在 vllm-ascend 仓库根目录执行，宿主机已 source 站点环境（CANN / vendor `set_env.bash`），`MODEL_PATH` 指向 GLM-5.2 权重。

```bash
# 0) 一键自检（不需要 NPU；跑真实验之前先跑这个，跑完看 [verdict]）
python tools/cp_balance_compare/selfcheck.py
#    跳过最慢的 mock 自测 / 指定站点 launcher：
#    python tools/cp_balance_compare/selfcheck.py --skip-selftest \
#        --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}"

# 0b) 无 NPU 自测（改 driver 后先跑这个）
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

# 3) T1：回退成 prev/next 两次调用
bash tools/cp_balance_compare/run_cp_diag.sh 2call

# 4) CPU 侧判读 dump + 并排对比两轮
bash tools/cp_balance_compare/run_cp_diag.sh check
python tools/cp_balance_compare/check_zigzag_dumps.py --dir /dev/shm/cp_balance_dump --kind both
python tools/cp_balance_compare/compare_cp_rounds.py \
    baseline=/dev/shm/cp_ab/r1_baseline 2call=/dev/shm/cp_ab/r2_2call

# 5) 收集整轮证据（HEAD / dump 清单 / summary 指标 / 日志关键行）到一个文件里
python tools/cp_balance_compare/selfcheck.py --collect
```

常用覆盖（`run_cp_diag.sh` 的环境变量）：`PROMPT_LENS`、`MIN_TOKENS`、`OUT_ROOT`、`BASE_PORT`、`DUMP_SPEC`（置空=不 dump）、`DUMP_DIR`、`LAUNCHER`。

已经有 server 在跑时，可以跳过拉起，直接驱动两个/三个地址（注意此模式无法读 server 日志，指纹与 zigzag 检查会被跳过）：

```bash
python tools/cp_balance_compare/ab_cp_compare.py --out /dev/shm/cp_ab \
    --urls B=http://n1:8034,C=http://n2:8034,B2=http://n3:8034 --repeat-a
```

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
```

## 七、注意事项

- **顺序执行是设计前提**：TP=N 的 server 独占整机，所以 driver 一个一个拉起/停掉；多机场景请自己起 server 后用 `--urls`。
- **C 没进 zigzag 就没有结论**：`VLLM_ASCEND_CP_BALANCE_MIN_TOKENS` 被 driver 显式钉住并写入指纹（源码默认是 8192），否则盒子默认值会让 C 悄悄留在连续切片路径上，看起来像“cp_balance 对精度无影响”。`run_cp_diag.sh` 用 `--zigzag-check strict --on-zigzag-miss skip` 兜底。
- **阈值不是绝对精度判据**：`0.05` 只是起步线，真正的判据是它和 5×噪声地板取大者；没有 B2 就没有噪声地板，`--repeat-a` 应默认带上。
- **dump 目录会跨轮累积**：判读只取每个 `(layer, rank, cpbal)` 的最新一份，旧文件被忽略（会打印提示），不要求手动清理。
- **`--kind topk` 对短 prompt 才最有信息量**：长 prompt 下 indexer 不做 `validS2Len < topkCount_` 快捷路径，identity 断言不再适用。
- driver 需要 `requests` + `numpy`；`check_zigzag_dumps.py` 需要 `torch`（跑在 vLLM 宿主机上）；画图需要 `matplotlib`（缺了只警告跳过）。
- 非 Linux 主机上，`selftest_mock.py` 里依赖 bash 的 launcher 用例会被跳过（`[skip] bash not usable`）或失败（Windows 上给假 `vllm` 置可执行位无效，`test_shipped_launchers_print_complete_fingerprint` 报 `vllm not found in PATH`），都属宿主差异，不是 driver 的问题。

## 八、工具与脚本的四条硬要求

> 这四条是对本目录所有工具（含后续新增的诊断脚本）的固定要求，新增或修改代码时都必须满足。

1. **修改代码直接 commit。** 不留未提交的改动：无论改的是 `vllm_ascend/` 里的实现还是本目录的诊断工具，改完就在当前仓库 commit，提交信息写清动机、改动点和验证方式。诊断轮次靠 `git rev-parse HEAD` 对齐版本（`selfcheck.py` 会打印 HEAD 与工作区改动），未提交的改动会让“这一轮结果对应哪份代码”无法追溯。

2. **执行脚本尽量做到一行执行。** 每一个测试/诊断动作都要有一条可直接复制粘贴、不需要手工拼参数的命令；默认值覆盖常见场景，环境差异用参数或环境变量覆盖。步骤多于一步的，用一个脚本把它们串成一条命令（例如 `python tools/cp_balance_compare/selfcheck.py` 一次完成现场体检 + driver 自测 + 配置门 + 轮次预演），不要写成需要人工分多步交互的形式。

3. **必须说明需要得到的结果。** 每个脚本、每一轮测试都要明确给出“期望看到什么”：判据是哪个数字/哪个阈值/哪一行标记，`PASS`/`WARN`/`FAIL`/`[skip]` 各代表什么，以及不满足时下一步去查什么。只打印数据、不给判据的脚本视为未完成——没有判据就无法下结论。范例见 `selfcheck.py`：每个检查项都会打印一行 `期望结果:`。

4. **一次只给一步。** 每次只说明当前要执行的一步：一条命令 + 这一步需要得到的结果；上一步的结果确认后再给下一步，不预先罗列后续步骤。每一步的产出决定下一步走向，一次给多步会诱导跳过判据、在错误的假设上继续跑。

