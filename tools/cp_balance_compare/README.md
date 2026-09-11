# cp_balance 开关对比工具（B/C，104 单节点）

用于定位 **DSA-CP cp_balance（zigzag）** 开启后 prefill 精度异常的对比工具。它把
"精度不对"拆成可量化的问题：**第几个 token 开始不一致、差多少、落在哪个 rank 的哪个块**。

只对比 **cp_balance 开/关**两种配置，DSA-CP 始终打开（不再跑关 DSA-CP 的"金标准"，
那条路径不同、只能当粗参照）：配合 `vllm-ascend` 上的 `VLLM_ASCEND_CP_BALANCE` 开关跑两组：

| 配置 | 含义 | 角色 |
| --- | --- | --- |
| `B` | `enable_dsa_cp=true` + `VLLM_ASCEND_CP_BALANCE=0` | **基线**：同 DSA-CP、同 kernel、连续切块 |
| `C` | `enable_dsa_cp=true` + `VLLM_ASCEND_CP_BALANCE=1` | zigzag cp_balance，待排查对象 |
| `B2` | 再独立起一次 B（`--repeat-a`） | **噪声底**（同配置第二次加载，与 `C-B` 最有可比性） |

判断标准：

- **`C-B` 就是 cp_balance 的账**（同 DSA-CP、同 SP、同 C8，只差 token 布局）。如果
  `C-B` 和 `B2-B` 同量级，说明看到的差异只是归约顺序噪声；
- 默认 `--configs B,C`，`--repeat-a` 再补一次 B 得到 `B2-B` 噪声底；`A` 已不支持。

## 1. 目录内容

| 文件 | 作用 |
| --- | --- |
| `ab_cp_compare.py` | 主 driver：拉起/连接 server、采集逐位置 logprob、算 diff、出图 |
| `launcher_glm52_w4a4c8_mxfp4.sh` | 本机（`/opt/its/z30055003`、eth2、GLM-5.2-W4A8C8）可直接用的 launcher |
| `launcher_template.sh` | 通用参考 launcher：演示 driver 要求的 env 覆盖 + 配置指纹约定 |
| `mock_vllm_server.py` | 模拟 vLLM `/v1/completions` 的 mock server（无 NPU 自测/联调用） |
| `selftest_mock.py` | 用 mock server 端到端自测 driver（`pytest` 或直接运行） |

依赖：`requests`、`numpy`；画图需要 `matplotlib`（`--no-plot` 可不装）。

## 2. 快速开始

### 2.1 104 单节点（GLM-5.2-w4a4c8-mxfp4，默认端口 8034）

TP=8 的 server 会占满 8 张卡，两组配置无法同时在线；driver 会在**同一条命令里顺序**
完成"拉起 → `/health` 就绪 → 打请求 → 杀进程 → 下一个 → 对比"。默认端口 **8034**
（`--base-port` 可改），拉起的 server 也从同一个端口起。

本仓库自带按这台机器定制的 launcher（eth2 / `7.246.78.76` /
`/opt/its/z30055003/vllm-ascend` / `/opt/its/model/...`，均可被环境变量覆盖）：

```bash
cd /opt/its/z30055003/vllm-ascend

# 0) 先在节点上自检 driver（不需要 NPU）
python tools/cp_balance_compare/selftest_mock.py

# 0.5) preflight：用 DRY_RUN 校验 launcher/env/指纹/vllm/模型路径，不加载模型
python tools/cp_balance_compare/ab_cp_compare.py \
  --preflight \
  --repo-root /opt/its/z30055003/vllm-ascend \
  --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh x {port}" \
  --no-kv-connector
# 期望输出：[preflight] B: OK ... [preflight] C: OK ... [preflight] all configs OK

# 1) 基线(B, CP_BALANCE=0) vs zigzag(C, CP_BALANCE=1) + B/B2 噪声底。
#    默认关 MTP、关 PD connector，先隔离变量；默认端口 8034。
#    --zigzag-check strict：C 没进 zigzag 直接判失败（默认跳过剩下的配置，避免白加载）
python tools/cp_balance_compare/ab_cp_compare.py \
  --out /dev/shm/cp_ab \
  --repo-root /opt/its/z30055003/vllm-ascend \
  --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh x {port}" \
  --config-check strict \
  --zigzag-check strict \
  --prompt-lens 2048,2049,4096 \
  --multi-lens "2048,2048;3000,1500" \
  --repeat-a \
  --no-kv-connector

# 2) 结果、曲线、CSV
ls /dev/shm/cp_ab
# summary.json / results_*.json / single_L2048_req0.png / single_L2048_req0.csv ...

# 3) 复现线上 MTP + Mooncake 组合（可选，第二轮再做）
python tools/cp_balance_compare/ab_cp_compare.py \
  --out /dev/shm/cp_ab_mtp \
  --repo-root /opt/its/z30055003/vllm-ascend \
  --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh x {port}" \
  --config-check strict \
  --prompt-lens 2048,2049 \
  --spec-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager":true}' \
  --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1","kv_role": "kv_producer","kv_port": "30000","engine_id": "0","kv_connector_extra_config": {"use_ascend_direct": true,"prefill": {"dp_size": 1,"tp_size": 8},"decode": {"dp_size": 32,"tp_size": 1},"ascend_local_comm_res_path": "/etc/hixlep"}}'
```

两组配置 = **2 次模型加载** + 采集（`--repeat-a` 再加 1 次）；`--max-num-batched-tokens 16384`
下 `--prompt-lens 2048,2049,4096`、`--multi-lens "2048,2048;3000,1500"` 都能一次装下。
先用 `--prompt-lens 2048,2049 --repeat-a` 快速跑一轮定位，再扩长度。

单节点顺序拉起时，driver 杀掉上一个 server 后会等待 `--restart-wait`（默认 30s）再起下一个。
如果看到某个配置 `server exited`（通常是 HCCL/HBM 还没释放），把等待时间加大即可：

```bash
python tools/cp_balance_compare/ab_cp_compare.py ... --restart-wait 90
```

**注意**：`VLLM_ASCEND_CP_BALANCE_MIN_TOKENS`（prompt 低于它就完全不进 zigzag）由 driver
按 `--cp-balance-min-tokens`（默认 2048）统一注入并写进指纹校验，launcher 里的默认值只对
手工启动生效——所以不要把对比用的 prompt 长度设得比它更小，否则 C 会一直走连续切块路径、
看起来"和 B 完全一致"。如果 C 因为任何原因没进 zigzag，`[runtime] C: ... forward_zigzag=False`
（或 `--zigzag-check strict` 直接失败）就是判据。

### 2.2 复用自己的启动脚本

如果不想用仓库里的 launcher，只要把 `/opt/its/z30055003/script/start_server_prefill-w4a4c8-mxfp4.sh`
改成满足两个约定：

1. **所有 cp_balance 开关用 `${VAR-default}`（不要硬编码）**：
   ```bash
   export VLLM_ASCEND_CP_BALANCE=${VLLM_ASCEND_CP_BALANCE:-1}
   export VLLM_ASCEND_CP_BALANCE_MIN_TOKENS=${VLLM_ASCEND_CP_BALANCE_MIN_TOKENS:-2048}
   export VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL=${VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL:-0}

   additional_config="${VLLM_ASCEND_ADDITIONAL_CONFIG:-'<原来的 additional_config JSON>'}"
   spec_config="${VLLM_ASCEND_SPEC_CONFIG-'<原来的 MTP JSON>'}"
   kv_config="${VLLM_ASCEND_KV_TRANSFER_CONFIG-'<原来的 Mooncake JSON>'}"
   ```
2. **启动前打印指纹**（driver 逐字段比对，包括 `MIN_TOKENS`）：
   ```bash
   dsa_cp=1
   if printf '%s' "$additional_config" | grep -q '"enable_dsa_cp"[[:space:]]*:[[:space:]]*false'; then
     dsa_cp=0
   fi
   echo "[cp-ab] CP_BALANCE=${VLLM_ASCEND_CP_BALANCE} DSA_CP=${dsa_cp} \
   MIN_TOKENS=${VLLM_ASCEND_CP_BALANCE_MIN_TOKENS} \
   EMBED_LOCAL=${VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL} \
   SPEC=$([ -n "$spec_config" ] && echo 1 || echo 0) KV=$([ -n "$kv_config" ] && echo 1 || echo 0)"
   ```
   并把 `--speculative-config` / `--kv-transfer-config` 改成条件参数
   （为空就不传），否则 driver 默认的"MTP off / connector off"无法生效。

然后：

```bash
python tools/cp_balance_compare/ab_cp_compare.py \
  --repo-root /opt/its/z30055003/vllm-ascend \
  --launcher "bash /opt/its/z30055003/script/start_server_prefill-w4a4c8-mxfp4.sh x {port}" \
  --config-check strict \
  --out /dev/shm/cp_ab
```

`--config-check strict` 会比对 `logs/server_<name>.log` 里的 `[cp-ab]` 指纹和 `[cp-ab-cfg]`
完整 additional_config；如果 launcher 没真正应用 env 覆盖（B/C 实际是同一个配置），
driver 会直接报错退出，而不是给出"精度完全一致"的假结论。

### 2.3 两台 server 同时在线（可选）

单节点 8 卡放不下两台，但如果你有两台 104（或一台机器上起了第二个实例），可以让它们分别
以 `CP_BALANCE=0/1` 常驻，driver 只负责采集。注意 `--repeat-a` 需要一个额外的 B 实例 URL，
否则 B2 指回同一台 server，噪声底会恒等于 0（`--urls` 模式无法重启 server，这是预期行为）：

```bash
# node1: CP_BALANCE=0 起 server；node2: CP_BALANCE=1 起 server，端口都是 8034
python tools/cp_balance_compare/ab_cp_compare.py \
  --urls B=http://node1:8034,C=http://node2:8034 \
  --out /dev/shm/cp_ab
# 需要噪声底时：--urls B=...,C=...,B2=http://node1b:8034 --repeat-a
```

### 2.4 用真实 prompt（JSONL / JSON）

```jsonl
{"case": "realchat", "prompt": "请介绍一下华为昇腾 NPU 的架构"}
{"case": "realchat", "prompt": [151644, 872, 198, 1]}
{"case": "longctx",  "prompt": "……长文本……"}
{"case": "batch",    "prompt": [[1,2,3,...], [4,5,6,...]]}
```

```bash
python tools/cp_balance_compare/ab_cp_compare.py \
  --prompts-file prompts.jsonl --out /dev/shm/cp_ab
```

- `prompt` 可以是字符串（服务端 tokenizer 处理）或 token id 数组；
- 同一个 `case` 的多条记录会被放在**一个请求**里（`prompt: [[...],[...]]`），
  保证它们进同一个 prefill batch，这条路径比"并发发多个请求"更严格；
- token 数从响应里的 `prompt_token_ids` 推导，文本 prompt 也能逐位置对齐。

## 3. 请求契约与失败语义

### 3.1 服务可用性 / 请求是否真的成功

就绪判定只有一个：`GET http://127.0.0.1:<port>/health` 返回 200（`wait_ready`）。之后每次
请求都按下面的方式保证"这一份数据可用"：

1. `POST /v1/completions` → `raise_for_status()`：非 2xx 直接判失败；
2. 响应里 `choices` 数量必须等于请求里的 prompt 数量（batch 请求会返回多个 choice）；
3. 每个 choice 必须能对齐：`prompt_token_ids` 必须**逐 token 等于我们发出去的 ids**
   （token-id prompt），`token_logprobs` 必须覆盖全部 prompt 位置且除第 0 位外都非空；
   任何一条不满足就抛错，不会"带着坏数据继续算"；
4. 单条 case 失败会重试 `--http-retries`（默认 3，退避 2s/4s/6s），仍失败则把该 case 记为
   `{"error": ...}`，并在终端打印 `case: FAILED (...)`；summary 里该 case 是
   `{"status": "missing results"}`。

**注意**：目前 driver 对"部分 case 失败"仍以退出码 0 结束（这是已知限制，见"已知限制"一节），
所以**不要只看退出码**，要看 `[compare]` 之后每个 case 是否都有指标行、以及
`summary.json` 里有没有 `status: missing results`。

### 3.2 怎么保证真的超过了 min_tokens

`prompt_token_ids` 的**长度**（不是字符数、不是我们本地数出来的词数）就是服务端真正
prefill 的 token 数：

- 合成 token-id prompt：长度由我们给定（`--prompt-lens`），服务端原样回显；
- 文本 prompt：长度是服务端 tokenizer 的结果，driver 从响应里读，不猜；
- 每次查询都会打印 `case: N prompt(s) OK, prompt_tokens=[...]`；
- 每个配置查完后打印 `[tokens] <config>: prompt_tokens per request = [...] (max=..., min_tokens=...)`；
- **低于 `--cp-balance-min-tokens`（默认 2048，driver 会注入并写进指纹）的 case 会直接告警**：
  `[warn] <case>: N prompt tokens < VLLM_ASCEND_CP_BALANCE_MIN_TOKENS=2048; zigzag stays off for
  this case, so C is a copy of B here`——这种 case 的 `C-B` 恒为 ~0，不能当成"cp_balance 没问题"。
- 另外运行时还会检查每条请求 `query_len >= 2 * tp_size`（TP=8 即 ≥16 token），
  以及可比较位置数是否 ≥ `--run-len`（否则 `first_div` 打印告警而不是"无差异"）。

### 3.3 用什么定位精度问题

逐位置 `token_logprobs` 是最干净的信号：B/C 的目标 token 相同，所以
`d[i] = lp_C[i] - lp_B[i]` 只包含数值差异，然后：

| 看什么 | 说明 |
| --- | --- |
| `max|d|` / `p99|d|` | 差多大；先和 `B2-B` 噪声底比 |
| `first_div_C-B` | 连续 `--run-len`（默认 8）个位置超过阈值（`max(--delta-threshold, 5×噪声 p99)`）的第一处 |
| `block@first_div_C-B` | 这个位置落在哪个 rank 的哪一段，例如 `rank2/prev(b3072-3328)` |
| `top1%C~B` / `top5_ovl%` | token 级是否翻转；近似并列时配合 top5 软重叠看 |
| `gen_top1%C~B` | 首生成 token 是否一致，区分 prefill 主体 vs 模型出口/gather |
| `runtime.C.forward` | 证明 C 真进了 zigzag（否则上面的对比无意义） |

### 3.4 本地手工验证单次请求

你给的 curl 只验证"服务通不通"，不能用于精度定位（没有 `echo`/`logprobs`）。
要拿逐位置数据，用 driver 完全相同的形式：

```bash
curl -sS http://127.0.0.1:8034/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "glm",
        "prompt": [151644, 872, 198],
        "max_tokens": 1,
        "temperature": 0,
        "echo": true,
        "logprobs": 20,
        "prompt_logprobs": 20,
        "add_special_tokens": false
      }' | python -m json.tool | head -40
```

要点：`echo=true` 才会回显 `prompt_token_ids`；`logprobs`/`prompt_logprobs` 才有逐位置
logprob；`add_special_tokens=false` 保证"回显的 token 就是我们发的 token"；
`max_tokens=1` 只为拿首生成 token 的 top-1。文本 prompt（如 `"你是什么模型"`）也能用，
但"一个汉字是一个 token 还是多个"由服务端 tokenizer 决定，**以响应里的
`prompt_token_ids` 长度为准**——先跑一次这条 curl 看清长度，再决定
`--prompt-lens` / `--cp-balance-min-tokens`。

## 4. 采集与指标

每个位置请求 `/v1/completions`（`echo=true`、`logprobs=K`）：

- `token_logprobs[i]`：第 i 个真实 prompt token 的 logprob。B/C 的目标 token 相同，
  差值 `d[i] = lp_X[i] - lp_Y[i]` 就是纯数值信号；
- `top_logprobs[i]`：top-K 分布，用于 top-1 一致率、top-5 软重叠。

输出目录：

```text
/dev/shm/cp_ab/
├── results_B.json / results_C.json                    # 原始逐位置数据
├── results_all.json
├── summary.json                                       # 每个 case/请求的全部指标 + 噪声底
├── single_L2048_req0.png                              # 3 panel：曲线 / |Δ| / 滑窗 top-1 一致率
└── single_L2048_req0.csv
```

`summary.json` 中每个请求的关键字段：

| 字段 | 含义 | 怎么用 |
| --- | --- | --- |
| `top1%C~B` | top-1 token 一致率（%） | 结构性错误的直接指标，掉到 99% 以下要警惕 |
| `top5_ovl%*` | top-5 互相包含率 | 排除近似并列 token 造成的假不一致 |
| `p99|d|C-B`、`max|d|C-B` | 逐位置 logprob 差 | 与 `B2-B` 噪声底比较，超过 5×p99 才当 bug |
| `first_div_C-B` | 连续 8 个位置超过阈值的第一处 | 与 `block@first_div_C-B` 一起定位 |
| `block@first_div_C-B` | 该位置属于 `rankX/prev|next(bstart-bend)` | 直接指向某个 rank 的某段，例如 `rank2/prev(b3072-3328)`（仅 `single_*` case） |
| `gen_top1%C~B` | 首生成 token 的 top-1 是否一致（0/1） | 判断问题在 prefill 主体还是模型出口/gather |

`summary.json` 顶层的 `runtime` 记录每个配置的运行时证据：

```json
"runtime": {
  "B": {"metadata": false, "forward": false},
  "C": {"metadata": true, "forward": true}
}
```

- `metadata=true`：`AscendSFAMetadataBuilder` 真的构建了 zigzag metadata；
- `forward=true`：`set_ascend_forward_context` 真的把 `zigzag_cp_active` 置为 1；
- 两个标记由 `VLLM_ASCEND_CP_BALANCE_DEBUG_LOG=1`（driver 自动为 B/C 导出）
  触发的 `[CP_BALANCE]` 日志产生，默认关闭，对正常推理没有影响；
- 注意这是"进程内至少有一个 batch 进了 zigzag"的证据（日志用 `logger.info_once`），
  不会逐个请求重复打印。

`--config-check strict` 还会校验 launcher 打印的 `[cp-ab-cfg]`（逐字段对比
additional_config），避免"改了 launcher 但 driver 用环境变量覆盖"这类静默不一致；
指纹里的 `MIN_TOKENS` 保证 B/C 的 zigzag 激活门槛一致。

配合 `--zigzag-check strict` 使用：如果 C 没有打出 forward 标记，driver 默认
（`--on-zigzag-miss skip`）**不再拉起剩余配置**（例如 B2，省一次模型加载），
但会把已经采到的 B/C 数据照常写出 `summary.json`/图，`summary.json.skipped`
记录被跳过的配置，`summary.json.runtime.C.forward=false`，最后以非 0 退出。
传 `--on-zigzag-miss continue` 可强制跑完全部配置。

这样既避免"C 其实静默回退到连续路径、却因为 C-B 一致被误判成修复成功"，
也不浪费一次加载、不丢已采数据。

终端会直接打印一行汇总：

```text
[runtime] C: metadata_zigzag=True forward_zigzag=True
[single_L4096.0] len=4096 top1% C~B=99.98 |
  p99|d| C-B=1.2e-03 first_div C-B=3073 @ rank2/prev(b3072-3328) gen_top1% C~B=1
```

## 5. 无 NPU 自测

```bash
python tools/cp_balance_compare/selftest_mock.py
# 或
pytest tools/cp_balance_compare/selftest_mock.py -q
```

自测覆盖：token-id / 文本 prompt、`prompt_logprobs` 兜底解析、配置指纹校验（OK /
mismatch / 缺失）、zigzag 激活日志解析、`--preflight` 逻辑与真实 bash launcher、
JSONL 解析、A 组被拒、`--configs` 行为、**请求契约校验（回显 token 被改写 / logprob 缺失 /
条数不足都要报错）**、**min_tokens 告警**、两个真实 launcher 的指纹行完整性，
以及 mock server 下的完整 B/C/B2 端到端流程。

也可以手动起 mock server 调 driver（两个端口分别扮演 `CP_BALANCE=0/1`）：

```bash
python tools/cp_balance_compare/mock_vllm_server.py --ports 18034,18035 \
  --offsets 1e-6,0.2
python tools/cp_balance_compare/ab_cp_compare.py \
  --urls B=http://127.0.0.1:18034,C=http://127.0.0.1:18035 \
  --prompt-lens 1024 --cp-balance-min-tokens 512 --out /tmp/cp_ab
```

（mock 只有几十个 token，`--cp-balance-min-tokens` 要调小，否则会看到 3.2 的告警。）

## 6. 已知限制

- **104 单节点无法并行**：TP=8 的 server 独占 8 卡，driver 只能顺序拉起，一轮耗时 =
  2~3 次模型加载 + 采集。先用 3 层脚本 + 短 prompt 迭代；
- **部分 case 失败时退出码仍是 0**：只有 zigzag 未命中 / 启动失败才返回非 0；要判断是否
  真的完整采集，看终端 `case: FAILED (...)` 和 `summary.json` 的 `status: missing results`
  （见 3.1）；
- **不看 DSA-CP 自身误差**：工具只对比 cp_balance 开/关，DSA-CP 始终打开，所以
  `C-B` 里也包含 "DSA-CP 本身在两个布局下的固有差异"，无法再拆一层（要看那一层
  得手工起一组 `enable_dsa_cp=false` 的 server 用 `--urls` 比）；
- **`multi_*` case 的块定位只是近似**：运行时按"批量内每个请求各自的 16 块"排布，
  driver 的 `block@first_div` 用单请求 plan 重建绝对位置，所以只在 `single_*` case 上有意义；
- **`--urls` 模式下噪声底可能是 0**：B2 指回同一台 server 时 `B2-B ≡ 0`，不代表真实
  跨加载噪声；
- **top-1 在近似并列时会翻转**：同时看 `top5_ovl%`；
- **超长 prompt 响应很大**（100k × 20 条 logprob）：建议对比时控制在 8k 以内；
- **`C-B` 与 `B2-B` 同量级时不要下结论**：先看 `summary.json` 的
  `runtime.C.forward`；只有它 `true` 才说明 C 真的走了 zigzag（可加
  `--zigzag-check strict` 强制校验）；
- driver 只做 prefill 对比，不覆盖 decode / PD 传输。

## 7. 常见报错

### 7.1 `recompute_scheduler_enable can only be enabled on PD-disaggregated D nodes`

- **原因**：`vllm_ascend/platform.py` 规定 `recompute_scheduler_enable=true` 只允许
  `kv_role='kv_consumer'`（PD 的 D 节点）。B/C 默认 `--no-kv-connector`，
  `kv_transfer_config=None`，于是启动时直接抛
  `ValueError: ... got kv_role=None`。
- **为什么改 launcher 没用**：driver 每次都会导出
  `VLLM_ASCEND_ADDITIONAL_CONFIG` / `VLLM_ASCEND_SPEC_CONFIG` /
  `VLLM_ASCEND_KV_TRANSFER_CONFIG`，launcher 里的默认值只在环境变量"未设置"时生效；
  改 launcher 的 `DEFAULT_ADDITIONAL_CONFIG` 不会影响 driver 拉起的进程。
- **正确做法**：
  - 默认：driver 的 `BASE_ADDITIONAL_CONFIG` 已移除 `recompute_scheduler_enable`，
    直接重跑即可；
  - 临时加回 / 覆盖其它键：`--extra-additional-config '{"recompute_scheduler_enable": true}'`
    （merge 进基础配置，B/C 两组一致）；
  - 整体替换成你们的真实 additional_config：
    `--env VLLM_ASCEND_ADDITIONAL_CONFIG='{...}'`（driver 的 `--env` 优先级最高）。
- **怎么确认 launcher 最终吃到的配置**：`--config-check strict` 除了 6 个开关指纹
  （`CP_BALANCE` / `DSA_CP` / `MIN_TOKENS` / `EMBED_LOCAL` / `SPEC` / `KV`），
  还会校验 launcher 打印的 `[cp-ab-cfg]` 行；不一致时会把 expected / launcher 两份
  JSON 都打出来。

### 7.2 `[preflight] ... FAILED` / `server exited`

- `MODEL_PATH does not exist` / `vllm not found in PATH`：按提示修 launcher 的
  `MODEL_PATH`、`source /root/.bashrc`；
- `additional_config mismatch`：driver 传入的配置和 launcher 实际使用的不同，
  按打印的两份 JSON 调整 launcher 或改用 `--extra-additional-config`；
- 某个配置 `server exited`：多半是 8 卡未释放或端口被占，`npu-smi info` +
  `ps -ef | grep "[v]llm serve"` 后重跑，或 `--restart-wait 90`。

### 7.3 拉起后长时间没输出，driver 怎么判断"可以发请求了"

就绪判定只有一个：`GET http://127.0.0.1:<port>/health` 返回 200。

```
launch  ->  [wait] B: polling http://127.0.0.1:8034/health (startup timeout=3600s)
            [wait] B: /health not ready yet (elapsed=30s/3600s, last=ConnectionError)
            ...
            [ready] B: /health -> 200 (elapsed=420s)     ← 从这里才开始
            [check] B fingerprint OK: {...}              ← 校验 launcher 实际配置
            [query] B -> http://127.0.0.1:8034           ← 真正开始发请求
```

- 轮询间隔 3s，上限 `--startup-timeout`（默认 3600s）；每 `--wait-log-every`（默认 30s）
  打一次心跳，所以"很久没输出"现在会变成一行行 `[wait]`；
- 首次请求一定在 `[ready]` + `[check] ... OK` 之后；`[query]` 那一行才是发请求；
- 如果 launcher 进程提前退出（配置错误/HBM 没释放），driver 立即报错并打印日志尾部，
  不会等满 1 小时。

长时间停在 `[wait]` 时按下面排查：

```bash
# 1) 模型是否在加载（正常会看到权重加载/compile/warmup 日志，耗时几分钟）
tail -f /dev/shm/cp_ab/logs/server_B.log
npu-smi info                     # 8 张卡是否被 8 个进程占用

# 2) 端口是否真的监听（应与 launcher 的 --port 一致，默认 8034）
ss -ltnp | grep 8034
curl --noproxy '*' -sv http://127.0.0.1:8034/health

# 3) 代理问题（driver 已用 trust_env=False 规避；如果你用旧版 driver 才有此问题）
env | grep -i proxy
```

心跳里 `last=` 的含义：`ConnectionError` = 端口还没监听（多在加载）；`ProxyError` =
请求被代理劫持（旧版 driver）；`Timeout` = 服务响应慢。

### 7.4 C 组没有进 zigzag（`forward_zigzag=False` / `--on-zigzag-miss skip`）

C 和 B 输出逐位完全一致时，第一件事是确认 C 真的走了 zigzag。运行时的门控都在
`vllm_ascend/layers/cp_zigzag.py: can_enable_zigzag_for_batch`：

- `VLLM_ASCEND_CP_BALANCE=1`（C 组由 driver 注入）；
- `num_actual_tokens >= VLLM_ASCEND_CP_BALANCE_MIN_TOKENS`：**最常见的原因**。
  driver 现在会显式注入并按 `MIN_TOKENS` 校验指纹（默认 2048），所以把
  `--prompt-lens` 设得比它更小就会命中；
- 每条请求 `query_len >= 2 * tp_size`（TP=8 时即 ≥16 token）；
- `SP-padded token 数` 是 `2 * tp_size` 的倍数（driver 用相同公式重建，一般自动满足）；
- 纯 prefill 状态、无 MTP draft、非 V2 model runner、`dp_size == 1`、
  `dcp_replicated=False`、且 `full_o_proj`（`kv_transfer_config is None` 或 kv_producer）——
  注意 `--kv-transfer-config` 给成 `kv_consumer` 会让 C 永远退回连续切块。

排查顺序：先看 `summary.json.runtime.C`，再看
`grep '\[CP_BALANCE\]' /dev/shm/cp_ab/logs/server_C.log`（driver 会自动设
`VLLM_ASCEND_CP_BALANCE_DEBUG_LOG=1`），最后回到上面这几条门控。

## 8. 相关代码

- `vllm_ascend/layers/cp_zigzag.py`：zigzag plan / shard / gather
- `vllm_ascend/attention/sfa_v1.py`：`DSACPContext`、merged metadata、KV/indexer 写回
- `vllm_ascend/patch/worker/patch_deepseek_v2.py`：模型入口 shard / 出口 gather
- `vllm_ascend/ops/vocab_parallel_embedding.py`：embedding 入口

## 9. 精度追查用的脚本（本目录）

driver 只回答"cp_balance 开关有没有影响"，定位"影响在哪一环"用下面这套。
设计背景与判据见仓库根目录 `CP_BALANCE_精度问题_下一步行动计划.md`。

| 脚本 | 用途 |
| --- | --- |
| `run_cp_diag.sh baseline` | T0+T2+T3：基线 B/C/B2 + topk/KV dump（2~3 次模型加载） |
| `run_cp_diag.sh 2call` | T1：回退 prev/next 两次调用（`VLLM_ASCEND_CP_BALANCE_MERGED_CALL=0`） |
| `run_cp_diag.sh l1024` | T4：`--prompt-lens 1024 --cp-balance-min-tokens 1024` |
| `run_cp_diag.sh check` | CPU 侧判读已有 dump（不需要 NPU） |
| `run_cp_diag.sh <mode> -n` | 只打印将要执行的命令（`--dry-run` 同效），防止误触发真实运行 |
| `check_zigzag_dumps.py` | dump 判读：`--kind topk` 判"top-k 是否等于因果集"，`--kind kv` 判"逐 token KV 是否一致" |
| `compare_cp_rounds.py` | 并排比较多轮 `summary.json`，直接输出"是否回到噪声级"的结论 |

可覆盖的变量：`LAUNCHER` / `PROMPT_LENS` / `MIN_TOKENS` / `OUT_ROOT` / `BASE_PORT` /
`DUMP_SPEC` / `DUMP_DIR`。典型用法：

```bash
bash tools/cp_balance_compare/run_cp_diag.sh baseline -n     # 先核对命令
bash tools/cp_balance_compare/run_cp_diag.sh baseline        # 跑第一轮
bash tools/cp_balance_compare/run_cp_diag.sh check           # 判 P1/P2
bash tools/cp_balance_compare/run_cp_diag.sh 2call           # 跑 T1
python tools/cp_balance_compare/compare_cp_rounds.py \
    baseline=/dev/shm/cp_ab/r1_baseline 2call=/dev/shm/cp_ab/r2_2call
```

引擎侧只有一个新开关控制 dump：`VLLM_ASCEND_CP_BALANCE_DUMP`（`kind:layers[,...]`，
如 `topk:6,kv:0,6`；空=关）。dump 落到 `/dev/shm/cp_balance_dump/`，文件名带
`cpbal{0|1}`，所以一轮 A/B 的 B/C 两份数据可以同时拿到、互不覆盖。

