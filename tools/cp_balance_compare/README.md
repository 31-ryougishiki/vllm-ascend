# cp_balance A/B/C 精度对比工具

用于定位 **DSA-CP cp_balance（zigzag）** 开启后 prefill 精度异常的对比工具。它把
"精度不对"拆成可量化的问题：**第几个 token 开始不一致、差多少、落在哪个 rank 的哪个块**。

配合 `vllm-ascend` 上的 `enable_dsa_cp` / `VLLM_ASCEND_CP_BALANCE` 两个开关，跑三组配置：

| 配置 | 含义 | 作用 |
| --- | --- | --- |
| `A` | `enable_dsa_cp=false` | 非 CP prefill 基线（金标准） |
| `B` | `enable_dsa_cp=true` + `VLLM_ASCEND_CP_BALANCE=0` | 连续切块 DSA-CP，隔离"CP 本身"的误差 |
| `C` | `enable_dsa_cp=true` + `VLLM_ASCEND_CP_BALANCE=1` | zigzag cp_balance，待排查对象 |
| `A2` | 再跑一遍 A（`--repeat-a`） | 测 bf16/fp8 prefill 的噪声底 |

判断标准：**`C-B` 才是 cp_balance 的账**。如果 `C-B` 和 `A-A2` 同量级，说明之前看到的
差异只是归约顺序噪声；如果 `C-B` 出现结构性大偏差、且 `first_div` 落在某个 rank 的
`prev/next` 块上，才说明布局/边界有 bug。

## 1. 目录内容

| 文件 | 作用 |
| --- | --- |
| `ab_cp_compare.py` | 主 driver：拉起/连接 server、采集逐位置 logprob、算 diff、出图 |
| `launcher_glm52_w4a4c8_mxfp4.sh` | 本机（`/home/z30055003`、eth2、GLM-5.2-w4a4c8-mxfp4）可直接用的 launcher |
| `launcher_template.sh` | 通用参考 launcher：演示 driver 要求的 env 覆盖 + 配置指纹约定 |
| `mock_vllm_server.py` | 模拟 vLLM `/v1/completions` 的 mock server（无 NPU 自测/联调用） |
| `selftest_mock.py` | 用 mock server 端到端自测 driver（`pytest` 或直接运行） |

依赖：`requests`、`numpy`；画图需要 `matplotlib`（`--no-plot` 可不装）。

## 2. 快速开始

### 2.1 单节点 8 卡（GLM-5.2-w4a4c8-mxfp4）

TP=8 的 server 会占满 8 张卡，三个配置无法同时在线；driver 会在**同一条命令里顺序**
完成"拉起 → `/health` 就绪 → 打请求 → 杀进程 → 下一个 → 对比"。

本仓库自带按这台机器定制的 launcher（eth2 / `141.61.133.104` /
`/home/z30055003/vllm-ascend` / `/mnt/share/weights/...`，均可被环境变量覆盖）：

```bash
cd /home/z30055003/vllm-ascend

# 0) 先在节点上自检 driver（不需要 NPU）
python tools/cp_balance_compare/selftest_mock.py

# 0.5) preflight：用 DRY_RUN 校验 launcher/env/指纹/vllm/模型路径，不加载模型
python tools/cp_balance_compare/ab_cp_compare.py \
  --preflight \
  --repo-root /home/z30055003/vllm-ascend \
  --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh x {port}" \
  --no-kv-connector
# 期望输出：[preflight] A: OK ... [preflight] all configs OK

# 1) A/B/C + A/A 噪声底。默认关 MTP、关 PD connector，先隔离变量
python tools/cp_balance_compare/ab_cp_compare.py \
  --out /dev/shm/cp_ab \
  --repo-root /home/z30055003/vllm-ascend \
  --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh x {port}" \
  --config-check strict \
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
  --repo-root /home/z30055003/vllm-ascend \
  --launcher "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh x {port}" \
  --config-check strict \
  --prompt-lens 2048,2049 \
  --spec-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager":true}' \
  --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1","kv_role": "kv_producer","kv_port": "30000","engine_id": "0","kv_connector_extra_config": {"use_ascend_direct": true,"prefill": {"dp_size": 1,"tp_size": 8},"decode": {"dp_size": 32,"tp_size": 1},"ascend_local_comm_res_path": "/etc/hixlep"}}'
```

一轮 A/B/C 的耗时 ≈ 4 次模型加载 + 采集；`--max-num-batched-tokens 16384` 下
`--prompt-lens 2048,2049,4096`、`--multi-lens "2048,2048;3000,1500"` 都能一次装下。
先用 `--prompt-lens 2048,2049 --repeat-a` 快速跑一轮定位，再扩长度。

单节点顺序拉起时，driver 杀掉上一个 server 后会等待 `--restart-wait`（默认 30s）再起下一个。
如果看到某个配置 `server exited`（通常是 HCCL/HBM 还没释放），把等待时间加大即可：

```bash
python tools/cp_balance_compare/ab_cp_compare.py ... --restart-wait 90
```

### 2.2 复用自己的启动脚本

如果不想用仓库里的 launcher，只要把 `/home/z30055003/script/start_server_prefill-w4a4c8-mxfp4.sh`
改成满足两个约定：

1. **all cp_balance knobs 用 `${VAR-default}`（不要硬编码）**：
   ```bash
   export VLLM_ASCEND_CP_BALANCE=${VLLM_ASCEND_CP_BALANCE:-1}
   export VLLM_ASCEND_CP_BALANCE_MIN_TOKENS=${VLLM_ASCEND_CP_BALANCE_MIN_TOKENS:-2048}
   export VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL=${VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL:-0}

   additional_config="${VLLM_ASCEND_ADDITIONAL_CONFIG:-'<原来的 additional_config JSON>'}"
   spec_config="${VLLM_ASCEND_SPEC_CONFIG-'<原来的 MTP JSON>'}"
   kv_config="${VLLM_ASCEND_KV_TRANSFER_CONFIG-'<原来的 Mooncake JSON>'}"
   ```
2. **启动前打印指纹**：
   ```bash
   dsa_cp=1
   if printf '%s' "$additional_config" | grep -q '"enable_dsa_cp"[[:space:]]*:[[:space:]]*false'; then
     dsa_cp=0
   fi
   echo "[cp-ab] CP_BALANCE=${VLLM_ASCEND_CP_BALANCE} DSA_CP=${dsa_cp} \
   EMBED_LOCAL=${VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL} \
   SPEC=$([ -n "$spec_config" ] && echo 1 || echo 0) KV=$([ -n "$kv_config" ] && echo 1 || echo 0)"
   ```
   并把 `--speculative-config` / `--kv-transfer-config` 改成条件参数
   （为空就不传），否则 driver 默认的"MTP off / connector off"无法生效。

然后：

```bash
python tools/cp_balance_compare/ab_cp_compare.py \
  --repo-root /home/z30055003 \
  --launcher "bash /home/z30055003/script/start_server_prefill-w4a4c8-mxfp4.sh x {port}" \
  --config-check strict \
  --out /dev/shm/cp_ab
```

`--config-check strict` 会比对 `logs/server_<name>.log` 里的 `[cp-ab]` 指纹；如果 launcher
没真正应用 env 覆盖（A/B/C 实际是同一个配置），driver 会直接报错退出，而不是给出
"精度完全一致"的假结论。

### 2.3 多节点：三台各起一个 server

```bash
# node1/2/3 上各自按 2.2 的方式起 server，端口一致
python tools/cp_balance_compare/ab_cp_compare.py \
  --urls A=http://node1:12800,B=http://node2:12800,C=http://node3:12800 \
  --out /dev/shm/cp_ab
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

## 3. 采集与指标

每个位置请求 `/v1/completions`（`echo=true`、`logprobs=K`）：

- `token_logprobs[i]`：第 i 个真实 prompt token 的 logprob。A/B/C 的目标 token 相同，
  差值 `d[i] = lp_X[i] - lp_Y[i]` 就是纯数值信号；
- `top_logprobs[i]`：top-K 分布，用于 top-1 一致率、top-5 软重叠。

输出目录：

```text
/dev/shm/cp_ab/
├── results_A.json / results_B.json / results_C.json   # 原始逐位置数据
├── results_all.json
├── summary.json                                       # 每个 case/请求的全部指标 + 噪声底
├── single_L2048_req0.png                              # 3 panel：曲线 / |Δ| / 滑窗 top-1 一致率
└── single_L2048_req0.csv
```

`summary.json` 中每个请求的关键字段：

| 字段 | 含义 | 怎么用 |
| --- | --- | --- |
| `top1%B~A`、`top1%C~B` | top-1 token 一致率（%） | 结构性错误的直接指标，`C~B` 掉到 99% 以下要警惕 |
| `top5_ovl%*` | top-5 互相包含率 | 排除近似并列 token 造成的假不一致 |
| `p99|d|C-B`、`max|d|C-B` | 逐位置 logprob 差 | 与 `A2-A` 噪声底比较，超过 5×p99 才当 bug |
| `first_div_C-B` | 连续 8 个位置超过阈值的第一处 | 与 `block@first_div_C-B` 一起定位 |
| `block@first_div_C-B` | 该位置属于 `rankX/prev|next(bstart-bend)` | 直接指向某个 rank 的某段，例如 `rank0/next(b1024-1280)` |
| `gen_top1%C~B` | 首生成 token 的 top-1 是否一致 | 判断问题在 prefill 主体还是模型出口/gather |

终端会直接打印一行汇总：

```text
[single_L4096.0] len=4096 top1% B~A=100.00 C~B=99.98 C~A=99.98 |
  p99|d| B-A=8.1e-04 C-B=1.2e-03 C-A=1.7e-03 |
  first_div C-B=3073 @ rank2/prev(b3072-3328) gen_top1% C~B=1
```

## 4. 无 NPU 自测

```bash
python tools/cp_balance_compare/selftest_mock.py
# 或
pytest tools/cp_balance_compare/selftest_mock.py -q
```

自测覆盖：token-id / 文本 prompt、`prompt_logprobs` 兜底解析、配置指纹校验（OK /
mismatch / 缺失）、JSONL 解析、以及 mock server 下的完整 A/B/C/A2 端到端流程。

也可以手动起 mock server 调 driver：

```bash
python tools/cp_balance_compare/mock_vllm_server.py --ports 18001,18002,18003 \
  --offsets 0,1e-6,0.2
python tools/cp_balance_compare/ab_cp_compare.py \
  --urls A=http://127.0.0.1:18001,B=http://127.0.0.1:18002,C=http://127.0.0.1:18003 \
  --prompt-lens 1024 --out /tmp/cp_ab
```

## 5. 已知限制

- **单机无法并行**：TP=N 的 server 独占 N 卡，driver 只能顺序拉起，一轮耗时 =
  3~4 次模型加载 + 采集。先用小模型 / 3 层脚本 + 短 prompt 迭代；
- **A 组不是"同一 kernel 换布局"**：A 关了 DSA-CP，`B-A` 只能当粗基线；判断
  cp_balance 必须看 `C-B`；
- **top-1 在近似并列时会翻转**：同时看 `top5_ovl%`；
- **超长 prompt 响应很大**（100k × 20 条 logprob）：建议对比时控制在 8k 以内；
- **`C-B` 与 `A2-A` 同量级时不要下结论**：那只是不同的 reduce / 量化顺序；
- driver 只做 prefill 对比，不覆盖 decode / PD 传输。

## 6. 相关代码

- `vllm_ascend/layers/cp_zigzag.py`：zigzag plan / shard / gather
- `vllm_ascend/attention/sfa_v1.py`：`DSACPContext`、merged metadata、KV/indexer 写回
- `vllm_ascend/patch/worker/patch_deepseek_v2.py`：模型入口 shard / 出口 gather
- `vllm_ascend/ops/vocab_parallel_embedding.py`：embedding 入口
