# CP_BALANCE 精度问题 · 交接文档

> 用途：接手的人只看这一份就能继续干活。
> 需要深挖推理过程时再看文末"参考文档"里的三份底稿。
> 最后更新：接手人在 `/opt/its/z30055003/vllm-ascend`（branch `glm52_cp_balance_v3`）跑 `baseline -n` 之后。
>
> 位置说明：本副本随代码走（在 `tools/cp_balance_compare/` 下，与工具同级）。
> 分析工作区根目录另有一份同名副本；**以本仓库副本为准**，改动请先改这里再同步过去。

---

## 0. 一页看懂

**问题**：GLM-5.2 在 vllm-ascend 上打开 `VLLM_ASCEND_CP_BALANCE=1`（zigzag，一个让 8 卡 prefill 负载更均衡的加速开关）后，
prefill 的 logprob 与关闭时不一致；关掉就正常。

**实测数字**（单请求 L2048/L2049）：

| 指标 | 值 | 含义 |
|---|---|---|
| `p99 |Δlogprob|` | 0.56~0.58 | 偏差幅度 |
| `max |Δlogprob|` | 2.8~3.0 | |
| top-1 一致率 | 82%~83% | 约 1/6 的位置首选 token 变了 |
| top-5 重叠率 | 96%~98% | 说明不是崩坏，是"小扰动" |
| 首次持续超阈位置 | 294(L2048) / 263(L2049) | 落在 rank2 的 prev 块附近 |
| `B2−B` | 0 | 同一配置跑两遍完全一致 ⇒ 差异是结构性的，不是随机噪声 |

**已把可疑面缩到 3 个**（源码层面排除了 2 个，见 §2）：

1. 索引器交给 attention 的"该看哪些历史 token"的清单错了；
2. KV 缓存里的内容存错或读错；
3. attention 算子"一次算两段"（合并调用）有 bug。

**现在卡在**：需要一次实测同时判定这三者。**没有阻塞，可以立刻开跑。**

---

## 1. 你要做的：3 条命令

```bash
cd /opt/its/z30055003/vllm-ascend

# ① 1 分钟，不加载模型：检查环境/路径
python tools/cp_balance_compare/ab_cp_compare.py --preflight \
  --launcher 'bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}' --no-kv-connector
#   期望最后一行： [preflight] all configs OK
#   出 ERROR 就把那行原样贴出来（最常见：MODEL_PATH / VLLM_ASCEND_REPO / vendor set_env.bash 路径不对）

# ② 2~3 小时：跑第一轮（B/C/B2 三组配置 + 抓"清单"和"KV 缓存"）
tmux new -s cpab                    # 必须用 tmux/nohup，断线会留下占卡的孤儿进程
bash tools/cp_balance_compare/run_cp_diag.sh baseline 2>&1 | tee /tmp/r1_baseline.log
#   中途看 /dev/shm/cp_ab/r1_baseline/logs/server_*.log
#   看到 [runtime] C: forward_zigzag=True 才说明这轮有效

# ③ 几秒，不需要 NPU：看结论
bash tools/cp_balance_compare/run_cp_diag.sh check
```

**把 ③ 的输出贴给负责人**，然后按 §3 的表格决定下一步。

---

## 2. 已经做过的判断（别重复劳动）

| 怀疑对象 | 结论 | 依据（想深究再看） |
|---|---|---|
| 合并元数据的 seqlen 约定（query 长度、KV 长度怎么传） | **不是 bug**。query 必须前缀和、KV 必须原始值，是算子内部规定；现在的写法正好符合 | `quant_lightning_indexer_kernel.h:165-170,214-227`、`..._kvcache.h:38-70` |
| 因果掩码窗口（Δ = kv−q） | **正确**。zigzag 下 Δ 恰好等于该块第一个 token 的绝对位置 | `..._kvcache.h:74-79,217-241` |
| 索引器的**打分质量/top-k 选择**（≤2048 token 时） | **完全不可能**。此时算子直接输出"完整历史位置"的清单，不做任何打分排序 | `quant_lightning_indexer_service_vector.h:450-464,479-551` |
| "把 KV 的 slot 打乱顺序再写"这种做法 | **数学上与 SGLang 的写法等价**（只差 `-1` 跳过语义），不是错位 | `sfa_v1.py:2320-2332` vs `cp_zigzag.py:489-499` |
| 0-token 请求导致 block_table 错位 | **是真缺陷，但触发条件很窄**：只有 batch 中间存在 0-token 请求时才错；本次 case 不触发。修法是"不过滤 query_lens"，不是"过滤 block_table" | `sfa_v1.py:649-667` + `:733` |
| 依赖 `slot == -1` 跳过 padding 行 | **只影响带 padding 的请求**；L2048 没有 padding，L2049 只有 15 行。仍需实测确认，但不是本 case 主因 | `sfa_v1.py:2311-2332` |
| MoE 的 `input_ids` 重排 | **死代码**：本模型 `scoring_func=sigmoid`，路由根本不读 `input_ids` | `experts_selector.py:247-278` |
| 剩下真正要查的 | ① 清单内容对不对（P1）② 每 token 的 KV 是否一致（P2）③ 合并调用本身（P3，唯一没被源码证明等价的路径） | — |

**一句话**：索引器"挑得准不准"这件事已经结案；要查的是"搬运对不对"和"算子算得对不对"。

---

## 3. 结果怎么解读（决策表）

`run_cp_diag.sh check` 会分两段打印结论：

| check 的输出 | 含义 | 下一步 |
|---|---|---|
| `[topk] RESULT: P1 holds` | 清单 = 完整历史位置，搬运没问题 | 看 kv 段 |
| `[topk] ... mismatched=N` | 清单内容错了 → 索引器/搬运 | 查 `indiceOutCoreOffset`、`ProcessInvalid`、`SplitCore`（QLI kernel）；此时 ②轮可不跑 |
| `[kv] ... identical for every token` | 两种布局下 KV 完全一致 | → 跑下面的 ②轮 |
| `[kv] first differing token=... layer=0` | 入口就有差异（embedding / 切片 / RoPE） | 查模型入口 |
| `[kv] first differing token=... layer=k` | 第 k 层算 KV 或写缓存有问题 | 定位到层 k |

判定"只剩算子"后，跑第二轮（唯一变量是调用形状）：

```bash
bash tools/cp_balance_compare/run_cp_diag.sh 2call        # 2~3 小时
python tools/cp_balance_compare/compare_cp_rounds.py \
    baseline=/dev/shm/cp_ab/r1_baseline 2call=/dev/shm/cp_ab/r2_2call
```

| 第二轮结果 | 含义 | 下一步 |
|---|---|---|
| `C−B` 回到噪声级（脚本会给 `[verdict]`） | 根因 = "合并 2B 单次调用" | 保留两次调用即可上线；同时把这次实验作为 repro 提给算子侧 |
| 仍 ≈0.56 | 合并调用无罪 | 回到 KV 那条线按层细查 |

可选第三轮（只在前面没定位到时跑）：`run_cp_diag.sh l1024` —— 用 1024 长度证明"与索引器/长度无关"。

---

## 4. 代码与工具交接

### 4.1 提交记录（branch `glm52_cp_balance_v3`）

| commit | 内容 |
|---|---|
| `a5507f5e3 提交A3上调试的修改` | 引擎侧开关 + dump + 契约检查；tools 三个脚本；工具 README |
| `e5051fef0 [fix] cp_balance launcher: 站点/挂载点更新 + vendor set_env 候选链 + preflight 自检` | launcher 的机器相关路径与自检 |

### 4.2 新增的 2 个环境变量（`vllm_ascend/envs.py`）

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_ASCEND_CP_BALANCE_MERGED_CALL` | `1` | `1`=现状（一次调用算两段）；`0`=回退成 prev/next 两次调用。**这是唯一的 A/B 开关，也是候选修复** |
| `VLLM_ASCEND_CP_BALANCE_DUMP` | 空=关 | 抓数据。语法 `kind:layers`，如 `topk:6,kv:0,6`（第 6 层抓清单、第 0/6 层抓 KV）。文件落在 `/dev/shm/cp_balance_dump/`，文件名带 `cpbal0/1`，B/C 两份不会互相覆盖 |

> 引擎里还新增了**无条件执行的 log-only 契约检查**（不改变数值、只在违反时打
> `[CP_BALANCE][check][...]` 告警，含 0-token 请求错位检测）。看到这类告警要留意，正常情况应当没有。

### 4.3 文件清单

| 文件 | 状态 | 说明 | 若要清理 |
|---|---|---|---|
| `vllm_ascend/envs.py` | 改（+23 行） | 上面 2 个变量 | 删变量即回到原状 |
| `vllm_ascend/attention/sfa_v1.py` | 改（+562 行） | 两次调用路径 `_indexer_select_post_process_zigzag`、dump（`_parse_dump_spec`/`_maybe_dump_topk`/`_maybe_dump_kv`）、契约检查（`_check_merged_zigzag_contracts`/`_check_zigzag_request_alignment`） | 默认开关下行为与改动前一致；脚本默认关 |
| `tools/cp_balance_compare/run_cp_diag.sh` | 新增 | 一键跑 baseline / 2call / l1024 / check，`-n` 只打印命令 | — |
| `tools/cp_balance_compare/check_zigzag_dumps.py` | 新增 | 判读 dump（`--kind topk｜kv`） | — |
| `tools/cp_balance_compare/compare_cp_rounds.py` | 新增 | 多轮 summary 并排 + 给结论 | — |
| `tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh` | 改 | 站点路径 + vendor 候选链 + dry-run 自检 | — |
| `tools/cp_balance_compare/README.md` | 改 | §9 脚本说明 + 覆盖项表格 | — |

### 4.4 工具用法速查

```bash
bash tools/cp_balance_compare/run_cp_diag.sh baseline -n    # 只打印将执行的命令（跑之前先看这个）
bash tools/cp_balance_compare/run_cp_diag.sh check          # 判读已有 dump（不用 NPU）
DRY_RUN=1 bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh 8034   # 单看 launcher 站点自检
```

---

## 5. 这台机器的环境（都可在 launcher SITE 段覆盖）

| 项 | 值 |
|---|---|
| 仓库 | `/opt/its/z30055003/vllm-ascend` |
| 权重 | `/opt/its/model/GLM-5.2-W4A8C8` |
| IP / 网卡 | `7.246.78.76` / `eth2` |
| vendor 环境（`vllm serve` 前 source） | `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash`（launcher 里是候选链自动探测，也可 `VENDOR_SET_ENV=<路径>` 指定，或置空跳过） |
| profiler 输出 | `/opt/its/z30055003/profiling_no_pooling` |
| 对比数据 / dump | `/dev/shm/cp_ab/r*` / `/dev/shm/cp_balance_dump/` |
| 运行配置 | TP8 + EP + DP1、SFA C8 + LI C8、enable_dsa_cp、MTP 1 token、enforce_eager、无 prefix caching、`MIN_TOKENS=2048` |

---

## 6. 已知的坑（踩过，别再来一次）

1. **`bash -lc` 会重建 PATH**：driver 用它起 launcher，所以"靠前置 PATH 注入工具"的做法在真机上无效。
2. **必须用 tmux/nohup 跑长任务**：driver 挂了会留下占卡的 vllm 进程，下一轮起不来（先 `pkill -f vllm`）。
3. `--repeat-a` 会把 B 再跑一遍（B2），于是每个层会有**两份 `cpbal0` 的 dump**；checker 取最新并提示忽略旧的，属正常现象。
4. dump 是"每层每进程只写一次"；重跑一轮直接跑即可（按时间戳取最新）。
5. **zigzag 只在 `num_actual_tokens >= MIN_TOKENS(=2048)` 时启用**，低于阈值 C 组会静默等于 B（driver 会告警）。
6. **不要"顺手修" seqlen 约定**（query 前缀和 / KV 原始值），改了必然更糟——这是算子内部规定。
7. A3 与 A5 的 kernel 是两套实现：**A3 的数值不能和 A5 横比**；A3 上 L4096 会走一条 A5 没有的 LD 路径，异常时先换 L2048 复测。
8. 单节点 8 卡无法并行两台 TP8 server，所以"关闭 DSA-CP 的 A 组对照"做不了，先跳过。

---

## 7. 未决 / 待办

| 事项 | 说明 |
|---|---|
| `repo` 与 `vendor` 是否同一个 checkout | `--preflight` 的 OK 行同时打印 `repo=` 和 `vendor=`，若不同源会导致"python 包与 custom ops 版本错配" |
| `/etc/hixlep` 是否存在 | 只在复现 "MTP + Mooncake PD" 组合时才需要（launcher 的 KV connector 配置用） |
| 0-token 请求对齐（T6） | 确认的缺陷，等诊断结论后再改（改 metadata 语义，会污染 A/B 对照） |
| 契约断言 + UT（T8） | 目前只有 log-only 告警；稳定后升级为断言并补 UT |
| 索引器侧的复现用例 | 若最终定位到算子，需要给算子侧一个最小 repro |

---

## 8. 参考文档（深挖用，不必先读）

全部归档在 `history_docs/`：

| 文档 | 内容 |
|---|---|
| `history_docs/CP_BALANCE_精度异常分析.md` | 最早的一份分析（doc1），列出 5 个候选根因 |
| `history_docs/ops-transformer_KvQuantSparseFlashAttention与QuantLightningIndexer实现分析.md` | 两个算子的实现级分析（doc2），本项目的"算子语义字典" |
| `history_docs/CP_BALANCE_精度问题_算子语义交叉分析.md` | 用 doc2 的算子语义逐条判定 doc1 的候选，给出本文 §2 的结论与源码依据 |
| `history_docs/CP_BALANCE_精度问题_下一步行动计划.md` | 任务卡（T0~T9）、里程碑、决策树（比本文详细） |
| `history_docs/A3验证_与A5的差异与可迁移结论.md` | 若要在 A3 上验证，先读这份（哪些结论可迁移、哪些不可） |
| `history_docs/GLM-5.2_*.md`、`vllm-ascend_cp_balance_代码实现文档.md` 等 | 更早的方案/实现记录，仅作背景 |

---

## 9. 东西都在哪（接手第一件事）

```
/opt/its/z30055003/vllm-ascend/          # 代码（branch glm52_cp_balance_v3），工具在 tools/cp_balance_compare/
D:\code\glm5.2_950_optim\                # 分析用的工作区（Windows 侧）
├── CP_BALANCE_精度问题_交接文档.md        # ← 本文，唯一需要先读的
├── history_docs/                        # 历史分析文档（§8）
├── unuse/                               # 已弃用的旧启动脚本（run_script/*、run_w4a8_cp_balance.sh 等），不要用
├── vllm-ascend/                         # 代码副本（与节点上是同一分支）
├── vllm/ ops-transformer/ sglang/       # 上游参考代码，只读
└── tmp/                                 # 调试期间的临时探针脚本
```

> 除代码仓库外，本文与 `history_docs/` 都**未纳入 git**（工作区根目录不是仓库）。
> 若希望交接材料随代码走，可把本文拷进 `vllm-ascend/tools/cp_balance_compare/` 并提交。

