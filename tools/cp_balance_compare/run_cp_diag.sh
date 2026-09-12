#!/usr/bin/env bash
# CP_BALANCE zigzag 精度诊断：一轮 A/B 的标准入口。
#
# 在 vllm-ascend 仓库根目录执行：
#   bash tools/cp_balance_compare/run_cp_diag.sh baseline   # T0+T2+T3：基线 B/C/B2 + topk/KV dump
#   bash tools/cp_balance_compare/run_cp_diag.sh sweep      # 多层扫描：B/C（无 B2）+ 全层 KV+FP dump
#   bash tools/cp_balance_compare/run_cp_diag.sh probe      # op 级打点：B/C（无 B2）+ 逐层激活/量化输入剖面
#   bash tools/cp_balance_compare/run_cp_diag.sh probe2     # probe + C 的重复跑（--configs C,B --repeat-a ⇒ C,B,C2）
#   bash tools/cp_balance_compare/run_cp_diag.sh check      # CPU 侧判读已有 dump（不需要 NPU）
#   bash tools/cp_balance_compare/run_cp_diag.sh list       # 只打印将要执行的命令
#   加 --no-repeat 可跳过 B2（等价于 REPEAT_A=0）
#
# 真跑前会先执行 CPU code-path gate（tools/cp_balance_compare/selftest_cp_logic.py）：
# 布局/对齐/固定顺序归约任一不通过就直接退出 4，不再花 20+ 分钟加载模型。
#
# 可覆盖的环境变量：
#   LAUNCHER     默认 bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}
#   CP_AB_SITE   站点档案（sites.sh）：its = A3 7.246.78.75/16 卡（默认）；share = 141.61.133.104/8 卡。
#                它决定 IP/repo/权重/PROFILER_DIR/TP_SIZE/可见卡；TP_SIZE 与 CP_SIZE 由档案给出，
#                显式 export 仍然优先。切换节点只改这一个变量。
#   PROMPT_LENS  默认 2048,2049,4096        MIN_TOKENS 默认 2048
#   TP_SIZE      默认取站点档案（its=16、share=8；覆盖它必须同时覆盖 CP_SIZE）
#   REPEAT_A     默认 1（=0 跳过 B2 重复跑：少一次模型加载。噪声地板已确认是 0 时可用，
#                此时 driver 的阈值仍是 max(0.05, 5×0)=0.05，各 case 指标不受影响）
#   CP_SIZE      默认取 TP_SIZE（driver 的 --cp-size，必须等于 TP_SIZE）
#   OUT_ROOT     默认 /dev/shm/cp_ab        BASE_PORT  默认 8034
#   KIND         check 模式的判读类型（默认 both；probe 轮用 KIND=act）
#   DUMP_SPEC    默认 topk:6,kv:0,6（置空=不 dump；支持 kv:all / topk:all / act:0,1,2,3 /
#                mlp:0,1 / qin:0；probe 模式自带一套默认值）
#   DUMP_DIR     默认 /dev/shm/cp_balance_dump（**writer 与 checker 共用这一个**；
#                kv:all 是 GB 级、act 是百 MB 级，/dev/shm 小就指到真实磁盘——写满
#                /dev/shm 还会把 server 自己搞死，它的 IPC/prometheus 目录都在那里）
#   HCCL_DET     默认空（关）。开启集合通信确定性（CANN 环境变量参考 HCCL_DETERMINISTIC /
#                LCCL_DETERMINISTIC；仓库自身用法见 vllm_ascend/batch_invariant.py:82-87
#                与 docs/source/faqs.md §15）。取值：
#                  true   → HCCL_DETERMINISTIC=true   + LCCL_DETERMINISTIC=1   ← **先用这个**
#                  atb    → 上面 + ATB_MATMUL_SHUFFLE_K_ENABLE=0 + ATB_LLM_LCOC_ENABLE=0
#                           + CLOSE_MATMUL_K_SHIFT=1（torchtitan-npu 的 NPU 确定性配方用的就是 true 这档）
#                  expand → 上面(true 档) + HCCL_OP_EXPANSION_MODE=2（AICPU 展开：A3 文档称
#                           该模式下归约类算子本身即确定性）
#                  strict → HCCL_DETERMINISTIC=strict（**本站不可用**：2026-09-12 实测在
#                           profile run 的 MoE dispatch 里炸 —— HcclReduceScatter 的 AICPU
#                           kernel RunAicpuIndOpCommInit "get kernel failed"(11003)，见 README §七）
#                用途：B/C 差异已收敛到跨 rank 归约（tensor_model_parallel_reduce_scatter），
#                这一轮回答"归约顺序的不确定性/排布相关性是不是根因"。
#                判据：`[cp-ab-hccl]` 行里能看到这些变量，且该轮 `p99|d|C-B <= 阈值`（0.05）
#                ⇒ 确定性配置即缓解；仍超阈 ⇒ 换归约实现（all_reduce + slice）再 A/B。
#                另可单独 `export HCCL_OP_EXPANSION_MODE=<0|1|2|3>` / `HCCL_ALGO=…` 做正交实验，
#                launcher 会把它们打进 `[cp-ab-hccl]` 行（环境本身是透传的）。
#
# 说明：digest 变量在 B/C 两个 server 上取值相同，dump 文件名自带 cpbal{0|1}，
# 所以一轮就能同时拿到 B 与 C 的数据。每轮输出到 $OUT_ROOT/<round>。
set -euo pipefail

# 参数解析：第一个非开关参数是模式；list/-n/--dry-run/--list 只打印命令不执行。
# 这样 `run_cp_diag.sh probe --dry-run` 不会误触发一轮真实运行（2~3 次模型加载）。
mode=""
dry_run=false
no_repeat=false
for arg in "$@"; do
  case "${arg}" in
    -n|--dry-run|list|--list) dry_run=true ;;
    help|-h|--help) dry_run=true; mode="help" ;;
    --no-repeat|--fast) no_repeat=true ;;
    *)
      if [[ -z "${mode}" ]]; then
        mode="${arg}"
      else
        echo "[run_cp_diag] ignoring extra argument: ${arg}" >&2
      fi
      ;;
  esac
done
mode="${mode:-baseline}"

readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# Site profile = single source of truth for node parameters (IP/repo/model/TP/
# visible chips).  Applying it here is what makes `CP_SIZE` follow this site's
# TP_SIZE instead of a hard-coded 8 -- a mismatch between the server's TP and the
# driver's --cp-size invalidates the whole round.
# shellcheck source=sites.sh
source "${script_dir}/sites.sh"
if ! cp_ab_site_apply "${CP_AB_SITE:-}"; then
  echo "[run_cp_diag] 站点档案无效；先 export CP_AB_SITE=$(cp_ab_site_list | tr '\n' '|' | sed 's/|$//')" >&2
  exit 2
fi

driver="tools/cp_balance_compare/ab_cp_compare.py"
checker="tools/cp_balance_compare/check_zigzag_dumps.py"
comparer="tools/cp_balance_compare/compare_cp_rounds.py"

launcher="${LAUNCHER:-bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}}"
prompt_lens="${PROMPT_LENS:-2048,2049,4096}"
min_tokens="${MIN_TOKENS:-2048}"
cp_size="${CP_SIZE:-${TP_SIZE:-8}}"
repeat_a="${REPEAT_A:-1}"
if [[ "${no_repeat}" == true ]]; then
  repeat_a=0
fi
out_root="${OUT_ROOT:-/dev/shm/cp_ab}"
base_port="${BASE_PORT:-8034}"
dump_spec="${DUMP_SPEC:-topk:6,kv:0,6}"
dump_dir="${DUMP_DIR:-/dev/shm/cp_balance_dump}"

# Collective-communication determinism (opt-in).  `HCCL_DET` is not a config
# knob: it applies to every config of the round, so B and C are still compared
# under identical settings -- only the token layout differs.
hccl_det="${HCCL_DET:-}"
hccl_det_env=()
case "${hccl_det}" in
  ""|0|off|false) hccl_det="" ;;
  # ``strict`` is kept only so the known-bad round can be reproduced on purpose: on
  # the site it kills the server in the profile run (AICPU kernel
  # RunAicpuIndOpCommInit "get kernel failed", see README §七).
  strict) hccl_det_env=(HCCL_DETERMINISTIC=strict LCCL_DETERMINISTIC=1) ;;
  true|1) hccl_det_env=(HCCL_DETERMINISTIC=true LCCL_DETERMINISTIC=1) ;;
  atb)
    hccl_det_env=(HCCL_DETERMINISTIC=true LCCL_DETERMINISTIC=1
                  ATB_MATMUL_SHUFFLE_K_ENABLE=0 ATB_LLM_LCOC_ENABLE=0
                  CLOSE_MATMUL_K_SHIFT=1)
    ;;
  expand)
    hccl_det_env=(HCCL_DETERMINISTIC=true LCCL_DETERMINISTIC=1
                  HCCL_OP_EXPANSION_MODE=2)
    ;;
  *)
    echo "[run_cp_diag] unknown HCCL_DET=${hccl_det}（可选 true|atb|expand|strict，或留空关闭）" >&2
    exit 2
    ;;
esac

round_name=""
extra_env=()
# Which configs the driver runs, in order.  ``--repeat-a`` repeats the *first*
# one as <name>2 (that is how the noise floor is measured), so a C-determinism
# round has to put C first: `--configs C,B --repeat-a` -> C, B, C2.
configs="B,C"
# True only when this round actually injects the DUMP env, so the "no dump was
# produced" guard cannot fire a false alarm.
dump_active=false

case "${mode}" in
  baseline)
    round_name="r1_baseline"
    if [[ -n "${dump_spec}" ]]; then
      extra_env+=(--env "VLLM_ASCEND_CP_BALANCE_DUMP=${dump_spec}")
      dump_active=true
    fi
    ;;
  sweep)
    # One-command multi-layer diagnostic round: no B2 (the noise floor is
    # already known to be 0), KV dump for every layer (the dump also carries the
    # pre-quantization FP copy), and its own output root so an earlier round is
    # never overwritten.  Deliberately driven by the mode word instead of
    # env prefixes: a long paste that loses "VAR=..." would silently fall back
    # to the defaults, which is exactly how a round ends up running B2 again.
    round_name="r_sweep"
    repeat_a=0
    dump_spec="${DUMP_SPEC:-kv:all}"
    out_root="${OUT_ROOT:-/dev/shm/cp_ab_sweep}"
    if [[ -n "${dump_spec}" ]]; then
      extra_env+=(--env "VLLM_ASCEND_CP_BALANCE_DUMP=${dump_spec}")
      dump_active=true
    fi
    ;;
  probe | probe2)
    # op 级全精度打点：attention 的输入（= 上一层的输出）与输出（= 本层 attention
    # 的贡献）各存一份，按**全局 token 位置**打点，所以 B/C 可以直接逐 token 比。
    # 判据见 README「激活剖面」：in 相同 + out 不同 => attention 内部先不等；
    # out 相同 + 下一层 in 不同 => 中间那层的 MoE/MLP 先不等。
    # MLP 边界（mlp:<层>）：MLP 的输入与输出各一份，把一层切成
    # attention out → mlp in → mlp out → 下一层 attention in。
    # qin:<层>：该层两个 MLP GEMM **实际吃到的量化输入**（fp8 + e8m0 scale）。
    # 它和 mlp_in/gu_out/dn_q 一起把「同一个 bf16 输入却算出不同结果」拆成两种根因：
    #   量化输入也变 => 根因在量化/融合内核；量化输入相同而 GEMM 输出不同 => 根因在 GEMM 内核。
    # 只打 layer 0/1（0 是根因所在，1 作对照）：层数越多 dump 越大，而 2 层已足够定位。
    # 规模：每层每 rank ≈16MB × 8 rank × 2 配置 ⇒ 2 层 ≈ 510MB，加 qin ≈ 575MB
    # → 必须落真实磁盘；**每轮一个独立子目录**（下面按时间戳生成），否则判读会把
    # 上一轮的旧文件按"最新一份"混进来 —— 不同轮次的 spec 不同，混了就等于把两次
    # 测量拼成一张表（现场发生过：新轮的 gu_q/dn_q 配旧轮的 mlp_out）。
    #
    # probe2 = probe + C 自身的重复跑（--configs C,B --repeat-a ⇒ C, B, C2，3 次加载）：
    # 除 B/C 判读外，额外拿到 `[noise] C2 vs C` —— 用来区分"内核与行序相关"（C2−C=0）
    # 与"内核不可复现"（C2−C≠0）。注意 C 与 C2 的 dump 同名 cpbal1，判读取最新一份。
    dump_spec="${DUMP_SPEC:-act:0,1,mlp:0,1,qin:0,topk:0,kv:0,1}"
    out_root="${OUT_ROOT:-/dev/shm/cp_ab_probe}"
    if [[ "${mode}" == "probe2" ]]; then
      round_name="r_probe_c2"
      repeat_a=1
      configs="C,B"
    else
      round_name="r_probe"
      repeat_a=0
    fi
    if [[ -z "${DUMP_DIR:-}" ]]; then
      dump_dir="/root/cp_probe/$(date +%m%d_%H%M%S)"
    else
      # A stale `export DUMP_DIR=/root/cp_dump` from a sweep round silently
      # redirects this round's dumps into the sweep's directory.
      echo "[run_cp_diag] WARN: DUMP_DIR=${DUMP_DIR} 已设置，本轮 dump 会写进它（默认是 /root/cp_probe/<时间戳>）" >&2
      echo "[run_cp_diag]       若该目录里已有别的轮次，判读按 (layer, rank, cpbal) 取最新一份 → 会串味；" >&2
      echo "[run_cp_diag]       要用默认值就先 unset DUMP_DIR" >&2
    fi
    if [[ -n "${dump_spec}" ]]; then
      extra_env+=(--env "VLLM_ASCEND_CP_BALANCE_DUMP=${dump_spec}")
      dump_active=true
    fi
    ;;
  check)
    kind="${KIND:-both}"
    check_args=(--dir "${dump_dir}" --kind "${kind}")
    if [[ "${kind}" == "act" ]]; then
      # 一层两行（in/out）的紧凑表才对得上判据
      check_args+=(--summary-only)
    fi
    echo "[run_cp_diag] analysing ${dump_dir} (kind=${kind})"
    exec python "${checker}" "${check_args[@]}"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 {baseline|sweep|probe|check|list} [--no-repeat]" >&2
    exit 2
    ;;
esac

if [[ -n "${hccl_det}" ]]; then
  # Distinct round name: compare_cp_rounds.py compares rounds by directory, and a
  # determinism round must not overwrite the baseline round's summary.json --
  # that would erase the "before" it exists to be compared against.
  round_name="${round_name}_det"
fi

# One knob for both sides: the writer must put its files where the checker looks.
# A full-layer sweep is GBs, so point DUMP_DIR at a real disk when /dev/shm is
# small -- filling /dev/shm also kills the server (its IPC lives there).
if [[ "${dump_active}" == true ]]; then
  extra_env+=(--env "VLLM_ASCEND_CP_BALANCE_DUMP_DIR=${dump_dir}")
fi

# Determinism is a round-wide setting: the driver hands it to the launcher for
# every config, so B and C still differ in nothing but the token layout.
if (( ${#hccl_det_env[@]} )); then
  for kv in "${hccl_det_env[@]}"; do
    extra_env+=(--env "${kv}")
  done
fi

out="${out_root}/${round_name}"
cmd=(python "${driver}"
     --launcher "${launcher}"
     --configs "${configs}"
     --prompt-lens "${prompt_lens}"
     --cp-balance-min-tokens "${min_tokens}"
     --cp-size "${cp_size}"
     --base-port "${base_port}"
     --config-check strict --zigzag-check strict --on-zigzag-miss skip
     --out "${out}")
# B2 (the noise-floor repeat) is a full extra model load: skip it when the
# caller already knows the floor (see REPEAT_A in the header).
if [[ "${repeat_a}" != "0" ]]; then
  cmd+=(--repeat-a)
fi
if (( ${#extra_env[@]} )); then
  cmd+=("${extra_env[@]}")
fi

echo "[run_cp_diag] mode=${mode} round=${round_name}"
echo "[run_cp_diag] site=${CP_AB_SITE_RESOLVED} ip=${LOCAL_IP} tp_size=${TP_SIZE} cp_size=${cp_size} visible=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[run_cp_diag] prompt_lens=${prompt_lens} min_tokens=${min_tokens} cp_size=${cp_size} repeat_a=${repeat_a} out=${out}"
[[ -n "${dump_spec}" ]] && echo "[run_cp_diag] dump=${dump_spec} dir=${dump_dir}"
if (( ${#hccl_det_env[@]} )); then
  echo "[run_cp_diag] hccl_det=${hccl_det} env=${hccl_det_env[*]}（B/C 同轮同设置；判据见头部注释）"
fi

# Which way this round's dumps are judged (probe/probe2 = the activation profile).
check_kind="both"
check_extra=""
if [[ "${mode}" == "probe" || "${mode}" == "probe2" ]]; then
  check_kind="act"
  # Keep the preview identical to the judged command in README/HANDOVER:
  # --block-size 128 is what turns "first differing token" into "which zigzag block".
  check_extra=" --summary-only --block-size 128"
fi

if [[ "${dry_run}" == true ]]; then
  printf '[run_cp_diag] command:\n  '
  printf '%q ' "${cmd[@]}"
  printf '\n[run_cp_diag] then:\n'
  printf '  python %q --dir %s --kind %s%s\n' "${checker}" "${dump_dir}" "${check_kind}" "${check_extra}"
  printf '  # 与另一轮并排比较（把 <round1>/<round2> 换成 $OUT_ROOT 下的目录名）：\n'
  printf '  python %q <round1>=%s/<round1> <round2>=%s/<round2>\n' "${comparer}" "${out_root}" "${out_root}"
  exit 0
fi

# CPU code-path gate: a mismatch in the zigzag plan / padding alignment / fixed
# reduction wiring can never be found by another model load, so fail before
# spending 20+ minutes on the NPU.  ``selftest_cp_logic.py`` runs in
# milliseconds and prints one [logic] line per checked property.
if ! python tools/cp_balance_compare/selftest_cp_logic.py; then
  echo "[run_cp_diag] ERROR: CPU code-path gate failed; a remote A/B round cannot fix a layout/reduction mismatch." >&2
  echo "[run_cp_diag]        先把上面的 selftest_cp_logic.py 输出贴回来，不要继续加载模型。" >&2
  exit 4
fi

# 一轮 = 2~3 次模型加载，先把已有 dump 归档，避免和上一轮混淆
dump_before=0
if [[ "${dump_active}" == true && -d "${dump_dir}" ]]; then
  dump_before=$(find "${dump_dir}" -maxdepth 1 -name '*.pt' | wc -l)
  if (( dump_before > 0 )); then
    echo "[run_cp_diag] note: ${dump_dir} already holds ${dump_before} dump(s);"
    echo "[run_cp_diag]       the newest one per (layer, rank, cp_balance) wins, older ones are ignored."
  fi
fi

"${cmd[@]}"
rc=$?

# 一轮 20+ 分钟，最不该发生的失败是"跑完了但没有 dump"：判读没有数据，而日志里
# 可能只有一条容易被刷掉的 warning。这里明确报出来并把退出码标成 3。
if [[ "${dump_active}" == true ]]; then
  dump_after=$(find "${dump_dir}" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)
  if (( dump_after <= dump_before )); then
    echo "[run_cp_diag] ERROR: DUMP_SPEC=${dump_spec} 但这一轮没有产生任何 dump（dir=${dump_dir}，之前 ${dump_before} 个）" >&2
    echo "[run_cp_diag]        先确认 dir 是不是你要的那个：上面那行 '[run_cp_diag] dump=... dir=...' 就是实际写入目录；" >&2
    echo "[run_cp_diag]        若 shell 里继承了一个旧 DUMP_DIR（例如 sweep 用的 /root/cp_dump），数据就写到那儿去了 → unset DUMP_DIR 再跑。" >&2
    echo "[run_cp_diag]        判读没有数据；去 server 日志里找这几类线索：" >&2
    echo "[run_cp_diag]          - [CP_BALANCE][dump] ... -> path         （说明真的写了）" >&2
    echo "[run_cp_diag]          - dump_no_layer_idx / could not be parsed（layer 名解析失败）" >&2
    echo "[run_cp_diag]          - dump prep failed / natural-order ...    （dump 准备阶段抛错）" >&2
    echo "[run_cp_diag]          - 一条都没有                                （DUMP 环境变量没到 worker）" >&2
    if (( rc == 0 )); then
      rc=3
    fi
  else
    echo "[run_cp_diag] dump: +$((dump_after - dump_before)) file(s) under ${dump_dir}"
  fi
fi

echo
echo "[run_cp_diag] round finished (rc=${rc}); analyse with:"
echo "  python ${checker} --dir ${dump_dir} --kind ${check_kind}${check_extra}"
echo "  python ${comparer} <round1>=${out_root}/<round1> <round2>=${out_root}/<round2>"
exit "${rc}"
