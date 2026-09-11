#!/usr/bin/env bash
# CP_BALANCE zigzag 精度诊断：一轮 A/B 的标准入口。
#
# 在 vllm-ascend 仓库根目录执行：
#   bash tools/cp_balance_compare/run_cp_diag.sh baseline   # T0+T2+T3：基线 B/C/B2 + topk/KV dump
#   bash tools/cp_balance_compare/run_cp_diag.sh sweep      # 多层扫描：B/C（无 B2）+ 全层 KV+FP dump
#   bash tools/cp_balance_compare/run_cp_diag.sh 2call      # T1：回退 prev/next 两次调用
#   bash tools/cp_balance_compare/run_cp_diag.sh l1024      # T4：MIN_TOKENS=1024
#   bash tools/cp_balance_compare/run_cp_diag.sh check      # CPU 侧判读已有 dump（不需要 NPU）
#   bash tools/cp_balance_compare/run_cp_diag.sh list       # 只打印将要执行的命令
#   加 --no-repeat 可跳过 B2（等价于 REPEAT_A=0）
#
# 可覆盖的环境变量：
#   LAUNCHER     默认 bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}
#   PROMPT_LENS  默认 2048,2049,4096        MIN_TOKENS 默认 2048
#   REPEAT_A     默认 1（=0 跳过 B2 重复跑：少一次模型加载。噪声地板已确认是 0 时可用，
#                此时 driver 的阈值仍是 max(0.05, 5×0)=0.05，各 case 指标不受影响）
#   TP_SIZE      默认 8（launcher 的 tensor-parallel-size，同时决定 zigzag 的 cp_size）
#   CP_SIZE      默认取 TP_SIZE（driver 的 --cp-size，必须等于 TP_SIZE）
#   OUT_ROOT     默认 /dev/shm/cp_ab        BASE_PORT  默认 8034
#   DUMP_SPEC    默认 topk:6,kv:0,6（置空=不 dump；支持 kv:all / topk:all）
#   DUMP_DIR     默认 /dev/shm/cp_balance_dump
#
# 说明：digest 变量在 B/C 两个 server 上取值相同，dump 文件名自带 cpbal{0|1}，
# 所以一轮就能同时拿到 B 与 C 的数据。每轮输出到 $OUT_ROOT/<round>。
set -euo pipefail

# 参数解析：第一个非开关参数是模式；list/-n/--dry-run/--list 只打印命令不执行。
# 这样 `run_cp_diag.sh 2call --dry-run` 不会误触发一轮真实运行（2~3 次模型加载）。
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

round_name=""
extra_env=()
# True only when this round actually injects the DUMP env (2call deliberately
# does not), so the "no dump was produced" guard cannot fire a false alarm.
dump_active=false

case "${mode}" in
  baseline)
    round_name="r1_baseline"
    if [[ -n "${dump_spec}" ]]; then
      extra_env+=(--env "VLLM_ASCEND_CP_BALANCE_DUMP=${dump_spec}")
      dump_active=true
    fi
    ;;
  2call)
    round_name="r2_2call"
    extra_env+=(--env "VLLM_ASCEND_CP_BALANCE_MERGED_CALL=0")
    ;;
  l1024)
    round_name="r3_l1024"
    prompt_lens="1024"
    min_tokens="1024"
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
  check)
    echo "[run_cp_diag] analysing ${dump_dir}"
    exec python "${checker}" --dir "${dump_dir}" --kind both
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 {baseline|sweep|2call|l1024|check|list} [--no-repeat]" >&2
    exit 2
    ;;
esac

out="${out_root}/${round_name}"
cmd=(python "${driver}"
     --launcher "${launcher}"
     --configs B,C
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
echo "[run_cp_diag] prompt_lens=${prompt_lens} min_tokens=${min_tokens} cp_size=${cp_size} repeat_a=${repeat_a} out=${out}"
[[ -n "${dump_spec}" ]] && echo "[run_cp_diag] dump=${dump_spec} dir=${dump_dir}"

if [[ "${dry_run}" == true ]]; then
  printf '[run_cp_diag] command:\n  '
  printf '%q ' "${cmd[@]}"
  printf '\n[run_cp_diag] then:\n'
  printf '  python %q --dir %s --kind both\n' "${checker}" "${dump_dir}"
  printf '  python %q baseline=%s 2call=%s\n' "${comparer}" "${out_root}/r1_baseline" "${out_root}/r2_2call"
  exit 0
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
echo "  python ${checker} --dir ${dump_dir} --kind both"
echo "  python ${comparer} baseline=${out_root}/r1_baseline 2call=${out_root}/r2_2call"
exit "${rc}"
