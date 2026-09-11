#!/usr/bin/env bash
# CP_BALANCE zigzag 精度诊断：一轮 A/B 的标准入口。
#
# 在 vllm-ascend 仓库根目录执行：
#   bash tools/cp_balance_compare/run_cp_diag.sh baseline   # T0+T2+T3：基线 B/C/B2 + topk/KV dump
#   bash tools/cp_balance_compare/run_cp_diag.sh 2call      # T1：回退 prev/next 两次调用
#   bash tools/cp_balance_compare/run_cp_diag.sh l1024      # T4：MIN_TOKENS=1024
#   bash tools/cp_balance_compare/run_cp_diag.sh check      # CPU 侧判读已有 dump（不需要 NPU）
#   bash tools/cp_balance_compare/run_cp_diag.sh list       # 只打印将要执行的命令
#
# 可覆盖的环境变量：
#   LAUNCHER     默认 bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}
#   PROMPT_LENS  默认 2048,2049,4096        MIN_TOKENS 默认 2048
#   OUT_ROOT     默认 /dev/shm/cp_ab        BASE_PORT  默认 8034
#   DUMP_SPEC    默认 topk:6,kv:0,6（置空=不 dump）
#   DUMP_DIR     默认 /dev/shm/cp_balance_dump
#
# 说明：digest 变量在 B/C 两个 server 上取值相同，dump 文件名自带 cpbal{0|1}，
# 所以一轮就能同时拿到 B 与 C 的数据。每轮输出到 $OUT_ROOT/<round>。
set -euo pipefail

# 参数解析：第一个非开关参数是模式；list/-n/--dry-run/--list 只打印命令不执行。
# 这样 `run_cp_diag.sh 2call --dry-run` 不会误触发一轮真实运行（2~3 次模型加载）。
mode=""
dry_run=false
for arg in "$@"; do
  case "${arg}" in
    -n|--dry-run|list|--list) dry_run=true ;;
    help|-h|--help) dry_run=true; mode="help" ;;
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
out_root="${OUT_ROOT:-/dev/shm/cp_ab}"
base_port="${BASE_PORT:-8034}"
dump_spec="${DUMP_SPEC:-topk:6,kv:0,6}"
dump_dir="${DUMP_DIR:-/dev/shm/cp_balance_dump}"

round_name=""
extra_env=()

case "${mode}" in
  baseline)
    round_name="r1_baseline"
    if [[ -n "${dump_spec}" ]]; then
      extra_env+=(--env "VLLM_ASCEND_CP_BALANCE_DUMP=${dump_spec}")
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
    fi
    ;;
  check)
    echo "[run_cp_diag] analysing ${dump_dir}"
    exec python "${checker}" --dir "${dump_dir}" --kind both
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 {baseline|2call|l1024|check|list}" >&2
    exit 2
    ;;
esac

out="${out_root}/${round_name}"
cmd=(python "${driver}"
     --launcher "${launcher}"
     --configs B,C --repeat-a
     --prompt-lens "${prompt_lens}"
     --cp-balance-min-tokens "${min_tokens}"
     --base-port "${base_port}"
     --config-check strict --zigzag-check strict --on-zigzag-miss skip
     --out "${out}")
if (( ${#extra_env[@]} )); then
  cmd+=("${extra_env[@]}")
fi

echo "[run_cp_diag] mode=${mode} round=${round_name}"
echo "[run_cp_diag] prompt_lens=${prompt_lens} min_tokens=${min_tokens} out=${out}"
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
if [[ -n "${dump_spec}" && "${mode}" == "baseline" && -d "${dump_dir}" ]]; then
  kept=$(find "${dump_dir}" -maxdepth 1 -name '*.pt' | wc -l)
  if (( kept > 0 )); then
    echo "[run_cp_diag] note: ${dump_dir} already holds ${kept} dump(s);"
    echo "[run_cp_diag]       the newest one per (layer, rank, cp_balance) wins, older ones are ignored."
  fi
fi

"${cmd[@]}"
rc=$?

echo
echo "[run_cp_diag] round finished (rc=${rc}); analyse with:"
echo "  python ${checker} --dir ${dump_dir} --kind both"
echo "  python ${comparer} baseline=${out_root}/r1_baseline 2call=${out_root}/r2_2call"
exit "${rc}"
