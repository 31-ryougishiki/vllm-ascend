#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# GLM-5.2 w4a4c8-mxfp4 single-node launcher for
# tools/cp_balance_compare/ab_cp_compare.py.
#
# It is a drop-in replacement for
# /home/z30055003/script/start_server_prefill-w4a4c8-mxfp4.sh with two extra
# properties required by the B/C driver:
#   1. every cp_balance knob can be overridden through the environment,
#   2. it prints one "[cp-ab] ..." fingerprint line (--config-check).
#
# TP=8 by default: the zigzag layout's cp_size IS the tensor parallel size
# (vllm_ascend passes global_tp_size as cp_size and pads to 2 * tp_size), so
# `TP_SIZE` here must stay in sync with the driver's `--cp-size`, and the run
# needs 8 visible NPUs.
#
# Usage (port can be $1 or $2, so both call styles work):
#   bash launcher_glm52_w4a4c8_mxfp4.sh <port>
#   bash launcher_glm52_w4a4c8_mxfp4.sh x <port>

# ---------------------------------------------------------------------------
# SITE section: the values come from a switchable profile (sites.sh), so moving
# to another test node is one variable, not an edit:
#
#   export CP_AB_SITE=its     # A3 站点 7.246.78.75（16 卡）—— 默认
#   export CP_AB_SITE=share   # 当前站点 141.61.133.104（8 卡）
#
# Everything below keeps the `:=` override semantics: an explicit `export` of any
# single value still wins over the profile.
# ---------------------------------------------------------------------------
# shellcheck source=sites.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sites.sh"
if ! cp_ab_site_apply "${CP_AB_SITE:-}"; then
  echo "[cp-ab] 站点档案无效，先 export CP_AB_SITE=$(cp_ab_site_list | tr '\n' '|' | sed 's/|$//') 再试" >&2
  exit 2
fi
# Vendor environment that must be sourced before `vllm serve` (CANN custom
# transformer ops).  It is a build artifact of a vllm-ascend checkout
# (vllm_ascend/_cann_ops_custom/vendors/... is not in git), so the candidates
# below hold the site path first and the repo-local one second.
#   profile 决定默认：auto = 自动探测候选链；none = 显式跳过（A3 历来如此）
#   VENDOR_SET_ENV=                      -> skip the vendor env entirely
#   VENDOR_SET_ENV=/path/to/set_env.bash -> use exactly that file
VENDOR_SET_ENV_FALLBACKS=(
  "/mnt/share/l00622059/vendors/custom_transformer/bin/set_env.bash"
  "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
  "${VLLM_ASCEND_REPO}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
)
vendor_auto=false
if [ "${SITE_VENDOR_SET_ENV:-auto}" = "none" ]; then
  # Profile says this site does not use the vendor env: pin it empty (which the
  # code below and the [cp-ab-site] line both read as "skipped"), and let an
  # explicit `export VENDOR_SET_ENV=...` override it.
  : "${VENDOR_SET_ENV:=}"
elif [ -z "${VENDOR_SET_ENV+set}" ]; then
  vendor_auto=true
  VENDOR_SET_ENV=""
  for candidate in "${VENDOR_SET_ENV_FALLBACKS[@]}"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
      VENDOR_SET_ENV="${candidate}"
      break
    fi
  done
fi

# ---------------------------------------------------------------------------
# Site environment (same as the original prefill script).
# ---------------------------------------------------------------------------
unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY
# The site rc is sourced below and it *does* override round knobs: on the A3 site
# (2026-09-12) `/root/.bashrc` reset ``LCCL_DETERMINISTIC`` to 0 and left
# ``ATB_MATMUL_SHUFFLE_K_ENABLE=1``, so an ``HCCL_DET=...`` round ran with the
# determinism knob silently neutralised -- indistinguishable from "the knob has no
# effect".  Snapshot the caller's values here and re-assert them after the rc; any
# value the rc tried to change is printed, so a round can never be misread again.
CP_AB_REASSERT_KEYS=(
  HCCL_DETERMINISTIC LCCL_DETERMINISTIC ATB_MATMUL_SHUFFLE_K_ENABLE
  ATB_LLM_LCOC_ENABLE CLOSE_MATMUL_K_SHIFT HCCL_OP_EXPANSION_MODE HCCL_ALGO
  HCCL_BUFFSIZE HCCL_EXEC_TIMEOUT HCCL_CONNECT_TIMEOUT
  VLLM_ASCEND_CP_BALANCE VLLM_ASCEND_CP_BALANCE_MIN_TOKENS
  VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL VLLM_ASCEND_CP_BALANCE_DEBUG_LOG
  VLLM_ASCEND_CP_BALANCE_DUMP VLLM_ASCEND_CP_BALANCE_DUMP_DIR
  VLLM_ASCEND_ENABLE_FLASHCOMM1 ASCEND_RT_VISIBLE_DEVICES
)
declare -A CP_AB_REASSERT_VALUES=()
for _key in "${CP_AB_REASSERT_KEYS[@]}"; do
  if [ -n "${!_key+set}" ]; then
    CP_AB_REASSERT_VALUES["${_key}"]="${!_key}"
  fi
done
# CP_AB_SITE_RC: the rc path is a knob purely so the CPU selftest can point it at a
# fake rc and check the re-assert below (the real sites all use /root/.bashrc).
site_rc="${CP_AB_SITE_RC:-/root/.bashrc}"
# CP_AB_SKIP_SOURCE=1: the caller already sourced the site rc + vendor env once
# (see README "省掉每次 source 的启动时间") and the driver passes that
# environment straight to this process, so sourcing again only costs time.
# Everything below (HCCL ifnames, timeouts, ...) is still applied.
skip_source="${CP_AB_SKIP_SOURCE:-0}"
if [ "${skip_source}" = "1" ]; then
  echo "[cp-ab] CP_AB_SKIP_SOURCE=1: 跳过 source ${site_rc} 与 vendor set_env（沿用调用者环境）"
  echo "[cp-ab] env check: ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>} LD_LIBRARY_PATH=${#LD_LIBRARY_PATH}B PYTHONPATH=${PYTHONPATH:-<unset>}"
elif [ -f "${site_rc}" ]; then
  # shellcheck disable=SC1091
  # Site rc files are not written for `set -u` (e.g. /etc/bashrc reads
  # BASHRCSOURCED unguarded): relax -u while sourcing, restore it after.
  set +u; source "${site_rc}"; set -u
  for _key in "${!CP_AB_REASSERT_VALUES[@]}"; do
    _wanted="${CP_AB_REASSERT_VALUES[${_key}]}"
    if [ "${!_key-}" != "${_wanted}" ]; then
      echo "[cp-ab] NOTE: 站点 rc 把 ${_key} 从 '${_wanted}' 改成 '${!_key-<unset>}'，按本轮要求改回 '${_wanted}'"
      export "${_key}=${_wanted}"
    fi
  done
  unset _key _wanted
fi
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "${PROMETHEUS_MULTIPROC_DIR}"
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
unset HCCL_INTRA_ROCE_ENABLE

export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
# Overridable on purpose: the B-vs-C precision triage has to be able to A/B the
# collective algorithm together with the determinism knobs below.
export HCCL_ALGO="${HCCL_ALGO:-level0:fullmesh}"

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=180
export HCCL_BUFFSIZE=1200

export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP="${VLLM_ASCEND_ENABLE_PREFETCH_MLP:-1}"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Visible chips come from the site profile (its: 0..15 for TP=16, share: 0..7 for
# TP=8); a manual `export` still wins.  Keep it in sync with TP_SIZE -- a short
# list makes vllm fail at startup, a long one silently wastes chips.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:?site profile must set it (see sites.sh)}"
export TASK_QUEUE_ENABLE=1

plog_dir="${PWD}/$(date +%Y%m%d_%H%M%S)/plog"
mkdir -p "${plog_dir}"
export ASCEND_PROCESS_LOG_PATH="${plog_dir}"

# NOTE: fastokens (VLLM_USE_FASTOKENS) is deliberately NOT used by this
# launcher: the B/C comparison is tokenizer-sensitive and must run on the stock
# HF tokenizer.  The site rc (sourced above unless CP_AB_SKIP_SOURCE=1) may still
# export it, so it is dropped explicitly instead of relying on the vllm default
# (which is off).
unset VLLM_USE_FASTOKENS

if [ "${skip_source}" = "1" ]; then
  # Already sourced by the caller; only report which vendor env that was.
  echo "[cp-ab] vendor env: ${VENDOR_SET_ENV:-none} (CP_AB_SKIP_SOURCE=1, 未重复 source)"
elif [ -n "${VENDOR_SET_ENV}" ] && [ -f "${VENDOR_SET_ENV}" ]; then
  # shellcheck disable=SC1090
  # Vendor setup scripts may also reference unset variables.
  echo "[cp-ab] vendor env: ${VENDOR_SET_ENV}"
  set +u; source "${VENDOR_SET_ENV}"; set -u
elif [ "${vendor_auto}" = true ]; then
  echo "[cp-ab] WARN: vendor set_env.bash not found; tried:" >&2
  for candidate in "${VENDOR_SET_ENV_FALLBACKS[@]}"; do
    echo "[cp-ab]   - ${candidate}" >&2
  done
  echo "[cp-ab]   set VENDOR_SET_ENV=/path/to/set_env.bash, or VENDOR_SET_ENV= to skip it" >&2
elif [ -n "${VENDOR_SET_ENV}" ]; then
  echo "[cp-ab] WARN: VENDOR_SET_ENV does not exist, vendor environment is NOT applied: ${VENDOR_SET_ENV}" >&2
fi
# VENDOR_SET_ENV= (explicitly empty) means "this site does not need it": stay quiet.

export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONPATH="${VLLM_ASCEND_REPO}:${PYTHONPATH:-}"

# Extra `vllm serve` flags, space separated, e.g.
#   EXTRA_SERVE_ARGS="--enable-return-routed-experts"
# The routed-experts probe needs one of these, and other site knobs can be
# injected the same way without editing this file.  Echoed below so every
# round's log says exactly what was requested.
extra_serve_args=()
if [ -n "${EXTRA_SERVE_ARGS:-}" ]; then
  # shellcheck disable=SC2206  # word splitting is the point here
  extra_serve_args=(${EXTRA_SERVE_ARGS})
fi
echo "[cp-ab] EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-<unset>}"

# ---------------------------------------------------------------------------
# cp_balance knobs (overridable; defaults match the original script).
# MIN_TOKENS is pinned by the driver and checked in the fingerprint; here it is
# only the fallback for manual runs (vllm_ascend source default is 8192).
# ---------------------------------------------------------------------------
export VLLM_ASCEND_CP_BALANCE="${VLLM_ASCEND_CP_BALANCE:-1}"
export VLLM_ASCEND_CP_BALANCE_MIN_TOKENS="${VLLM_ASCEND_CP_BALANCE_MIN_TOKENS:-2048}"
export VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL="${VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL:-0}"

# NOTE: no PD-only knobs here. `recompute_scheduler_enable=true` is rejected by
# vllm_ascend.platform unless kv_role='kv_consumer', and the B/C comparison by
# default runs without the PD connector. Add it back with
# `--extra-additional-config '{"recompute_scheduler_enable": true}'` only when
# the run really is a PD decode node.
DEFAULT_ADDITIONAL_CONFIG='{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": "True", "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_dsa_cp": true}'
DEFAULT_SPEC_CONFIG='{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager":true}'
DEFAULT_KV_TRANSFER_CONFIG='{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {
                        "dp_size": 1,
                        "tp_size": 8
                },
                "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                },
                "ascend_local_comm_res_path": "/etc/hixlep"
        }
    }'

# NOTE: `${VAR-default}` (no ':') keeps an explicitly exported empty value,
# which is how the driver disables MTP / the PD connector.
additional_config="${VLLM_ASCEND_ADDITIONAL_CONFIG:-${DEFAULT_ADDITIONAL_CONFIG}}"
spec_config="${VLLM_ASCEND_SPEC_CONFIG-${DEFAULT_SPEC_CONFIG}}"
kv_config="${VLLM_ASCEND_KV_TRANSFER_CONFIG-${DEFAULT_KV_TRANSFER_CONFIG}}"

spec_args=()
if [ -n "${spec_config}" ]; then
  spec_args=(--speculative-config "${spec_config}")
fi
kv_args=()
if [ -n "${kv_config}" ]; then
  kv_args=(--kv-transfer-config "${kv_config}")
fi

# ---------------------------------------------------------------------------
# Configuration fingerprint read by ab_cp_compare.py --config-check.
# It must describe the configuration that is actually launched.
# ---------------------------------------------------------------------------
dsa_cp=1
if printf '%s' "${additional_config}" | grep -q '"enable_dsa_cp"[[:space:]]*:[[:space:]]*false'; then
  dsa_cp=0
fi
spec_flag=0
if [ -n "${spec_config}" ]; then spec_flag=1; fi
kv_flag=0
if [ -n "${kv_config}" ]; then kv_flag=1; fi
echo "[cp-ab] CP_BALANCE=${VLLM_ASCEND_CP_BALANCE} DSA_CP=${dsa_cp} MIN_TOKENS=${VLLM_ASCEND_CP_BALANCE_MIN_TOKENS} EMBED_LOCAL=${VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL} SPEC=${spec_flag} KV=${kv_flag}"
# Effective additional_config, verified verbatim by ab_cp_compare.py --config-check.
echo "[cp-ab-cfg] ${additional_config}"
# Diagnostic dump switch: the driver injects it per round; echoing it here makes
# "round finished but no dump was written" traceable to the env that was applied.
echo "[cp-ab] DUMP=${VLLM_ASCEND_CP_BALANCE_DUMP:-<unset>}"
# Which site profile this round actually ran with.  A round on the wrong node (or
# with the wrong TP) is otherwise indistinguishable in the log -- and TP is the
# zigzag's cp_size, so a mismatch invalidates every conclusion of the round.
echo "[cp-ab-site] CP_AB_SITE=${CP_AB_SITE_RESOLVED} LOCAL_IP=${LOCAL_IP} TP_SIZE=${TP_SIZE} VISIBLE=${ASCEND_RT_VISIBLE_DEVICES} REPO=${VLLM_ASCEND_REPO} MODEL=${MODEL_PATH} VENDOR=${VENDOR_SET_ENV:-none}"
# HCCL/ATB determinism knobs -- deliberately pass-through: this launcher never
# forces them, the round decides (`run_cp_diag.sh` with HCCL_DET=...).  The
# candidate root cause for the B-vs-C prefill difference is the cross-rank
# reduction (tensor_model_parallel_reduce_scatter), and whether that reduction is
# deterministic / layout independent is exactly what these knobs control, so the
# round has to say which setting it ran with.  NOT part of the [cp-ab] line:
# that line is parsed as int KEY=VALUE pairs by the driver's --config-check.
echo "[cp-ab-hccl] HCCL_ALGO=${HCCL_ALGO} HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-<unset>} LCCL_DETERMINISTIC=${LCCL_DETERMINISTIC:-<unset>} HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-<unset>} ATB_MATMUL_SHUFFLE_K_ENABLE=${ATB_MATMUL_SHUFFLE_K_ENABLE:-<unset>} ATB_LLM_LCOC_ENABLE=${ATB_LLM_LCOC_ENABLE:-<unset>} CLOSE_MATMUL_K_SHIFT=${CLOSE_MATMUL_K_SHIFT:-<unset>}"

port="${2:-${1:-8034}}"

# DRY_RUN=1: validate the environment without touching the NPUs (used by
# `ab_cp_compare.py --preflight`).  Also checks the site paths from the SITE
# section above, so a machine move (IP / repo / weights) fails here instead of
# half-way through a model load.
if [ -n "${DRY_RUN:-}" ]; then
  dry_rc=0
  if [ ! -d "${MODEL_PATH}" ]; then
    echo "[cp-ab][dry-run] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    dry_rc=2
  fi
  if [ ! -d "${VLLM_ASCEND_REPO}" ]; then
    echo "[cp-ab][dry-run] ERROR: VLLM_ASCEND_REPO does not exist: ${VLLM_ASCEND_REPO}" >&2
    dry_rc=2
  fi
  if [ -n "${VENDOR_SET_ENV}" ] && [ ! -f "${VENDOR_SET_ENV}" ]; then
    echo "[cp-ab][dry-run] ERROR: VENDOR_SET_ENV does not exist: ${VENDOR_SET_ENV}" >&2
    dry_rc=2
  elif [ "${vendor_auto}" = true ] && [ -z "${VENDOR_SET_ENV}" ]; then
    echo "[cp-ab][dry-run] WARN: no vendor set_env.bash auto-detected; custom ops may be missing" >&2
    echo "[cp-ab][dry-run]   set VENDOR_SET_ENV=/path/to/set_env.bash, or VENDOR_SET_ENV= to skip it" >&2
  fi
  if [ -n "${PROFILER_DIR}" ] && [ ! -d "$(dirname "${PROFILER_DIR}")" ]; then
    echo "[cp-ab][dry-run] WARN: PROFILER_DIR parent does not exist: ${PROFILER_DIR}" >&2
  fi
  if ! command -v vllm >/dev/null 2>&1; then
    echo "[cp-ab][dry-run] ERROR: vllm not found in PATH" >&2
    dry_rc=2
  fi
  if [ "${dry_rc}" -ne 0 ]; then
    exit "${dry_rc}"
  fi
  echo "[cp-ab][dry-run] OK vllm=$(command -v vllm) model=${MODEL_PATH} repo=${VLLM_ASCEND_REPO} vendor=${VENDOR_SET_ENV:-none} port=${port} tp=${TP_SIZE:-8}"
  exit 0
fi

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${port}" \
  --tensor-parallel-size "${TP_SIZE:-8}" \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --max_model_len "${MAX_MODEL_LEN:-135000}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --served-model-name "${SERVED_MODEL_NAME:-glm}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --max-num-seqs "${MAX_NUM_SEQS:-500}" \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --async-scheduling \
  --no-enable-prefix-caching \
  ${spec_args[@]+"${spec_args[@]}"} \
  --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"${PROFILER_DIR}\", \"torch_profiler_with_stack\": false}" \
  ${kv_args[@]+"${kv_args[@]}"} \
  ${extra_serve_args[@]+"${extra_serve_args[@]}"} \
  --additional_config "${additional_config}"
