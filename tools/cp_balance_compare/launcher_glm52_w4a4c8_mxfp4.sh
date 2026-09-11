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
# SITE section: adapt these to the machine.
# ---------------------------------------------------------------------------
NIC_NAME="${NIC_NAME:-eth2}"
LOCAL_IP="${LOCAL_IP:-141.61.133.104}"
VLLM_ASCEND_REPO="${VLLM_ASCEND_REPO:-/home/z30055003/vllm-ascend}"
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4}"
# Vendor environment that must be sourced before `vllm serve` (CANN custom
# transformer ops).  It is a build artifact of a vllm-ascend checkout
# (vllm_ascend/_cann_ops_custom/vendors/... is not in git), so the candidates
# below hold the site path first and the repo-local one second.
#   unset                                -> auto-detect, first hit wins
#   VENDOR_SET_ENV=                      -> skip the vendor env entirely
#   VENDOR_SET_ENV=/path/to/set_env.bash -> use exactly that file
VENDOR_SET_ENV_FALLBACKS=(
  "/mnt/share/l00622059/vendors/custom_transformer/bin/set_env.bash"
  "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
  "${VLLM_ASCEND_REPO}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
)
vendor_auto=false
if [ -z "${VENDOR_SET_ENV+set}" ]; then
  vendor_auto=true
  VENDOR_SET_ENV=""
  for candidate in "${VENDOR_SET_ENV_FALLBACKS[@]}"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
      VENDOR_SET_ENV="${candidate}"
      break
    fi
  done
fi
PROFILER_DIR="${PROFILER_DIR:-/home/z30055003/profiling_no_pooling}"

# ---------------------------------------------------------------------------
# Site environment (same as the original prefill script).
# ---------------------------------------------------------------------------
unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY
if [ -f /root/.bashrc ]; then
  # shellcheck disable=SC1091
  # Site rc files are not written for `set -u` (e.g. /etc/bashrc reads
  # BASHRCSOURCED unguarded): relax -u while sourcing, restore it after.
  set +u; source /root/.bashrc; set -u
fi
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "${PROMETHEUS_MULTIPROC_DIR}"
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
unset HCCL_INTRA_ROCE_ENABLE

export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_ALGO=level0:fullmesh

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

# 8 chips by default: TP_SIZE=8 needs ASCEND_RT_VISIBLE_DEVICES to cover them
# (override if the box exposes a different set, e.g. 0..15 for TP=16).
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TASK_QUEUE_ENABLE=1

plog_dir="${PWD}/$(date +%Y%m%d_%H%M%S)/plog"
mkdir -p "${plog_dir}"
export ASCEND_PROCESS_LOG_PATH="${plog_dir}"

# NOTE: fastokens (VLLM_USE_FASTOKENS) is deliberately NOT used by this
# launcher: the B/C comparison is tokenizer-sensitive and must run on the stock
# HF tokenizer.  /root/.bashrc is sourced above and may still export it, so it is
# dropped explicitly instead of relying on the vllm default (which is off).
unset VLLM_USE_FASTOKENS

if [ -n "${VENDOR_SET_ENV}" ] && [ -f "${VENDOR_SET_ENV}" ]; then
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
  --additional_config "${additional_config}"
