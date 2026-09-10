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
# Reference launcher for tools/cp_balance_compare/ab_cp_compare.py.
#
# The driver starts this script as `bash launcher_template.sh <port>` and
# injects the B/C environment variables:
#
#   VLLM_ASCEND_CP_BALANCE             0 = continuous CP, 1 = zigzag
#   VLLM_ASCEND_ADDITIONAL_CONFIG      full additional_config JSON
#   VLLM_ASCEND_SPEC_CONFIG            speculative config JSON, empty = MTP off
#   VLLM_ASCEND_KV_TRANSFER_CONFIG     KV connector JSON, empty = PD off
#   VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL embedding entry A/B switch
#
# Two rules for any launcher used by the driver:
#   1. honour the overrides above (use ${VAR-default}, never hard-code them),
#   2. print one line starting with "[cp-ab]" that reflects what was applied;
#      the driver compares it against the expected config (--config-check).
#
# Adapt MODEL_PATH / PRE_LAUNCH_SCRIPT / TP_SIZE and any site specific flags
# (HCCL, custom_transformer set_env.bash, ...) to your environment.

set -euo pipefail

port="${1:-8034}"
: "${MODEL_PATH:?set MODEL_PATH to the checkpoint directory}"

# Optional site specific environment (e.g. HCCL ifnames, vendor set_env.bash).
# Site rc files are not written for `set -u` (e.g. /etc/bashrc reads
# BASHRCSOURCED unguarded), so relax `-u` while sourcing and restore it after.
if [ -n "${PRE_LAUNCH_SCRIPT:-}" ] && [ -f "${PRE_LAUNCH_SCRIPT}" ]; then
  # shellcheck disable=SC1090
  set +u; source "${PRE_LAUNCH_SCRIPT}"; set -u
fi
if [ -f /root/.bashrc ]; then
  # shellcheck disable=SC1091
  set +u; source /root/.bashrc; set -u
fi

export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_CP_BALANCE="${VLLM_ASCEND_CP_BALANCE:-1}"
# Zigzag only turns on at/above this token count.  The driver pins
# VLLM_ASCEND_CP_BALANCE_MIN_TOKENS for every config and verifies it in the
# fingerprint, so the value here is just a fallback for manual runs (the
# vllm_ascend source default is 8192).
export VLLM_ASCEND_CP_BALANCE_MIN_TOKENS="${VLLM_ASCEND_CP_BALANCE_MIN_TOKENS:-2048}"
export VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL="${VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL:-0}"

# NOTE: no PD-only knobs here. `recompute_scheduler_enable=true` is rejected by
# vllm_ascend.platform unless kv_role='kv_consumer', and the B/C comparison by
# default runs without the PD connector.
DEFAULT_ADDITIONAL_CONFIG='{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": "True", "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_dsa_cp": true}'
additional_config="${VLLM_ASCEND_ADDITIONAL_CONFIG:-${DEFAULT_ADDITIONAL_CONFIG}}"
spec_config="${VLLM_ASCEND_SPEC_CONFIG-}"
kv_config="${VLLM_ASCEND_KV_TRANSFER_CONFIG-}"

spec_args=()
if [ -n "${spec_config}" ]; then
  spec_args=(--speculative-config "${spec_config}")
fi
kv_args=()
if [ -n "${kv_config}" ]; then
  kv_args=(--kv-transfer-config "${kv_config}")
fi

# ---------------------------------------------------------------------------
# Fingerprint consumed by ab_cp_compare.py --config-check.  It must describe
# the configuration that is actually launched, not the requested one.
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

# DRY_RUN=1: validate the environment without touching the NPUs (used by
# `ab_cp_compare.py --preflight`).
if [ -n "${DRY_RUN:-}" ]; then
  if [ ! -d "${MODEL_PATH}" ]; then
    echo "[cp-ab][dry-run] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 2
  fi
  if ! command -v vllm >/dev/null 2>&1; then
    echo "[cp-ab][dry-run] ERROR: vllm not found in PATH" >&2
    exit 2
  fi
  echo "[cp-ab][dry-run] OK vllm=$(command -v vllm) model=${MODEL_PATH} port=${port}"
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
  --async-scheduling \
  --no-enable-prefix-caching \
  ${spec_args[@]+"${spec_args[@]}"} \
  ${kv_args[@]+"${kv_args[@]}"} \
  --additional_config "${additional_config}"
