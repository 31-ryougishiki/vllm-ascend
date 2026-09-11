#!/usr/bin/env bash
# 一次性准备 cp_balance 测试环境（**必须 source，不要直接执行**）：
#
#   source tools/cp_balance_compare/prepare_env.sh
#
# 它做三件事：
#   1. source /root/.bashrc（站点 rc）；
#   2. source vendor 的 set_env.bash（路径直接从站点 launcher 的 DRY_RUN 输出里取，
#      与 launcher 用同一套候选链，避免两处漂移）；
#   3. export CP_AB_SKIP_SOURCE=1 —— 之后在同一个 shell 里跑 run_cp_diag.sh /
#      run_single.py，launcher 就不会再重复 source，每轮省掉这部分启动时间。
#
# 为什么可以省：driver 起 server 时是 `env = os.environ.copy()` 再交给 launcher，
# 所以"在当前 shell 里 export 好"和"launcher 自己 source"对 server 是等价的。
#
# 恢复原行为：`unset CP_AB_SKIP_SOURCE`
# 注意：必须 export（不是仅赋值），且要在**运行 run_cp_diag.sh 的那个 shell** 里执行。
# 它只改环境变量、不 exec、不 set -e，可以安全地 source 进交互 shell。
set +u

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo="$(cd "${_here}/../.." && pwd)"
_launcher="${_here}/launcher_glm52_w4a4c8_mxfp4.sh"

# 候选链与 VENDOR_SET_ENV 覆盖都交给 launcher 自己判断（DRY_RUN 只做路径检查，不加载模型）
_vendor="$(cd "${_repo}" && DRY_RUN=1 bash "${_launcher}" "${BASE_PORT:-8034}" 2>/dev/null \
  | sed -n 's/^\[cp-ab\] vendor env: //p' | head -1)"

_start=${SECONDS}
if [ -f /root/.bashrc ]; then
  source /root/.bashrc
fi
_r1=${SECONDS}

_ok=true
if [ -n "${_vendor}" ] && [ -f "${_vendor}" ]; then
  source "${_vendor}"
elif [ -n "${VENDOR_SET_ENV:-}" ] && [ -f "${VENDOR_SET_ENV}" ]; then
  source "${VENDOR_SET_ENV}"
else
  _ok=false
fi
_r2=${SECONDS}

if [ "${_ok}" = true ]; then
  export CP_AB_SKIP_SOURCE=1
  echo "[cp-ab] prepare: /root/.bashrc $((_r1 - _start))s, vendor $((_r2 - _r1))s -> CP_AB_SKIP_SOURCE=1 已导出"
  echo "[cp-ab] env check: ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>} LD_LIBRARY_PATH=${#LD_LIBRARY_PATH}B PYTHONPATH=${PYTHONPATH:-<unset>}"
  echo "[cp-ab] 现在可以直接跑，例如: bash tools/cp_balance_compare/run_cp_diag.sh sweep"
else
  echo "[cp-ab] prepare FAILED: vendor set_env.bash 没找到（launcher 也没探测到），CP_AB_SKIP_SOURCE 未导出" >&2
  echo "[cp-ab]   显式指定后重试: export VENDOR_SET_ENV=/path/to/set_env.bash; source ${BASH_SOURCE[0]}" >&2
  echo "[cp-ab]   或该站点确实不需要 vendor env: export VENDOR_SET_ENV= 后再 source" >&2
fi

unset _here _repo _launcher _vendor _start _r1 _r2 _ok
