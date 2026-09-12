#!/usr/bin/env bash
# 站点档案（site profiles）—— 一次 export 就能换测试节点。
#
#   export CP_AB_SITE=its      # A3 站点 7.246.78.75（16 卡）—— 默认
#   export CP_AB_SITE=share    # 当前站点 141.61.133.104（8 卡）
#   unset CP_AB_SITE           # 等价于默认档案
#
# 语义（两条硬规则）：
#   1. **显式 export 的变量优先**：档案只用 `:=` 填默认值，所以临时换权重/端口/卡数
#      直接 export 即可（`export MODEL_PATH=…`），档案不会覆盖它；
#   2. 档案是**唯一**的站点参数来源：launcher 与 run_cp_diag.sh 都 source 本文件，
#      避免"两边各写一份、改一处忘一处"（43a8b336d 就是这样漏掉 TP/可见卡的）。
#
# 用法（脚本里）：
#   source "$(dirname "$0")/sites.sh"; cp_ab_site_apply "${CP_AB_SITE:-}"
#
# 站点参数来自 git 历史：`a5507f5e3`（A3 站点）与 `43a8b336d`（切回 141 站点，
# 本文件把那次"反向改回"固化成可切换的两份档案，而不是再改一次默认值）。

CP_AB_SITE_DEFAULT="${CP_AB_SITE_DEFAULT:-its}"

cp_ab_site_list() {
  printf '%s\n' its share
}

cp_ab_site_apply() {
  local name="${1:-${CP_AB_SITE_DEFAULT}}"
  case "${name}" in
    its)
      # A3 站点：/opt/its 挂载点、16 卡；`a5507f5e3` 的 SITE 段原样恢复。
      # ⚠️ IP 用现场值 **7.246.78.75**：git 历史里写的是 7.246.78.76（末位 6），
      #    现场确认末位是 5；别再照历史"改回去"。
      # ⚠️ 历史档案用的权重是 GLM-5.2-W4A8C8；若该机器上挂的是 w4a4c8-mxfp4，
      #    用 `export MODEL_PATH=/opt/its/model/GLM-5.2-w4a4c8-mxfp4` 覆盖（档案不会挡）。
      : "${NIC_NAME:=eth2}"
      : "${LOCAL_IP:=7.246.78.75}"
      : "${VLLM_ASCEND_REPO:=/opt/its/z30055003/vllm-ascend}"
      : "${MODEL_PATH:=/opt/its/model/GLM-5.2-W4A8C8}"
      : "${PROFILER_DIR:=/opt/its/z30055003/profiling_no_pooling}"
      : "${TP_SIZE:=16}"
      : "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
      # A3 上历来不 source vendor set_env（a5507f5e3 就是显式跳过）；
      # 需要时 `export VENDOR_SET_ENV=/path/to/set_env.bash` 覆盖。
      : "${SITE_VENDOR_SET_ENV:=none}"
      ;;
    share)
      # 当前站点（141.61.133.104，8 卡，w4a4c8-mxfp4 权重）：`43a8b336d` 的 SITE 段。
      : "${NIC_NAME:=eth2}"
      : "${LOCAL_IP:=141.61.133.104}"
      : "${VLLM_ASCEND_REPO:=/home/z30055003/vllm-ascend}"
      : "${MODEL_PATH:=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4}"
      : "${PROFILER_DIR:=/home/z30055003/profiling_no_pooling}"
      : "${TP_SIZE:=8}"
      : "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
      : "${SITE_VENDOR_SET_ENV:=auto}"
      ;;
    *)
      echo "[sites] 未知站点 CP_AB_SITE='${name}'；可选：$(cp_ab_site_list | tr '\n' ' ')" >&2
      return 2
      ;;
  esac

  # TP/可见卡必须导出：run_cp_diag.sh 用 TP_SIZE 推 cp_size，driver 再把它透传给
  # launcher（两边读同一个值，才不会出现"服务是 TP=16、driver 以为是 8"）。
  export NIC_NAME LOCAL_IP VLLM_ASCEND_REPO MODEL_PATH PROFILER_DIR
  export TP_SIZE ASCEND_RT_VISIBLE_DEVICES SITE_VENDOR_SET_ENV
  export CP_AB_SITE_RESOLVED="${name}"
  return 0
}
