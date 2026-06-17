#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""
Proxy entry point for split attn/moe cross-group communication.
This module forwards all exports to the compiled binary module.
"""

from .split_attn_moe_communicator_core import (
    init_cross_group,
    init_p2p_groups,
    recv_from_attn,
    recv_from_moe,
    send_to_attn,
    send_to_moe,
)

__all__ = [
    "init_cross_group",
    "init_p2p_groups",
    "send_to_moe",
    "recv_from_attn",
    "send_to_attn",
    "recv_from_moe",
]
