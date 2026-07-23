#
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

import os

import torch.distributed as dist
from vllm.logger import init_logger

import vllm_ascend.envs as envs

logger = init_logger(__name__)

_LOGGED_EVENTS: set[str] = set()


def _is_rank_zero() -> bool:
    """Use global communication rank when available, with a startup fallback."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    rank = os.getenv("RANK")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return False


def claim_gqa_kv_fp8_debug_event(event: str) -> bool:
    """Claim a one-time diagnostic event on global rank 0.

    Callers must claim an event before collecting tensor samples. This avoids
    device synchronization and host copies on nonzero ranks and repeated
    forward calls.
    """
    if not envs.VLLM_ASCEND_GQA_KV_FP8_DEBUG or not _is_rank_zero() or event in _LOGGED_EVENTS:
        return False
    _LOGGED_EVENTS.add(event)
    return True


def log_gqa_kv_fp8(message: str) -> None:
    """Log a diagnostic message after its event has been claimed."""
    logger.info("[MiniMax M3 GQA KV FP8] %s", message)


def log_gqa_kv_fp8_once(event: str, message: str) -> None:
    """Log one diagnostic event on global rank 0 only."""
    if claim_gqa_kv_fp8_debug_event(event):
        log_gqa_kv_fp8(message)
