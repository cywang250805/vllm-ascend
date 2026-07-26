# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump the inputs/outputs of the fused-infer-attention operator for a single
target dense attention layer.

Controlled by environment variables:
  VLLM_ASCEND_DUMP_FIA_LAYER  : target decoder layer index (e.g. "0")
  VLLM_ASCEND_DUMP_FIA_DIR    : output directory (default /tmp/fia_dump)

For each rank the first forward of the target layer saves one file per mode
("bf16" uses npu_fused_infer_attention_score, "fp8" uses ..._v2):
  <dir>/fia_<mode>_layer<idx>_rank<rank>.pt
"""
import os

import torch
import torch.distributed as dist

_TARGET_LAYER = os.getenv("VLLM_ASCEND_DUMP_FIA_LAYER", "")
_DUMP_DIR = os.getenv("VLLM_ASCEND_DUMP_FIA_DIR", "/tmp/fia_dump")
_ENABLED = bool(_TARGET_LAYER)
_DUMPED: set[str] = set()


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    raw = os.getenv("RANK")
    try:
        return int(raw) if raw is not None else 0
    except ValueError:
        return 0


def should_dump(layer_name: str) -> bool:
    if not _ENABLED:
        return False
    return f"layers.{_TARGET_LAYER}." in layer_name


def _to_cpu(val):
    if torch.is_tensor(val):
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        return val.detach().cpu().clone()
    return val


def save_fia_io(
    mode: str,
    layer_name: str,
    inputs: dict,
    output,
    meta: dict,
) -> None:
    key = f"{mode}_layer{_TARGET_LAYER}_rank{_rank()}"
    if key in _DUMPED:
        return
    _DUMPED.add(key)
    os.makedirs(_DUMP_DIR, exist_ok=True)
    path = os.path.join(_DUMP_DIR, f"fia_{key}.pt")
    payload = {
        "mode": mode,
        "layer_name": layer_name,
        "rank": _rank(),
        "inputs": {k: _to_cpu(v) for k, v in inputs.items()},
        "output": _to_cpu(output),
        "meta": {k: _to_cpu(v) for k, v in meta.items()},
    }
    torch.save(payload, path)
    print(f"[fia_dump] saved {mode} layer {_TARGET_LAYER} rank {_rank()} -> {path}",
          flush=True)
