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
"""Utilities for the Hunyuan V3 FP8 KV-cache path on Ascend.

The first target is to make the PD-colocated prefill flow correct:
current-token attention keeps BF16 K/V, while cached history is stored as FP8
and dequantized back to BF16 before FIA.
"""

from __future__ import annotations

import math

import torch
import torch_npu

from vllm_ascend.device.device_op import DeviceOperator

_FP8_SCALE_BYTES = 4


def fp8_per_head_scale_elems_padded(
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> int:
    del num_kv_heads
    unit = head_size // math.gcd(block_size, head_size)
    return ((_FP8_SCALE_BYTES + unit - 1) // unit) * unit


def compute_padded_total_rows(
    block_size: int,
    head_size: int,
    pad_head_size: int,
) -> int:
    assert (block_size * pad_head_size) % head_size == 0
    return block_size * pad_head_size // head_size


def _fold_hnd_cache(
    kv_cache_fp8: torch.Tensor,
    block_size: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold physical HND ``[block_size, D_pad]`` storage to rows of ``D``."""
    num_blocks, _, num_kv_heads, pad_head_size = kv_cache_fp8.shape
    pad_total_rows = compute_padded_total_rows(
        block_size, head_size, pad_head_size
    )
    kv_cache_hnd = kv_cache_fp8.permute(0, 2, 1, 3)
    strides = kv_cache_hnd.stride()
    folded_hnd = torch.as_strided(
        kv_cache_hnd,
        (num_blocks, num_kv_heads, pad_total_rows, head_size),
        (strides[0], strides[1], head_size, 1),
    )
    data_hnd = folded_hnd[:, :, :block_size, :]
    scale_hnd = folded_hnd[:, :, block_size:, :]

    scale_bytes_hnd = _flatten_scale_hnd(scale_hnd)
    k_scale_cache = scale_bytes_hnd.view(torch.float32)

    return data_hnd.permute(0, 2, 1, 3), k_scale_cache.permute(0, 2, 1, 3)


def _flatten_scale_hnd(scale_hnd: torch.Tensor) -> torch.Tensor:
    # scale_hnd: (num_blocks, num_kv_heads, scale_rows, head_size)
    # 物理上 scale_rows * head_size 字节 = block_size 个 fp32 scale
    # block_size = scale_rows * head_size // 4
    num_blocks, num_kv_heads, scale_rows, head_size = scale_hnd.shape
    block_size = scale_rows * head_size // 4
    strides = scale_hnd.stride()
    return torch.as_strided(
        scale_hnd,
        (num_blocks, num_kv_heads, block_size, 4),
        (strides[0], strides[1], 4, 1),
    )


def _scatter_k_scale_to_cache(
    scale_storage_hnd: torch.Tensor,
    key_scale: torch.Tensor,
    slots: torch.Tensor,
    block_size: int,
) -> None:
    """Write FP32 K scale raw bytes with npu_scatter_nd_update_."""
    scale_cache_f16 = scale_storage_hnd.view(torch.float16)
    scale_updates = key_scale.contiguous().unsqueeze(-1).view(torch.float16)
    num_tokens, num_kv_heads, scale_half_elems = scale_updates.shape
    assert scale_half_elems == _FP8_SCALE_BYTES // 2

    block_ids = torch.div(slots, block_size, rounding_mode="floor").to(torch.long)
    block_offsets = torch.remainder(slots, block_size).to(torch.long)
    half_offsets = (
        block_offsets.view(-1, 1, 1) * scale_half_elems
        + torch.arange(scale_half_elems, device=slots.device).view(1, 1, -1)
    )
    head_ids = torch.arange(num_kv_heads, device=slots.device).view(1, -1, 1)

    indices = torch.stack(
        (
            block_ids.view(-1, 1, 1).expand(
                num_tokens, num_kv_heads, scale_half_elems
            ),
            head_ids.expand(num_tokens, num_kv_heads, scale_half_elems),
            half_offsets.expand(num_tokens, num_kv_heads, scale_half_elems),
        ),
        dim=-1,
    ).to(torch.int32)

    if hasattr(torch_npu, "npu_scatter_nd_update_"):
        torch_npu.npu_scatter_nd_update_(
            scale_cache_f16, indices, scale_updates
        )
        return

    out = torch_npu.npu_scatter_nd_update(
        scale_cache_f16, indices, scale_updates
    )
    scale_cache_f16.copy_(out)



def _gather_fp8_data_from_cache(
    key_data_cache: torch.Tensor,
    value_data_cache: torch.Tensor,
    block_table: torch.Tensor,
    history_lens: list[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather historical FP8 data from BNBD cache data rows."""
    data_dtype = key_data_cache.dtype
    if key_data_cache.dtype == torch.float8_e4m3fn:
        key_data_cache_new = key_data_cache.view(torch.int8)
        value_data_cache_new = value_data_cache.view(torch.int8)
    else:
        key_data_cache_new = key_data_cache
        value_data_cache_new = value_data_cache
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for seq_idx, history_len in enumerate(history_lens):
        if history_len <= 0:
            continue
        token_offsets = torch.arange(
            history_len, device=block_table.device, dtype=torch.long
        )
        block_offsets = torch.remainder(token_offsets, block_size)
        block_ids = block_table[
            seq_idx, torch.div(token_offsets, block_size, rounding_mode="floor")
        ].to(torch.long)
        key_parts.append(key_data_cache_new[block_ids, block_offsets])
        value_parts.append(value_data_cache_new[block_ids, block_offsets])
        if torch.distributed.get_rank()==0:
            print(f"key_parts is {key_parts},max is{key_parts[0].max()},min is {key_parts[0].min()}")
            print(f"value_parts is {value_parts}, max is {value_parts[0].max()}, min is {value_parts[0].min()}")

    if not key_parts:
        num_kv_heads = key_data_cache.shape[2]
        head_size = key_data_cache.shape[3]
        empty_shape = (0, num_kv_heads, head_size)
        return (
            torch.empty(
                empty_shape,
                dtype=key_data_cache.dtype,
                device=key_data_cache.device,
            ),
            torch.empty(
                empty_shape,
                dtype=value_data_cache.dtype,
                device=value_data_cache.device,
            ),
        )
    return (
            torch.cat(key_parts, dim=0).contiguous().view(torch.float8_e4m3fn) if data_dtype == torch.float8_e4m3fn else torch.cat(key_parts, dim=0).contiguous(),
            torch.cat(value_parts, dim=0).contiguous().view(torch.float8_e4m3fn) if data_dtype == torch.float8_e4m3fn else torch.cat(value_parts, dim=0).contiguous(),
            ) 

def _gather_fp8_data_from_cache_with_pa(
    key_data_cache: torch.Tensor,
    value_data_cache: torch.Tensor,
    block_table: torch.Tensor,
    history_lens: list[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather historical FP8 data from PA cache with npu_gather_pa_kv_cache."""
    total_history_len = sum(history_lens)
    num_kv_heads = key_data_cache.shape[2]
    head_size = key_data_cache.shape[3]
    output_shape = (total_history_len, num_kv_heads, head_size)
    if total_history_len == 0:
        return (
            torch.empty(
                output_shape,
                dtype=key_data_cache.dtype,
                device=key_data_cache.device,
            ),
            torch.empty(
                output_shape,
                dtype=value_data_cache.dtype,
                device=value_data_cache.device,
            ),
        )

    key_data = torch.empty(
        output_shape, dtype=key_data_cache.dtype, device=key_data_cache.device
    )
    value_data = torch.empty(
        output_shape, dtype=value_data_cache.dtype, device=value_data_cache.device
    )
    context_seq_len = torch.tensor(
        history_lens, dtype=torch.int32, device=block_table.device
    )
    seq_offset = torch.zeros_like(context_seq_len)
    torch_npu.npu_gather_pa_kv_cache(
        key_data_cache,
        value_data_cache,
        block_table.to(torch.int32).contiguous(),
        context_seq_len,
        seq_offset=seq_offset,
        key=key_data,
        value=value_data,
    )
    return key_data, value_data


def view_fp8_cache_as_hnd(
    kv_cache_fp8: torch.Tensor,
    block_size: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, _, num_kv_heads, pad_head_size = kv_cache_fp8.shape
    pad_total_rows = compute_padded_total_rows(
        block_size, head_size, pad_head_size
    )
    kv_cache_hnd = kv_cache_fp8.permute(0, 2, 1, 3)
    strides = kv_cache_hnd.stride()
    folded_hnd = torch.as_strided(
        kv_cache_hnd,
        (num_blocks, num_kv_heads, pad_total_rows, head_size),
        (strides[0], strides[1], head_size, 1),
    )
    # First ``block_size`` folded rows are data, the rest carry scale bytes.
    data_hnd = folded_hnd[:, :, :block_size, :]
    scale_hnd = folded_hnd[:, :, block_size:, :]
    
    scale_bytes_hnd = _flatten_scale_hnd(scale_hnd)
    k_scale_cache = scale_bytes_hnd.view(torch.float32)
    
    return data_hnd, k_scale_cache


def _to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float8_e4m3fn:
        return x
    return x.to(torch.float8_e4m3fn)


def dynamic_quant_k_per_token_head(
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric dynamic quantization for K: bf16 -> fp8 + per-token-head scale."""
    key_f32 = key.to(torch.float32)
    row_max = torch.amax(torch.abs(key_f32), dim=-1, keepdim=True)
    scale = torch.clamp(row_max / 127.0, min=torch.finfo(torch.float32).tiny)
    key_q = torch.round(key_f32 / scale).clamp(-127, 127)
    return _to_fp8_e4m3(key_q), scale.squeeze(-1).to(torch.float32)


def static_quant_v_per_head(
    value: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric static quantization for V using per-head scales."""
    scale = v_scale.to(torch.float32).reshape(1, -1, 1)
    value_q = torch.round(value.to(torch.float32) / scale).clamp(-127, 127)
    return _to_fp8_e4m3(value_q)


def write_fp8_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    v_scale: torch.Tensor,
    block_size: int,
    head_size: int,
) -> None:
    """Quantize BF16 K/V and write data plus K scale into FP8 paged cache."""
    key_cache_fp8 = (
        key_cache.view(torch.float8_e4m3fn)
        if key_cache.dtype != torch.float8_e4m3fn
        else key_cache
    )
    value_cache_fp8 = (
        value_cache.view(torch.float8_e4m3fn)
        if value_cache.dtype != torch.float8_e4m3fn
        else value_cache
    )

    slots = slot_mapping
    valid = slots >= 0
    if not torch.all(valid):
        key = key[valid]
        value = value[valid]
        slots = slots[valid]

    key_fp8, key_scale = dynamic_quant_k_per_token_head(key)
    value_fp8 = static_quant_v_per_head(value, v_scale)

    key_data_cache, key_scale_cache_hnd = _fold_hnd_cache(
        key_cache_fp8, block_size, head_size
    )
    value_data_cache, _ = _fold_hnd_cache(value_cache_fp8, block_size, head_size)

    DeviceOperator.reshape_and_cache(
        key=key_fp8,
        value=value_fp8,
        key_cache=key_data_cache,
        value_cache=value_data_cache,
        slot_mapping=slots,
    )

    _scatter_k_scale_to_cache(
        _flatten_scale_hnd(key_scale_cache_hnd),
        key_scale,
        slots,
        block_size,
    )


def gather_dequantize_fp8_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    history_lens: list[int],
    block_size: int,
    head_size: int,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather historical FP8 data from PA cache and dequantize to TND."""
    key_cache_fp8 = (
        key_cache.view(torch.float8_e4m3fn)
        if key_cache.dtype != torch.float8_e4m3fn
        else key_cache
    )
    value_cache_fp8 = (
        value_cache.view(torch.float8_e4m3fn)
        if value_cache.dtype != torch.float8_e4m3fn
        else value_cache
    )

    key_data_cache, key_scale_cache_hnd = _fold_hnd_cache(
        key_cache_fp8, block_size, head_size
    )
    value_data_cache, value_scale_cache_hnd = _fold_hnd_cache(
        value_cache_fp8, block_size, head_size
    )
    key_data, value_data = _gather_fp8_data_from_cache_with_pa(
        key_data_cache,
        value_data_cache,
        block_table,
        history_lens,
        block_size,
    )

    key_scale, _ = _gather_fp8_data_from_cache_with_pa(
        key_scale_cache_hnd,
        value_scale_cache_hnd,
        block_table,
        history_lens,
        block_size,
    )
    value_scale = v_scale.to(torch.float32).reshape(1, -1, 1)
    key_bf16 = (key_data.to(torch.float32) * key_scale).to(out_dtype)
    value_bf16 = (value_data.to(torch.float32) * value_scale).to(out_dtype)
    return key_bf16, value_bf16
