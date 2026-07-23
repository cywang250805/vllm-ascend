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
"""NPU fused RoPE + QK-Norm + KV-Cache-Write + FP8 Q quant."""

from __future__ import annotations

import os
import time

import torch
import torch_npu

from vllm.config.cache import KVCacheQuantConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.kv_cache_fp8 import view_fp8_cache_as_hnd

logger = init_logger(__name__)


def _debug_seq_lens_enabled() -> bool:
    return os.environ.get("HCF_DEBUG_SEQ_LENS", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _debug_seq_lens_compare_enabled() -> bool:
    return os.environ.get("HCF_DEBUG_SEQ_LENS_COMPARE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _debug_seq_lens_device_enabled() -> bool:
    return os.environ.get("HCF_DEBUG_SEQ_LENS_DEVICE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _debug_rank() -> str:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))


class NpuRopeNorm(torch.nn.Module):
    """Hunyuan V3 NPU equivalent of GPU HpcRopeNorm.

    The fused operator owns QK-Norm, RoPE, KV cache write, and FP8 Q/K/V
    quantization. The normal attention backend then consumes the processed
    query and the populated KV cache.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cos_sin_cache: torch.Tensor,
        use_qk_norm: bool,
        fallback_qnorm: torch.nn.Module | None,
        fallback_knorm: torch.nn.Module | None,
        kv_cache_dtype: str,
        qk_norm_policy: int = 1,
        enable_hadamard: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.cos_sin_cache = cos_sin_cache.float()
        self.fallback_qnorm = fallback_qnorm
        self.fallback_knorm = fallback_knorm
        self.head_per_group = num_heads // num_kv_heads
        self.qnorm_weight: torch.Tensor | None = None
        self.knorm_weight: torch.Tensor | None = None
        self.q_rotation: torch.Tensor | None = None
        self.k_rotation: torch.Tensor | None = None
        self.qk_norm_policy = qk_norm_policy
        self.enable_hadamard = enable_hadamard
        self.use_fp8 = "fp8" in kv_cache_dtype
        self.layer_name: str | None = None
        self._kv_cache_quant_config: KVCacheQuantConfig | None = None
        self._quant_type: int | None = None
        self.rms_norm_eps = 1e-6
        if fallback_qnorm is not None:
            self.rms_norm_eps = getattr(fallback_qnorm, "variance_epsilon", 1e-6)
        if self.use_fp8:
            self._resolve_quant_config()

    @classmethod
    def support(
        cls,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: str,
    ) -> bool:
        if kv_cache_dtype not in ("fp8_e4m3", "fp8", "bfloat16", "auto"):
            return False
        if head_dim != 128:
            return False
        head_per_group = num_heads // num_kv_heads
        return head_per_group in (4, 8)

    def _resolve_quant_config(self) -> None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.cache_config is not None:
            self._kv_cache_quant_config = (
                vllm_config.cache_config.kv_cache_quant_config
            )

        if self._kv_cache_quant_config is None:
            from vllm.config.cache import KVCacheQuantConfig, KVQuantSpec

            self._kv_cache_quant_config = KVCacheQuantConfig(
                k_quant=KVQuantSpec(
                    dtype="fp8_e4m3",
                    granularity="per_token_per_head",
                ),
                v_quant=KVQuantSpec(dtype="fp8_e4m3", granularity="per_head"),
            )

        # Policy 1 maps to Q per-token-per-head, K per-token-per-head, V per-head
        # in the current NPU fused operator prototype.
        self._quant_type = 1

    def process_weights_after_loading(self, act_dtype: torch.dtype | None = None) -> None:
        self._ensure_weights_ready(self.cos_sin_cache.device)

    def _ensure_weights_ready(self, device: torch.device) -> None:
        """Idempotently materialize rotation matrices and norm weights.

        ``process_weights_after_loading`` is not reliably invoked for this
        module by the loader, so ``forward`` also calls this lazily on the
        first step to make sure the fused operator's required tensors exist.
        """
        if self.q_rotation is not None:
            return
        # The fused operator requires q/k rotation matrices. Hadamard rotation
        # is not enabled yet, so feed an identity matrix (i.e. no rotation).
        if self.enable_hadamard:
            from scipy.linalg import hadamard
            import math
            H = torch.tensor(
                hadamard(self.head_dim), dtype=torch.bfloat16, device=device
            ) / math.sqrt(self.head_dim)
            self.q_rotation = H
            self.k_rotation = H
        else:
            self.q_rotation = torch.eye(
                self.head_dim, dtype=torch.bfloat16, device=device
            )
            self.k_rotation = torch.eye(
                self.head_dim, dtype=torch.bfloat16, device=device
            )
        if not self.use_qk_norm:
            return

        if self.fallback_qnorm is not None:
            self.qnorm_weight = self.fallback_qnorm.weight.detach().float()
        if self.fallback_knorm is not None:
            self.knorm_weight = self.fallback_knorm.weight.detach().float()

    def _get_v_scale(self, layer) -> torch.Tensor:
        """Per-head V scale as a 1-D ``[num_kv_heads]`` fp32 tensor.

        V scale 取自 ``layer.fa_v.scale``（与 attention_v1.py 中
        ``self._current_v_scale = layer.fa_v.scale`` 同源），它是量化"除数"
        语义(quant: v / scale)；而融合算子把 v_scale 当作"乘数"
        (quant: v * v_scale, 见 golden.py)，故取倒数。
        """
        v_scale = layer.fa_v.scale.to(torch.float32).reshape(-1)
        if v_scale.numel() == 1:
            v_scale = v_scale.expand(self.num_kv_heads).contiguous()
        # 融合算子内部用乘法出fp8，对应这里是除法出fp8
        v_scale = torch.reciprocal(
            v_scale.clamp(min=torch.finfo(torch.float32).tiny)
        )
        return v_scale.contiguous()

    def register_layer_name(self, layer_name: str) -> None:
        self.layer_name = layer_name

    def forward(
        self, qkv: torch.Tensor, layer_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.layer_name = layer_name
        self._ensure_weights_ready(qkv.device)
        result = self._forward_impl(qkv)
        return result

    def _get_kv_cache(self):
        forward_context = get_forward_context()
        assert self.layer_name is not None
        attn_layer = forward_context.no_compile_layers[self.layer_name]
        kv_cache = attn_layer.kv_cache[0]
        if isinstance(kv_cache, (tuple, list)):
            return attn_layer, kv_cache[0], kv_cache[1]
        return attn_layer, kv_cache[0], kv_cache[1]

    def _get_kv_scales(
        self,
        layer,
        k_scale_from_cache: torch.Tensor | None,
        v_scale_from_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_qcfg = self._kv_cache_quant_config
        if kv_qcfg is None:
            return layer._k_scale.reshape(1), layer._v_scale.reshape(1)

        if kv_qcfg.k_quant is not None and kv_qcfg.k_quant.granularity == "per_token_per_head":
            k_scale = (
                k_scale_from_cache.view(torch.float32)
                if k_scale_from_cache is not None
                else layer._k_scale
            )
        elif kv_qcfg.k_quant is not None and kv_qcfg.k_quant.granularity == "per_head":
            k_scale = layer._k_scale
        else:
            k_scale = layer._k_scale.reshape(1)

        if kv_qcfg.v_quant is not None and kv_qcfg.v_quant.granularity == "per_token_per_head":
            v_scale = (
                v_scale_from_cache.view(torch.float32)
                if v_scale_from_cache is not None
                else layer._v_scale
            )
        elif kv_qcfg.v_quant is not None and kv_qcfg.v_quant.granularity == "per_head":
            v_scale = layer._v_scale
        else:
            v_scale = layer._v_scale.reshape(1)
        return k_scale, v_scale

    def _zero_outputs(
            self, qkv: torch.Tensor, num_tokens: int | None = None
          ) -> tuple[torch.Tensor, torch.Tensor]:
          """Dummy/None 场景下的合法零返回，形状与正常输出一致。

          q_out: [T, num_heads, head_dim] fp8；q_scale: [T, num_heads] fp32。
          """
          t = qkv.shape[0] if num_tokens is None else num_tokens
          q_out = torch.zeros(
              (t, self.num_heads, self.head_dim),
              dtype=torch.float8_e4m3fn,
              device=qkv.device,
          )
          q_scale = torch.zeros(
              (t, self.num_heads), dtype=torch.float32, device=qkv.device
          )
          return q_out, q_scale

    def _forward_impl(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            assert self.layer_name is not None
            attn_metadata = attn_metadata[self.layer_name]

        if attn_metadata is None:
            return self._zero_outputs(qkv)

        attn_layer, key_cache, value_cache = self._get_kv_cache()
        if key_cache.numel() == 0:
            return self._zero_outputs(qkv)

        total_tokens = qkv.shape[0]
        num_actual_tokens = attn_metadata.num_actual_tokens
        qkv = qkv[:num_actual_tokens]
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
        # Logical cache shape is [num_blocks, block_size, num_kv_heads,
        # pad_head_size]; block_size is the second dim.
        block_size = key_cache.shape[1]
        k_cache_hnd, k_scale_cache = view_fp8_cache_as_hnd(
            key_cache_fp8, block_size, self.head_dim
        )
        v_cache_hnd, _ = view_fp8_cache_as_hnd(
            value_cache_fp8, block_size, self.head_dim
        )

        qkv_3d = qkv.view(
            -1, self.num_heads + 2 * self.num_kv_heads, self.head_dim
        )

        head_nums = [self.num_heads, self.num_kv_heads, self.num_kv_heads]
        if _debug_seq_lens_enabled():
            src_seq_lens = attn_metadata.seq_lens
            src_seq_lens_list = (
                src_seq_lens.tolist()
                if src_seq_lens.device.type == "cpu"
                else src_seq_lens.cpu().tolist()
            )
            print(
                f"[ROPE_SEQ_SRC] t={time.time():.6f} rank={_debug_rank()} "
                f"step={getattr(attn_metadata, 'debug_step_id', -1)} "
                f"layer={self.layer_name} "
                f"num_actual_tokens={num_actual_tokens} "
                f"seq_device={src_seq_lens.device} "
                f"seq_shape={tuple(src_seq_lens.shape)} "
                f"seq_ptr={src_seq_lens.data_ptr()} "
                f"seq={src_seq_lens_list}",
                flush=True,
            )
            if _debug_seq_lens_compare_enabled():
                unsafe_seq_lens = src_seq_lens.pin_memory().npu(
                    non_blocking=True
                )
                safe_seq_lens_cpu = src_seq_lens.detach().clone().pin_memory()
                safe_seq_lens = safe_seq_lens_cpu.npu(non_blocking=True)
                torch_npu.synchronize()
                unsafe_seq_lens_cpu = unsafe_seq_lens.cpu()
                safe_seq_lens_cpu_after = safe_seq_lens.cpu()
                print(
                    f"[ROPE_SEQ_CMP] t={time.time():.6f} rank={_debug_rank()} "
                    f"step={getattr(attn_metadata, 'debug_step_id', -1)} "
                    f"layer={self.layer_name} "
                    f"src={src_seq_lens_list} "
                    f"unsafe={unsafe_seq_lens_cpu.tolist()} "
                    f"safe={safe_seq_lens_cpu_after.tolist()} "
                    f"unsafe_eq_safe={torch.equal(unsafe_seq_lens_cpu, safe_seq_lens_cpu_after)}",
                    flush=True,
                )
        seq_lens_for_op = None
        pinned_seq_lens_for_op = None
        if _debug_seq_lens_device_enabled():
            pinned_seq_lens_for_op = attn_metadata.seq_lens.pin_memory()
            seq_lens_for_op = pinned_seq_lens_for_op.npu(non_blocking=True)
        q_out, q_scale = torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
            qkv_3d,
            self.qnorm_weight,
            self.knorm_weight,
            self.cos_sin_cache,
            attn_metadata.slot_mapping[:num_actual_tokens],
            k_cache_hnd,
            v_cache_hnd,
            k_scale_cache,
            attn_metadata.query_start_loc,
            seq_lens_for_op
            if seq_lens_for_op is not None
            else attn_metadata.seq_lens.pin_memory().npu(non_blocking=True),
            head_nums,
            q_rotation=self.q_rotation,
            k_rotation=self.k_rotation,
            v_scale=self._get_v_scale(attn_layer),
            epsilon=self.rms_norm_eps,
        )
        if _debug_seq_lens_device_enabled():
            assert seq_lens_for_op is not None
            assert pinned_seq_lens_for_op is not None
            seq_lens_for_op_cpu = seq_lens_for_op.cpu()
            src_seq_lens_now = attn_metadata.seq_lens
            src_seq_lens_now_list = (
                src_seq_lens_now.tolist()
                if src_seq_lens_now.device.type == "cpu"
                else src_seq_lens_now.cpu().tolist()
            )
            print(
                f"[ROPE_SEQ_DEV] t={time.time():.6f} rank={_debug_rank()} "
                f"step={getattr(attn_metadata, 'debug_step_id', -1)} "
                f"layer={self.layer_name} "
                f"src_now={src_seq_lens_now_list} "
                f"pinned={pinned_seq_lens_for_op.tolist()} "
                f"device={seq_lens_for_op_cpu.tolist()} "
                f"src_ptr={src_seq_lens_now.data_ptr()} "
                f"pinned_ptr={pinned_seq_lens_for_op.data_ptr()} "
                f"device_ptr={seq_lens_for_op.data_ptr()}",
                flush=True,
            )
        # TPSP: pad back to full length for reduce_scatter alignment
        if q_out.shape[0] < total_tokens:
            full_q = torch.zeros(
                (total_tokens, q_out.shape[1], q_out.shape[2]),
                dtype=q_out.dtype, device=q_out.device,
            )
            full_q[:num_actual_tokens] = q_out
            full_scale = torch.zeros(
                (total_tokens, q_scale.shape[1]),
                dtype=q_scale.dtype, device=q_scale.device,
            )
            full_scale[:num_actual_tokens] = q_scale
            return full_q, full_scale
        return q_out, q_scale
