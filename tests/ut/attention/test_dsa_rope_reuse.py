# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.attention.test_dsa_v1 import (
    _make_decode_builder,
    _make_dspark_common_metadata,
    _make_dspark_draft_builder,
)
from vllm_ascend.device.device_op import DeviceOperator


@pytest.mark.parametrize("padded_tokens", [6, 8])
@pytest.mark.parametrize("mixed", [False, True])
def test_draft_reuses_only_pure_decode_rope(padded_tokens, mixed):
    builder = _make_dspark_draft_builder()
    builder.metadata_cls = SimpleNamespace
    builder.decode_threshold = 4
    builder._device_metadata_enabled = False
    common = _make_dspark_common_metadata(2, padded_tokens)
    common.num_input_tokens = padded_tokens
    common.num_actual_tokens = 6
    common.slot_mapping = torch.zeros(padded_tokens, dtype=torch.int32)
    common.attn_state = None
    cos, sin = torch.arange(padded_tokens), -torch.arange(padded_tokens)
    split = (1, 1, 3, 3) if mixed else (2, 0, 6, 0)
    with (
        patch("vllm_ascend.attention.dsa_v1.split_decodes_and_prefills", return_value=split),
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size", return_value=1),
        patch("vllm_ascend.attention.dsa_v1.get_cos_and_sin_dsa", return_value=(cos, sin)) as lookup,
        patch.object(builder, "build_prefill_metadata_for_drafting", return_value=object()),
        patch.object(DeviceOperator, "format_dsa_slot_mapping", return_value=torch.zeros((padded_tokens, 2))),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", return_value={}),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_op",
            return_value=MagicMock(return_value=torch.zeros(1024, dtype=torch.int32)),
        ),
        patch("vllm_ascend.attention.dsa_v1.build_dspark_swa_indices", return_value=(torch.zeros(8, 1, 4), None)),
    ):
        metadata = builder.build_for_drafting(common, draft_index=1)
    assert lookup.call_count == (2 if mixed else 1)
    assert all(call.kwargs["layer_names"] == builder.rope_layer_names for call in lookup.call_args_list)
    if not mixed:
        assert metadata.decode.cos.shape[0] == 6
        assert metadata.decode.cos.data_ptr() == metadata.cos.data_ptr()
        torch.testing.assert_close(metadata.decode.sin, sin[:6])


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("padded_tokens", [2, 8])
def test_target_reuses_common_decode_prefix(mixed, padded_tokens):
    builder = _make_decode_builder(1, False)
    builder.num_prefills = int(mixed)
    builder.decode_ratio_to_sas_metadata = {}
    cos, sin = torch.arange(padded_tokens), -torch.arange(padded_tokens)
    builder.common_ratio_to_sas_metadata = {"cos": cos, "sin": sin}
    common = SimpleNamespace(
        positions=torch.arange(padded_tokens),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([8, 9], dtype=torch.int32),
    )
    with (
        patch("vllm_ascend.attention.dsa_v1.get_cos_and_sin_dsa", return_value=(cos[:2], sin[:2])) as lookup,
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size", return_value=1),
        patch.object(DeviceOperator, "pad_dsa_decode_slot_mapping", return_value=builder.slot_mapping),
        patch.object(DeviceOperator, "get_dsa_decode_cu_seqlens_ori_kv", return_value=torch.tensor([0, 8, 17])),
        patch.object(DeviceOperator, "get_dsa_decode_cu_seqlens_cmp_kv", return_value=None),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", return_value={}),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_op",
            return_value=MagicMock(return_value=torch.zeros(1024, dtype=torch.int32)),
        ),
    ):
        metadata = builder.build_decode_metadata(0, common, 2)
    assert lookup.call_count == int(mixed)
    torch.testing.assert_close(metadata.cos, cos[:2])
    if not mixed:
        assert metadata.cos.data_ptr() == cos.data_ptr()
