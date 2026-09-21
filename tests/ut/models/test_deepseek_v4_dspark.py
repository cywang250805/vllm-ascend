# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.deepseek_v4_dspark import DeepseekV4DSparkModel, _apply_dsv4_rope


@pytest.mark.parametrize("num_layers", [1, 3, 4])
def test_context_kv_queries_all_layer_configs_once(num_layers):
    names = [f"mtp.{index}.self_attn.attn" for index in range(num_layers)]
    model = SimpleNamespace(
        layers={
            name: SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=SimpleNamespace(layername=name)))
            for name in names
        },
        _project_shared_kv=MagicMock(),
        _store_standard_swa_kv=MagicMock(),
    )
    states, positions = torch.ones(4, 8), torch.arange(4)
    rope = (object(), object())
    with patch("vllm_ascend.models.deepseek_v4_dspark.get_cos_and_sin_dsa", return_value=rope) as lookup:
        DeepseekV4DSparkModel.precompute_and_store_context_kv(model, states, positions, [positions] * num_layers)
    lookup.assert_called_once_with(positions, layer_names=names)
    assert model._project_shared_kv.call_count == num_layers
    assert all(call.kwargs["rope"] is rope for call in model._project_shared_kv.call_args_list)
    assert model._store_standard_swa_kv.call_count == num_layers


@pytest.mark.parametrize("inverse", [False, True])
def test_apply_rope_reuses_lookup_and_preserves_inverse(inverse):
    rotary = MagicMock(layername="draft")
    cos, sin = torch.ones(2, 4), torch.full((2, 4), 0.5)
    x, positions = torch.zeros(2, 4), torch.arange(2)
    with patch("vllm_ascend.models.deepseek_v4_dspark.get_cos_and_sin_dsa") as lookup:
        _apply_dsv4_rope(rotary, positions, x, inverse=inverse, rope=({"draft": cos}, {"draft": sin}))
    lookup.assert_not_called()
    torch.testing.assert_close(rotary.call_args.args[2], -sin if inverse else sin)
    torch.testing.assert_close(sin, torch.full((2, 4), 0.5))


def test_empty_context_skips_lookup():
    with patch("vllm_ascend.models.deepseek_v4_dspark.get_cos_and_sin_dsa") as lookup:
        DeepseekV4DSparkModel.precompute_and_store_context_kv(None, torch.empty(0), torch.empty(0), [])
    lookup.assert_not_called()
