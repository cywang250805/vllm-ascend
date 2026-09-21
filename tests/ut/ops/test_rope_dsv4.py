# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops import rope_dsv4 as rope


@pytest.fixture
def state(monkeypatch):
    state = rope.RopeGlobalState()
    state.layer_info = {
        "draft.0": ("normal", ["default"]),
        "draft.1": ("normal", ["default"]),
        "target": ("compressed", ["default", "c4"]),
    }
    for index, (key, groups) in enumerate([("normal", {"default"}), ("compressed", {"default", "c4"})]):
        state.registry_summary[key] = groups
        table = torch.arange(64, dtype=torch.float32).reshape(16, 1, 1, 4) + index * 100
        state.full_rope_cache[key] = (table, -table)
        state.runtime_buffer[key] = {group: (torch.zeros_like(table), torch.zeros_like(table)) for group in groups}
        state.spec_runtime_buffer[key] = {
            group: ([torch.zeros_like(table) for _ in range(2)], [torch.zeros_like(table) for _ in range(2)])
            for group in groups
        }
    monkeypatch.setattr(rope, "_ROPE_STATE", state)
    return state


@pytest.mark.parametrize("use_cache,draft_index", [(False, None), (True, None), (True, 1), (True, 2)])
def test_filtered_lookup_matches_all_configs_and_deduplicates(state, use_cache, draft_index):
    positions = torch.tensor([4, 1, 4, 7])
    expected = rope.get_cos_and_sin_dsa(positions, use_cache, draft_index)
    normal = state.full_rope_cache["normal"]
    counted = tuple(MagicMock() for _ in range(2))
    for mock, table in zip(counted, normal):
        mock.__getitem__.side_effect = table.__getitem__
    state.full_rope_cache["normal"] = counted
    unused = tuple(MagicMock() for _ in range(2))
    state.full_rope_cache["compressed"] = unused

    actual = rope.get_cos_and_sin_dsa(positions, use_cache, draft_index, layer_names=["draft.0", "draft.1"])
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result["draft.0"], reference["draft.0"])
        assert result["draft.0"].data_ptr() == result["draft.1"].data_ptr()
    for mock in counted:
        mock.__getitem__.assert_called_once()
    for mock in unused:
        mock.__getitem__.assert_not_called()


def test_union_of_layer_groups_and_position_maps(state):
    positions = {"default": torch.tensor([1, 2]), "c4": torch.tensor([3])}
    cos, sin = rope.get_cos_and_sin_dsa(positions, layer_names=["draft.0", "target"])
    torch.testing.assert_close(cos["draft.0"], state.full_rope_cache["normal"][0][positions["default"]])
    for group, indices in positions.items():
        torch.testing.assert_close(sin["target"][group], state.full_rope_cache["compressed"][1][indices])


def test_filter_does_not_modify_registry_and_rejects_unknown_layer(state):
    rope.get_cos_and_sin_dsa(torch.tensor([], dtype=torch.long), layer_names=[])
    assert set(state.registry_summary) == {"normal", "compressed"}
    with pytest.raises(KeyError, match="not registered"):
        rope.get_cos_and_sin_dsa(torch.tensor([0]), layer_names=["unknown"])


def test_speculative_steps_have_distinct_stable_buffers(state):
    first, _ = rope.get_cos_and_sin_dsa(torch.tensor([1, 2]), True, 1, ["draft.0"])
    saved = first["draft.0"].clone()
    second, _ = rope.get_cos_and_sin_dsa(torch.tensor([3, 4]), True, 2, ["draft.0"])
    assert first["draft.0"].data_ptr() != second["draft.0"].data_ptr()
    torch.testing.assert_close(first["draft.0"], saved)
    # Uncached context-KV lookups must not overwrite attention runtime buffers.
    rope.get_cos_and_sin_dsa(torch.tensor([5, 6]), layer_names=["draft.0"])
    torch.testing.assert_close(first["draft.0"], saved)
