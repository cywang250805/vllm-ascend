import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.msa_m3 import (
    AscendMiniMaxM3IndexerMetadata,
    AscendMiniMaxM3SparseMetadata,
    MiniMaxM3SparseAttention,
)


class TestMiniMaxM3SparseKVCache(unittest.TestCase):
    def _build_layer(self, cache_dtype: torch.dtype) -> MiniMaxM3SparseAttention:
        layer = MiniMaxM3SparseAttention.__new__(MiniMaxM3SparseAttention)
        torch.nn.Module.__init__(layer)
        layer.layer_name = "model.layers.1.self_attn.attn"
        layer.requested_kv_cache_dtype = "fp8"
        layer.kv_cache_dtype = "bfloat16"
        layer.kv_cache_torch_dtype = torch.bfloat16
        layer.num_kv_heads = 2
        layer.head_dim = 4
        layer.idx_head_dim = 3
        layer.kv_cache = (
            torch.zeros((1, 2, 16, 4), dtype=cache_dtype),
            torch.zeros((1, 2, 16, 4), dtype=cache_dtype),
        )
        indexer_layer_name = f"{layer.layer_name}.indexer"
        layer.indexer = SimpleNamespace(
            index_cache=SimpleNamespace(
                prefix=indexer_layer_name,
                kv_cache=torch.zeros((1, 16, 1, 3), dtype=torch.bfloat16),
            )
        )
        return layer

    def _build_forward_context(
        self,
        layer: MiniMaxM3SparseAttention,
    ) -> SimpleNamespace:
        slot_mapping = torch.tensor([0], dtype=torch.int32)
        common_fields = {
            "seq_lens": torch.tensor([1], dtype=torch.int32),
            "max_seq_len": 1,
            "slot_mapping": slot_mapping,
            "num_actual_tokens": 1,
            "num_decodes": 0,
            "num_decode_tokens": 0,
            "num_prefills": 1,
            "num_prefill_tokens": 1,
        }
        return SimpleNamespace(
            attn_metadata={
                layer.layer_name: AscendMiniMaxM3SparseMetadata(**common_fields),
                layer.indexer.index_cache.prefix: AscendMiniMaxM3IndexerMetadata(
                    **common_fields
                ),
            }
        )

    @patch("vllm_ascend.attention.msa_m3.log_gqa_kv_fp8")
    @patch(
        "vllm_ascend.attention.msa_m3.claim_gqa_kv_fp8_debug_event",
        return_value=True,
    )
    @patch("vllm_ascend.attention.msa_m3.get_forward_context")
    @patch(
        "vllm_ascend.device.device_op.DeviceOperator.reshape_and_cache",
    )
    def test_insert_kv_uses_matching_bfloat16_input_and_cache(
        self,
        mock_reshape_and_cache,
        mock_get_forward_context,
        _mock_claim_debug_event,
        mock_log,
    ):
        layer = self._build_layer(torch.bfloat16)
        mock_get_forward_context.return_value = self._build_forward_context(layer)
        key = torch.ones((1, 8), dtype=torch.bfloat16)
        value = torch.ones((1, 8), dtype=torch.bfloat16)
        index_key = torch.ones((1, 3), dtype=torch.bfloat16)

        layer._insert_kv(key, value, index_key)

        args = mock_reshape_and_cache.call_args.args
        self.assertEqual(args[0].dtype, torch.bfloat16)
        self.assertEqual(args[1].dtype, torch.bfloat16)
        self.assertEqual(args[2].dtype, torch.bfloat16)
        self.assertEqual(args[3].dtype, torch.bfloat16)
        self.assertEqual(args[0].dtype, args[2].dtype)
        self.assertEqual(args[1].dtype, args[3].dtype)
        self.assertIn("event=sparse_cache_insert", mock_log.call_args.args[0])
        self.assertIn(layer.layer_name, mock_log.call_args.args[0])

    @patch("vllm_ascend.attention.msa_m3.get_forward_context")
    @patch(
        "vllm_ascend.device.device_op.DeviceOperator.reshape_and_cache",
        side_effect=RuntimeError("dtype mismatch"),
    )
    def test_insert_kv_error_includes_layer_and_dtypes(
        self,
        _mock_reshape_and_cache,
        mock_get_forward_context,
    ):
        layer = self._build_layer(torch.uint8)
        mock_get_forward_context.return_value = self._build_forward_context(layer)
        key = torch.ones((1, 8), dtype=torch.bfloat16)
        value = torch.ones((1, 8), dtype=torch.bfloat16)
        index_key = torch.ones((1, 3), dtype=torch.bfloat16)

        with self.assertRaisesRegex(
            RuntimeError,
            "event=sparse_cache_insert_error",
        ) as exc_info:
            layer._insert_kv(key, value, index_key)

        message = str(exc_info.exception)
        self.assertIn(layer.layer_name, message)
        self.assertIn("key_input=(1, 2, 4)/torch.bfloat16", message)
        self.assertIn("key_cache=(1, 2, 16, 4)/torch.uint8", message)


if __name__ == "__main__":
    unittest.main()
