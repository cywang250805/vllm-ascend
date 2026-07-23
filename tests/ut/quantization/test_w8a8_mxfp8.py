from unittest.mock import Mock, patch

import torch
from vllm.config import CompilationMode

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod


class TestAscendW8A8MXFP8FusedMoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.ensure_mxfp8_moe_available")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_ep_group")
    def setUp(self, mock_get_ep_group, mock_get_ascend_config, mock_ensure_mxfp8_moe_available):
        with patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_current_vllm_config") as mock_config:
            mock_vllm_config = Mock()
            mock_vllm_config.quant_config = Mock(quant_description={"group_size": 32})
            mock_vllm_config.compilation_config = Mock(mode=CompilationMode.NONE)
            mock_vllm_config.model_config = Mock(enforce_eager=True)
            mock_config.return_value = mock_vllm_config

            mock_ascend_config = Mock()
            mock_ascend_config.eplb_config = Mock(dynamic_eplb=False)
            mock_ascend_config.multistream_overlap_gate = False
            mock_get_ascend_config.return_value = mock_ascend_config
            mock_get_ep_group.return_value = Mock()

            self.quant_method = AscendW8A8MXFP8DynamicFusedMoEMethod()

    def build_layer(self):
        layer = torch.nn.Module()
        layer.w13_weight = torch.empty(
            self.num_experts,
            2 * self.intermediate_size,
            self.hidden_size,
            dtype=torch.float32,
        )
        layer.w2_weight = torch.empty(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            dtype=torch.float32,
        )
        layer.w13_weight_scale = torch.empty(
            self.num_experts,
            2 * self.intermediate_size,
            self.hidden_size // self.quant_method.group_size,
            dtype=torch.uint8,
        )
        layer.w2_weight_scale = torch.empty(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size // self.quant_method.group_size,
            dtype=torch.uint8,
        )
        return layer

    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_flash_common3_context")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.select_experts")
    def test_apply_overlap_gate_uses_fc3_context(
        self,
        mock_select_experts,
        mock_extra_ctx,
        mock_get_flash_common3_context,
    ):
        tokens = 4
        x = torch.randn(tokens, self.hidden_size, dtype=torch.float32)
        router_logits = torch.randn(tokens, self.num_experts, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2, dtype=torch.float32)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2), dtype=torch.int64)

        self.quant_method.multistream_overlap_gate = True
        mock_get_flash_common3_context.return_value = Mock(
            shared_experts=Mock(), topk_weights=topk_weights, topk_ids=topk_ids
        )
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, self.hidden_size, dtype=torch.float32)
        mock_extra_ctx.flashcomm_v3_enabled = True
        mock_extra_ctx.moe_comm_method = mock_comm

        self.quant_method.apply(
            layer=self.build_layer(),
            x=x,
            router_logits=router_logits,
            top_k=2,
            renormalize=True,
            global_num_experts=self.num_experts,
            enable_force_load_balance=False,
        )

        mock_select_experts.assert_not_called()
        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIs(fused_experts_input.topk_weights, topk_weights)
        self.assertIs(fused_experts_input.topk_ids, topk_ids)

    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_flash_common3_context")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w8a8_mxfp8.select_experts")
    def test_apply_overlap_gate_falls_back_when_fc3_disabled(
        self,
        mock_select_experts,
        mock_extra_ctx,
        mock_get_flash_common3_context,
    ):
        tokens = 4
        x = torch.randn(tokens, self.hidden_size, dtype=torch.float32)
        router_logits = torch.randn(tokens, self.num_experts, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2, dtype=torch.float32)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2), dtype=torch.int64)

        self.quant_method.multistream_overlap_gate = True
        mock_get_flash_common3_context.return_value = Mock(shared_experts=Mock())
        mock_select_experts.return_value = (topk_weights, topk_ids)
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, self.hidden_size, dtype=torch.float32)
        mock_extra_ctx.flashcomm_v3_enabled = False
        mock_extra_ctx.moe_comm_method = mock_comm

        self.quant_method.apply(
            layer=self.build_layer(),
            x=x,
            router_logits=router_logits,
            top_k=2,
            renormalize=True,
            global_num_experts=self.num_experts,
            enable_force_load_balance=False,
        )

        mock_select_experts.assert_called_once()
        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIs(fused_experts_input.topk_weights, topk_weights)
        self.assertIs(fused_experts_input.topk_ids, topk_ids)
