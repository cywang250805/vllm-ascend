import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

# 形状定义
num_tokens = 4       # 本次需要写入的token数量
num_head = 8         # 注意力头数
k_head_size = 128    # key头维度
v_head_size = 128    # value头维度
num_blocks = 2       # KV cache分块总数
block_size = 16      # 每个分块包含的token数

# FP8 dtype（float8_e5m2 与 float8_e4m3fn 均支持，此处以 e4m3fn 为例）
kv_dtype = torch.float8_e4m3fn

# 构造输入：key/value 为待写入的新数据
key = torch.randn(num_tokens, num_head, k_head_size, dtype=torch.float32, device="npu").to(kv_dtype)
value = torch.randn(num_tokens, num_head, v_head_size, dtype=torch.float32, device="npu").to(kv_dtype)

# 构造KV cache（被inplace更新的目标，初始置0便于校验）
key_cache = torch.zeros(num_blocks, num_head, block_size, k_head_size, dtype=kv_dtype, device="npu")
value_cache = torch.zeros(num_blocks, num_head, block_size, v_head_size, dtype=kv_dtype, device="npu")

# 构造slot_mapping：每个token在cache中的偏移，取值范围 [0, num_blocks*block_size-1]
slot_mapping = torch.tensor([0, 1, 16, 17], dtype=torch.int32, device="npu")

# 构造key_scale及其cache（per-token-head的FP8反量化scale）
key_scale = torch.randn(num_tokens, num_head, dtype=torch.float32, device="npu")
key_scale_cache = torch.zeros(num_blocks, num_head, block_size, 1, dtype=torch.float32, device="npu")

# 调用算子，将key/value/key_scale按slot_mapping写入cache
key_cache_out, value_cache_out, key_scale_cache_out = cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    key_scale,
    key_scale_cache,
    cache_layout='BNBD',
)

torch_npu.npu.synchronize()
print(key_cache_out.shape, key_cache_out.dtype)
print(value_cache_out.shape, value_cache_out.dtype)
print(key_scale_cache_out.shape, key_scale_cache_out.dtype)

