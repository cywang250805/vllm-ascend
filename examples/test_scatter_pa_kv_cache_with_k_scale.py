"""Single-op smoke test for ScatterPaKvCacheWithKScale on an Ascend device.

Run this file only after installing the custom 4-in-1 operator package and
exporting its op_api library directory through LD_LIBRARY_PATH.
"""

import inspect
import os

import torch
import torch_npu  # noqa: F401

# Load torch_npu before the separately installed custom-op binding.
# isort: off
import cann_ops_transformer

# isort: on

NUM_BLOCKS = 2
NUM_HEADS = 2
BLOCK_SIZE = 16
HEAD_SIZE = 64
NUM_TOKENS = 3
CACHE_LAYOUT = "BNBD"


def _print_binding_info() -> None:
    op = cann_ops_transformer.scatter_pa_kv_cache_with_k_scale
    print(f"cann_ops_transformer: {getattr(cann_ops_transformer, '__file__', '<built-in>')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    print(f"operator: {op!r}")
    try:
        print(f"signature: {inspect.signature(op)}")
    except (TypeError, ValueError):
        print("signature: unavailable for this compiled binding")


def _assert_slot(
    cache: torch.Tensor,
    expected: torch.Tensor,
    slot: int,
) -> None:
    block_index = slot // BLOCK_SIZE
    block_offset = slot % BLOCK_SIZE
    actual = cache[block_index, :, block_offset, :]
    torch.testing.assert_close(actual.float().cpu(), expected.float().cpu(), rtol=0, atol=0)


def main() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is visible to torch_npu")

    torch.npu.set_device(0)
    _print_binding_info()

    key_source = torch.arange(
        NUM_TOKENS * NUM_HEADS * HEAD_SIZE,
        dtype=torch.float32,
        device="npu",
    ).reshape(NUM_TOKENS, NUM_HEADS, HEAD_SIZE)
    key = ((key_source % 31) - 15).to(torch.float8_e4m3fn)
    value = ((key_source % 29) - 14).to(torch.float8_e4m3fn)
    key_cache = torch.zeros(
        (NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE),
        dtype=torch.float8_e4m3fn,
        device="npu",
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.tensor([0, 17, 5], dtype=torch.int32, device="npu")
    key_scale = torch.tensor(
        [[1.0, 1.25], [1.5, 1.75], [2.0, 2.25]],
        dtype=torch.float32,
        device="npu",
    )
    key_scale_cache = torch.zeros(
        (NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, 1),
        dtype=torch.float32,
        device="npu",
    )

    result = cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_scale,
        key_scale_cache,
        CACHE_LAYOUT,
    )
    torch.npu.synchronize()
    print(f"operator return value: {result!r}")

    for token_index, slot in enumerate((0, 17, 5)):
        _assert_slot(key_cache, key[token_index], slot)
        _assert_slot(value_cache, value[token_index], slot)
        block_index = slot // BLOCK_SIZE
        block_offset = slot % BLOCK_SIZE
        actual_scale = key_scale_cache[block_index, :, block_offset, 0]
        torch.testing.assert_close(
            actual_scale.cpu(),
            key_scale[token_index].cpu(),
            rtol=0,
            atol=0,
        )

    assert torch.count_nonzero(key_cache.float()).item() == torch.count_nonzero(key.float()).item()
    assert torch.count_nonzero(value_cache.float()).item() == torch.count_nonzero(value.float()).item()
    assert torch.count_nonzero(key_scale_cache).item() == key_scale.numel()
    print("PASS: K cache, V cache, and K-scale cache were updated at all requested slots.")


if __name__ == "__main__":
    main()
