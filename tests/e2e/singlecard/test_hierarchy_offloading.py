# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end NPU test for the full hierarchy KV-offload chain:

    L1 NPU  <->  L2 CPU (LRU)  <->  L3 KVStore (file://)

It drives a real ``vllm.LLM`` on Ascend with
``OffloadingConnector`` + ``NPUHierarchyOffloadingSpec`` (registered in
``vllm.v1.kv_offload.factory``).  Two complementary cases are covered:

  * ``test_npu_hierarchy_offloading_cpu_tier`` -- a large L2 budget so every
    offloaded block stays in fast CPU memory.  Mirrors the upstream
    ``test_cpu_offloading`` latency + accuracy checks: after ``reset_prefix_cache``
    the request must reload from L2 (proving the NPU<->CPU swap path is exercised
    and beneficial), and the generated text must stay correct.

  * ``test_npu_hierarchy_offloading_kvstore_tier`` -- a deliberately tiny L2
    budget so a long prompt overflows CPU and spills into the L3 file KVStore.
    Asserts that both ``CPU`` and ``KVStore`` store events fire (NPU->CPU->disk
    write path works), and that after dropping the GPU prefix cache the request
    reloads from L2+L3 and reproduces byte-identical output (NPU<-CPU/KVStore
    read path is correct end to end).

These run only on Ascend NPU hardware (they spin up the real model + worker).
"""

import os
import socket
import tempfile
import time
from collections import Counter

import msgspec
import msgspec.msgpack
import pytest
import zmq
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import BlockStored, KVEventBatch

try:
    import torch_npu  # type: ignore  # noqa: F401

    _HAS_NPU = True
except Exception:  # pragma: no cover - import guard
    _HAS_NPU = False

pytestmark = pytest.mark.skipif(not _HAS_NPU, reason="requires Ascend NPU (torch_npu)")

# Overridable so the test can point at a locally-available model (e.g. an
# air-gapped NPU box): ``VLLM_TEST_MODEL=/mnt/models/Qwen3-8B pytest ...``
MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")
# offload block size (must be a multiple of the engine --block-size)
OFFLOAD_BLOCK_SIZE = 128


class MockSubscriber:
    """Subscribe to the engine's KV-cache event stream and collect store events."""

    def __init__(self, endpoint: str, topic: str):
        self.ctx = zmq.Context.instance()  # type: ignore
        self.topic_bytes = topic.encode("utf-8")
        self.sub = self.ctx.socket(zmq.SUB)  # type: ignore
        self.sub.setsockopt(zmq.SUBSCRIBE, self.topic_bytes)  # type: ignore
        self.sub.connect(endpoint)
        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def get_new_stored_events(self) -> list[BlockStored]:
        """Drain currently-available BlockStored events (across all mediums)."""
        stored: list[BlockStored] = []
        poller = zmq.Poller()  # type: ignore
        poller.register(self.sub, zmq.POLLIN)  # type: ignore
        timeout = 1000  # ms; wait up to 1s for the first event, then short-poll
        while True:
            events = dict(poller.poll(timeout))
            if events.get(self.sub) != zmq.POLLIN:  # type: ignore
                return stored
            topic_bytes, _, payload = self.sub.recv_multipart()
            assert topic_bytes == self.topic_bytes
            event_batch = self.decoder.decode(payload)
            assert isinstance(event_batch, KVEventBatch)
            for event in event_batch.events:
                if isinstance(event, BlockStored):
                    stored.append(event)
                    timeout = 100

    def drain(self) -> None:
        self.get_new_stored_events()

    def close(self) -> None:
        self.sub.close()


def _medium_counts(events: list[BlockStored]) -> Counter:
    return Counter(e.medium for e in events)


def _make_llm(cpu_bytes_to_use: int, kvstore_dir: str):
    """Build an LLM wired with OffloadingConnector + NPUHierarchyOffloadingSpec."""
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            # NPUHierarchyOffloadingSpec is registered in vllm.v1.kv_offload.factory,
            # so spec_name alone is enough (spec_module_path not required).
            "spec_name": "NPUHierarchyOffloadingSpec",
            "block_size": OFFLOAD_BLOCK_SIZE,
            "cpu_bytes_to_use": cpu_bytes_to_use,
            "kvstore_url": f"file://{kvstore_dir}",
        },
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    events_endpoint = f"tcp://*:{port}"
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=events_endpoint,
        topic="test",
    )

    llm = LLM(
        model=MODEL,
        block_size=OFFLOAD_BLOCK_SIZE,
        gpu_memory_utilization=0.5,
        max_model_len=12288,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
    )
    subscriber = MockSubscriber(
        events_endpoint.replace("*", "127.0.0.1"), topic=kv_events_config.topic
    )
    return llm, subscriber


# --------------------------------------------------------------------------- #
# L2 (CPU) tier: latency + accuracy, mirroring upstream test_cpu_offloading.
# --------------------------------------------------------------------------- #
def _latency_test(llm: LLM, subscriber: MockSubscriber) -> None:
    sampling_params = SamplingParams(max_tokens=1)
    num_tests = 10
    num_times_hit_better_than_cold = 0
    total_cold = total_gpu_hit = total_offload_hit = 0.0
    prompt_token_ids = [0] * 10001
    for i in range(num_tests):
        prompt_token_ids[0] = i
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

        start = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_time = time.time() - start
        total_cold += cold_time

        start = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        total_gpu_hit += time.time() - start

        # drop the GPU prefix cache so the next run must reload from L2/L3
        llm.reset_prefix_cache()
        assert subscriber.get_new_stored_events(), "expected blocks to be offloaded"

        start = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        offload_hit = time.time() - start
        total_offload_hit += offload_hit

        if offload_hit < cold_time:
            num_times_hit_better_than_cold += 1

    print("Average times:")
    print(f"    Cold:        {total_cold * 1000 / num_tests:.2f}ms")
    print(f"    GPU hit:     {total_gpu_hit * 1000 / num_tests:.2f}ms")
    print(f"    Offload hit: {total_offload_hit * 1000 / num_tests:.2f}ms")

    # reloading from CPU should beat full recompute most of the time
    assert num_times_hit_better_than_cold >= 0.8 * num_tests


def _accuracy_test(llm: LLM, subscriber: MockSubscriber) -> None:
    sampling_params = SamplingParams(max_tokens=1)
    subscriber.drain()

    # block-align the prompt so it produces full offloaded blocks
    prompt = "Let's count to 10. One, two, three, four,"
    while (
        len(llm.generate(prompt, use_tqdm=False)[0].prompt_token_ids)
        % OFFLOAD_BLOCK_SIZE
        != 0
    ):
        prompt = ". " + prompt

    assert subscriber.get_new_stored_events()

    test_count = 100
    success_count = 0
    for _ in range(test_count):
        llm.reset_prefix_cache()
        out = llm.generate(prompt, sampling_params, use_tqdm=False)[0].outputs[0].text
        if out == " five":
            success_count += 1

    assert success_count >= 0.5 * test_count


def test_npu_hierarchy_offloading_cpu_tier() -> None:
    """Full chain with a large L2 budget: everything stays in fast CPU memory."""
    kvstore_dir = tempfile.mkdtemp(prefix="vllm_kvs_cpu_")
    # ~2 GiB of host memory -> enough CPU blocks to hold the 10k-token prompt.
    llm, subscriber = _make_llm(
        cpu_bytes_to_use=2 * 1024 * 1024 * 1024, kvstore_dir=kvstore_dir
    )
    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        subscriber.close()
        del llm


# --------------------------------------------------------------------------- #
# L3 (KVStore) tier: force CPU eviction to the file store, verify byte-exact
# reload through the whole NPU <-> CPU <-> KVStore chain.
# --------------------------------------------------------------------------- #
def test_npu_hierarchy_offloading_kvstore_tier() -> None:
    """Tiny L2 budget forces eviction into the L3 file KVStore."""
    kvstore_dir = tempfile.mkdtemp(prefix="vllm_kvs_l3_")
    # 64 MiB -> only a handful of CPU blocks; a long prompt overflows into L3.
    llm, subscriber = _make_llm(
        cpu_bytes_to_use=64 * 1024 * 1024, kvstore_dir=kvstore_dir
    )
    try:
        subscriber.drain()

        # 32 full offloaded blocks of distinct tokens; far exceeds the tiny L2
        # budget so most blocks are evicted to the L3 KVStore.
        num_blocks = 32
        prompt_token_ids = list(range(1, num_blocks * OFFLOAD_BLOCK_SIZE + 1))
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        sampling_params = SamplingParams(max_tokens=8, temperature=0.0)

        ref = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
        ref_ids = list(ref.outputs[0].token_ids)

        counts = _medium_counts(subscriber.get_new_stored_events())
        print(f"store events by medium: {dict(counts)}")
        assert counts.get("CPU", 0) > 0, "expected L2 CPU stores"
        assert counts.get("KVStore", 0) > 0, (
            f"expected L3 KVStore eviction, got mediums={dict(counts)}"
        )
        # the L3 file store should now physically hold block files
        assert os.listdir(kvstore_dir), "L3 file KVStore is empty"

        # drop the GPU prefix cache: the rerun must reload KV from L2 + L3
        llm.reset_prefix_cache()
        again = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
        again_ids = list(again.outputs[0].token_ids)

        # byte-exact reload through NPU <- CPU/KVStore: identical greedy output
        assert again_ids == ref_ids, (
            f"reload mismatch after L2+L3 round-trip: {again_ids} != {ref_ids}"
        )
    finally:
        subscriber.close()
        del llm
