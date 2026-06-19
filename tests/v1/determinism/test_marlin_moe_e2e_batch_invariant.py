# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end batch invariance test for the WNA16 Marlin MoE path.

The default model is a compressed-tensors W4A16 MoE checkpoint with *all*
expert weights quantized, so every MoE layer routes through ``MarlinExperts``
(``fused_marlin_moe`` -> ``moe_wna16_marlin_gemm``). A model that quantizes
only some layers would leave the remaining experts on a different kernel, so
the result would not isolate the Marlin path. This validates the full
load/select/kernel stack under ``VLLM_BATCH_INVARIANT``.
"""

import contextlib
import os
import random

import pytest
import torch
from utils import (
    _extract_step_logprobs,
    _random_prompt,
    skip_unsupported,
)

from vllm import LLM, SamplingParams

MARLIN_MOE_MODEL = os.getenv(
    "VLLM_TEST_MARLIN_MOE_MODEL",
    "nm-testing/Qwen1.5-MoE-A2.7B-Chat-quantized.w4a16",
)


def _make_llm(max_num_seqs: int, backend: str) -> LLM:
    return LLM(
        model=MARLIN_MOE_MODEL,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(
            os.getenv("VLLM_MARLIN_MOE_TEST_GPU_MEMORY_UTILIZATION", "0.6")
        ),
        max_model_len=int(os.getenv("VLLM_MARLIN_MOE_TEST_MAX_MODEL_LEN", "2048")),
        dtype="auto",
        tensor_parallel_size=int(os.getenv("VLLM_MARLIN_MOE_TEST_TP_SIZE", "1")),
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_config={"backend": backend},
    )


@skip_unsupported
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("backend", ["FLASH_ATTN"])
def test_marlin_moe_bs1_vs_bsN_is_bitwise_invariant(backend):
    """BS=1 vs BS=N bitwise equality on the Marlin MoE path."""
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    num_trials = int(os.getenv("VLLM_MARLIN_MOE_NEEDLE_TRIALS", "2"))
    max_batch_size = int(os.getenv("VLLM_MARLIN_MOE_NEEDLE_BATCH_SIZE", "8"))
    min_random_prompt = int(os.getenv("VLLM_MARLIN_MOE_MIN_PROMPT", "32"))
    max_random_prompt = int(os.getenv("VLLM_MARLIN_MOE_MAX_PROMPT", "96"))
    assert max_batch_size >= 2, "Batch size should be >= 2 to test invariance."

    sampling = SamplingParams(
        temperature=float(os.getenv("VLLM_MARLIN_MOE_NEEDLE_TEMPERATURE", "0.6")),
        top_p=float(os.getenv("VLLM_MARLIN_MOE_NEEDLE_TOP_P", "0.95")),
        max_tokens=int(os.getenv("VLLM_MARLIN_MOE_NEEDLE_MAX_TOKENS", "32")),
        seed=20240919,
        logprobs=5,
    )
    needle_prompt = "Write one factual sentence about the moon."

    llm = None
    try:
        llm = _make_llm(max_num_seqs=max_batch_size, backend=backend)

        # BS=1 baseline.
        baseline_output = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        baseline_completion = baseline_output.outputs[0]
        baseline_logprobs, _ = _extract_step_logprobs(baseline_output)
        assert baseline_logprobs is not None

        for _ in range(num_trials):
            batch_size = random.randint(max_batch_size // 2, max_batch_size)
            needle_pos = random.randint(0, batch_size - 1)
            prompts: list[str] = []
            for idx in range(batch_size):
                if idx == needle_pos:
                    prompts.append(needle_prompt)
                else:
                    prompts.append(
                        _random_prompt(min_random_prompt, max_random_prompt)
                    )

            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            needle_output = outputs[needle_pos]
            needle_completion = needle_output.outputs[0]
            needle_logprobs, _ = _extract_step_logprobs(needle_output)
            assert needle_logprobs is not None
            assert needle_output.prompt == needle_prompt

            assert needle_completion.token_ids == baseline_completion.token_ids
            assert needle_completion.text == baseline_completion.text
            torch.testing.assert_close(
                needle_logprobs, baseline_logprobs, rtol=0.0, atol=0.0
            )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()
