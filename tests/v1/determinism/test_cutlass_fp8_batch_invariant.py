# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batch invariance test for the CutlassFP8ScaledMMLinearKernel.

Verifies that logprobs are bitwise identical for the same prompt regardless
of batch size, when VLLM_BATCH_INVARIANT=1 pins the CUTLASS FP8 config
to avoid M-dependent kernel dispatch.

Target: SM89 (Ada Lovelace / L40S) and SM90 (Hopper).
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

# Default to a small FP8-quantized model; override via env var for your model.
FP8_TEST_MODEL = os.getenv(
    "VLLM_TEST_FP8_MODEL", "neuralmagic/Meta-Llama-3-8B-Instruct-FP8"
)


def _make_llm(max_num_seqs: int, backend: str) -> LLM:
    return LLM(
        model=FP8_TEST_MODEL,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(
            os.getenv("VLLM_FP8_TEST_GPU_MEMORY_UTILIZATION", "0.9")
        ),
        max_model_len=int(os.getenv("VLLM_FP8_TEST_MAX_MODEL_LEN", "2048")),
        dtype="auto",
        tensor_parallel_size=int(os.getenv("VLLM_FP8_TEST_TP_SIZE", "1")),
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_config={"backend": backend},
    )


@skip_unsupported
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "TRITON_ATTN"])
def test_cutlass_fp8_logprobs_bitwise_batch_invariance(backend):
    """
    Verifies that the same needle prompt produces bitwise-identical logprobs
    whether run alone (BS=1) or batched with other prompts (BS=N).
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    num_trials = int(os.getenv("VLLM_FP8_NEEDLE_TRIALS", "3"))
    max_batch_size = int(os.getenv("VLLM_FP8_NEEDLE_BATCH_SIZE", "16"))
    min_random_prompt = int(os.getenv("VLLM_FP8_MIN_PROMPT", "32"))
    max_random_prompt = int(os.getenv("VLLM_FP8_MAX_PROMPT", "128"))
    assert max_batch_size >= 2, "Batch size should be >= 2 to test invariance."

    sampling = SamplingParams(
        temperature=float(os.getenv("VLLM_FP8_NEEDLE_TEMPERATURE", "0.6")),
        top_p=float(os.getenv("VLLM_FP8_NEEDLE_TOP_P", "0.95")),
        max_tokens=int(os.getenv("VLLM_FP8_NEEDLE_MAX_TOKENS", "16")),
        seed=20240919,
        logprobs=5,
    )
    needle_prompt = "Write one factual sentence about the moon."

    llm = None
    baseline_logprobs = None
    baseline_token_ids = None
    try:
        llm = _make_llm(max_num_seqs=max_batch_size, backend=backend)

        # Baseline: run needle alone
        baseline_output = llm.generate(
            [needle_prompt], sampling, use_tqdm=False
        )[0]
        baseline_completion = baseline_output.outputs[0]
        baseline_logprobs, baseline_token_ids = _extract_step_logprobs(
            baseline_output
        )
        assert baseline_logprobs is not None
        assert baseline_token_ids is not None

        # Trials: run needle in a batch with random filler prompts
        for trial in range(num_trials):
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
            needle_logprobs, needle_token_ids = _extract_step_logprobs(
                needle_output
            )
            assert needle_logprobs is not None
            assert needle_token_ids is not None

            assert needle_output.prompt == needle_prompt, (
                f"Trial {trial}: prompt mismatch at position {needle_pos}"
            )
            assert needle_completion.token_ids == baseline_completion.token_ids, (
                f"Trial {trial}: token IDs differ. "
                f"BS=1: {baseline_completion.token_ids}, "
                f"BS={batch_size}: {needle_completion.token_ids}"
            )
            assert needle_completion.text == baseline_completion.text, (
                f"Trial {trial}: text differs. "
                f"BS=1: {baseline_completion.text!r}, "
                f"BS={batch_size}: {needle_completion.text!r}"
            )
            torch.testing.assert_close(
                needle_logprobs,
                baseline_logprobs,
                atol=0,
                rtol=0,
                msg=lambda msg: (
                    f"Trial {trial}: logprobs not bitwise equal. {msg}"
                ),
            )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()
