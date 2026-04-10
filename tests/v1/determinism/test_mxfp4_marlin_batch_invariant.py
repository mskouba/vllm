# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance test for the Marlin MXFP4 MoE path.

Forces the Marlin backend (via ``VLLM_MXFP4_USE_MARLIN=1``) so the test
exercises ``fused_marlin_moe`` / ``_fused_marlin_moe`` regardless of the
host GPU's compute capability, then compares BS=1 vs BS=N logprobs
bitwise on a MXFP4 MoE model.

The test is intentionally gated behind ``VLLM_TEST_MXFP4_MOE_MODEL``:
the reference target (``openai/gpt-oss-20b``) is too large for default
CI, but the logic applies to any MXFP4 MoE checkpoint.
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

import vllm.envs as envs
from vllm import LLM, SamplingParams

MXFP4_MOE_MODEL = os.getenv("VLLM_TEST_MXFP4_MOE_MODEL", "")

pytestmark = pytest.mark.skipif(
    not MXFP4_MOE_MODEL,
    reason=(
        "Set VLLM_TEST_MXFP4_MOE_MODEL to a MXFP4 MoE checkpoint "
        "(e.g. openai/gpt-oss-20b) to run this test."
    ),
)


@pytest.fixture(autouse=True)
def force_marlin_mxfp4_backend(monkeypatch: pytest.MonkeyPatch):
    """Force the Marlin MXFP4 MoE backend regardless of device capability.

    The oracle at ``vllm/model_executor/layers/fused_moe/oracle/mxfp4.py``
    short-circuits to the Marlin backend when
    ``VLLM_MXFP4_USE_MARLIN`` is set.
    """
    monkeypatch.setattr(envs, "VLLM_MXFP4_USE_MARLIN", True)
    monkeypatch.setenv("VLLM_MXFP4_USE_MARLIN", "1")


def _make_llm(max_num_seqs: int, backend: str) -> LLM:
    return LLM(
        model=MXFP4_MOE_MODEL,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(
            os.getenv("VLLM_MXFP4_MOE_TEST_GPU_MEMORY_UTILIZATION", "0.9")
        ),
        max_model_len=int(os.getenv("VLLM_MXFP4_MOE_TEST_MAX_MODEL_LEN", "2048")),
        dtype="auto",
        tensor_parallel_size=int(os.getenv("VLLM_MXFP4_MOE_TEST_TP_SIZE", "1")),
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_config={"backend": backend},
    )


@skip_unsupported
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
def test_mxfp4_marlin_moe_bs1_vs_bsN_is_bitwise_invariant(backend):
    """BS=1 vs BS=N bitwise equality on the Marlin MXFP4 MoE path.

    Without the batch-invariant block_size_m / thread_config pinning in
    ``fused_marlin_moe.py``, gpt-oss-20b on L40S flips ~2.9% of tokens
    between BS=1 and BS=N. After the fix, the flip rate should be zero.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    num_trials = int(os.getenv("VLLM_MXFP4_MOE_NEEDLE_TRIALS", "2"))
    max_batch_size = int(os.getenv("VLLM_MXFP4_MOE_NEEDLE_BATCH_SIZE", "8"))
    min_random_prompt = int(os.getenv("VLLM_MXFP4_MOE_MIN_PROMPT", "32"))
    max_random_prompt = int(os.getenv("VLLM_MXFP4_MOE_MAX_PROMPT", "96"))
    assert max_batch_size >= 2, "Batch size should be >= 2 to test invariance."

    sampling = SamplingParams(
        temperature=float(os.getenv("VLLM_MXFP4_MOE_NEEDLE_TEMPERATURE", "0.6")),
        top_p=float(os.getenv("VLLM_MXFP4_MOE_NEEDLE_TOP_P", "0.95")),
        max_tokens=int(os.getenv("VLLM_MXFP4_MOE_NEEDLE_MAX_TOKENS", "32")),
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
        assert baseline_logprobs is not None, (
            "logprobs must be enabled to compare bitwise invariance."
        )

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
            needle_logprobs, _ = _extract_step_logprobs(needle_output)
            assert needle_logprobs is not None
            assert needle_output.prompt == needle_prompt

            # Token-id flip count diagnostic, surfaced on mismatch so
            # we can see how far off we are (e.g. ~2.9% on current main
            # for gpt-oss-20b on L40S).
            base_ids = list(baseline_completion.token_ids)
            needle_ids = list(needle_completion.token_ids)
            n = min(len(base_ids), len(needle_ids))
            flips = sum(1 for i in range(n) if base_ids[i] != needle_ids[i])
            flip_rate = flips / max(n, 1)

            assert needle_completion.token_ids == baseline_completion.token_ids, (
                f"[trial={trial}] token-id mismatch under batch invariance: "
                f"flips={flips}/{n} ({flip_rate:.3%}). Marlin MXFP4 MoE path "
                f"is not batch-invariant."
            )
            assert needle_completion.text == baseline_completion.text
            torch.testing.assert_close(
                needle_logprobs,
                baseline_logprobs,
                rtol=0.0,
                atol=0.0,
                msg=lambda m: (
                    f"[trial={trial}] logprob mismatch under batch "
                    f"invariance on Marlin MXFP4 MoE path:\n{m}"
                ),
            )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()
