# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic chunk-boundary invariance probe.

A single long prompt is generated at batch size 1 (no other requests, no async
scheduling, no timing) through engines that differ *only* in
``max_num_batched_tokens``. That budget controls how many prefill chunks the
prompt is split into (``ceil(prompt_len / budget)``), so it isolates the
prefix-vs-current-chunk split of attention from batch composition.

With batch-invariant kernels the sampled tokens and per-step logprobs must be
identical across chunkings. A mismatch is a deterministic, race-free repro of
an attention reduction that depends on where the prefill is chunked -- a kernel
bug, not a scheduler effect. This is the axis the static bs1-vs-bsN test never
exercises, because there the checked request is never chunked.
"""

import contextlib
import os

import pytest
import torch
from utils import (
    BACKENDS,
    TEST_MODEL,
    _extract_step_logprobs,
    long_probe_prompt,
    skip_if_not_cuda,
)

from vllm import LLM, SamplingParams


def _make_llm(max_num_batched_tokens: int, backend: str, max_model_len: int) -> LLM:
    return LLM(
        model=TEST_MODEL,
        max_num_seqs=1,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5")),
        dtype="auto",
        tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_config={"backend": backend},
    )


@skip_if_not_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_chunk_boundary_invariance(backend: str) -> None:
    max_model_len = int(os.getenv("VLLM_CHUNK_MAX_MODEL_LEN", "16384"))
    max_tokens = int(os.getenv("VLLM_PROBE_MAX_TOKENS", "128"))
    # Ascending budgets: several chunks, fewer chunks, then a single pass
    # (budget == max_model_len, so the whole prompt prefills at once).
    default_budgets = f"2048,4096,{max_model_len}"
    raw_budgets = os.getenv("VLLM_CHUNK_BUDGETS", default_budgets)
    budgets = [int(b) for b in raw_budgets.split(",")]

    probe = long_probe_prompt()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=1,
        seed=20240919,
    )

    baseline_tokens: list[int] | None = None
    baseline_logprobs: torch.Tensor | None = None

    for budget in budgets:
        llm = None
        try:
            llm = _make_llm(budget, backend, max_model_len)
            out = llm.generate([probe], sampling)[0]
            logprobs, token_ids = _extract_step_logprobs(out)
            if logprobs is None or token_ids is None:
                raise AssertionError(
                    "logprobs not returned; SamplingParams(logprobs=...) required"
                )
            token_ids = list(token_ids)

            if baseline_tokens is None:
                baseline_tokens = token_ids
                baseline_logprobs = logprobs
                print(f"[chunk-boundary] baseline budget={budget}: {len(token_ids)}")
                continue

            assert baseline_logprobs is not None
            if token_ids != baseline_tokens:
                raise AssertionError(
                    f"budget={budget} vs baseline budget={budgets[0]}: token "
                    f"divergence -- attention is not chunk-boundary invariant.\n"
                    f"baseline={baseline_tokens}\n"
                    f"got     ={token_ids}"
                )
            if not torch.equal(baseline_logprobs, logprobs):
                max_diff = (baseline_logprobs - logprobs).abs().max().item()
                raise AssertionError(
                    f"budget={budget} vs baseline budget={budgets[0]}: logprob "
                    f"bitwise mismatch (max abs diff={max_diff:.3e}) despite "
                    f"identical tokens -- chunk-boundary reduction drift."
                )
            print(f"[chunk-boundary] budget={budget}: matches baseline")
        finally:
            if llm is not None:
                with contextlib.suppress(Exception):
                    llm.shutdown()
