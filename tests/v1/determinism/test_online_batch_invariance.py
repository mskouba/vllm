# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HTTP-based batch invariance test: send requests to a running
vLLM server and compare BS=1 vs BS=N results (tokens and per-step logprobs).

Environment variables:
  - VLLM_TEST_MODEL: served model name (e.g., Qwen/Qwen3-1.7B / DeepSeek-R1)
  - VLLM_TP_SIZE: tensor parallelism size (e.g., 4)

"""

import concurrent.futures as cf
import os
import random
import sys
from typing import Any

import openai
import pytest
from utils import (
    BACKENDS,
    TEST_MODEL,
    _random_prompt,
    long_probe_prompt,
    skip_if_not_cuda,
)

from tests.utils import RemoteOpenAIServer


def _request_completion(
    client: openai.OpenAI,
    model: str,
    prompt: Any,
    sp: dict[str, Any],
    max_retries: int = 3,
    retry_backoff: float = 0.5,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"model": model, "prompt": prompt}
    payload.update(sp)

    for attempt in range(max_retries + 1):
        try:
            completion = client.completions.create(**payload)
            # Convert to plain dict so downstream logic can keep using
            # dict-style access just like with raw HTTP JSON.
            return completion.model_dump()
        except Exception as e:  # pragma: no cover
            if attempt < max_retries:
                import time as _t

                _t.sleep(retry_backoff * (2**attempt))
                continue
            sys.stderr.write(f"Error: {e}\n")
            return None
    return None


def _extract_tokens_and_logprobs(
    choice: dict[str, Any],
) -> tuple[list[Any], list[float] | None]:
    tokens: list[Any] = []
    token_logprobs: list[float] | None = None
    lp = choice.get("logprobs")
    if lp and isinstance(lp, dict):
        tokens = lp.get("token_ids") or lp.get("tokens") or []
        token_logprobs = lp.get("token_logprobs", None)
    return tokens, token_logprobs


def _compare_bs1_vs_bsn_single_process(
    prompts: list[str],
    sp_kwargs: dict[str, Any],
    client: openai.OpenAI,
    model_name: str,
) -> None:
    # BS=1
    bs1_tokens_per_prompt: list[list[Any]] = []
    bs1_logprobs_per_prompt: list[list[float] | None] = []
    for p in prompts:
        resp = _request_completion(client, model_name, p, sp_kwargs)
        if resp is None or not resp.get("choices"):
            raise AssertionError("BS=1 empty/failed response")
        choice = resp["choices"][0]
        toks, lps = _extract_tokens_and_logprobs(choice)
        if lps is None:
            raise AssertionError(
                "logprobs not returned; ensure server supports 'logprobs'"
            )
        bs1_tokens_per_prompt.append(list(toks))
        bs1_logprobs_per_prompt.append(list(lps))

    # BS=N
    bsN_tokens_per_prompt: list[list[Any]] = [None] * len(prompts)  # type: ignore[list-item]
    bsN_logprobs_per_prompt: list[list[float] | None] = [None] * len(prompts)
    resp = _request_completion(client, model_name, prompts, sp_kwargs)
    if resp is None or not resp.get("choices"):
        raise AssertionError("BS=N empty/failed batched response")
    choices = resp.get("choices", [])
    if len(choices) != len(prompts):
        raise AssertionError(
            f"BS=N choices length {len(choices)} != num prompts {len(prompts)}"
        )
    for idx, choice in enumerate(choices):
        toks, lps = _extract_tokens_and_logprobs(choice)
        if lps is None:
            raise AssertionError(f"BS=N missing logprobs for prompt {idx}")
        bsN_tokens_per_prompt[idx] = list(toks)
        bsN_logprobs_per_prompt[idx] = list(lps)

    # compare
    for i, (tokens_bs1, tokens_bsN, logprobs_bs1, logprobs_bsN) in enumerate(
        zip(
            bs1_tokens_per_prompt,
            bsN_tokens_per_prompt,
            bs1_logprobs_per_prompt,
            bsN_logprobs_per_prompt,
        )
    ):
        if tokens_bs1 != tokens_bsN:
            raise AssertionError(
                f"Prompt {i} (sampling): Different tokens sampled. "
                f"BS=1 tokens: {tokens_bs1} BS=N tokens: {tokens_bsN}"
            )
        if logprobs_bs1 is None or logprobs_bsN is None:
            raise AssertionError(f"Prompt {i}: Missing logprobs in one of the runs")
        if len(logprobs_bs1) != len(logprobs_bsN):
            raise AssertionError(
                f"Prompt {i}: Different number of steps: "
                f"{len(logprobs_bs1)} (BS=1) vs {len(logprobs_bsN)} (BS=N)."
            )
        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
            if a != b:
                diff = abs(a - b)
                raise AssertionError(
                    f"Prompt {i} Step {t}: Bitwise mismatch "
                    f"(abs diff={diff:.6e}). "
                    f"BS=1 tokens: {tokens_bs1} BS=N tokens: {tokens_bsN}"
                )


@skip_if_not_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(
    backend: str,
) -> None:
    random.seed(int(os.getenv("VLLM_TEST_SEED", "12345")))
    prompts_all = [_random_prompt(10, 50) for _ in range(32)]

    sp_kwargs: dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 1.0,
        "max_tokens": 8,
        "seed": 42,
        "logprobs": 5,
    }

    tp_size = os.getenv("VLLM_TP_SIZE", "1")
    server_args: list[str] = [
        "--max-model-len=8192",
        "--max-num-seqs=32",
        f"--attention-backend={backend}",
    ]
    if tp_size:
        server_args += ["-tp", tp_size]

    with RemoteOpenAIServer(TEST_MODEL, server_args) as server:
        client = server.get_client()
        _compare_bs1_vs_bsn_single_process(
            prompts=prompts_all,
            sp_kwargs=sp_kwargs,
            client=client,
            model_name=TEST_MODEL,
        )


def _filler_prompt(idx: int, num_words: int = 2000) -> str:
    words = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
    ).split()
    body = " ".join(words[(i + idx) % len(words)] for i in range(num_words))
    return f"Request {idx}. Continue this list of terms:\n\n" + body


@skip_if_not_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_probe_invariance_under_async_scheduling_churn(backend: str) -> None:
    """Probe stays bitwise-identical while batch composition churns mid-decode.

    Unlike the bs1-vs-bsN test above, this keeps async scheduling ON (the
    default) and fires concurrent fillers with *staggered* ``max_tokens`` so
    fillers retire at different decode steps. Batch composition therefore keeps
    changing while the long probe is still generating.

    A complete batch-invariant configuration is expected to return the exact
    same probe tokens and logprobs every round despite that churn. A single
    residual composition-dependent reduction (an incompletely invariant kernel)
    shows up here as a probe flip across rounds, even though the static
    bs1-vs-bsN test above passes. See the discussion on PR #51287.
    """
    num_rounds = int(os.getenv("VLLM_PROBE_ROUNDS", "3"))
    num_fillers = int(os.getenv("VLLM_PROBE_FILLERS", "24"))
    probe_max_tokens = int(os.getenv("VLLM_PROBE_MAX_TOKENS", "256"))

    probe = long_probe_prompt()
    probe_sp: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": probe_max_tokens,
        "seed": 20240919,
        "logprobs": 1,
    }

    server_args: list[str] = [
        "--max-model-len=16384",
        f"--max-num-seqs={num_fillers + 1}",
        f"--attention-backend={backend}",
        # The path under test. Async scheduling is on by default; pass it
        # explicitly so this test still churns composition if the default flips.
        "--async-scheduling",
        # Isolate kernel numerics from cache-reuse effects.
        "--no-enable-prefix-caching",
    ]

    baseline_tokens: list[Any] | None = None
    baseline_logprobs: list[float] | None = None

    with RemoteOpenAIServer(TEST_MODEL, server_args) as server:
        client = server.get_client()

        for rnd in range(num_rounds):
            jobs: list[tuple[str, dict[str, Any]]] = [(probe, probe_sp)]
            for i in range(num_fillers):
                jobs.append(
                    (
                        _filler_prompt(i),
                        {
                            "temperature": 0.0,
                            # Staggered retirement: fillers stop at different
                            # decode steps, so composition churns under the probe.
                            "max_tokens": 64 + (i % 8) * 24,
                            "seed": 1234 + i,
                        },
                    )
                )

            with cf.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                results = list(
                    pool.map(
                        lambda j: _request_completion(
                            client, TEST_MODEL, j[0], j[1]
                        ),
                        jobs,
                    )
                )

            probe_resp = results[0]
            if probe_resp is None or not probe_resp.get("choices"):
                raise AssertionError(f"Round {rnd}: probe request failed")
            tokens, logprobs = _extract_tokens_and_logprobs(probe_resp["choices"][0])
            if logprobs is None:
                raise AssertionError(
                    "logprobs not returned; ensure server supports 'logprobs'"
                )

            if baseline_tokens is None:
                baseline_tokens = tokens
                baseline_logprobs = logprobs
                continue

            if tokens != baseline_tokens:
                raise AssertionError(
                    f"Round {rnd}: probe token divergence under async-scheduling "
                    f"composition churn.\n"
                    f"baseline={baseline_tokens}\n"
                    f"got     ={tokens}"
                )
            assert baseline_logprobs is not None
            for t, (a, b) in enumerate(zip(baseline_logprobs, logprobs)):
                if a != b:
                    raise AssertionError(
                        f"Round {rnd} step {t}: probe logprob bitwise mismatch "
                        f"(abs diff={abs(a - b):.3e})."
                    )
