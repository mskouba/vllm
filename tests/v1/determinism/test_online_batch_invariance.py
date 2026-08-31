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
import hashlib
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
    probe_text,
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


def _int_list(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def _chat_answer(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    max_retries: int = 3,
    retry_backoff: float = 0.5,
) -> str:
    """One temperature-0 chat turn, returning reasoning + content concatenated.

    Matches PR #51287's ``answer()``: reasoning models emit their chain in
    ``reasoning_content``, so hashing content alone would miss drift there.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message
            reasoning = getattr(msg, "reasoning_content", "") or ""
            return reasoning + (msg.content or "")
        except Exception as e:  # pragma: no cover
            if attempt < max_retries:
                import time as _t

                _t.sleep(retry_backoff * (2**attempt))
                continue
            raise AssertionError(f"chat request failed after retries: {e}") from e
    raise AssertionError("unreachable")


def _probe_hash(
    client: openai.OpenAI,
    model: str,
    probe: str,
    probe_max_tokens: int,
    batch: int,
) -> str:
    """Hash the probe's output when co-resident with ``batch - 1`` fillers.

    The probe is job 0; fillers use *staggered* ``max_tokens`` so they retire at
    different decode steps, churning batch composition while the probe is still
    generating. Mirrors ``probe_hash()`` in PR #51287's repro.
    """
    jobs: list[tuple[str, int]] = [(probe, probe_max_tokens)]
    for i in range(batch - 1):
        jobs.append((probe_text(3000, i), 400 + (i % 8) * 150))

    with cf.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        outputs = list(
            pool.map(lambda j: _chat_answer(client, model, j[0], j[1]), jobs)
        )
    return hashlib.sha256(outputs[0].encode()).hexdigest()[:12]


@skip_if_not_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_probe_invariance_under_async_scheduling_churn(backend: str) -> None:
    """Reproduce PR #51287's async-scheduling batch-invariance probe.

    This mirrors that PR's end-to-end repro as closely as the pytest harness
    allows: async scheduling stays ON (the path under test), a long 9000-word
    probe generates at temperature 0 while co-resident fillers with *staggered*
    ``max_tokens`` retire at different decode steps, so batch composition churns
    while the probe is still generating. Across every batch size and repeat the
    probe output must hash identically. A single residual composition-dependent
    reduction shows up as more than one distinct hash for the same input, even
    though the static bs1-vs-bsN test above passes. See PR #51287.
    """
    batches = _int_list(os.getenv("VLLM_PROBE_BATCHES", "1,5,10,50"))
    repeats = int(os.getenv("VLLM_PROBE_ROUNDS", "3"))
    probe_max_tokens = int(os.getenv("VLLM_PROBE_MAX_TOKENS", "600"))

    probe = long_probe_prompt()  # 9000-word probe, matching the PR

    server_args: list[str] = [
        "--max-model-len=16384",
        f"--max-num-seqs={max(batches)}",
        f"--attention-backend={backend}",
        # The path under test: async scheduling on (the default). Passed
        # explicitly so composition still churns if the default ever flips.
        "--async-scheduling",
        # Isolate kernel numerics from cache-reuse effects (matches the PR).
        "--no-enable-prefix-caching",
    ]
    tp_size = os.getenv("VLLM_TP_SIZE")
    if tp_size:
        server_args += ["-tp", tp_size]

    hashes: dict[str, list[str]] = {}
    with RemoteOpenAIServer(TEST_MODEL, server_args) as server:
        client = server.get_client()
        for rep in range(repeats):
            for batch in batches:
                h = _probe_hash(client, TEST_MODEL, probe, probe_max_tokens, batch)
                hashes.setdefault(h, []).append(f"rep{rep} batch{batch}")

    if len(hashes) != 1:
        detail = "\n".join(f"  {h}: {', '.join(v)}" for h, v in hashes.items())
        raise AssertionError(
            "Probe not batch-invariant under async scheduling: "
            f"{len(hashes)} distinct outputs for one input.\n{detail}"
        )
