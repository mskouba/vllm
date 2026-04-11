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
def test_mxfp4_marlin_moe_same_content_batch_greedy(backend):
    """Same-content BS=2 with GREEDY decoding (no RNG, no top-p/top-k).

    Removes all sampling nondeterminism:
      temperature=0 -> argmax
      top_p=1.0, top_k=-1 -> no filtering
      no seed dependency

    If this still fails intra-batch, the divergence is 100% in the
    model forward pass (or lm_head / RMSNorm). If this passes while
    the non-greedy same-content test fails, the bug is in vLLM's
    sampler or logprob extraction, not in the Marlin MoE path.
    """
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=8,
        logprobs=5,
    )
    needle_prompt = "Write one factual sentence about the moon."
    llm = None
    try:
        llm = _make_llm(max_num_seqs=8, backend=backend)
        base = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        lp_base, _ = _extract_step_logprobs(base)

        pair = llm.generate(
            [needle_prompt, needle_prompt], sampling, use_tqdm=False
        )
        lp0, _ = _extract_step_logprobs(pair[0])
        lp1, _ = _extract_step_logprobs(pair[1])

        print("\n[greedy same-content intra-batch] per-step:", flush=True)
        for i in range(lp0.numel()):
            d = abs(lp0[i].item() - lp1[i].item())
            print(
                f"  step {i:2d}: lp0={lp0[i].item():+.9f} "
                f"lp1={lp1[i].item():+.9f} abs_diff={d:.6e} "
                f"{'MATCH' if d == 0.0 else 'DIFF'}",
                flush=True,
            )
        print("\n[greedy same-content cross-batch vs BS=1] per-step:", flush=True)
        for i in range(lp0.numel()):
            d = abs(lp0[i].item() - lp_base[i].item())
            print(
                f"  step {i:2d}: lp0={lp0[i].item():+.9f} "
                f"lp_base={lp_base[i].item():+.9f} abs_diff={d:.6e} "
                f"{'MATCH' if d == 0.0 else 'DIFF'}",
                flush=True,
            )

        assert pair[0].outputs[0].token_ids == pair[1].outputs[0].token_ids
        torch.testing.assert_close(lp0, lp1, rtol=0.0, atol=0.0)
        torch.testing.assert_close(lp0, lp_base, rtol=0.0, atol=0.0)
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


@skip_unsupported
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
def test_mxfp4_marlin_moe_bs1_self_consistency(backend):
    """BS=1 run twice — should be bitwise identical to itself.

    If this fails, the problem is not batching at all; it's some form of
    hidden state (KV cache reuse, RNG drift, caching) that makes even a
    single-prompt run non-reproducible.
    """
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=32,
        seed=20240919,
        logprobs=5,
    )
    needle_prompt = "Write one factual sentence about the moon."
    llm = None
    try:
        llm = _make_llm(max_num_seqs=8, backend=backend)
        a = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        b = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        lp_a, _ = _extract_step_logprobs(a)
        lp_b, _ = _extract_step_logprobs(b)
        assert a.outputs[0].token_ids == b.outputs[0].token_ids
        torch.testing.assert_close(lp_a, lp_b, rtol=0.0, atol=0.0)
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


@skip_unsupported
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
def test_mxfp4_marlin_moe_same_content_batch(backend):
    """BS=2 with two identical copies of the same prompt.

    Three sub-checks, each more specific:
      1. intra-batch: outputs[0] == outputs[1] (same content, different
         position within the batch must produce identical results).
      2. cross-batch vs BS=1: either copy matches a separate BS=1 run.

    If (1) fails, something in the forward pass is picking up
    position-within-batch as a signal (likely a kernel bug or
    padding/scheduling effect).

    If (1) passes but (2) fails, the extra row in the batch is
    perturbing a reduction somewhere (attention, MoE align, etc.),
    even though the row's content is identical.

    If both pass, the failure mode of the main heterogeneous-batch test
    is specifically content-dependent, which strongly implicates
    ``moe_align_block_size`` ordering or a topk-tie break influenced
    by the other prompts' tokens sharing an expert.
    """
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=32,
        seed=20240919,
        logprobs=5,
    )
    needle_prompt = "Write one factual sentence about the moon."
    llm = None
    try:
        llm = _make_llm(max_num_seqs=8, backend=backend)
        # BS=1 baseline
        base = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        lp_base, _ = _extract_step_logprobs(base)

        # BS=2 with identical content
        pair = llm.generate(
            [needle_prompt, needle_prompt], sampling, use_tqdm=False
        )
        lp0, _ = _extract_step_logprobs(pair[0])
        lp1, _ = _extract_step_logprobs(pair[1])

        # Per-index diff dump before the hard assert so we can see exactly
        # which decode step first diverges and by how much.
        diff01 = (lp0 - lp1).abs()
        print(
            "\n[same-content intra-batch] per-step abs(lp0 - lp1):",
            flush=True,
        )
        for i in range(lp0.numel()):
            print(
                f"  step {i:2d}: lp0={lp0[i].item():+.9f} "
                f"lp1={lp1[i].item():+.9f} "
                f"abs_diff={diff01[i].item():.6e} "
                f"{'MATCH' if diff01[i].item() == 0.0 else 'DIFF'}",
                flush=True,
            )

        diff0b = (lp0 - lp_base).abs()
        print(
            "\n[same-content cross-batch vs BS=1] per-step abs(lp0 - lp_base):",
            flush=True,
        )
        for i in range(lp0.numel()):
            print(
                f"  step {i:2d}: lp0={lp0[i].item():+.9f} "
                f"lp_base={lp_base[i].item():+.9f} "
                f"abs_diff={diff0b[i].item():.6e} "
                f"{'MATCH' if diff0b[i].item() == 0.0 else 'DIFF'}",
                flush=True,
            )

        # (1) intra-batch invariance
        assert pair[0].outputs[0].token_ids == pair[1].outputs[0].token_ids, (
            "intra-batch token ids differ for identical prompts"
        )
        torch.testing.assert_close(
            lp0,
            lp1,
            rtol=0.0,
            atol=0.0,
            msg=lambda m: f"intra-batch (same content) logprob mismatch:\n{m}",
        )

        # (2) cross-batch vs BS=1
        assert pair[0].outputs[0].token_ids == base.outputs[0].token_ids, (
            "cross-batch (same content) token ids differ from BS=1"
        )
        torch.testing.assert_close(
            lp0,
            lp_base,
            rtol=0.0,
            atol=0.0,
            msg=lambda m: (
                f"cross-batch (same content) logprob mismatch vs BS=1:\n{m}"
            ),
        )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


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

        # Defer all assertions to the end of the loop so a single failing
        # trial does not hide the distribution of behavior across the
        # remaining trials. We track per-trial diagnostics and only raise
        # after the loop completes.
        token_flip_failures: list[str] = []
        logprob_drift_failures: list[str] = []

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

            base_ids = list(baseline_completion.token_ids)
            needle_ids = list(needle_completion.token_ids)
            n = min(len(base_ids), len(needle_ids))
            flips = sum(1 for i in range(n) if base_ids[i] != needle_ids[i])
            flip_rate = flips / max(n, 1)

            # Bucketed histogram of per-step abs diffs. Designed to be
            # transcribable from a remote terminal: four counts plus the
            # worst-step index and value.
            diffs = (needle_logprobs - baseline_logprobs).abs()
            n_diff = diffs.numel()
            b_ulp = int((diffs < 1e-5).sum().item())
            b_small = int(((diffs >= 1e-5) & (diffs < 1e-3)).sum().item())
            b_med = int(((diffs >= 1e-3) & (diffs < 1e-1)).sum().item())
            b_large = int((diffs >= 1e-1).sum().item())
            worst_idx = int(diffs.argmax().item())
            worst_val = float(diffs[worst_idx].item())

            print(
                f"[trial={trial:02d}] flips={flips}/{n} ({flip_rate:.1%}) "
                f"lp_diff_buckets[<1e-5,<1e-3,<1e-1,>=1e-1]="
                f"[{b_ulp},{b_small},{b_med},{b_large}]/{n_diff} "
                f"worst=(idx={worst_idx}, val={worst_val:.4e})",
                flush=True,
            )

            if needle_completion.token_ids != baseline_completion.token_ids:
                token_flip_failures.append(
                    f"[trial={trial}] flips={flips}/{n} ({flip_rate:.3%})"
                )
            if b_small + b_med + b_large > 0:
                logprob_drift_failures.append(
                    f"[trial={trial}] worst={worst_val:.4e} at idx={worst_idx}"
                )

        # Summary line for transcription.
        print(
            f"\n[summary] token_flip_trials={len(token_flip_failures)}/"
            f"{num_trials} logprob_drift_trials="
            f"{len(logprob_drift_failures)}/{num_trials}",
            flush=True,
        )

        # Token-ID equality is the customer-facing bar; assert hard.
        assert not token_flip_failures, (
            "token-id mismatch under batch invariance:\n  "
            + "\n  ".join(token_flip_failures)
        )
        # Bitwise logprob equality is the strict invariance bar; assert
        # after token-id so a residual logprob drift is reported with the
        # full distribution above.
        assert not logprob_drift_failures, (
            "logprob drift under batch invariance (token ids matched):\n  "
            + "\n  ".join(logprob_drift_failures)
        )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()
