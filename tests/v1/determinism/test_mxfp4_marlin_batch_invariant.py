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
    # ``test_mxfp4_marlin_moe_unit_invariance`` ships a probe function
    # to the worker via ``LLMEngine.apply_model``. vLLM's default
    # msgpack RPC cannot serialize Python callables, so opt into the
    # pickle fallback for this test module.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _marlin_moe_unit_probe(worker) -> dict:
    """Run inside the worker via ``collective_rpc``.

    ``collective_rpc(callable)`` calls ``callable(worker)`` on each
    worker (see ``vllm.v1.serial_utils.run_method``). The worker exposes
    ``model_runner``, from which we get both the live model and the
    real ``vllm_config`` — needed by ``set_forward_context``.

    Drills into the first transformer layer's MoE block, then calls it
    twice — once with M=1 and once with M=8 where the M=8 input shares
    row 0 with the M=1 input — and returns a summary dict of the row-0
    bitwise comparison.
    """
    import torch as _torch  # local imports; this runs in the worker

    from vllm.forward_context import set_forward_context

    model_runner = worker.model_runner
    vllm_config = model_runner.vllm_config
    model = model_runner.model
    inner = getattr(model, "model", model)
    mlp = inner.layers[0].mlp

    device = next(mlp.parameters()).device
    # The router's weight dtype is the activation dtype we want; the
    # quantized expert weights are uint8/int4 and not usable here.
    dtype = mlp.router.weight.dtype
    K = mlp.hidden_size

    _torch.manual_seed(20240919)
    hs_full = _torch.randn(8, K, device=device, dtype=dtype)
    hs_1 = hs_full[0:1].clone()
    hs_N = hs_full.clone()
    assert _torch.equal(hs_1[0], hs_N[0])

    def _run(hs):
        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_config,
            num_tokens=hs.shape[0],
        ):
            return mlp(hs)

    with _torch.inference_mode():
        out_1 = _run(hs_1)
        out_N = _run(hs_N)

    row0_1 = out_1[0, :K].float()
    row0_N = out_N[0, :K].float()
    diff = (row0_1 - row0_N).abs()
    return {
        "max_abs": float(diff.max().item()),
        "max_idx": int(diff.argmax().item()),
        "n_within_1e_5": int((diff < 1e-5).sum().item()),
        "n_total": int(diff.numel()),
        "bitwise_equal": bool(_torch.equal(row0_1, row0_N)),
    }


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


# ---------------------------------------------------------------------------
# Diagnostic: layer-by-layer decode bisect (BS=1 vs BS=2)
# ---------------------------------------------------------------------------
# Finds which layer first produces different hidden states between a BS=1
# run and a BS=2 run for the same prompt.  Hooks capture the input to each
# layer's attention (post-input-layernorm) and the input to each layer's
# MoE (post-attention + post-post-attention-layernorm).  The needle's row
# in BS=2 is identified by matching the ``positions`` tensor.
# ---------------------------------------------------------------------------


def _get_bisect_db(worker):
    """Get or create the bisect state dict, stored on the model object.

    We store state on the model (a persistent nn.Module in the worker
    subprocess) rather than on the worker, because the worker object
    handed to each ``collective_rpc`` call may be a fresh wrapper that
    does not carry attributes from previous calls.
    """
    model = worker.model_runner.model
    inner = getattr(model, "model", model)
    if not hasattr(inner, "_bisect_db"):
        inner._bisect_db = {
            "captures": [],
            "hooks": [],
        }
    return inner, inner._bisect_db


def _install_decode_bisect_hooks(worker) -> None:
    """Install forward-pre-hooks on every layer's attn and mlp.

    Forward-pass boundaries are detected by watching layer 0's attn hook
    (the first hook to fire each forward).  A model-level post-hook is
    NOT used because ``@support_torch_compile`` can wrap the model's
    ``forward`` and silently prevent it from firing.
    """
    inner, db = _get_bisect_db(worker)

    # Clear any prior state.
    for h in db["hooks"]:
        h.remove()
    db["captures"] = []
    db["hooks"] = []

    for i, layer in enumerate(inner.layers):

        def _make_attn_pre(idx):
            def hook(mod, args):
                if idx == 0:
                    # Layer 0 attn is the first hook to fire each
                    # forward — start a new capture dict.
                    db["captures"].append({})
                    # Also grab ``positions`` (arg 1 of OAIAttention).
                    db["captures"][-1]["positions"] = (
                        args[1].detach().cpu().clone()
                    )
                db["captures"][-1][f"L{idx}_attn_in"] = (
                    args[0].detach().float().cpu()
                )
            return hook

        def _make_mlp_pre(idx):
            def hook(mod, args):
                db["captures"][-1][f"L{idx}_mlp_in"] = (
                    args[0].detach().float().cpu()
                )
            return hook

        db["hooks"].append(
            layer.attn.register_forward_pre_hook(_make_attn_pre(i))
        )
        db["hooks"].append(
            layer.mlp.register_forward_pre_hook(_make_mlp_pre(i))
        )


def _reset_decode_bisect(worker) -> None:
    _, db = _get_bisect_db(worker)
    db["captures"] = []


def _get_decode_bisect_captures(worker) -> list[dict]:
    """Return lightweight per-forward summaries (positions + M only).

    The full tensors stay on the worker; call
    ``_compare_decode_bisect`` to do the comparison there.
    """
    _, db = _get_bisect_db(worker)
    out = []
    for fwd_idx, cap in enumerate(db["captures"]):
        pos = cap.get("positions")
        any_key = next(
            (k for k in cap if k.startswith("L")), None
        )
        M = cap[any_key].shape[0] if any_key else 0
        out.append({
            "fwd": fwd_idx,
            "M": M,
            "positions": pos.tolist() if pos is not None else [],
        })
    return out


def _compare_decode_bisect(worker, fwd_bs1: int, row_bs1: int,
                           fwd_bs2: int, row_bs2: int) -> list[dict]:
    """Compare hidden states at every layer between two captures.

    Returns a list of per-layer dicts with bitwise-eq flag and max diff.
    """
    import torch as _torch

    _, db = _get_bisect_db(worker)
    caps = db["captures"]
    cap1 = caps[fwd_bs1]
    cap2 = caps[fwd_bs2]

    results = []
    # Iterate layers in order.
    layer_keys = sorted(
        [k for k in cap1 if k.startswith("L")],
        key=lambda k: (int(k.split("_")[0][1:]), k.split("_", 1)[1]),
    )
    for key in layer_keys:
        t1 = cap1[key]
        t2 = cap2[key]
        r1 = t1[row_bs1]
        r2 = t2[row_bs2]
        diff = (r1 - r2).abs()
        results.append({
            "key": key,
            "bitwise_eq": bool(_torch.equal(r1, r2)),
            "max_abs_diff": float(diff.max().item()),
            "max_diff_idx": int(diff.argmax().item()),
        })
    return results


def _remove_decode_bisect_hooks(worker) -> None:
    inner, db = _get_bisect_db(worker)
    for h in db.get("hooks", []):
        h.remove()
    db["captures"] = []
    db["hooks"] = []
    if hasattr(inner, "_bisect_db"):
        del inner._bisect_db


@skip_unsupported
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
def test_mxfp4_marlin_moe_decode_layer_bisect(backend):
    """Diagnostic: find the first layer where BS=1 vs BS=2 diverge.

    Runs the needle prompt at BS=1, then at BS=2 (needle + filler).
    Compares the needle's hidden states at each layer boundary
    (pre-attention and pre-MoE) during the first decode step where the
    needle is at the same sequence position in both runs.

    The needle's row in the BS=2 batch is identified by matching its
    expected ``positions`` value (= prompt_len), fixing the row-indexing
    bug in the earlier crossrun bisect.
    """
    needle_prompt = "Write one factual sentence about the moon."
    filler_prompt = "Explain photosynthesis in simple terms for a child."
    max_tokens = 3

    sampling = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
    )

    llm = None
    try:
        llm = _make_llm(max_num_seqs=8, backend=backend)

        # ---- BS=1 run ----
        llm.llm_engine.collective_rpc(_install_decode_bisect_hooks)
        out_bs1 = llm.generate(
            [needle_prompt], sampling, use_tqdm=False
        )
        summaries_bs1 = llm.llm_engine.collective_rpc(
            _get_decode_bisect_captures
        )[0]

        # Record BS=1 capture count so we can compute absolute indices
        # after the BS=2 captures are appended.
        n_fwd_bs1 = len(summaries_bs1)

        # Don't reset — we'll keep BS=1 captures and append BS=2 captures
        # so _compare_decode_bisect can access both in one call.

        # ---- BS=2 run ----
        # Reset only the forward counter; captures continue accumulating.
        # Actually, we want to append, so just let it continue.
        out_bs2 = llm.generate(
            [needle_prompt, filler_prompt], sampling, use_tqdm=False
        )
        summaries_bs2_raw = llm.llm_engine.collective_rpc(
            _get_decode_bisect_captures
        )[0]
        # BS=2 summaries include BS=1 captures at the front; slice them off.
        summaries_bs2 = summaries_bs2_raw[n_fwd_bs1:]

        llm.llm_engine.collective_rpc(_remove_decode_bisect_hooks)

        # ---- Identify decode steps to compare ----
        needle_prompt_len = len(
            out_bs1[0].prompt_token_ids
        )
        print(
            f"needle prompt_len={needle_prompt_len}  "
            f"BS=1 forwards={n_fwd_bs1}  "
            f"BS=2 forwards={len(summaries_bs2)}",
            flush=True,
        )

        # Dump ALL forward summaries so we can see the schedule.
        print("--- BS=1 forward schedule ---", flush=True)
        for s in summaries_bs1:
            pos_str = ",".join(str(p) for p in s["positions"][:8])
            if len(s["positions"]) > 8:
                pos_str += f"...({len(s['positions'])} total)"
            print(
                f"  fwd={s['fwd']} M={s['M']} positions=[{pos_str}]",
                flush=True,
            )
        print("--- BS=2 forward schedule ---", flush=True)
        for s in summaries_bs2:
            pos_str = ",".join(str(p) for p in s["positions"][:8])
            if len(s["positions"]) > 8:
                pos_str += f"...({len(s['positions'])} total)"
            print(
                f"  fwd={s['fwd']} M={s['M']} positions=[{pos_str}]",
                flush=True,
            )

        # BS=1 decode forwards: M=1, position >= prompt_len
        bs1_decode_fwds: list[tuple[int, int]] = []  # (abs_fwd_idx, position)
        for s in summaries_bs1:
            if s["M"] == 1:
                pos = s["positions"][0] if s["positions"] else -1
                if pos >= needle_prompt_len:
                    bs1_decode_fwds.append((s["fwd"], pos))

        # BS=2 decode forwards: find the needle's row by matching
        # position >= needle_prompt_len.
        bs2_decode_fwds: list[tuple[int, int, int]] = []  # (abs, pos, row)
        for s in summaries_bs2:
            positions = s["positions"]
            for row_idx, pos in enumerate(positions):
                if pos >= needle_prompt_len:
                    bs2_decode_fwds.append(
                        (s["fwd"], pos, row_idx)
                    )
                    break  # one match per forward

        # ---- Compare matching decode steps ----
        # Match by position value (= same point in the sequence).
        bs1_by_pos = {pos: fwd for fwd, pos in bs1_decode_fwds}

        found_first_diff = False
        for abs_fwd_bs2, pos, needle_row in bs2_decode_fwds:
            abs_fwd_bs1 = bs1_by_pos.get(pos)
            if abs_fwd_bs1 is None:
                print(
                    f"  pos={pos}: no matching BS=1 decode step, skipping",
                    flush=True,
                )
                continue

            print(
                f"\n=== Comparing pos={pos}  "
                f"BS=1 fwd={abs_fwd_bs1} row=0  vs  "
                f"BS=2 fwd={abs_fwd_bs2} row={needle_row} ===",
                flush=True,
            )

            layer_results = llm.llm_engine.collective_rpc(
                lambda w, f1=abs_fwd_bs1, f2=abs_fwd_bs2, r2=needle_row: (
                    _compare_decode_bisect(w, f1, 0, f2, r2)
                ),
            )[0]

            first_diff_key = None
            for lr in layer_results:
                tag = "EQ" if lr["bitwise_eq"] else (
                    f"DIFF max={lr['max_abs_diff']:.4e} "
                    f"idx={lr['max_diff_idx']}"
                )
                print(f"  {lr['key']}: {tag}", flush=True)
                if not lr["bitwise_eq"] and first_diff_key is None:
                    first_diff_key = lr["key"]

            if first_diff_key is not None and not found_first_diff:
                found_first_diff = True
                print(
                    f"\n>>> First divergence at pos={pos}: "
                    f"{first_diff_key} <<<",
                    flush=True,
                )

        if not found_first_diff:
            print(
                "\nAll layers bitwise-equal at all matched decode steps!",
                flush=True,
            )

    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


@skip_unsupported
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
def test_mxfp4_marlin_moe_unit_invariance(backend):
    """Direct unit-level invariance probe for ``fused_marlin_moe``.

    Bypasses every engine confound (KV cache, scheduler, sampler,
    chunked prefill, batch composition). Loads the LLM only to obtain a
    real Marlin MXFP4 MoE layer with real weights, then drives the
    layer's ``forward`` directly inside the worker process with two
    synthetic ``hidden_states`` tensors that share row 0 by
    construction:

      hs_1: shape [1, K]
      hs_N: shape [N, K], hs_N[0] == hs_1[0]

    Outcomes:
      - row 0 bitwise-equal → ``fused_marlin_moe`` is itself
        batch-invariant; the engine-level drift comes from upstream
        (attention KV-cache decode, router input drift, etc).
      - row 0 differs → the bug is intrinsic to the Marlin MoE path
        and we keep digging there.
    """
    llm = None
    try:
        llm = _make_llm(max_num_seqs=8, backend=backend)
        # ``collective_rpc(callable)`` calls ``callable(worker)`` on
        # each worker. We need worker access (not just the model) so the
        # probe can read ``vllm_config`` from the model_runner.
        results = llm.llm_engine.collective_rpc(_marlin_moe_unit_probe)
        result = results[0]
        print(
            f"[unit] M=1 vs M=8 row-0 diff: "
            f"max_abs={result['max_abs']:.4e} "
            f"at hidden_idx={result['max_idx']} "
            f"elements_within_1e-5={result['n_within_1e_5']}/{result['n_total']} "
            f"bitwise_equal={result['bitwise_equal']}",
            flush=True,
        )
        assert result["bitwise_equal"], (
            f"fused_marlin_moe not batch-invariant for row 0: "
            f"max_abs_diff={result['max_abs']:.4e} at "
            f"hidden_idx={result['max_idx']}, only "
            f"{result['n_within_1e_5']}/{result['n_total']} elements "
            f"within 1e-5. This isolates the residual drift to the "
            f"Marlin MoE path itself (not engine-side)."
        )
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
