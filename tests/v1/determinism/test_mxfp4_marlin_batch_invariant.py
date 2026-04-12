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


def _install_bisect_hooks(worker) -> None:
    """Install per-sub-module forward hooks on every transformer block.

    For each TransformerBlock we hook ``input_layernorm``, ``attn``,
    ``post_attention_layernorm``, and ``mlp`` separately so we can pin
    the divergence to a single sub-step within a layer (input norm vs.
    attention vs. post norm vs. MoE). We also hook the embedding and the
    final ``norm`` for completeness.

    Captures *only* forward calls whose primary output has shape
    ``[2, hidden]`` — that uniquely identifies the same-content BS=2
    decode steps and skips prefill (which has shape [2*L, hidden]).
    """
    model = worker.model_runner.model
    inner = getattr(model, "model", model)
    layers = inner.layers

    worker._bisect_storage = {}
    storage = worker._bisect_storage

    def _primary(tensor_or_tuple):
        if isinstance(tensor_or_tuple, tuple):
            for t in tensor_or_tuple:
                if hasattr(t, "shape"):
                    return t
            return None
        return tensor_or_tuple

    def make_hook(name):
        def hook(_module, _inputs, output):
            try:
                out = _primary(output)
                if out is None or out.dim() != 2 or out.shape[0] != 2:
                    return
                storage.setdefault(name, []).append(out.detach().clone())
            except Exception:
                pass

        return hook

    # Capture EVERY embedding forward unconditionally so we can see the
    # full schedule (prefill chunk shapes, decode batch shapes, mixed
    # batches, etc.) rather than guessing what shape filter to use. The
    # earlier shape==[2, hidden] filter was producing misleading
    # captures: all dumps showed row0_id != row1_id even though the two
    # generations are identical token-for-token, meaning the captured
    # forwards weren't "two-seq decode" forwards at all.
    worker._bisect_embed_inputs = []
    embed_inputs = worker._bisect_embed_inputs

    def embedding_hook(_module, inputs, output):
        try:
            out = _primary(output)
            if out is None:
                return
            # store the embedding output too, in case we want to do
            # row-level comparisons later (only the bisect_storage path
            # uses it; for now we just need inputs).
            if out.dim() == 2 and out.shape[0] == 2:
                storage.setdefault("embedding", []).append(out.detach().clone())
            if inputs and hasattr(inputs[0], "shape"):
                embed_inputs.append(inputs[0].detach().clone())
            else:
                embed_inputs.append(None)
        except Exception:
            pass

    handles = []
    if hasattr(inner, "embedding"):
        handles.append(inner.embedding.register_forward_hook(embedding_hook))
    for i, layer in enumerate(layers):
        for sub in ("input_layernorm", "attn", "post_attention_layernorm", "mlp"):
            mod = getattr(layer, sub, None)
            if mod is not None:
                handles.append(
                    mod.register_forward_hook(make_hook(f"layer_{i:02d}.{sub}"))
                )
    if hasattr(inner, "norm"):
        handles.append(inner.norm.register_forward_hook(make_hook("final_norm")))
    worker._bisect_handles = handles


def _collect_bisect_results(worker) -> dict:
    """Compute per-step row0 vs row1 diff for each hooked sub-module."""
    storage = getattr(worker, "_bisect_storage", {})
    out: dict[str, list[dict]] = {}
    for name, tensors in storage.items():
        per_step = []
        for t in tensors:
            d = (t[0].float() - t[1].float()).abs()
            max_v = float(d.max().item())
            arg = int(d.argmax().item())
            per_step.append(
                {
                    "max": max_v,
                    "argmax": arg,
                    "bitwise_eq": max_v == 0.0,
                }
            )
        out[name] = per_step
    # Surface the captured embedding input_ids alongside the per-layer
    # diff so the test can print actual token IDs at row 0 / row 1 of
    # each [2, hidden] forward.
    embed_inputs = getattr(worker, "_bisect_embed_inputs", [])
    embed_input_dump = []
    for t in embed_inputs:
        if t is None:
            embed_input_dump.append(None)
        else:
            try:
                embed_input_dump.append(t.detach().cpu().tolist())
            except Exception:
                embed_input_dump.append(None)
    out["__embed_input_ids__"] = embed_input_dump  # type: ignore[assignment]

    for h in getattr(worker, "_bisect_handles", []):
        h.remove()
    worker._bisect_handles = []
    worker._bisect_storage = {}
    worker._bisect_embed_inputs = []
    return out


@skip_unsupported
@pytest.mark.parametrize("backend", ["TRITON_ATTN"])
def test_mxfp4_marlin_moe_layer_bisect_same_content_bs2(backend):
    """Per-sub-module bisect of same-content BS=2 greedy decode.

    Marlin MoE has been independently proven invariant by the unit-level
    test. The remaining drift on gpt-oss-20b must originate upstream of
    the MoE block. This test installs forward hooks on every
    sub-module of every transformer block, runs a same-content BS=2
    greedy generation, and reports — per decode step — the first
    sub-module whose output for row 0 differs from row 1.

    The first divergence point is the offending sub-module. Suspects on
    gpt-oss specifically are: attention sinks, alternating SWA, the
    pre/post-attention RMSNorms with residual fusion.
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
        llm.llm_engine.collective_rpc(_install_bisect_hooks)

        # Run a same-content BS=2 greedy generation. Decode steps will be
        # forward calls with [2, hidden] inputs/outputs (one token per
        # sequence per step) — the hooks select exactly those.
        pair = llm.generate(
            [needle_prompt, needle_prompt], sampling, use_tqdm=False
        )

        # Print the actual generated token IDs of both sequences. This
        # is the unambiguous ground truth: if pair[0].token_ids ==
        # pair[1].token_ids, the two sequences are on the same token
        # stream and any "embedding row0 != row1" reading must be a
        # capture artifact (mixed batch / wrong forwards). If they
        # differ at position 0, prefill itself sampled different T0.
        ids0 = list(pair[0].outputs[0].token_ids)
        ids1 = list(pair[1].outputs[0].token_ids)
        print(f"\n[bisect] pair[0].token_ids = {ids0}", flush=True)
        print(f"[bisect] pair[1].token_ids = {ids1}", flush=True)
        first_div_pos = next(
            (i for i in range(min(len(ids0), len(ids1))) if ids0[i] != ids1[i]),
            None,
        )
        print(
            f"[bisect] first divergent generated position: {first_div_pos}",
            flush=True,
        )

        results = llm.llm_engine.collective_rpc(_collect_bisect_results)[0]

        # Pull out the embedding input_ids dump (token IDs that were
        # actually looked up at each captured [2, hidden] forward) so we
        # can print row 0 / row 1 token IDs alongside the bisect.
        embed_input_ids = results.pop("__embed_input_ids__", [])

        # Assemble per-step view: for each captured decode step index,
        # walk sub-modules in topological order and find the first one
        # whose row0 != row1.
        # We assume all sub-modules captured the same number of steps;
        # take the min to be safe.
        ordered_names = []
        if "embedding" in results:
            ordered_names.append("embedding")
        # layers in numeric order
        layer_keys = sorted(
            k for k in results if k.startswith("layer_") and "." in k
        )
        # within each layer, fixed sub-order:
        sub_order = ["input_layernorm", "attn", "post_attention_layernorm", "mlp"]
        layer_indices = sorted(
            {int(k.split("_")[1].split(".")[0]) for k in layer_keys}
        )
        for li in layer_indices:
            for sub in sub_order:
                key = f"layer_{li:02d}.{sub}"
                if key in results:
                    ordered_names.append(key)
        if "final_norm" in results:
            ordered_names.append("final_norm")

        if not ordered_names:
            pytest.fail(
                "no [2, hidden] sub-module outputs captured — model "
                "structure may differ from gpt-oss; check hook filters."
            )

        n_steps = min(len(results[n]) for n in ordered_names)
        print(
            f"\n[bisect] captured {n_steps} decode step(s) across "
            f"{len(ordered_names)} sub-modules",
            flush=True,
        )

        # Dump every captured embedding forward unfiltered: shape, full
        # input_ids list (or compact summary if long). This is the
        # ground truth — we can read the schedule directly from these
        # rows rather than guessing.
        print(
            f"[bisect] captured {len(embed_input_ids)} embedding "
            f"forward(s) total",
            flush=True,
        )
        for i, ids in enumerate(embed_input_ids):
            if ids is None:
                print(f"  fwd {i:2d}: <unavailable>", flush=True)
                continue
            n = len(ids)
            if n <= 32:
                print(f"  fwd {i:2d}: shape=[{n}] ids={ids}", flush=True)
            else:
                # Long forward (likely prefill): show the first few,
                # last few, and any natural midpoint.
                head = ids[:8]
                tail = ids[-8:]
                mid = ids[n // 2 - 4 : n // 2 + 4]
                print(
                    f"  fwd {i:2d}: shape=[{n}] head={head} "
                    f"mid={mid} tail={tail}",
                    flush=True,
                )

        first_div_per_step: list[tuple[int, str | None, float]] = []
        for step in range(n_steps):
            first_div: str | None = None
            first_max = 0.0
            for name in ordered_names:
                entry = results[name][step]
                if not entry["bitwise_eq"]:
                    first_div = name
                    first_max = entry["max"]
                    break
            first_div_per_step.append((step, first_div, first_max))
            if first_div is None:
                print(f"  step {step:2d}: all sub-modules bitwise-equal", flush=True)
            else:
                print(
                    f"  step {step:2d}: first divergence at "
                    f"{first_div} max_abs={first_max:.4e}",
                    flush=True,
                )

        # Find the earliest step that diverges and dump the full
        # sub-module ladder for it so we can see whether the residual
        # path also drifts (input_layernorm sees the residual fused in).
        first_bad = next(
            (s for s, d, _ in first_div_per_step if d is not None), None
        )
        if first_bad is not None:
            print(
                f"\n[bisect] full ladder for earliest divergent step "
                f"{first_bad}:",
                flush=True,
            )
            for name in ordered_names:
                entry = results[name][first_bad]
                tag = "EQ " if entry["bitwise_eq"] else "DIFF"
                print(
                    f"  {tag} {name}: max={entry['max']:.4e} "
                    f"argmax={entry['argmax']}",
                    flush=True,
                )

        # This test is diagnostic — it should *fail* (and dump the
        # ladder) until the upstream divergence is fixed. Once fixed,
        # this is also the regression guard.
        assert first_bad is None, (
            f"row 0 vs row 1 diverges at decode step {first_bad}; "
            f"see ladder above for the offending sub-module."
        )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


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


        print("First response Token Ids: ",pair[0].outputs[0].token_ids)
        print("Second response Token Ids: ",pair[1].outputs[0].token_ids)


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
