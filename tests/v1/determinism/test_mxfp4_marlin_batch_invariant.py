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
                if not db["captures"]:
                    return  # guard: partial/warmup forward
                db["captures"][-1][f"L{idx}_attn_in"] = (
                    args[0].detach().float().cpu()
                )
            return hook

        def _make_mlp_pre(idx):
            def hook(mod, args):
                if not db["captures"]:
                    return  # guard: partial/warmup forward
                db["captures"][-1][f"L{idx}_mlp_in"] = (
                    args[0].detach().float().cpu()
                )
            return hook

        def _make_mlp_post(idx):
            def hook(mod, args, output):
                if not db["captures"]:
                    return
                db["captures"][-1][f"L{idx}_mlp_out"] = (
                    output.detach().float().cpu()
                )
            return hook

        db["hooks"].append(
            layer.attn.register_forward_pre_hook(_make_attn_pre(i))
        )
        db["hooks"].append(
            layer.mlp.register_forward_pre_hook(_make_mlp_pre(i))
        )
        db["hooks"].append(
            layer.mlp.register_forward_hook(_make_mlp_post(i))
        )

    # Hook the final RMSNorm (input to lm_head) to capture the model's
    # output hidden states.  This catches divergence in the last layer's
    # MoE output that the per-layer pre-hooks would miss.
    if hasattr(inner, "norm"):
        def _final_norm_hook(mod, args):
            if not db["captures"]:
                return
            db["captures"][-1]["final_norm_in"] = (
                args[0].detach().float().cpu()
            )
        db["hooks"].append(
            inner.norm.register_forward_pre_hook(_final_norm_hook)
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
        n_keys = sum(1 for k in cap if k.startswith("L"))
        out.append({
            "fwd": fwd_idx,
            "M": M,
            "n_keys": n_keys,
            "positions": pos.tolist() if pos is not None else [],
        })
    return out


def _replay_mlp_at_layer(worker, layer_idx: int,
                         fwd_bs1: int, fwd_bs2: int,
                         row_bs2: int) -> dict:
    """Replay the MoE at a specific layer with captured inputs.

    Takes the BS=1 input (M=1) and the BS=2 input (M=M_bs2) from the
    captures.  Runs the MoE layer standalone in both configurations and
    checks if row 0 of the M=1 output equals row ``row_bs2`` of the
    M=M_bs2 output.  Then also runs M=1 vs M=1 (just the needle row
    from the BS=2 capture) to see if the layer is invariant for M=1.
    """
    import torch as _torch

    from vllm.forward_context import set_forward_context

    _, db = _get_bisect_db(worker)
    caps = db["captures"]
    if fwd_bs1 >= len(caps) or fwd_bs2 >= len(caps):
        return {"error": "index out of range"}

    key = f"L{layer_idx}_mlp_in"
    if key not in caps[fwd_bs1] or key not in caps[fwd_bs2]:
        return {"error": f"key {key} not found in captures"}

    model_runner = worker.model_runner
    vllm_config = model_runner.vllm_config
    model = model_runner.model
    inner = getattr(model, "model", model)
    mlp = inner.layers[layer_idx].mlp
    device = next(mlp.parameters()).device
    dtype = mlp.router.weight.dtype

    # Get captured inputs (they're stored as float32 CPU tensors).
    inp_bs1_f32 = caps[fwd_bs1][key]  # [1, K]
    inp_bs2_f32 = caps[fwd_bs2][key]  # [M_bs2, K]

    # Convert to model dtype on device.
    inp_bs1 = inp_bs1_f32.to(dtype=dtype, device=device)
    inp_bs2 = inp_bs2_f32.to(dtype=dtype, device=device)
    K = mlp.hidden_size

    # Also build an M=1 version of just the needle row from BS=2.
    inp_needle_only = inp_bs2[row_bs2:row_bs2 + 1].clone()

    def _run(hs):
        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_config,
            num_tokens=hs.shape[0],
        ):
            return mlp(hs)

    with _torch.inference_mode():
        out_bs1 = _run(inp_bs1)        # M=1
        out_bs2 = _run(inp_bs2)        # M=M_bs2
        out_needle = _run(inp_needle_only)  # M=1, same input as needle row

    r_bs1 = out_bs1[0, :K].float()
    r_bs2 = out_bs2[row_bs2, :K].float()
    r_needle = out_needle[0, :K].float()

    diff_main = (r_bs1 - r_bs2).abs()
    diff_needle = (r_bs1 - r_needle).abs()

    return {
        "M_bs1": int(inp_bs1.shape[0]),
        "M_bs2": int(inp_bs2.shape[0]),
        "row_bs2": row_bs2,
        # Main comparison: M=1 vs M=M_bs2
        "bitwise_eq": bool(_torch.equal(r_bs1, r_bs2)),
        "max_abs_diff": float(diff_main.max().item()),
        # Control: M=1 vs M=1 (needle row only)
        "needle_m1_bitwise_eq": bool(_torch.equal(r_bs1, r_needle)),
        "needle_m1_max_diff": float(diff_needle.max().item()),
        # Are the BS=1 and BS=2 needle inputs actually equal?
        "inputs_bitwise_eq": bool(_torch.equal(
            inp_bs1[0].float(), inp_bs2[row_bs2].float()
        )),
    }


def _decomposed_replay_mlp_at_layer(worker, layer_idx: int,
                                    fwd_bs1: int, fwd_bs2: int,
                                    row_bs2: int) -> dict:
    """Decomposed MoE replay: test each sub-operation independently.

    Breaks the MoE layer into: router → topk → moe_align → GEMM1 →
    activation → GEMM2 → moe_sum, and checks each intermediate result
    between M=1 and M=2 for the needle row.  This identifies exactly
    which sub-operation introduces the batch-dependent divergence.
    """
    import torch as _torch

    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        _fused_marlin_moe,
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    _, db = _get_bisect_db(worker)
    caps = db["captures"]
    if fwd_bs1 >= len(caps) or fwd_bs2 >= len(caps):
        return {"error": "index out of range"}

    key = f"L{layer_idx}_mlp_in"
    if key not in caps[fwd_bs1] or key not in caps[fwd_bs2]:
        return {"error": f"key {key} not found in captures"}

    model_runner = worker.model_runner
    vllm_config = model_runner.vllm_config
    model = model_runner.model
    inner = getattr(model, "model", model)
    mlp = inner.layers[layer_idx].mlp
    device = next(mlp.parameters()).device
    dtype = mlp.router.weight.dtype

    inp_bs1_f32 = caps[fwd_bs1][key]
    inp_bs2_f32 = caps[fwd_bs2][key]
    inp_bs1 = inp_bs1_f32.to(dtype=dtype, device=device)
    inp_bs2 = inp_bs2_f32.to(dtype=dtype, device=device)
    K = mlp.hidden_size

    results = {}

    with _torch.inference_mode():
        # ---- Step 1: Router ----
        with set_forward_context(attn_metadata=None, vllm_config=vllm_config,
                                 num_tokens=1):
            g_bs1 = mlp.router(inp_bs1)
        with set_forward_context(attn_metadata=None, vllm_config=vllm_config,
                                 num_tokens=2):
            g_bs2 = mlp.router(inp_bs2)
        # ReplicatedLinear returns (output, bias) tuple
        if isinstance(g_bs1, tuple):
            g_bs1 = g_bs1[0]
        if isinstance(g_bs2, tuple):
            g_bs2 = g_bs2[0]
        results["router_logits_eq"] = bool(
            _torch.equal(g_bs1[0].float(), g_bs2[row_bs2].float()))
        results["router_max_diff"] = float(
            (g_bs1[0].float() - g_bs2[row_bs2].float()).abs().max().item())

        # ---- Step 2: TopK routing ----
        experts_mod = mlp.experts
        router = experts_mod.router
        tw_bs1, ti_bs1 = router.select_experts(
            hidden_states=inp_bs1, router_logits=g_bs1)
        tw_bs2, ti_bs2 = router.select_experts(
            hidden_states=inp_bs2, router_logits=g_bs2)

        results["topk_ids_eq"] = bool(
            _torch.equal(ti_bs1[0], ti_bs2[row_bs2]))
        results["topk_weights_eq"] = bool(
            _torch.equal(tw_bs1[0].float(), tw_bs2[row_bs2].float()))
        results["topk_ids_bs1"] = ti_bs1[0].tolist()
        results["topk_ids_bs2_needle"] = ti_bs2[row_bs2].tolist()
        results["topk_weights_bs1"] = [
            f"{x:.6f}" for x in tw_bs1[0].tolist()]
        results["topk_weights_bs2_needle"] = [
            f"{x:.6f}" for x in tw_bs2[row_bs2].tolist()]

        # ---- Step 3: moe_align_block_size ----
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        block_size_m = 64
        E = ti_bs1.shape[1]  # This is topk, not num_experts
        global_num_experts = experts_mod.global_num_experts

        stids_bs1, eids_bs1, ntpp_bs1 = moe_align_block_size(
            ti_bs1, block_size_m, global_num_experts, None)
        stids_bs2, eids_bs2, ntpp_bs2 = moe_align_block_size(
            ti_bs2, block_size_m, global_num_experts, None)

        # For each expert in BS=1, find which block-local position the
        # needle's slot is at
        topk = ti_bs1.shape[1]
        results["moe_align_num_blocks_bs1"] = int(ntpp_bs1.item()) // block_size_m
        results["moe_align_num_blocks_bs2"] = int(ntpp_bs2.item()) // block_size_m

        # ---- Step 4: Call fused_marlin_moe with identical routing ----
        # Use BS=1's routing for BOTH calls — this isolates whether the
        # kernel itself (given identical routing) produces M-dependent output.
        #
        # Build a synthetic M=2 input where row 0 = BS=1 input and
        # row 1 = BS=1 input (duplicate), with BS=1's topk_ids/weights
        # replicated.
        hs_dup = inp_bs1.expand(2, -1).contiguous()  # [2, K]
        ti_dup = ti_bs1.expand(2, -1).contiguous()    # [2, topk]
        tw_dup = tw_bs1.expand(2, -1).contiguous()    # [2, topk]

        # Call the full MoE (mlp) with controlled inputs
        def _run_mlp(hs):
            with set_forward_context(attn_metadata=None,
                                     vllm_config=vllm_config,
                                     num_tokens=hs.shape[0]):
                return mlp(hs)

        out_bs1_full = _run_mlp(inp_bs1)
        # Also try calling with duplicated M=2 using same input
        out_dup = _run_mlp(hs_dup)

        r_bs1 = out_bs1_full[0, :K].float()
        r_dup = out_dup[0, :K].float()
        diff_dup = (r_bs1 - r_dup).abs()

        results["dup_m2_bitwise_eq"] = bool(_torch.equal(r_bs1, r_dup))
        results["dup_m2_max_diff"] = float(diff_dup.max().item())

        # Sweep multiple random row-1 seeds to find the failure pattern.
        # For each seed, build M=2 = [real_row0, random_row1] and check
        # whether the MoE output for row 0 matches the M=1 baseline.
        sweep_results = []
        for seed in range(20):
            _torch.manual_seed(seed)
            rand_row = _torch.randn(1, inp_bs1.shape[1],
                                    device=device, dtype=dtype)
            mixed = _torch.cat([inp_bs1, rand_row], dim=0)
            # Get routing for the random row to see expert overlap
            with set_forward_context(attn_metadata=None,
                                     vllm_config=vllm_config,
                                     num_tokens=2):
                g_mixed = mlp.router(mixed)
            if isinstance(g_mixed, tuple):
                g_mixed = g_mixed[0]
            _, ti_mixed = router.select_experts(
                hidden_states=mixed, router_logits=g_mixed)
            row1_experts = ti_mixed[1].tolist()
            row0_experts = set(ti_bs1[0].tolist())
            overlap = len(row0_experts & set(row1_experts))

            out_mixed = _run_mlp(mixed)
            r_mixed = out_mixed[0, :K].float()
            diff = (r_bs1 - r_mixed).abs()
            eq = bool(_torch.equal(r_bs1, r_mixed))
            md = float(diff.max().item())
            argmax = int(diff.argmax().item()) if md > 0 else -1
            # For failures, also get the actual values at the diff point
            val_bs1 = float(r_bs1[argmax].item()) if argmax >= 0 else 0
            val_mixed = float(r_mixed[argmax].item()) if argmax >= 0 else 0
            n_diff_elems = int((diff > 0).sum().item())
            sweep_results.append({
                "seed": seed, "eq": eq, "max_diff": md,
                "overlap": overlap,
                "row1_experts": row1_experts,
                "argmax": argmax,
                "val_bs1": val_bs1,
                "val_mixed": val_mixed,
                "n_diff_elems": n_diff_elems,
            })

        n_fail = sum(1 for s in sweep_results if not s["eq"])
        results["sweep_n_fail"] = n_fail
        results["sweep_n_total"] = len(sweep_results)
        # Collect failure details for printing
        fail_details = []
        for s in sweep_results:
            if not s["eq"]:
                fail_details.append(
                    f"seed={s['seed']:02d} diff={s['max_diff']:.4f} "
                    f"argmax={s['argmax']} "
                    f"val_bs1={s['val_bs1']:.6f} "
                    f"val_mixed={s['val_mixed']:.6f} "
                    f"n_diff_elems={s['n_diff_elems']} "
                    f"overlap={s['overlap']} "
                    f"row1_experts={s['row1_experts']}"
                )
        results["sweep_failures"] = fail_details

        # Also try the actual BS=2 input
        out_bs2_full = _run_mlp(inp_bs2)
        r_bs2 = out_bs2_full[row_bs2, :K].float()
        diff_real = (r_bs1 - r_bs2).abs()
        results["real_m2_bitwise_eq"] = bool(_torch.equal(r_bs1, r_bs2))
        results["real_m2_max_diff"] = float(diff_real.max().item())

    return results


def _compare_decode_bisect(worker, fwd_bs1: int, row_bs1: int,
                           fwd_bs2: int, row_bs2: int) -> list[dict]:
    """Compare hidden states at every layer between two captures.

    Returns a list of per-layer dicts with bitwise-eq flag and max diff.
    If an index is out of range, returns a single-element error list.
    """
    import torch as _torch

    _, db = _get_bisect_db(worker)
    caps = db["captures"]
    n_caps = len(caps)
    if fwd_bs1 >= n_caps or fwd_bs2 >= n_caps:
        return [{
            "key": "__ERROR__",
            "bitwise_eq": False,
            "max_abs_diff": -1.0,
            "max_diff_idx": -1,
            "error": (f"index out of range: fwd_bs1={fwd_bs1}, "
                      f"fwd_bs2={fwd_bs2}, n_caps={n_caps}"),
        }]
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
    max_tokens = 16

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

        # NOTE: Do NOT remove hooks/captures here — _compare_decode_bisect
        # needs the captures that live on the worker.  Clean up after all
        # comparisons are done (see below).

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
                f"  fwd={s['fwd']} M={s['M']} "
                f"n_keys={s['n_keys']} positions=[{pos_str}]",
                flush=True,
            )
        print("--- BS=2 forward schedule ---", flush=True)
        for s in summaries_bs2:
            pos_str = ",".join(str(p) for p in s["positions"][:8])
            if len(s["positions"]) > 8:
                pos_str += f"...({len(s['positions'])} total)"
            print(
                f"  fwd={s['fwd']} M={s['M']} "
                f"n_keys={s['n_keys']} positions=[{pos_str}]",
                flush=True,
            )

        # BS=1 decode forwards: M=1, position >= prompt_len, complete
        bs1_decode_fwds: list[tuple[int, int]] = []  # (abs_fwd_idx, position)
        for s in summaries_bs1:
            if s["M"] == 1 and s["n_keys"] > 0:
                pos = s["positions"][0] if s["positions"] else -1
                if pos >= needle_prompt_len:
                    bs1_decode_fwds.append((s["fwd"], pos))

        # BS=2 decode forwards: find the needle's row by matching
        # position >= needle_prompt_len. Skip incomplete captures.
        bs2_decode_fwds: list[tuple[int, int, int]] = []  # (abs, pos, row)
        for s in summaries_bs2:
            if s["n_keys"] == 0:
                continue  # skip incomplete warmup forwards
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

            # Check for error sentinel from bounds-checking.
            if layer_results and layer_results[0].get("error"):
                print(
                    f"  ERROR: {layer_results[0]['error']}",
                    flush=True,
                )
                continue

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
                # If the first divergence is at an mlp_out, replay the
                # MoE standalone to confirm the bug is inside the layer.
                if "_mlp_out" in first_diff_key:
                    layer_idx = int(
                        first_diff_key.split("_")[0][1:]
                    )
                    print(
                        f"\n--- Replaying MoE at layer {layer_idx} "
                        f"with captured inputs ---",
                        flush=True,
                    )
                    replay = llm.llm_engine.collective_rpc(
                        lambda w, li=layer_idx, f1=abs_fwd_bs1,
                        f2=abs_fwd_bs2, r2=needle_row: (
                            _replay_mlp_at_layer(
                                w, li, f1, f2, r2
                            )
                        ),
                    )[0]
                    for k, v in sorted(replay.items()):
                        print(f"  {k}: {v}", flush=True)

                    # Decomposed replay: test each sub-op
                    print(
                        f"\n--- Decomposed MoE replay at layer "
                        f"{layer_idx} ---",
                        flush=True,
                    )
                    decomp = llm.llm_engine.collective_rpc(
                        lambda w, li=layer_idx, f1=abs_fwd_bs1,
                        f2=abs_fwd_bs2, r2=needle_row: (
                            _decomposed_replay_mlp_at_layer(
                                w, li, f1, f2, r2
                            )
                        ),
                    )[0]
                    for k, v in sorted(decomp.items()):
                        print(f"  {k}: {v}", flush=True)

        if not found_first_diff:
            print(
                "\nAll layers bitwise-equal at all matched decode steps!",
                flush=True,
            )

        # Clean up hooks and captures now that comparisons are done.
        llm.llm_engine.collective_rpc(_remove_decode_bisect_hooks)

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
