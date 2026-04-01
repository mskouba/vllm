# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diagnostic logging for MoE non-determinism debugging.

Enable via VLLM_MOE_DETERMINISM_DEBUG=1. Optionally set
VLLM_MOE_DETERMINISM_DEBUG_PATH to write JSONL to a file
instead of the vLLM logger.

When disabled, all public functions are effectively no-ops
(single boolean check).
"""

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

import torch

import vllm.envs as envs

logger = logging.getLogger("vllm.moe_determinism")

_enabled: bool | None = None
_file_logger: logging.Logger | None = None
_step_counter = 0
_step_lock = threading.Lock()


def is_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = envs.VLLM_MOE_DETERMINISM_DEBUG
        if _enabled:
            _setup_logger()
    return _enabled


def _setup_logger():
    global _file_logger
    path = envs.VLLM_MOE_DETERMINISM_DEBUG_PATH
    _file_logger = logging.getLogger("vllm.moe_determinism.jsonl")
    _file_logger.setLevel(logging.DEBUG)
    _file_logger.propagate = False

    if path:
        handler = logging.FileHandler(path, mode="a")
    else:
        import sys
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(logging.Formatter("%(message)s"))
    _file_logger.addHandler(handler)


def _next_step() -> int:
    global _step_counter
    with _step_lock:
        _step_counter += 1
        return _step_counter


def _emit(event: dict[str, Any]):
    if _file_logger is not None:
        _file_logger.debug(json.dumps(event, default=str))


def tensor_fingerprint(t: torch.Tensor) -> dict[str, Any]:
    """Compute a cheap scalar fingerprint of a tensor."""
    ft = t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "l2_norm": float(ft.norm().item()),
        "abs_max": float(ft.abs().max().item()),
        "sum": float(ft.sum().item()),
    }


def log_routing_decision(
    layer_name: str,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    num_tokens: int,
) -> None:
    if not is_enabled():
        return

    step = _next_step()

    # Expert histogram
    expert_ids = topk_ids.flatten().tolist()
    histogram: dict[str, int] = {}
    for eid in expert_ids:
        key = str(eid)
        histogram[key] = histogram.get(key, 0) + 1

    # Weight stats
    w = topk_weights.float()
    weight_stats = {
        "mean": float(w.mean().item()),
        "std": float(w.std().item()),
        "min": float(w.min().item()),
        "max": float(w.max().item()),
    }

    # Sample topk_ids and weights for small batches
    if num_tokens <= 64:
        ids_sample = topk_ids.tolist()
        weights_sample = topk_weights.float().tolist()
    else:
        ids_sample = (topk_ids[:8].tolist() + ["..."] +
                      topk_ids[-8:].tolist())
        weights_sample = (topk_weights[:8].float().tolist() + ["..."] +
                          topk_weights[-8:].float().tolist())

    _emit({
        "ts": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "layer": layer_name,
        "event": "routing_decision",
        "num_tokens": num_tokens,
        "expert_histogram": histogram,
        "topk_weights_stats": weight_stats,
        "topk_ids_sample": ids_sample,
        "topk_weights_sample": weights_sample,
        "router_logits_fingerprint": tensor_fingerprint(router_logits),
    })


def log_kernel_config(
    layer_name: str,
    M: int,
    E: int,
    N: int,
    K: int,
    config: Any,
    source: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if not is_enabled():
        return

    event: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "step": _step_counter,
        "layer": layer_name,
        "event": "kernel_config",
        "M": M,
        "E": E,
        "N": N,
        "K": K,
        "config": str(config),
        "source": source,
    }
    if extra:
        event["extra"] = extra
    _emit(event)


def log_tensor_checkpoint(
    layer_name: str,
    checkpoint_name: str,
    tensor: torch.Tensor,
) -> None:
    if not is_enabled():
        return

    _emit({
        "ts": datetime.now(timezone.utc).isoformat(),
        "step": _step_counter,
        "layer": layer_name,
        "event": "tensor_checkpoint",
        "checkpoint_name": checkpoint_name,
        "fingerprint": tensor_fingerprint(tensor),
    })


def log_moe_output(
    layer_name: str,
    hidden_states_in: torch.Tensor,
    hidden_states_out: torch.Tensor,
) -> None:
    if not is_enabled():
        return

    _emit({
        "ts": datetime.now(timezone.utc).isoformat(),
        "step": _step_counter,
        "layer": layer_name,
        "event": "moe_output",
        "input_fingerprint": tensor_fingerprint(hidden_states_in),
        "output_fingerprint": tensor_fingerprint(hidden_states_out),
    })
