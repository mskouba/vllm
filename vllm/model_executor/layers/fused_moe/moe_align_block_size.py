# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


def _deterministic_sort_expert_tokens(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size: int,
    numel: int,
) -> None:
    """Sort token indices within each expert group to make ordering
    deterministic. The CUDA kernel uses atomicAdd to assign positions within
    each expert's group, which produces non-deterministic ordering. This
    function sorts each expert's token indices in ascending order so that the
    same set of tokens always appears in the same order regardless of GPU
    thread scheduling.

    Args:
        sorted_ids: Token indices grouped by expert, with padding.
            Padding tokens have value >= numel.
        expert_ids: Expert index for each block of block_size tokens.
        num_tokens_post_pad: Total tokens after padding (scalar tensor).
        block_size: Block size used for alignment.
        numel: Total number of real token assignments (num_tokens * top_k).
    """
    total = num_tokens_post_pad.item()
    num_blocks = total // block_size

    # Find expert boundaries using expert_ids on GPU.
    eid = expert_ids[:num_blocks]
    # Detect where expert changes: compare adjacent expert_ids.
    if num_blocks <= 1:
        # Only one block, sort it directly.
        if num_blocks == 1:
            segment = sorted_ids[:block_size]
            sorted_ids[:block_size] = segment.sort().values
        return

    changes = (eid[1:] != eid[:-1]).nonzero(as_tuple=True)[0] + 1
    # boundaries: [0, change1, change2, ..., num_blocks]
    boundaries = torch.cat([
        torch.zeros(1, dtype=changes.dtype, device=changes.device),
        changes,
        torch.tensor([num_blocks], dtype=changes.dtype, device=changes.device),
    ])
    boundaries_cpu = boundaries.cpu()

    sorted_ids_view = sorted_ids[:total]
    for idx in range(len(boundaries_cpu) - 1):
        start_block = boundaries_cpu[idx].item()
        end_block = boundaries_cpu[idx + 1].item()
        # Skip invalid experts.
        if expert_ids[start_block].item() == -1:
            continue
        start = start_block * block_size
        end = end_block * block_size
        # Sort on GPU — this is a small sort per expert group.
        segment = sorted_ids_view[start:end]
        sorted_ids_view[start:end] = segment.sort().values


def _pad_expert_tokens_to_fixed_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None,
    pad_sorted_ids: bool,
    ignore_invalid_experts: bool,
    min_expert_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad each expert's token allocation to at least min_expert_tokens.

    This ensures that per-expert matmul dimensions are more consistent across
    batches, reducing cuBLAS tiling non-determinism within experts.

    The approach: count tokens per expert, then inflate each expert's count
    to at least min_expert_tokens before computing the aligned layout.

    We do this by creating synthetic topk_ids entries that route padding tokens
    to under-represented experts. Then we run the standard moe_align_block_size
    on the inflated topk_ids.
    """
    numel = topk_ids.numel()
    flat_topk = topk_ids.view(-1)

    # Count tokens per expert.
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    for e in range(num_experts):
        counts[e] = (flat_topk == e).sum()

    # Compute how many padding tokens each expert needs.
    min_aligned = round_up(min_expert_tokens, block_size)
    pad_counts = (min_aligned - counts).clamp(min=0)
    total_pad = pad_counts.sum().item()

    if total_pad == 0:
        # All experts already have enough tokens, use standard path.
        return _moe_align_block_size_core(
            topk_ids, block_size, num_experts,
            expert_map, pad_sorted_ids, ignore_invalid_experts,
        )

    # Run the standard kernel, then re-pad each expert group in sorted_ids
    # to min_aligned.
    max_num_tokens_padded = numel + num_experts * (min_aligned - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)

    # Ensure we have enough space: each expert gets at least min_aligned slots.
    max_num_tokens_padded = max(max_num_tokens_padded, num_experts * min_aligned)

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )

    # Run the standard CUDA kernel first.
    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )

    # Now re-layout: ensure each expert has at least min_aligned slots.
    _repad_expert_groups(
        sorted_ids, expert_ids, num_tokens_post_pad,
        block_size, num_experts, min_aligned, numel,
        max_num_tokens_padded,
    )

    if expert_map is not None and not ignore_invalid_experts:
        total = num_tokens_post_pad.item()
        num_blocks_total = total // block_size
        expert_ids[:num_blocks_total] = expert_map[
            expert_ids[:num_blocks_total]
        ]

    return sorted_ids, expert_ids, num_tokens_post_pad


def _repad_expert_groups(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size: int,
    num_experts: int,
    min_aligned: int,
    numel: int,
    max_num_tokens_padded: int,
) -> None:
    """Re-layout sorted_ids and expert_ids so each expert gets at least
    min_aligned token slots. Experts with fewer real tokens get padding
    (sentinel values = numel) to fill the remaining slots.

    This function also deterministically sorts tokens within each expert
    group for ordering consistency.
    """
    total = num_tokens_post_pad.item()
    num_blocks = total // block_size

    # Find expert boundaries and gather tokens per expert.
    # Do this on CPU since we need to rebuild the layout.
    eid_cpu = expert_ids[:num_blocks].cpu().tolist()
    sid_cpu = sorted_ids[:total].cpu().tolist()

    # Collect tokens per expert from the kernel output.
    expert_tokens: list[list[int]] = [[] for _ in range(num_experts)]
    for i in range(num_blocks):
        e = eid_cpu[i]
        if e == -1:
            continue
        start = i * block_size
        end = start + block_size
        expert_tokens[e].extend(sid_cpu[start:end])

    # Rebuild layout with min_aligned padding per expert.
    new_sorted: list[int] = []
    new_expert_ids: list[int] = []
    for e in range(num_experts):
        tokens = sorted_ids.new_tensor(expert_tokens[e]).sort().values.tolist()
        padded_len = max(round_up(len(tokens), block_size), min_aligned)
        tokens.extend([numel] * (padded_len - len(tokens)))
        new_sorted.extend(tokens)
        for _ in range(padded_len // block_size):
            new_expert_ids.append(e)

    new_total = len(new_sorted)
    new_num_blocks = len(new_expert_ids)

    # Write back to GPU tensors.
    sorted_ids[:new_total] = torch.tensor(
        new_sorted, dtype=torch.int32, device=sorted_ids.device
    )
    if new_total < max_num_tokens_padded:
        sorted_ids[new_total:max_num_tokens_padded] = numel
    expert_ids[:new_num_blocks] = torch.tensor(
        new_expert_ids, dtype=torch.int32, device=expert_ids.device
    )
    if new_num_blocks < expert_ids.size(0):
        expert_ids[new_num_blocks:] = -1
    num_tokens_post_pad.fill_(new_total)


def _moe_align_block_size_core(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core implementation: runs the CUDA kernel and optionally applies
    deterministic sorting. Does NOT dispatch to per-expert padding."""
    deterministic = envs.VLLM_DETERMINISTIC_BATCH_PADDING

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )

    if deterministic:
        _deterministic_sort_expert_tokens(
            sorted_ids, expert_ids, num_tokens_post_pad,
            block_size, topk_ids.numel(),
        )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
        should be padded to a multiple of block_size,
    - ignore_invalid_experts: A flag indicating whether to ignore invalid
        experts. When False, all expert_ids in topk_ids will participate in
        counting and ranking, but invalid experts in expert_ids will be marked
        as -1. When True, all invalid expert_ids in topk_ids will be ignored
        and will not participate in counting or ranking, and there will be no
        -1 in expert_ids.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    # Per-expert padding: ensure each expert gets a minimum number of tokens
    # so that per-expert matmul dimensions are consistent across batches.
    if (envs.VLLM_DETERMINISTIC_BATCH_PADDING
            and envs.VLLM_MOE_EXPERT_MIN_TOKENS > 0):
        return _pad_expert_tokens_to_fixed_size(
            topk_ids, block_size, num_experts,
            expert_map, pad_sorted_ids, ignore_invalid_experts,
            min_expert_tokens=envs.VLLM_MOE_EXPERT_MIN_TOKENS,
        )

    return _moe_align_block_size_core(
        topk_ids, block_size, num_experts,
        expert_map, pad_sorted_ids, ignore_invalid_experts,
    )


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Parameters:
    - max_tokens_per_batch (int): Number of tokens in each batch (both
        valid and invalid).
    - block_size (int): block_size to align the data to.
    - expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
        the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.
    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.
    """

    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
