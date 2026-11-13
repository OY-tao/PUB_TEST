"""CPU golden for TND backward with seqused prefixes.

Allocated lengths (sq/sk) define the packed region and cu_seqlens.
Used lengths (uq/uk) are the prefixes the kernel is allowed to read.
Tokens past the prefix stay in the packed tensor and must not affect dq/dk/dv.
Gradients on those unused tokens are zero: the device kernel leaves them
unspecified, and the ATK compare is on the full packed tensor.
"""

import torch

from fa_small_op_golden_same_input import (
    golden_tnd_bwd_grads,
    golden_tnd_fwd_only,
)


def cumulative(seqlens):
    cu = [0]
    for length in seqlens:
        cu.append(cu[-1] + int(length))
    return cu


def take_lengths(used_q, used_k):
    """Drop a batch from the compact golden when either used side is empty."""
    take_q = []
    take_k = []
    for uq, uk in zip(used_q, used_k):
        uq = int(uq)
        uk = int(uk)
        if uq > 0 and uk > 0:
            take_q.append(uq)
            take_k.append(uk)
        else:
            take_q.append(0)
            take_k.append(0)
    return take_q, take_k


def zero_unused_prefix_(tensor, region_lens, used_lens):
    """Zero token rows [used, region) inside each packed batch region."""
    cursor = 0
    for region, used in zip(region_lens, used_lens):
        region = int(region)
        used = int(used)
        if used < 0 or used > region:
            raise ValueError(f"used length {used} is outside allocated length {region}")
        if used < region:
            tensor[cursor + used : cursor + region].zero_()
        cursor += region
    if cursor != tensor.shape[0]:
        raise ValueError(
            f"region lengths sum to {cursor}, tensor token axis is {tensor.shape[0]}"
        )
    return tensor


def _compact(tensor, region_lens, take_lens):
    parts = []
    cursor = 0
    for region, take in zip(region_lens, take_lens):
        region = int(region)
        take = int(take)
        if take > 0:
            parts.append(tensor[cursor : cursor + take])
        cursor += region
    if not parts:
        return tensor[:0].contiguous()
    return torch.cat(parts, dim=0).contiguous()


def _scatter(compact, region_lens, take_lens, like):
    out = torch.zeros_like(like)
    cursor = 0
    offset = 0
    for region, take in zip(region_lens, take_lens):
        region = int(region)
        take = int(take)
        if take > 0:
            out[cursor : cursor + take] = compact[offset : offset + take]
            offset += take
        cursor += region
    return out


def _scatter_lse(lse_nt, region_q, take_q, total_q):
    nheads = lse_nt.shape[0]
    full = torch.zeros(nheads, int(total_q), dtype=lse_nt.dtype)
    cursor = 0
    offset = 0
    for region, take in zip(region_q, take_q):
        region = int(region)
        take = int(take)
        if take > 0:
            full[:, cursor : cursor + take] = lse_nt[:, offset : offset + take]
            offset += take
        cursor += region
    return full.contiguous()


def golden_tnd_seqused(
    q,
    k,
    v,
    dout,
    nheads,
    nheads_k,
    seqlens_q,
    seqlens_k,
    used_q,
    used_k,
    scale,
    is_causal,
    window_size_left,
    window_size_right,
    *,
    softcap=0.0,
    native_dtype=None,
    dropout_p=0.0,
    compute_dtype=None,
    return_softmax_stats=False,
):
    """Forward+backward on used prefixes, scattered back into allocated TND.

    Returns out, softmax_lse, dq, dk, dv. out/lse/grads are zero on unused tails.
    softmax_lse is float32 NT (N, total_allocated_q), matching the 950 bwd ABI.
    With return_softmax_stats=True also returns softmax_max/sum as float32
    TND (total_allocated_q, N, 8) for npu_fusion_attention_grad.
    compute_dtype defaults to fp64 (golden). Pass native fp16/bf16 for a
    low-precision double-benchmark reference.
    """
    seqlens_q = [int(x) for x in seqlens_q]
    seqlens_k = [int(x) for x in seqlens_k]
    used_q = [int(x) for x in used_q]
    used_k = [int(x) for x in used_k]
    if not (len(seqlens_q) == len(seqlens_k) == len(used_q) == len(used_k)):
        raise ValueError("sq/sk/uq/uk batch sizes differ")
    if sum(seqlens_q) != q.shape[0] or sum(seqlens_k) != k.shape[0]:
        raise ValueError(
            f"packed shapes {(q.shape[0], k.shape[0])} != "
            f"sum(sq), sum(sk) = {(sum(seqlens_q), sum(seqlens_k))}"
        )
    for uq, sq, uk, sk in zip(used_q, seqlens_q, used_k, seqlens_k):
        if uq < 0 or uq > sq or uk < 0 or uk > sk:
            raise ValueError(
                f"seqused {(uq, uk)} outside allocated region {(sq, sk)}"
            )

    take_q, take_k = take_lengths(used_q, used_k)
    total_q = q.shape[0]
    if sum(take_q) == 0 or sum(take_k) == 0:
        out = torch.zeros_like(q)
        lse = torch.zeros(nheads, total_q, dtype=torch.float32)
        base = (
            out,
            lse,
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
        )
        if return_softmax_stats:
            zeros_tn8 = torch.zeros(total_q, nheads, 8, dtype=torch.float32)
            return base + (zeros_tn8, torch.ones_like(zeros_tn8))
        return base

    q_c = _compact(q, seqlens_q, take_q)
    k_c = _compact(k, seqlens_k, take_k)
    v_c = _compact(v, seqlens_k, take_k)
    dout_c = _compact(dout, seqlens_q, take_q)
    active_q = [length for length in take_q if length > 0]
    active_k = [length for length in take_k if length > 0]

    fwd_ret = golden_tnd_fwd_only(
        q_c,
        k_c,
        v_c,
        nheads,
        nheads_k,
        active_q,
        active_k,
        scale,
        dropout_p,
        is_causal,
        window_size_left,
        window_size_right,
        softcap=softcap,
        native_dtype=native_dtype,
        compute_dtype=compute_dtype,
        return_softmax_stats=return_softmax_stats,
    )
    if return_softmax_stats:
        out_c, lse_c, softmax_res, drop_masks, softmax_max_c, softmax_sum_c = fwd_ret
    else:
        out_c, lse_c, softmax_res, drop_masks = fwd_ret
    # When compute runs in native fp16/bf16, quantize softmax_res the same way
    # so the BM path differs from the fp64 golden (needed for double-benchmark).
    if compute_dtype is not None and compute_dtype in (torch.float16, torch.bfloat16):
        softmax_res = [
            None if s is None else s.to(compute_dtype).to(torch.float32)
            for s in softmax_res
        ]
    dq_c, dk_c, dv_c = golden_tnd_bwd_grads(
        q_c,
        k_c,
        v_c,
        dout_c,
        softmax_res,
        drop_masks,
        nheads,
        nheads_k,
        active_q,
        active_k,
        scale,
        dropout_p,
        softcap=softcap,
        compute_dtype=torch.float32 if compute_dtype in (torch.float16, torch.bfloat16) else compute_dtype,
    )
    out = _scatter(out_c, seqlens_q, take_q, q)
    dq = _scatter(dq_c, seqlens_q, take_q, q)
    dk = _scatter(dk_c, seqlens_k, take_k, k)
    dv = _scatter(dv_c, seqlens_k, take_k, v)
    lse = _scatter_lse(lse_c, seqlens_q, take_q, total_q)
    if not return_softmax_stats:
        return out, lse, dq, dk, dv
    like_tn8 = torch.zeros(total_q, nheads, 8, dtype=torch.float32)
    softmax_max = _scatter(softmax_max_c.to(torch.float32), seqlens_q, take_q, like_tn8)
    softmax_sum = _scatter(
        softmax_sum_c.to(torch.float32),
        seqlens_q,
        take_q,
        torch.ones_like(like_tn8),
    )
    return out, lse, dq, dk, dv, softmax_max, softmax_sum
