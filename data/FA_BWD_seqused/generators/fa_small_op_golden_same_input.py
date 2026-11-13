"""
小算子 FlashAttention 标杆（same_input）。

正向：
  - 全程在 CPU 上、float64 计算
  - out：先 cast 到原始 dtype（fp16/bf16），再 cast 到当前输入 dtype
  - lse：max/sum 经 native quantize→upcast 后计算为 fp32（对齐 torch_fag）
"""

import torch


def cast_via_native(tensor, native_dtype, current_dtype):
    """fp64/... -> native(fp16/bf16) -> current_dtype（如 fp32）。"""
    if native_dtype is None:
        native_dtype = current_dtype
    t = tensor.detach()
    if t.dtype != native_dtype:
        t = t.to(native_dtype)
    if t.dtype != current_dtype:
        t = t.to(current_dtype)
    return t.contiguous()


def broadcast_kv_single(num_heads, num_kv_heads, kv_tensor, dtype):
    factor = num_heads // num_kv_heads
    b, _, s, d = kv_tensor.shape
    kv_res = torch.zeros(
        [b, num_heads, s, d], dtype=dtype, device=kv_tensor.device
    )
    for i in range(num_heads):
        j = i // factor
        kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    return kv_res


def normalize_window(seq_q, seq_k, is_causal, window_size_left, window_size_right):
    wl = window_size_left
    wr = window_size_right
    if wl >= seq_k - 1:
        wl = -1
    if wr >= seq_q - 1:
        wr = -1
    if is_causal:
        wr = 0
    is_causal_out = wl < 0 and wr == 0
    is_local = (wl >= 0 or wr >= 0) and not is_causal_out
    return wl, wr, is_causal_out, is_local


def make_window_atten_mask(
    seq_q,
    seq_k,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
):
    wl, wr, is_causal_out, is_local = normalize_window(
        seq_q, seq_k, is_causal, window_size_left, window_size_right
    )
    if not is_causal_out and not is_local:
        return torch.tensor(0)

    offset = seq_k - seq_q
    shape = (seq_q, seq_k)
    if is_causal_out:
        return torch.triu(torch.ones(shape), diagonal=offset + 1)

    mask = torch.zeros(shape)
    if wr >= 0:
        mask = mask + torch.triu(torch.ones(shape), diagonal=wr + 1 + offset)
    if wl >= 0:
        mask = mask + torch.tril(torch.ones(shape), diagonal=-wl - 1 + offset)
    return mask


def _resolve_atten_mask(q, k, atten_mask, is_causal, window_size_left, window_size_right):
    if atten_mask is not None and len(atten_mask.shape) != 0:
        return atten_mask
    seq_q = q.shape[2]
    seq_k = k.shape[2]
    return make_window_atten_mask(
        seq_q, seq_k, is_causal, window_size_left, window_size_right
    )


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    return (dp - muls_r) * softmax_res


def _keep_prob(dropout_p):
    return 1.0 - dropout_p


def make_rng_state(seed, offset=0):
    """CPU int64[2]=(seed, offset); fag_general_host requires CPU tensor when p_dropout>0."""
    return torch.tensor([int(seed), int(offset)], dtype=torch.int64, device="cpu")


def get_drop_mask(q, B, N, S1, S2, dropout_p, seed, offset=0, device=None):
    """
    用 NPU torch 接口生成 drop_mask（1=keep），对齐:
      flash_api.cpp / fag_general_host:
        bit_num = B*N*S1*align8(S2)
        npu_dropout_gen_mask(empty_float, {bit_num}, p, seed, offset, false, false)
      dailyauto_fusion_grad_only.get_drop_mask:
        torch_npu._npu_dropout_gen_mask + np.unpackbits(..., bitorder='little')

    Returns:
      float mask (B,N,S1,S2) on CPU，或标量 tensor(1)（无 dropout）
    """
    if dropout_p is None or float(dropout_p) <= 0.0:
        return torch.tensor(1)

    import numpy as np
    import torch_npu  # noqa: F401

    B, N, S1, S2 = int(B), int(N), int(S1), int(S2)
    sk_aligned = ((S2 + 7) // 8) * 8
    bit_num = B * N * S1 * sk_aligned
    seed = int(seed)
    offset = int(offset)

    torch.npu.set_compile_mode(jit_compile=False)
    if device is not None:
        torch.npu.set_device(device)

    if q is not None and torch.is_tensor(q):
        template = q.detach()
        if template.device.type != "npu":
            template = template.to("npu")
    else:
        template = torch.empty(0, device="npu", dtype=torch.float32)

    packed = torch_npu._npu_dropout_gen_mask(
        template,
        [bit_num],
        p=float(dropout_p),
        seed=seed,
        offset=offset,
        parallel=False,
        sync=False,
    )
    torch.npu.synchronize()

    u8 = packed.detach().cpu().contiguous().numpy()
    if u8.dtype != np.uint8:
        u8 = packed.detach().cpu().contiguous().view(torch.uint8).numpy()
    bits = np.unpackbits(u8, count=bit_num, bitorder="little")
    bits = bits.reshape(B, N, S1, sk_aligned)[:, :, :, :S2]
    return torch.from_numpy(bits.copy()).to(dtype=torch.float32)


def get_drop_masks_varlen(
    seqlens_q, seqlens_k, nheads, max_seqlen_q, max_seqlen_k, dropout_p, seed, offset=0, q=None
):
    """
    Varlen：按 host 布局生成一块 dense mask
      B*N*max_Sq*align8(max_Sk)，再按 sample 切成 (1,N,sq,sk)。
    """
    batch = len(seqlens_q)
    if dropout_p is None or float(dropout_p) <= 0.0 or batch == 0:
        return [torch.tensor(1) for _ in seqlens_q]

    dense = get_drop_mask(
        q,
        batch,
        nheads,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        seed,
        offset=offset,
    )
    out = []
    for i, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k)):
        if sq == 0 or sk == 0:
            out.append(torch.tensor(1))
        else:
            out.append(dense[i : i + 1, :, :sq, :sk].contiguous())
    return out


def make_drop_mask(shape, dropout_p, seed, dtype=torch.float32):
    """
    确定性 Bernoulli drop_mask（1=keep, 0=drop）。
    shape: attention 分数布局，通常 (B, N, Sq, Sk)。
    dropout_p<=0 时返回标量 tensor(1)。
    """
    if dropout_p is None or float(dropout_p) <= 0.0:
        return torch.tensor(1)
    keep_prob = _keep_prob(float(dropout_p))
    if keep_prob >= 1.0:
        return torch.tensor(1)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) & 0xFFFFFFFF)
    u = torch.rand(shape, generator=rng, dtype=torch.float32)
    return (u < keep_prob).to(dtype)


def _softcap_backward_factor(q, k, scale, softcap, compute_dtype, *, gtype=torch.float64):
    if softcap <= 0.0:
        return None
    qb = q.to(compute_dtype)
    kb = k.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    tanh_qk = torch.tanh(qk.to(gtype) / softcap)
    return 1.0 - tanh_qk * tanh_qk


def sum_gqa_grad(dk_or_dv, nheads, nheads_k, batch, seq_k, headdim):
    if nheads == nheads_k:
        return dk_or_dv
    g = nheads // nheads_k
    return (
        torch.sum(
            dk_or_dv.reshape(batch, nheads_k, g, seq_k, headdim),
            dim=2,
            keepdim=True,
        ).reshape(batch, nheads_k, seq_k, headdim)
    )


def _matmul_bwd(a, b):
    """非 CPU 设备上用 out_dtype=float32 做反向 matmul，CPU 保持原样。"""
    if a.device.type != "cpu":
        return torch.matmul(a, b, out_dtype=torch.float32)
    return torch.matmul(a, b)


def tbackward_bsnd(dx, q, k, v, softmax_res, drop_mask, scale, softcap, dropout_p):
    keep_prob = _keep_prob(dropout_p)
    dp = _matmul_bwd(dx, v.permute(0, 1, 3, 2))
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = dp
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob).permute(0, 1, 3, 2)
        dp_drop = dp * drop_mask * (1.0 / keep_prob)
    dv = _matmul_bwd(drop_res, dx)
    softmax_grad_res = tsoftmax_grad(dp_drop, softmax_res) * scale
    softcap_factor = _softcap_backward_factor(q, k, scale, softcap, q.dtype)
    if softcap_factor is not None:
        softmax_grad_res = softmax_grad_res * softcap_factor
    softmax_grad_res = softmax_grad_res.to(k.dtype)
    dq = _matmul_bwd(softmax_grad_res, k)
    dk = _matmul_bwd(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def tbackward_tnd(dx, q, k, v, softmax_res, drop_mask, scale, softcap, dropout_p):
    keep_prob = _keep_prob(dropout_p)
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = _matmul_bwd(dx, v.permute(0, 1, 3, 2))
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob).permute(0, 1, 3, 2)
        dp = _matmul_bwd(dx, v.permute(0, 1, 3, 2))
        dp_drop = dp * drop_mask * (1.0 / keep_prob)
    dv = _matmul_bwd(drop_res, dx)
    softmax_grad_res = tsoftmax_grad(dp_drop, softmax_res) * scale
    softcap_factor = _softcap_backward_factor(q, k, scale, softcap, q.dtype)
    if softcap_factor is not None:
        softmax_grad_res = softmax_grad_res * softcap_factor
    softmax_grad_res = softmax_grad_res.to(k.dtype)
    dq = _matmul_bwd(softmax_grad_res, k)
    dk = _matmul_bwd(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    return y.div(x_sum), x_max, x_sum


def upload_to_npu_like(template, tensor):
    """CPU 标杆写到 template.device。A5 无 InplaceZero/ZerosLike 时用 empty+copy_，再退回 .to()。"""
    src = tensor.detach().to(dtype=template.dtype)
    if src.device == template.device:
        return src.contiguous()
    try:
        buf = torch.empty(
            template.shape, dtype=template.dtype, device=template.device
        )
        buf.copy_(src.to(device=template.device))
        return buf
    except RuntimeError:
        return src.to(device=template.device).contiguous()


def upload_lse_bns_to_npu(batch, nheads, seq_q, device, lse_cpu):
    """BSND arch22: softmax_lse shape (B, N, S)。"""
    src = lse_cpu.detach().to(dtype=torch.float32).contiguous()
    if torch.device(device).type == "cpu":
        return src
    try:
        buf = torch.empty(batch, nheads, seq_q, dtype=torch.float32, device=device)
        buf.copy_(src.to(device=device))
        return buf.contiguous()
    except RuntimeError:
        return src.to(device=device).contiguous()


def upload_lse_nt_to_npu(nheads, total_q, device, lse_cpu):
    """TND arch22: softmax_lse shape (N, total_q)。"""
    src = lse_cpu.detach().to(dtype=torch.float32).contiguous()
    if torch.device(device).type == "cpu":
        return src
    try:
        buf = torch.empty(nheads, total_q, dtype=torch.float32, device=device)
        buf.copy_(src.to(device=device))
        return buf.contiguous()
    except RuntimeError:
        return src.to(device=device).contiguous()


def fa_bwd_window_bsnd(seq_q, seq_k, is_causal, window_size_left, window_size_right):
    """flash_api bwd 的 window 规范化，返回 (causal_switch, wl, wr)。"""
    wl, wr, causal_switch, _ = normalize_window(
        seq_q, seq_k, is_causal, window_size_left, window_size_right
    )
    return causal_switch, wl, wr


def quantize_then_upcast(t, casedtype):
    """与 torch_fag / dailyauto 对齐：先 cast 到原始 fp16/bf16，再升回 fp32。"""
    return t.to(casedtype).to(torch.float32)


def lse_bns_from_max_sum(x_max, x_sum, casedtype):
    """小算子 tforward 的 x_max/x_sum (B,N,S,1) -> FA bwd BNS softmax_lse。

    max/sum 先量化到 case dtype 再升 fp32 后算 LSE（对齐 torch_fag）。
    """
    x_max_q = quantize_then_upcast(x_max, casedtype)
    x_sum_q = quantize_then_upcast(x_sum, casedtype)
    lse = x_max_q + torch.log(x_sum_q.clamp_min(1e-20))
    return lse.squeeze(-1).contiguous()


def lse_ns_from_max_sum(x_max, x_sum, casedtype):
    """单 batch 切片 x_max/x_sum (1,N,S,1) -> (N, S)。

    max/sum 先量化到 case dtype 再升 fp32 后算 LSE（对齐 torch_fag）。
    """
    x_max_q = quantize_then_upcast(x_max, casedtype)
    x_sum_q = quantize_then_upcast(x_sum, casedtype)
    lse = x_max_q + torch.log(x_sum_q.clamp_min(1e-20))
    return lse.squeeze(0).squeeze(-1).contiguous()


def _apply_full_mask_lse_inf(lse, atten_mask):
    """全 mask query 行 LSE 置 +inf（对齐 CANN / FA 行为）。"""
    if atten_mask is None or len(atten_mask.shape) == 0:
        return lse
    fully = atten_mask.bool().all(dim=-1)
    if lse.dim() == 3:
        mask = fully.view(1, 1, -1).expand_as(lse)
    elif lse.dim() == 2:
        mask = fully.view(1, -1).expand_as(lse)
    else:
        return lse
    return torch.where(mask, torch.full_like(lse, float("inf")), lse)


def tforward_fa_aligned(
    q,
    k,
    v,
    drop_mask,
    atten_mask,
    scale,
    softcap,
    dropout_p,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    compute_dtype=torch.float64,
    gtype=torch.float64,
):
    """
    same_input 正向：QKV / matmul / softmax 全程 compute_dtype（默认 fp64），不落 fp32。
    q/k/v shape: (B, N, S, D)。
    """
    keep_prob = _keep_prob(dropout_p)
    atten_mask = _resolve_atten_mask(
        q, k, atten_mask, is_causal, window_size_left, window_size_right
    )
    qb = q.to(compute_dtype)
    kb = k.to(compute_dtype)
    vb = v.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).mul(scale)
    if softcap > 0.0:
        qk = softcap * torch.tanh(qk / softcap)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        qk = qk + atten_mask.to(dtype=qk.dtype) * (-40000.0)
    softmax_res, x_max, x_sum = tsoftmax(qk)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask.to(softmax_res.dtype) * (1.0 / keep_prob)
    y = torch.matmul(drop_res.to(compute_dtype), vb)
    return y, softmax_res.to(gtype), x_max, x_sum, atten_mask


def golden_bsnd_fwd_only(
    q,
    k,
    v,
    nheads,
    nheads_k,
    scale,
    dropout_p,
    is_causal,
    window_size_left,
    window_size_right,
    *,
    drop_mask=None,
    drop_seed=None,
    return_softmax_stats=False,
    softcap=0.0,
    native_dtype=None,
):
    """
    BSND 小算子正向（same_input）：
      - CPU + float64 全程计算
      - out：先 cast 到 native_dtype(fp16/bf16)，再 cast 到当前输入 dtype(q.dtype)
      - lse：max/sum 经 native quantize→upcast 后计算，fp32 (B,N,S)（对齐 torch_fag）
      - 结果留在 CPU
    返回 (out, lse, softmax_res, drop_mask, bwd_causal, bwd_wl, bwd_wr)
    或再追加 (x_max, x_sum) 当 return_softmax_stats=True。
    """
    if softcap is None:
        softcap = 0.0
    batch, seq_q, _, headdim = q.shape
    seq_k = k.shape[1]
    fwd_dtype = torch.float64
    current_dtype = q.dtype
    if native_dtype is None:
        native_dtype = (
            current_dtype
            if current_dtype in (torch.float16, torch.bfloat16)
            else torch.float16
        )

    q_bn = q.detach().cpu().permute(0, 2, 1, 3).to(fwd_dtype)
    k_bn = k.detach().cpu().permute(0, 2, 1, 3).to(fwd_dtype)
    v_bn = v.detach().cpu().permute(0, 2, 1, 3).to(fwd_dtype)

    if nheads == nheads_k:
        k_new, v_new = k_bn, v_bn
    else:
        k_new = broadcast_kv_single(nheads, nheads_k, k_bn, fwd_dtype)
        v_new = broadcast_kv_single(nheads, nheads_k, v_bn, fwd_dtype)

    if drop_mask is None:
        seed = 0 if drop_seed is None else int(drop_seed)
        drop_mask = make_drop_mask(
            (batch, nheads, seq_q, seq_k), dropout_p, seed, dtype=fwd_dtype
        )
    elif torch.is_tensor(drop_mask) and drop_mask.numel() > 1:
        drop_mask = drop_mask.detach().cpu().to(fwd_dtype)

    out_bn, softmax_res, x_max, x_sum, atten_mask = tforward_fa_aligned(
        q_bn,
        k_new,
        v_new,
        drop_mask,
        None,
        scale,
        softcap,
        dropout_p,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        compute_dtype=fwd_dtype,
        gtype=fwd_dtype,
    )
    # LSE：与 torch_fag 一致，对 max/sum 做 native quantize→upcast 后计算（fp32）
    lse = _apply_full_mask_lse_inf(
        lse_bns_from_max_sum(x_max, x_sum, native_dtype), atten_mask
    )
    out_fp64 = out_bn.permute(0, 2, 1, 3).contiguous()

    # out：先量化到原始 dtype，再升到当前输入 dtype；始终留在 CPU
    out = cast_via_native(out_fp64.cpu(), native_dtype, current_dtype)
    bwd_causal, bwd_wl, bwd_wr = fa_bwd_window_bsnd(
        seq_q, seq_k, is_causal, window_size_left, window_size_right
    )
    base = (out, lse, softmax_res, drop_mask, bwd_causal, bwd_wl, bwd_wr)
    if return_softmax_stats:
        return base + (
            quantize_then_upcast(x_max, native_dtype),
            quantize_then_upcast(x_sum, native_dtype),
        )
    return base


def golden_bsnd_bwd_grads(
    q,
    k,
    v,
    dout,
    softmax_res,
    drop_mask,
    nheads,
    nheads_k,
    scale,
    dropout_p,
    *,
    softcap=0.0,
):
    """BSND 小算子反向：CPU + float64 计算（对齐 torch_fag），结果 cast 回 q.dtype。"""
    if softcap is None:
        softcap = 0.0
    batch, seq_q, _, headdim = q.shape
    seq_k = k.shape[1]
    gtype = torch.float64
    out_dtype = q.dtype

    q_bn = q.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    k_bn = k.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    v_bn = v.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    dx_bn = dout.detach().cpu().permute(0, 2, 1, 3).to(gtype)

    if nheads == nheads_k:
        k_new, v_new = k_bn, v_bn
    else:
        k_new = broadcast_kv_single(nheads, nheads_k, k_bn, gtype)
        v_new = broadcast_kv_single(nheads, nheads_k, v_bn, gtype)

    softmax_res_bwd = softmax_res.detach().cpu().to(dtype=gtype)
    drop_mask_bwd = (
        drop_mask.detach().cpu().to(dtype=gtype)
        if torch.is_tensor(drop_mask) and drop_mask.numel() > 1
        else drop_mask
    )
    dq_bn, dk_bn, dv_bn = tbackward_bsnd(
        dx_bn,
        q_bn,
        k_new,
        v_new,
        softmax_res_bwd,
        drop_mask_bwd,
        scale,
        softcap,
        dropout_p,
    )
    dk_bn = sum_gqa_grad(dk_bn, nheads, nheads_k, batch, seq_k, headdim)
    dv_bn = sum_gqa_grad(dv_bn, nheads, nheads_k, batch, seq_k, headdim)

    dq = dq_bn.permute(0, 2, 1, 3).contiguous().to(dtype=out_dtype)
    dk = dk_bn.permute(0, 2, 1, 3).contiguous().to(dtype=out_dtype)
    dv = dv_bn.permute(0, 2, 1, 3).contiguous().to(dtype=out_dtype)
    return dq, dk, dv


def _tnd_cu_offsets(seqlens_q, seqlens_k):
    seqlens_q = list(seqlens_q)
    seqlens_k = list(seqlens_k)
    cu_q = [0]
    cu_k = [0]
    for sq, sk in zip(seqlens_q, seqlens_k):
        cu_q.append(cu_q[-1] + int(sq))
        cu_k.append(cu_k[-1] + int(sk))
    return seqlens_q, seqlens_k, cu_q, cu_k


def golden_tnd_fwd_only(
    q,
    k,
    v,
    nheads,
    nheads_k,
    seqlens_q,
    seqlens_k,
    scale,
    dropout_p,
    is_causal,
    window_size_left,
    window_size_right,
    *,
    drop_masks=None,
    drop_seed=None,
    return_softmax_stats=False,
    softcap=0.0,
    native_dtype=None,
    compute_dtype=None,
):
    """
    TND 小算子正向（same_input）：
      - CPU + compute_dtype（默认 float64）全程计算
      - out：先 cast 到 native_dtype(fp16/bf16)，再 cast 到当前输入 dtype(q.dtype)
      - lse：max/sum 经 native quantize→upcast 后计算，fp32 NT (N, total_q)（对齐 torch_fag）
      - 结果留在 CPU
    返回 (out, lse, softmax_res_list, drop_masks)
    或再追加 (softmax_max, softmax_sum) 当 return_softmax_stats=True。
    fusion 用 softmax_max/sum 为连续 TND (T, N, 8)，需配合 softmax_layout=\"TND\"。
    """
    if softcap is None:
        softcap = 0.0
    seqlens_q, seqlens_k, cu_q, cu_k = _tnd_cu_offsets(seqlens_q, seqlens_k)
    headdim = q.shape[-1]
    fwd_dtype = torch.float64 if compute_dtype is None else compute_dtype
    current_dtype = q.dtype
    if native_dtype is None:
        native_dtype = (
            current_dtype
            if current_dtype in (torch.float16, torch.bfloat16)
            else torch.float16
        )
    base_seed = 0 if drop_seed is None else int(drop_seed)
    total_q = int(cu_q[-1]) if len(cu_q) else 0

    out_fp64 = torch.empty(
        q.shape, dtype=fwd_dtype, device="cpu"
    )
    lse_parts = []
    out_drop_masks = []
    softmax_res_list = []
    max_parts = []
    sum_parts = []

    for i, sq in enumerate(seqlens_q):
        sk = seqlens_k[i]
        if sq == 0 or sk == 0:
            lse_parts.append(torch.zeros(nheads, 0, dtype=torch.float32))
            out_drop_masks.append(torch.tensor(1))
            softmax_res_list.append(None)
            continue

        qi = q[cu_q[i] : cu_q[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(fwd_dtype)
        ki = k[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(fwd_dtype)
        vi = v[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(fwd_dtype)

        if nheads == nheads_k:
            ki_new, vi_new = ki, vi
        else:
            ki_new = broadcast_kv_single(nheads, nheads_k, ki, fwd_dtype)
            vi_new = broadcast_kv_single(nheads, nheads_k, vi, fwd_dtype)

        if drop_masks is not None and i < len(drop_masks) and drop_masks[i] is not None:
            drop_mask_i = drop_masks[i]
            if torch.is_tensor(drop_mask_i) and drop_mask_i.numel() > 1:
                drop_mask_i = drop_mask_i.detach().cpu().to(fwd_dtype)
        else:
            drop_mask_i = make_drop_mask(
                (1, nheads, sq, sk), dropout_p, base_seed + i * 10007, dtype=fwd_dtype
            )
        out_drop_masks.append(drop_mask_i)

        out_i, softmax_res_i, x_max_i, x_sum_i, atten_mask_i = tforward_fa_aligned(
            qi,
            ki_new,
            vi_new,
            drop_mask_i,
            None,
            scale,
            softcap,
            dropout_p,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            compute_dtype=fwd_dtype,
            gtype=fwd_dtype,
        )
        lse_i = _apply_full_mask_lse_inf(
            lse_ns_from_max_sum(x_max_i, x_sum_i, native_dtype), atten_mask_i
        )
        softmax_res_list.append(softmax_res_i)

        out_fp64[cu_q[i] : cu_q[i + 1]] = (
            out_i.permute(0, 2, 1, 3).reshape(sq, nheads, headdim)
        )
        lse_parts.append(lse_i)
        if return_softmax_stats and sq > 0:
            # (1,N,S,1) -> (S,N,8)；沿 token 维 cat 得到连续 TND (T,N,8)
            x_max_q = quantize_then_upcast(x_max_i, native_dtype)
            x_sum_q = quantize_then_upcast(x_sum_i, native_dtype)
            max_parts.append(
                x_max_q[0].broadcast_to(nheads, sq, 8).permute(1, 0, 2).contiguous()
            )
            sum_parts.append(
                x_sum_q[0].broadcast_to(nheads, sq, 8).permute(1, 0, 2).contiguous()
            )

    lse_nt = (
        torch.cat(lse_parts, dim=1)
        if lse_parts
        else torch.zeros(nheads, 0, dtype=torch.float32)
    )
    out = cast_via_native(out_fp64.cpu(), native_dtype, current_dtype)
    # LSE 已是 quantize→upcast 后的 fp32，不再对整段 LSE 做 native 再量化
    lse = lse_nt.contiguous()
    base = (out, lse, softmax_res_list, out_drop_masks)
    if return_softmax_stats:
        softmax_max = (
            torch.cat(max_parts, dim=0)
            if max_parts
            else torch.zeros(total_q, nheads, 8, dtype=torch.float32)
        )
        softmax_sum = (
            torch.cat(sum_parts, dim=0)
            if sum_parts
            else torch.ones(total_q, nheads, 8, dtype=torch.float32)
        )
        return base + (softmax_max.contiguous(), softmax_sum.contiguous())
    return base


def golden_tnd_bwd_grads(
    q,
    k,
    v,
    dout,
    softmax_res_list,
    drop_masks,
    nheads,
    nheads_k,
    seqlens_q,
    seqlens_k,
    scale,
    dropout_p,
    *,
    softcap=0.0,
    compute_dtype=None,
):
    """TND 小算子反向：CPU + compute_dtype（默认 float64）按 sample 计算，结果 cast 回 q.dtype。"""
    if softcap is None:
        softcap = 0.0
    seqlens_q, seqlens_k, cu_q, cu_k = _tnd_cu_offsets(seqlens_q, seqlens_k)
    headdim = q.shape[-1]
    gtype = torch.float64 if compute_dtype is None else compute_dtype
    out_dtype = q.dtype

    dq_golden = torch.empty(q.shape, dtype=out_dtype)
    dk_golden = torch.empty(k.shape, dtype=out_dtype)
    dv_golden = torch.empty(v.shape, dtype=out_dtype)

    for i, sq in enumerate(seqlens_q):
        sk = seqlens_k[i]
        if sq == 0 or sk == 0:
            continue

        qi = q[cu_q[i] : cu_q[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        ki = k[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        vi = v[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        dxi = dout[cu_q[i] : cu_q[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)

        if nheads == nheads_k:
            ki_new, vi_new = ki, vi
        else:
            ki_new = broadcast_kv_single(nheads, nheads_k, ki, gtype)
            vi_new = broadcast_kv_single(nheads, nheads_k, vi, gtype)

        drop_mask_i = drop_masks[i] if drop_masks is not None and i < len(drop_masks) else torch.tensor(1)
        drop_mask_bwd = (
            drop_mask_i.detach().cpu().to(gtype)
            if torch.is_tensor(drop_mask_i) and drop_mask_i.numel() > 1
            else drop_mask_i
        )
        softmax_res_i = softmax_res_list[i].detach().cpu().to(gtype)

        dqi, dki, dvi = tbackward_tnd(
            dxi, qi, ki_new, vi_new, softmax_res_i, drop_mask_bwd, scale, softcap, dropout_p
        )
        dki = sum_gqa_grad(dki, nheads, nheads_k, 1, sk, headdim)
        dvi = sum_gqa_grad(dvi, nheads, nheads_k, 1, sk, headdim)

        dq_golden[cu_q[i] : cu_q[i + 1]] = (
            dqi.permute(0, 2, 1, 3).reshape(sq, nheads, headdim).to(out_dtype)
        )
        dk_golden[cu_k[i] : cu_k[i + 1]] = (
            dki.permute(0, 2, 1, 3).reshape(sk, nheads_k, headdim).to(out_dtype)
        )
        dv_golden[cu_k[i] : cu_k[i + 1]] = (
            dvi.permute(0, 2, 1, 3).reshape(sk, nheads_k, headdim).to(out_dtype)
        )
    return dq_golden, dk_golden, dv_golden


def softmax_res_from_fa_lse_bsnd(
    q_bn,
    k_bn,
    softmax_lse,
    scale,
    softcap,
    is_causal,
    window_size_left,
    window_size_right,
    compute_dtype,
    *,
    gtype=torch.float64,
):
    """q_bn / k_bn: (B, N, S, D)。softmax_lse: FA BSND (B, N, S_q)。"""
    atten_mask = _resolve_atten_mask(
        q_bn, k_bn, None, is_causal, window_size_left, window_size_right
    )
    qb = q_bn.to(compute_dtype)
    kb = k_bn.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    if softcap > 0.0:
        qk = softcap * torch.tanh(qk / softcap)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        qk = qk + atten_mask.to(torch.float32) * (-40000.0)
    lse = softmax_lse.to(torch.float32).to(gtype)
    if lse.dim() != 3:
        raise ValueError(f"BSND softmax_lse 期望 3 维 (B,N,S)，实际 {tuple(lse.shape)}")
    lse = lse.unsqueeze(-1)
    softmax_res = torch.exp(qk.to(gtype) - lse)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    return softmax_res


def softmax_res_from_fa_lse_tnd_slice(
    q_bn,
    k_bn,
    softmax_lse_nt_slice,
    scale,
    softcap,
    is_causal,
    window_size_left,
    window_size_right,
    compute_dtype,
    *,
    gtype=torch.float64,
):
    """TND 单 batch 切片：FA varlen softmax_lse 切片 (N, S_q) NT。"""
    atten_mask = _resolve_atten_mask(
        q_bn, k_bn, None, is_causal, window_size_left, window_size_right
    )
    qb = q_bn.to(compute_dtype)
    kb = k_bn.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    if softcap > 0.0:
        qk = softcap * torch.tanh(qk / softcap)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        qk = qk + atten_mask.to(torch.float32) * (-40000.0)
    lse = softmax_lse_nt_slice.to(torch.float32).to(gtype).unsqueeze(0).unsqueeze(-1)
    softmax_res = torch.exp(qk.to(gtype) - lse)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    return softmax_res


def golden_bsnd_bwd_from_fwd(
    q,
    k,
    v,
    dout,
    out,
    softmax_lse,
    nheads,
    nheads_k,
    scale,
    softcap,
    dropout_p,
    is_causal,
    window_size_left,
    window_size_right,
    *,
    gtype=torch.float64,
):
    """BSND 反传标杆（用 FA 正向 out/lse）。softmax_lse: (B, N, S_q)。"""
    del out
    batch, seq_q, _, headdim = q.shape
    seq_k = k.shape[1]
    compute_dtype = q.dtype

    q_bn = q.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    k_bn = k.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    v_bn = v.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    dx_bn = dout.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    lse_cpu = softmax_lse.detach().cpu().to(torch.float32)

    if nheads == nheads_k:
        k_new, v_new = k_bn, v_bn
    else:
        k_new = broadcast_kv_single(nheads, nheads_k, k_bn, gtype)
        v_new = broadcast_kv_single(nheads, nheads_k, v_bn, gtype)

    softmax_res = softmax_res_from_fa_lse_bsnd(
        q_bn,
        k_new,
        lse_cpu,
        scale,
        softcap,
        is_causal,
        window_size_left,
        window_size_right,
        compute_dtype,
        gtype=gtype,
    )
    drop_mask = torch.tensor(1)
    dq_bn, dk_bn, dv_bn = tbackward_bsnd(
        dx_bn, q_bn, k_new, v_new, softmax_res, drop_mask, scale, softcap, dropout_p
    )
    dk_bn = sum_gqa_grad(dk_bn, nheads, nheads_k, batch, seq_k, headdim)
    dv_bn = sum_gqa_grad(dv_bn, nheads, nheads_k, batch, seq_k, headdim)

    dq = dq_bn.permute(0, 2, 1, 3).to(compute_dtype)
    dk = dk_bn.permute(0, 2, 1, 3).to(compute_dtype)
    dv = dv_bn.permute(0, 2, 1, 3).to(compute_dtype)
    return dq, dk, dv


def golden_tnd_bwd_from_fwd(
    q,
    k,
    v,
    dout,
    out,
    softmax_lse,
    nheads,
    nheads_k,
    seqlens_q,
    seqlens_k,
    scale,
    softcap,
    dropout_p,
    is_causal,
    window_size_left,
    window_size_right,
    *,
    gtype=torch.float64,
):
    """TND 反传标杆（用 FA 正向 out/lse）。softmax_lse: (N, total_q) NT。"""
    del out
    seqlens_q = list(seqlens_q)
    seqlens_k = list(seqlens_k)
    cu_q = [0]
    cu_k = [0]
    for sq, sk in zip(seqlens_q, seqlens_k):
        cu_q.append(cu_q[-1] + int(sq))
        cu_k.append(cu_k[-1] + int(sk))
    headdim = q.shape[-1]
    compute_dtype = q.dtype
    lse_nt = softmax_lse.detach().cpu().to(torch.float32)
    if lse_nt.dim() != 2:
        raise ValueError(f"TND softmax_lse 期望 NT (N, total_q)，实际 {tuple(lse_nt.shape)}")

    dq_golden = torch.empty_like(q, dtype=compute_dtype)
    dk_golden = torch.empty_like(k, dtype=compute_dtype)
    dv_golden = torch.empty_like(v, dtype=compute_dtype)
    drop_mask = torch.tensor(1)

    for i, sq in enumerate(seqlens_q):
        sk = seqlens_k[i]
        if sq == 0 or sk == 0:
            continue

        qi = q[cu_q[i] : cu_q[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        ki = k[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        vi = v[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        dxi = dout[cu_q[i] : cu_q[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        lse_i = lse_nt[:, cu_q[i] : cu_q[i + 1]]

        if nheads == nheads_k:
            ki_new, vi_new = ki, vi
        else:
            ki_new = broadcast_kv_single(nheads, nheads_k, ki, gtype)
            vi_new = broadcast_kv_single(nheads, nheads_k, vi, gtype)

        softmax_res_i = softmax_res_from_fa_lse_tnd_slice(
            qi,
            ki_new,
            lse_i,
            scale,
            softcap,
            is_causal,
            window_size_left,
            window_size_right,
            compute_dtype,
            gtype=gtype,
        )
        dqi, dki, dvi = tbackward_tnd(
            dxi, qi, ki_new, vi_new, softmax_res_i, drop_mask, scale, softcap, dropout_p
        )
        dki = sum_gqa_grad(dki, nheads, nheads_k, 1, sk, headdim)
        dvi = sum_gqa_grad(dvi, nheads, nheads_k, 1, sk, headdim)

        dq_golden[cu_q[i] : cu_q[i + 1]] = (
            dqi.permute(0, 2, 1, 3).reshape(sq, nheads, headdim).to(compute_dtype)
        )
        dk_golden[cu_k[i] : cu_k[i + 1]] = (
            dki.permute(0, 2, 1, 3).reshape(sk, nheads_k, headdim).to(k.dtype)
        )
        dv_golden[cu_k[i] : cu_k[i + 1]] = (
            dvi.permute(0, 2, 1, 3).reshape(sk, nheads_k, headdim).to(v.dtype)
        )

    return dq_golden, dk_golden, dv_golden
