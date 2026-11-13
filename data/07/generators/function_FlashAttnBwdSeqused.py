# Copyright (c) Huawei Technologies Co., Ltd. 2012-2026. All rights reserved.
"""
ATK API for FlashAttention v3 TND backward seqused_q/seqused_k.

cu_seqlens_* describes the packed allocation. seqused_q/seqused_k, when passed,
are the per-batch used prefixes.

NPU under test is selected by case meta ``backend`` (from yaml attr):
  ascend950 -> flash_attn_npu_3.flash_attn_npu_interface_950
               (csrc/ascend950/flash_attn_npu_3), kwargs seqused_q/seqused_k
  ascend910 -> flash_attn_npu_3.flash_attn_npu_interface
               (csrc/ascend910/flash_attn_npu_3), kwargs sequed_q/sequed_k

Node roles (nodes yaml), matching torch_fag / fag_A5 dual-NPU BM:
  - cpu (name cpu_bm): fp64 same_input golden
  - npu (no fusion/_bm): DUT with seqused_q/seqused_k via flash_attn_npu_3
  - npu (name contains fusion, e.g. npu_fusion / npu_fusion_bm):
      compact used-prefix + torch_npu.npu_fusion_attention_grad
      (fusion has no seqused_* ABI; used lengths go through actual_seq_*)
  - npu (name *_bm without fusion): compact-used-prefix FA bwd BM

Unused tails of dq/dk/dv are zero on both sides: the kernel does not define
gradients outside the used prefix.
"""
import json
import os

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from fa_small_op_golden_same_input import cast_via_native
from seqused_bwd_golden import golden_tnd_seqused, zero_unused_prefix_

_FUSION_INT_MAX = 2147483647

_BACKEND_ALIASES = {
    "950": "ascend950",
    "a5": "ascend950",
    "a5v3": "ascend950",
    "ascend950": "ascend950",
    "910": "ascend910",
    "910b": "ascend910",
    "910c": "ascend910",
    "ascend910": "ascend910",
}


def _normalize_backend(value, default="ascend950"):
    if value is None:
        return default
    key = str(value).strip().lower().replace("-", "").replace("_", "")
    for alias, canonical in _BACKEND_ALIASES.items():
        if key == alias.replace("-", "").replace("_", ""):
            return canonical
    raise ValueError(
        f"unsupported backend={value!r}; use ascend950 or ascend910"
    )


def _node_is_bm(node_name):
    """nodes yaml name containing '_bm' (e.g. npu_compact_bm, cpu_bm)."""
    return isinstance(node_name, str) and "_bm" in node_name.lower()


def _node_name_has_fusion(node_name):
    """nodes yaml name containing 'fusion' (e.g. npu_fusion, npu_fusion_bm)."""
    return isinstance(node_name, str) and "fusion" in node_name.lower()


def _as_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, (bool, int)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y")
    return default


def _as_float(x, default=0.0):
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _as_int(x, default=0):
    if x is None:
        return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _as_window_size(window_size):
    if window_size is None:
        return (-1, -1)
    if isinstance(window_size, (list, tuple)) and len(window_size) == 2:
        return (int(window_size[0]), int(window_size[1]))
    return (-1, -1)


def _resolve_softmax_scale(softmax_scale, head_size):
    if softmax_scale is None:
        return 1.0 / (head_size ** 0.5)
    try:
        scale = float(softmax_scale)
    except (TypeError, ValueError):
        return 1.0 / (head_size ** 0.5)
    if scale == 0.0:
        return 1.0 / (head_size ** 0.5)
    return scale


def _make_dout(shape, dtype, case_id):
    rng = torch.Generator(device="cpu")
    seed = int(case_id) if case_id is not None else 0
    rng.manual_seed(seed)
    return (2.0 * (torch.rand(shape, generator=rng) - 0.5)).to(dtype)


def _resolve_native_dtype(case_config, query):
    cfg_dtype = None
    try:
        cfg_dtype = str(case_config.inputs[0].dtype).lower()
    except Exception:
        pass
    if cfg_dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if cfg_dtype in ("fp16", "float16", "half"):
        return torch.float16
    if query.dtype in (torch.float16, torch.bfloat16):
        return query.dtype
    return torch.float16


def _align_bwd_tensor(t, native_dtype, current_dtype):
    return cast_via_native(t.detach().cpu(), native_dtype, current_dtype)


def _lse_for_fa_api(softmax_lse):
    if softmax_lse.dtype == torch.float32:
        return softmax_lse
    return softmax_lse.to(torch.float32)


def _to_fa_compute_dtype(t, native_dtype):
    if t.dtype == native_dtype:
        return t
    return t.to(native_dtype)


def _grads_to_current_dtype(dq, dk, dv, current_dtype):
    if dq.dtype == current_dtype:
        return dq, dk, dv
    return dq.to(current_dtype), dk.to(current_dtype), dv.to(current_dtype)


def _cu_seqlens_from_seqlens(seqlens):
    cu = [0]
    for length in seqlens:
        cu.append(cu[-1] + int(length))
    return torch.tensor(cu, dtype=torch.int32)


def _try_parse_seq_meta(meta):
    if not meta or not isinstance(meta, str):
        return None
    meta = meta.strip()
    if not meta.startswith("{"):
        return None
    try:
        info = json.loads(meta)
        layout = str(info.get("layout", "")).upper()
        sq = [int(x) for x in info["sq"]]
        sk = [int(x) for x in info["sk"]]
        uq = [int(x) for x in info.get("uq", sq)]
        uk = [int(x) for x in info.get("uk", sk)]
        pass_q = bool(info.get("pass_q", "uq" in info))
        pass_k = bool(info.get("pass_k", "uk" in info))
        backend = _normalize_backend(info.get("backend", "ascend950"))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if not sq or not sk or len(sq) != len(sk) or len(uq) != len(sq) or len(uk) != len(sk):
        return None
    if layout not in ("TND",):
        return None
    return {
        "layout": layout,
        "backend": backend,
        "sq": sq,
        "sk": sk,
        "uq": uq,
        "uk": uk,
        "pass_q": pass_q,
        "pass_k": pass_k,
        "mode": info.get("mode", ""),
    }


def _try_load_seq_meta_from_cases_json(case_id):
    candidates = []
    env_path = os.environ.get("ATK_CASES_JSON") or os.environ.get("FA_BWD_SEQUSED_CASES_JSON")
    if env_path:
        candidates.append(env_path)
    here = os.path.dirname(__file__)
    candidates.extend(
        [
            "FlashAttnBwdSeqused_Accu_Generalization.json",
            os.path.join(here, "FlashAttnBwdSeqused_Accu_Generalization.json"),
        ]
    )
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                cases = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(cases, list):
            continue
        for case in cases:
            if case.get("id") != case_id:
                continue
            for key in ("aclnn_name", "save_name"):
                parsed = _try_parse_seq_meta(case.get(key))
                if parsed is not None:
                    return parsed
    return None


def _resolve_seq_meta(case_config, case_id):
    for meta in (
        getattr(case_config, "aclnn_name", None),
        getattr(case_config, "save_name", None),
    ):
        parsed = _try_parse_seq_meta(meta)
        if parsed is not None:
            return parsed
    return _try_load_seq_meta_from_cases_json(case_id)


def _seqused_tensor(lengths, device):
    return torch.tensor([int(x) for x in lengths], dtype=torch.int32, device=device).contiguous()


def _mask_unused_grads(dq, dk, dv, seqlens_q, seqlens_k, used_q, used_k):
    zero_unused_prefix_(dq, seqlens_q, used_q)
    zero_unused_prefix_(dk, seqlens_k, used_k)
    zero_unused_prefix_(dv, seqlens_k, used_k)
    return dq, dk, dv


def _assert_grad_layout(dq, dk, dv, q, k, v, where):
    """Keep (dq, dk, dv) tied to (q, k, v) shapes.

    ATK multi-output double-benchmark pairs Actual/Benchmark by output index.
    A dq/dk swap here shows up as Benchmark metrics of output_0/output_1 being
    crossed in the report (seen on the 910 Accu run) even when NPU Absolute
    error is fine.
    """
    if tuple(dq.shape) != tuple(q.shape):
        raise RuntimeError(
            f"{where}: dq shape {tuple(dq.shape)} != q shape {tuple(q.shape)}"
        )
    if tuple(dk.shape) != tuple(k.shape):
        raise RuntimeError(
            f"{where}: dk shape {tuple(dk.shape)} != k shape {tuple(k.shape)}"
        )
    if tuple(dv.shape) != tuple(v.shape):
        raise RuntimeError(
            f"{where}: dv shape {tuple(dv.shape)} != v shape {tuple(v.shape)}"
        )
    return dq, dk, dv


def _fusion_sparse_args(is_causal, window_size):
    """Map FA causal/window to npu_fusion_attention_grad sparse args."""
    wl, wr = int(window_size[0]), int(window_size[1])
    if is_causal:
        return {
            "sparse_mode": 3,
            "pre_tockens": _FUSION_INT_MAX,
            "next_tockens": 0,
        }
    if wl >= 0 or wr >= 0:
        return {
            "sparse_mode": 4,
            "pre_tockens": _FUSION_INT_MAX if wl < 0 else wl,
            "next_tockens": _FUSION_INT_MAX if wr < 0 else wr,
        }
    return {
        "sparse_mode": 0,
        "pre_tockens": _FUSION_INT_MAX,
        "next_tockens": _FUSION_INT_MAX,
    }


def _cu_to_actual_seq_list(cu_or_seqlens):
    """cu_seqlens [0,s1,...] or seqlens -> fusion actual_seq_* cumulative list."""
    if torch.is_tensor(cu_or_seqlens):
        vals = cu_or_seqlens.detach().cpu().to(torch.int64).reshape(-1).tolist()
    else:
        vals = [int(x) for x in cu_or_seqlens]
    if len(vals) >= 2 and vals[0] == 0:
        return vals[1:]
    # bare seqlens -> cumulative without leading 0
    out = []
    acc = 0
    for length in vals:
        acc += int(length)
        out.append(acc)
    return out


def _fusion_causal_atten_mask(device):
    """sparse_mode=3 compressed mask must be [2048,2048] (CANN / torch_fag)."""
    return torch.triu(
        torch.ones(2048, 2048, device=device, dtype=torch.bool), diagonal=1
    )


def _run_fusion_attention_grad_tnd(
    dout,
    q,
    k,
    v,
    out,
    softmax_max,
    softmax_sum,
    *,
    head_num,
    softmax_scale,
    is_causal,
    window_size,
    actual_seq_qlen,
    actual_seq_kvlen,
    max_seqlen_k,
    deterministic=False,
    keep_prob=1.0,
):
    """TND npu_fusion_attention_grad; softmax_* are contiguous (T,N,8)."""
    import torch_npu  # noqa: F401

    scale = 1.0 if softmax_scale is None else float(softmax_scale)
    sm_max = softmax_max.detach().to(device=q.device, dtype=torch.float32).contiguous()
    sm_sum = softmax_sum.detach().to(device=q.device, dtype=torch.float32).contiguous()
    t_total = int(q.shape[0])
    if sm_max.dim() != 3 or sm_max.shape[0] != t_total or sm_max.shape[1] != int(head_num):
        raise ValueError(
            f"TND softmax_max expect (T,N,8)=({t_total},{head_num},8), "
            f"got {tuple(sm_max.shape)}"
        )
    if is_causal:
        sparse_mode = 3
        pre_tockens = int(max_seqlen_k) if max_seqlen_k is not None else _FUSION_INT_MAX
        next_tockens = 0
        atten_mask = _fusion_causal_atten_mask(q.device)
    else:
        sparse = _fusion_sparse_args(False, window_size)
        sparse_mode = int(sparse["sparse_mode"])
        pre_tockens = int(sparse["pre_tockens"])
        next_tockens = int(sparse["next_tockens"])
        atten_mask = None
    rets = torch_npu.npu_fusion_attention_grad(
        q,
        k,
        v,
        dout,
        int(head_num),
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask,
        softmax_max=sm_max,
        softmax_sum=sm_sum,
        softmax_in=None,
        attention_in=out,
        scale_value=scale,
        keep_prob=float(keep_prob),
        input_layout="TND",
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        inner_precise=1 if deterministic else 0,
        seed=0,
        offset=0,
        numels=0,
        prefix=None,
        actual_seq_qlen=tuple(actual_seq_qlen) if actual_seq_qlen is not None else None,
        actual_seq_kvlen=tuple(actual_seq_kvlen) if actual_seq_kvlen is not None else None,
        sparse_mode=sparse_mode,
        softmax_layout="TND",
        gen_mask_parallel=True,
        sync=False,
    )
    return rets[0], rets[1], rets[2]


def _run_flash_attn_backward_npu(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    *,
    backend,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    is_causal,
    window_size,
    softcap,
    deterministic,
):
    # torch_npu must load before the FA extension (aclrtGetFuncBySymbol).
    import torch_npu  # noqa: F401

    backend = _normalize_backend(backend)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    common = dict(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_lse=softmax_lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dq=dq,
        dk=dk,
        dv=dv,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=int(window_size[0]),
        window_size_right=int(window_size[1]),
        softcap=float(softcap),
        deterministic=bool(deterministic),
        sm_margin=0,
    )
    if backend == "ascend910":
        # 910 Python ABI keeps the historical sequed_* spellings.
        from flash_attn_npu_3.flash_attn_npu_interface import _flash_attn_backward

        _flash_attn_backward(
            **common,
            sequed_q=seqused_q,
            sequed_k=seqused_k,
        )
    else:
        from flash_attn_npu_3.flash_attn_npu_interface_950 import _flash_attn_backward

        _flash_attn_backward(
            **common,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
        )
    return dq, dk, dv


def _pad_bwd_headdim_fa2(dout, q, k, v, out):
    head_size_og = dout.size(-1)
    if head_size_og % 8 == 0:
        return dout, q, k, v, out, head_size_og
    pad = 8 - head_size_og % 8

    def _pad(t):
        return torch.nn.functional.pad(t, [0, pad])

    return _pad(dout), _pad(q), _pad(k), _pad(v), _pad(out), head_size_og


def _run_flash_attn_backward_fa2_tnd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    scale,
    causal,
):
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

    dout_p, q_p, k_p, v_p, out_p, head_size_og = _pad_bwd_headdim_fa2(dout, q, k, v, out)
    need_pad = dout_p.size(-1) != head_size_og
    if need_pad:
        dq_p = torch.zeros_like(q_p)
        dk_p = torch.zeros_like(k_p)
        dv_p = torch.zeros_like(v_p)
    else:
        dq_p, dk_p, dv_p = dq, dk, dv

    _flash_attn_varlen_backward(
        dout_p,
        q_p,
        k_p,
        v_p,
        out_p,
        softmax_lse,
        dq_p,
        dk_p,
        dv_p,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        0.0,
        scale,
        causal,
        -1,
        0 if causal else -1,
        0.0,
        None,
        False,
        rng_state=None,
    )
    if need_pad:
        dq.copy_(dq_p[..., :head_size_og])
        dk.copy_(dk_p[..., :head_size_og])
        dv.copy_(dv_p[..., :head_size_og])
    torch.cuda.synchronize()


@register("function_FlashAttnBwdSeqused")
class FlashAttnBwdSequsedApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query = input_data.kwargs["q"]
        key = input_data.kwargs["k"]
        value = input_data.kwargs["v"]
        if query.dim() != 3:
            raise ValueError(f"seqused bwd expects TND q rank 3, got {query.dim()}")

        causal = _as_bool(input_data.kwargs.get("causal", False), False)
        window_size = _as_window_size(input_data.kwargs.get("window_size", (-1, -1)))
        softcap = _as_float(input_data.kwargs.get("softcap", 0.0), 0.0)
        deterministic = _as_bool(input_data.kwargs.get("deterministic", False), False)
        case_id = self.task_result.case_config.id
        native_dtype = _resolve_native_dtype(self.task_result.case_config, query)
        current_dtype = query.dtype

        seq_meta = _resolve_seq_meta(self.task_result.case_config, case_id)
        if seq_meta is None:
            raise ValueError(
                "missing seqused metadata in aclnn_name/save_name; "
                "regenerate cases with ascend_generate_FlashAttnBwdSeqused"
            )
        backend = _normalize_backend(
            input_data.kwargs.get("backend", seq_meta.get("backend", "ascend950"))
        )

        total_q, num_heads, head_size = query.shape
        total_k = key.shape[0]
        kv_heads = key.shape[1]
        seqlens_q = seq_meta["sq"]
        seqlens_k = seq_meta["sk"]
        if sum(seqlens_q) != int(total_q) or sum(seqlens_k) != int(total_k):
            raise ValueError(
                f"case meta seqlens sum {(sum(seqlens_q), sum(seqlens_k))} "
                f"!= packed {(int(total_q), int(total_k))}"
            )
        used_q = seq_meta["uq"] if seq_meta["pass_q"] else list(seqlens_q)
        used_k = seq_meta["uk"] if seq_meta["pass_k"] else list(seqlens_k)
        if causal and any(uq > uk for uq, uk in zip(used_q, used_k)):
            raise ValueError(
                f"causal case has used_q > used_k: uq={used_q}, uk={used_k}"
            )

        max_seqlen_q = int(max(seqlens_q))
        max_seqlen_k = int(max(seqlens_k))
        softmax_scale = _resolve_softmax_scale(
            input_data.kwargs.get("softmax_scale", None), head_size
        )

        q = _align_bwd_tensor(query, native_dtype, current_dtype)
        k = _align_bwd_tensor(key, native_dtype, current_dtype)
        v = _align_bwd_tensor(value, native_dtype, current_dtype)
        dout = _align_bwd_tensor(
            _make_dout(query.shape, native_dtype, case_id),
            native_dtype,
            current_dtype,
        )
        # Padding tokens are not part of the backward. Zero their dout so the
        # full-row delta prologue cannot depend on unused outputs.
        zero_unused_prefix_(dout, seqlens_q, used_q)

        use_fusion = self.device == "npu" and _node_name_has_fusion(self.name)
        if use_fusion and softcap not in (None, 0, 0.0):
            raise ValueError(
                "npu_fusion_attention_grad path does not support softcap!=0"
            )

        # CPU / fwd reference: always fp64 golden (used for out/lse + cpu node).
        golden_ret = golden_tnd_seqused(
            q,
            k,
            v,
            dout,
            num_heads,
            kv_heads,
            seqlens_q,
            seqlens_k,
            used_q,
            used_k,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            softcap=softcap,
            native_dtype=native_dtype,
            compute_dtype=None,
            return_softmax_stats=use_fusion,
        )
        if use_fusion:
            out, softmax_lse, dq_g, dk_g, dv_g, softmax_max, softmax_sum = golden_ret
        else:
            out, softmax_lse, dq_g, dk_g, dv_g = golden_ret
            softmax_max = softmax_sum = None

        if self.device not in ("npu", "gpu") or all(
            int(uq) == 0 or int(uk) == 0 for uq, uk in zip(used_q, used_k)
        ):
            return _assert_grad_layout(
                dq_g.detach().cpu(),
                dk_g.detach().cpu(),
                dv_g.detach().cpu(),
                q,
                k,
                v,
                "cpu/empty golden",
            )

        # Compact-used path:
        #   - npu name contains fusion → npu_fusion_attention_grad
        #   - gpu / npu *_bm (no fusion) → FA2 / flash_attn_npu compact bwd
        # Fusion has no seqused_* ABI; used lengths are encoded as actual_seq_*.
        use_compact = use_fusion or self.device == "gpu" or (
            self.device == "npu" and _node_is_bm(self.name)
        )
        if use_compact:
            from seqused_bwd_golden import _compact, _scatter, take_lengths

            take_q, take_k = take_lengths(used_q, used_k)
            active_q = [length for length in take_q if length > 0]
            active_k = [length for length in take_k if length > 0]
            q_c = _to_fa_compute_dtype(_compact(q, seqlens_q, take_q), native_dtype)
            k_c = _to_fa_compute_dtype(_compact(k, seqlens_k, take_k), native_dtype)
            v_c = _to_fa_compute_dtype(_compact(v, seqlens_k, take_k), native_dtype)
            dout_c = _to_fa_compute_dtype(
                _compact(dout, seqlens_q, take_q), native_dtype
            )
            out_c = _to_fa_compute_dtype(
                _compact(out, seqlens_q, take_q), native_dtype
            )
            cu_q = _cu_seqlens_from_seqlens(active_q)
            cu_k = _cu_seqlens_from_seqlens(active_k)
            max_q_c = int(max(active_q) if active_q else 0)
            max_k_c = int(max(active_k) if active_k else 0)

            if use_fusion:
                sm_max_c = _compact(softmax_max, seqlens_q, take_q).to(torch.float32)
                sm_sum_c = _compact(softmax_sum, seqlens_q, take_q).to(torch.float32)
                q_c = q_c.contiguous().npu()
                k_c = k_c.contiguous().npu()
                v_c = v_c.contiguous().npu()
                dout_c = dout_c.contiguous().npu()
                out_c = out_c.contiguous().npu()
                sm_max_c = sm_max_c.contiguous().npu()
                sm_sum_c = sm_sum_c.contiguous().npu()
                torch.npu.synchronize()
                dq_c, dk_c, dv_c = _run_fusion_attention_grad_tnd(
                    dout_c,
                    q_c,
                    k_c,
                    v_c,
                    out_c,
                    sm_max_c,
                    sm_sum_c,
                    head_num=num_heads,
                    softmax_scale=softmax_scale,
                    is_causal=causal,
                    window_size=window_size,
                    actual_seq_qlen=_cu_to_actual_seq_list(active_q),
                    actual_seq_kvlen=_cu_to_actual_seq_list(active_k),
                    max_seqlen_k=max_k_c,
                    deterministic=deterministic,
                    keep_prob=1.0,
                )
                torch.npu.synchronize()
                dq_c, dk_c, dv_c = _grads_to_current_dtype(
                    dq_c, dk_c, dv_c, current_dtype
                )
            elif self.device == "gpu":
                lse_c = _lse_for_fa_api(
                    _compact(softmax_lse.transpose(0, 1), seqlens_q, take_q).transpose(
                        0, 1
                    )
                )
                q_c = q_c.contiguous().cuda()
                k_c = k_c.contiguous().cuda()
                v_c = v_c.contiguous().cuda()
                dout_c = dout_c.contiguous().cuda()
                out_c = out_c.contiguous().cuda()
                lse_c = lse_c.contiguous().cuda()
                cu_q = cu_q.contiguous().cuda()
                cu_k = cu_k.contiguous().cuda()
                torch.cuda.synchronize()
                dq_c = torch.zeros_like(q_c)
                dk_c = torch.zeros_like(k_c)
                dv_c = torch.zeros_like(v_c)
                _run_flash_attn_backward_fa2_tnd(
                    dout_c,
                    q_c,
                    k_c,
                    v_c,
                    out_c,
                    lse_c,
                    dq_c,
                    dk_c,
                    dv_c,
                    cu_q,
                    cu_k,
                    max_q_c,
                    max_k_c,
                    softmax_scale,
                    causal,
                )
                dq_c, dk_c, dv_c = _grads_to_current_dtype(
                    dq_c.cpu(), dk_c.cpu(), dv_c.cpu(), current_dtype
                )
            else:
                lse_c = _lse_for_fa_api(
                    _compact(softmax_lse.transpose(0, 1), seqlens_q, take_q).transpose(
                        0, 1
                    )
                )
                q_c = q_c.contiguous().npu()
                k_c = k_c.contiguous().npu()
                v_c = v_c.contiguous().npu()
                dout_c = dout_c.contiguous().npu()
                out_c = out_c.contiguous().npu()
                lse_c = lse_c.contiguous().npu()
                cu_q = cu_q.contiguous().npu()
                cu_k = cu_k.contiguous().npu()
                torch.npu.synchronize()
                dq_c, dk_c, dv_c = _run_flash_attn_backward_npu(
                    dout_c,
                    q_c,
                    k_c,
                    v_c,
                    out_c,
                    lse_c,
                    backend=backend,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    seqused_q=None,
                    seqused_k=None,
                    max_seqlen_q=max_q_c,
                    max_seqlen_k=max_k_c,
                    softmax_scale=softmax_scale,
                    is_causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    deterministic=deterministic,
                )
                torch.npu.synchronize()
                dq_c, dk_c, dv_c = _grads_to_current_dtype(
                    dq_c, dk_c, dv_c, current_dtype
                )

            dq = _scatter(dq_c, seqlens_q, take_q, q)
            dk = _scatter(dk_c, seqlens_k, take_k, k)
            dv = _scatter(dv_c, seqlens_k, take_k, v)
            return _assert_grad_layout(
                dq.cpu(), dk.cpu(), dv.cpu(), q, k, v, "compact BM/fusion/gpu"
            )

        # DUT: packed TND + seqused prefixes (flash_attn_npu_3).
        q = _to_fa_compute_dtype(q, native_dtype).contiguous().npu()
        k = _to_fa_compute_dtype(k, native_dtype).contiguous().npu()
        v = _to_fa_compute_dtype(v, native_dtype).contiguous().npu()
        dout = _to_fa_compute_dtype(dout, native_dtype).contiguous().npu()
        out = _to_fa_compute_dtype(out.detach(), native_dtype).contiguous().npu()
        softmax_lse = _lse_for_fa_api(softmax_lse).contiguous().npu()
        cu_seqlens_q = _cu_seqlens_from_seqlens(seqlens_q).contiguous().npu()
        cu_seqlens_k = _cu_seqlens_from_seqlens(seqlens_k).contiguous().npu()
        seqused_q = _seqused_tensor(used_q, q.device) if seq_meta["pass_q"] else None
        seqused_k = _seqused_tensor(used_k, q.device) if seq_meta["pass_k"] else None
        torch.npu.synchronize()
        dq, dk, dv = _run_flash_attn_backward_npu(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            backend=backend,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            is_causal=causal,
            window_size=window_size,
            softcap=softcap,
            deterministic=deterministic,
        )
        torch.npu.synchronize()
        dq, dk, dv = _grads_to_current_dtype(dq, dk, dv, current_dtype)
        _mask_unused_grads(dq, dk, dv, seqlens_q, seqlens_k, used_q, used_k)
        return _assert_grad_layout(
            dq.cpu(), dk.cpu(), dv.cpu(), q, k, v, f"DUT backend={backend}"
        )
