#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Boundary cases for FlashAttention v3 TND backward seqused_q/seqused_k.

Standalone generator: it does not depend on the atk package so it can run on a
plain workstation. The produced JSON is byte-compatible with the ones emitted
by ``ascend_generate_FlashAttnBwdSeqused`` (same 31 top-level keys, same
``aclnn_name`` / ``save_name`` seq_meta protocol), so ``function_
FlashAttnBwdSeqused`` consumes it unchanged.

Design intent
-------------
Every case is built around ONE boundary axis, and the axis is encoded in the
case ``name`` suffix ``_BND_<TAG>`` so the axis is greppable in the result
report. Cases stay fully valid inputs: boundary means "edge of the legal
domain", not "illegal input". Illegal inputs live in the -NEG generator.

Boundary axes covered
  L1_REGN1      allocated region == 1 for both q and k
  L1_MIXED      region 1 mixed with a large region in the same batch
  USED_EQ1      used prefix == 1 while region is large
  USED_EQREG    used == allocated on one side, strict on the other
  USED_DIFF1    used == allocated - 1 (single dropped token)
  TILE_*        used length on the 16/64/128/256 tile boundaries +-1
  HEADDIM_*     headdim 1 / 128 and the 64B-alignment edge
  ALIGN_*       headdim that makes row bytes exactly 32B/64B multiples
  GQA_*         kv_heads 1 and 8, num_heads/kv_heads ratio at both ends
  BATCH_1       single batch
  BATCH_6       maximum batch (6)
  MAXLEN_*      region at the generator's token/attn limits
  CAUSAL_EQ     causal with used_q == used_k (diagonal-only attention)
  CAUSAL_D1     causal with used_k == used_q + 1
  NONCAUSAL_ASYM very asymmetric used_q vs used_k, non-causal
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cases_boundary")

# Same budgets as the generalization generator: the CPU fp64 golden is
# O(sum(sq*sk) * heads * headdim), so anything larger times out.
TOKEN_LIMIT = 2_000_000
ATTN_LIMIT = 4_000_000

BACKEND_DEFAULT = "ascend950"


# --------------------------------------------------------------------------
# schema helpers (mirror one case emitted by the atk generator)
# --------------------------------------------------------------------------
def case_skeleton(case_id, backend):
    """Build the 31-key skeleton with placeholder inputs, then fill."""
    return {
        "id": case_id,
        "default_seed": None,
        "name": "flash_attn_npu_3._flash_attn_backward",
        "aclnn_name": "",
        "triton_name": None,
        "kernel_name": None,
        "version": "v1.0",
        "expected_error_msg": None,
        "api": "pytorch",
        "api_type": "function_FlashAttnBwdSeqused",
        "aclnn_api_type": "aclnn_function",
        "triton_api_type": "triton_function",
        "fusion_api_type": "fusion_function",
        "fusion_mode": None,
        "dist_api_type": "dist_function",
        "kernel_api_type": "kernel_function",
        "backward": False,
        "standard": {
            "acc": {
                "cv_fused_double_benchmark": {
                    "max_re_ratio": 5,
                    "avg_re_ratio": 1.5,
                    "root_mean_squared_ratio": 1.5,
                }
            },
            "perf": [0.95, 0.95],
            "mem": 1.1,
        },
        "outputs": None,
        "inputs": None,
        "acl_json": "",
        "method_inputs": None,
        "tensor_input": None,
        "compute_times": None,
        "save_name": "",
        "uuid": None,
        "downloaded": False,
        "is_boundary": True,
        "xrun_cs_name": None,
        "xrun_data": None,
        "strategy": None,
    }


def build_inputs(sq, sk, uq, uk, heads, kv_heads, headdim, dtype,
                 causal, backend, scale=None):
    total_q = int(sum(sq))
    total_k = int(sum(sk))
    batch = len(sq)
    if scale is None:
        scale = 1.0 / (headdim ** 0.5)
    return [
        _tensor("q", dtype, [total_q, heads, headdim]),
        _tensor("k", dtype, [total_k, kv_heads, headdim]),
        _tensor("v", dtype, [total_k, kv_heads, headdim]),
        _tensor("cu_seqlens_q", "int32", [batch + 1], [0, 1]),
        _tensor("cu_seqlens_k", "int32", [batch + 1], [0, 1]),
        _attr("max_seqlen_q", "int", int(max(sq))),
        _attr("max_seqlen_k", "int", int(max(sk))),
        _attr("layout", "string", "TND"),
        _attr("softmax_scale", "float", float(scale)),
        _attr("causal", "bool", bool(causal)),
        [
            _attr("window_size", "attr_tuple", -1, typ="attr_tuple"),
            _attr("window_size", "attr_tuple", -1, typ="attr_tuple"),
        ],
        _attr("softcap", "float", 0.0),
        _attr("deterministic", "bool", False),
        _attr("backend", "string", backend),
    ]


def _tensor(name, dtype, shape, range_values=None, backward=True):
    return {
        "name": name,
        "type": "tensor",
        "required": name in ("q", "k", "v"),
        "dtype": dtype,
        "shape": list(shape),
        "range_values": range_values if range_values is not None else [-1, 1],
        "backward": backward,
        "align_32B": None,
        "outlier_values": None,
    }


def _attr(name, dtype, value, typ="attr"):
    return {
        "name": name,
        "type": typ,
        "required": False,
        "dtype": dtype,
        "shape": None,
        "range_values": value,
        "backward": False,
        "align_32B": None,
        "outlier_values": None,
    }


def seq_meta(backend, mode, sq, sk, uq, uk, tag):
    return json.dumps(
        {
            "layout": "TND",
            "backend": backend,
            "mode": mode,
            "sq": [int(x) for x in sq],
            "sk": [int(x) for x in sk],
            "uq": [int(x) for x in uq],
            "uk": [int(x) for x in uk],
            "pass_q": mode in ("both", "q_only"),
            "pass_k": mode in ("both", "k_only"),
            "boundary": tag,
        },
        separators=(",", ":"),
    )


def make_case(case_id, tag, sq, sk, uq, uk, heads, kv_heads, headdim,
              dtype, causal, backend=BACKEND_DEFAULT, mode="both"):
    validate(sq, sk, uq, uk, causal, mode)
    check_budget(sq, sk, heads, headdim)
    c = case_skeleton(case_id, backend)
    c["inputs"] = build_inputs(
        sq, sk, uq, uk, heads, kv_heads, headdim, dtype, causal, backend
    )
    meta = seq_meta(backend, mode, sq, sk, uq, uk, tag)
    c["aclnn_name"] = meta
    c["save_name"] = meta
    c["name"] = "flash_attn_npu_3._flash_attn_backward:%s" % tag
    return c


# --------------------------------------------------------------------------
# invariants the function/golden will enforce at runtime; assert them here so
# a broken case fails at generation time instead of on the NPU.
# --------------------------------------------------------------------------
class CaseError(Exception):
    pass


def validate(sq, sk, uq, uk, causal, mode):
    """Reject anything the runtime (function/golden) would reject.

    The function computes used_q/used_k from the mode:
      both   -> uq, uk     q_only -> uq, sk     k_only -> sq, uk
    A boundary case is only meaningful if the PASSED side is a strict prefix.
    """
    n = len(sq)
    if not (len(sk) == len(uq) == len(uk) == n):
        raise CaseError("batch length mismatch")
    for i in range(n):
        if sq[i] < 1 or sk[i] < 1:
            raise CaseError("allocated region must be >= 1, got sq=%s sk=%s" % (sq, sk))
        if not (1 <= uq[i] <= sq[i]):
            raise CaseError("uq %d outside [1, %d]" % (uq[i], sq[i]))
        if not (1 <= uk[i] <= sk[i]):
            raise CaseError("uk %d outside [1, %d]" % (uk[i], sk[i]))
    if causal:
        for i in range(n):
            if uq[i] > uk[i]:
                raise CaseError(
                    "causal requires uq <= uk, got uq=%s uk=%s" % (uq, uk)
                )
    # what the function will actually pass to the kernel
    eff_uq = uq if mode in ("both", "q_only") else list(sq)
    eff_uk = uk if mode in ("both", "k_only") else list(sk)
    if causal:
        for a, b in zip(eff_uq, eff_uk):
            if a > b:
                raise CaseError(
                    "causal effective used_q > used_k: %s > %s" % (eff_uq, eff_uk)
                )
    # At least one PASSED side must be a STRICT prefix somewhere in the batch,
    # otherwise the case tests nothing about seqused. A length-1 region cannot
    # be strict (decode shapes), so the check is "any", not "all".
    strict_q = any(a < b for a, b in zip(uq, sq))
    strict_k = any(a < b for a, b in zip(uk, sk))
    ok = (mode in ("both", "q_only") and strict_q) or (
        mode in ("both", "k_only") and strict_k
    )
    if not ok:
        raise CaseError(
            "degenerate: no passed side is a strict prefix "
            "(mode=%s, uq==sq and uk==sk)" % mode
        )


def check_budget(sq, sk, heads, headdim):
    tokens = sum(sq) * heads * headdim
    attn = sum(a * b for a, b in zip(sq, sk)) * heads
    if tokens > TOKEN_LIMIT or attn > ATTN_LIMIT:
        raise CaseError(
            "over golden budget: tokens=%d (limit %d) attn=%d (limit %d)"
            % (tokens, TOKEN_LIMIT, attn, ATTN_LIMIT)
        )


# --------------------------------------------------------------------------
# boundary case definitions
#
# Each spec is a dict so the intent of every axis stays readable:
#   tag, sq, sk, uq, uk, heads, kv, hd  (see make_case)
#   causal : attention masking
#   mode   : which seqused side(s) are actually passed
#   dtype  : None -> alternate fp16/bf16 automatically
#
# Note on region==1: a length-1 region cannot host a strict prefix, so such a
# case is only meaningful as a MIXED batch (other regions carry the seqused
# semantics) or with the other side strict.
# --------------------------------------------------------------------------
def build_cases():
    S = []

    def add(tag, sq, sk, uq, uk, heads, kv, hd, causal=False, mode="both",
            dtype=None):
        S.append(dict(tag=tag, sq=sq, sk=sk, uq=uq, uk=uk, heads=heads, kv=kv,
                      hd=hd, causal=causal, mode=mode, dtype=dtype))

    # ---- L1: region length exactly 1 ----
    # mixed batch: region 0 is length 1 on both sides (no seqused work possible
    # there), region 1 carries the real strict prefix.
    add("L1_MIXED", [1, 512], [1, 512], [1, 300], [1, 300], 4, 2, 64)
    add("L1_MIXED_CAUSAL", [1, 512], [1, 512], [1, 256], [1, 256], 4, 2, 64,
        causal=True)
    # region 1 on q, large on k: strict on the k side only
    add("L1_Q_ONLY", [1], [256], [1], [128], 2, 2, 64, mode="k_only")

    # ---- used == 1 while region is large ----
    add("USED_EQ1_Q", [1024], [1024], [1], [1024], 2, 2, 64, mode="q_only")
    add("USED_EQ1_K", [1024], [1024], [1024], [1], 2, 2, 64, mode="k_only")
    add("USED_EQ1_CAUSAL", [1024], [1024], [1], [1], 2, 2, 64, causal=True)
    add("USED_EQ1_BOTH", [768], [768], [1], [1], 2, 2, 128)

    # ---- used == region on one side only (single-side modes) ----
    add("USED_EQREG_Q", [777], [777], [777], [400], 4, 2, 64, mode="k_only")
    add("USED_EQREG_K", [777], [777], [400], [777], 4, 2, 64, mode="q_only")

    # ---- used == region - 1 (single dropped tail token) ----
    add("USED_DIFF1", [513], [513], [512], [512], 4, 4, 64)
    add("USED_DIFF1_ASYM", [512], [1024], [511], [1023], 2, 2, 128)
    add("USED_DIFF1_CAUSAL", [513], [513], [512], [512], 4, 4, 64, causal=True)

    # ---- tile boundaries: used length sits on 16/64/128/256 +-1 ----
    for b in (16, 64, 128, 256):
        for delta, suffix in ((-1, "M1"), (0, "EQ"), (1, "P1")):
            used = b + delta
            region = used + 16
            add("TILE_%d_%s" % (b, suffix), [region], [region],
                [used], [used], 4, 4, 64)
    # tile boundary on the k side only
    add("TILE_K_128_EQ", [256], [256], [256], [128], 4, 4, 64, mode="k_only")
    add("TILE_K_128_P1", [256], [256], [256], [129], 4, 4, 64, mode="k_only")
    add("TILE_Q_64_M1", [300], [300], [63], [300], 4, 4, 64, mode="q_only")

    # ---- headdim edges ----
    add("HEADDIM_1", [64], [64], [32], [32], 2, 2, 1)
    add("HEADDIM_128", [64], [64], [32], [32], 2, 2, 128)
    add("HEADDIM_127", [64], [64], [32], [32], 2, 2, 127)
    add("HEADDIM_129", [64], [64], [32], [32], 2, 2, 129)

    # ---- row alignment: headdim * 2 bytes == 32/64/96/128 ----
    add("ALIGN_32B_H16", [128], [128], [64], [64], 4, 4, 16)
    add("ALIGN_64B_H32", [128], [128], [64], [64], 4, 4, 32)
    add("ALIGN_96B_H48", [128], [128], [64], [64], 4, 4, 48)
    add("ALIGN_128B_H64", [128], [128], [64], [64], 4, 4, 64)

    # ---- GQA edges ----
    add("GQA_R1", [256], [256], [128], [128], 8, 8, 64)
    add("GQA_R4", [256], [256], [128], [128], 8, 2, 64)
    add("GQA_R8", [256], [256], [128], [128], 8, 1, 64)
    add("GQA_KV1_H32", [256], [256], [128], [128], 32, 1, 64)
    add("GQA_HEAD1", [256], [256], [128], [128], 1, 1, 64)

    # ---- batch edges ----
    add("BATCH_1", [1024], [1024], [512], [512], 2, 2, 64)
    add("BATCH_6", [64] * 6, [64] * 6, [32] * 6, [32] * 6, 4, 4, 64)
    # one region collapses to used==region, the rest stay strict
    add("BATCH_6_MIXED", [64, 96, 128, 160, 192, 224],
        [64, 96, 128, 160, 192, 224],
        [64, 48, 64, 80, 96, 112],
        [64, 48, 64, 80, 96, 112], 4, 4, 64)

    # ---- long region at the golden budget ----
    add("MAXLEN_Q", [8192], [256], [8192], [128], 1, 1, 16, mode="k_only")
    add("MAXLEN_K", [256], [8192], [128], [8192], 1, 1, 16, mode="q_only")
    add("MAXLEN_BOTH", [1024], [1024], [512], [512], 1, 1, 32)

    # ---- causal edges ----
    add("CAUSAL_EQ", [512], [512], [256], [256], 4, 4, 64, causal=True)
    add("CAUSAL_D1", [512], [512], [256], [257], 4, 4, 64, causal=True)
    add("CAUSAL_K_LONG", [512], [1024], [256], [1024], 4, 4, 64, causal=True)

    # ---- non-causal asymmetry ----
    add("NONCAUSAL_ASYM_Q", [1024], [64], [512], [32], 4, 4, 64)
    add("NONCAUSAL_ASYM_K", [64], [1024], [32], [512], 4, 4, 64)

    # ---- dtype x causal matrix on a representative boundary ----
    for d in ("fp16", "bf16"):
        for ca in (False, True):
            add("DT_%s_%s" % (d, "CAUSAL" if ca else "NONCAUSAL"),
                [256], [256], [100], [128], 4, 2, 64, causal=ca, dtype=d)

    return S


def main():
    os.makedirs(OUT, exist_ok=True)
    specs = build_cases()
    cases = []
    cid = 90000  # disjoint from the generalization file (0..4999)
    tags = []

    for i, s in enumerate(specs):
        dtype = s["dtype"] or ("fp16" if i % 2 == 0 else "bf16")
        cases.append(
            make_case(cid, s["tag"], s["sq"], s["sk"], s["uq"], s["uk"],
                      s["heads"], s["kv"], s["hd"], dtype, s["causal"],
                      mode=s["mode"])
        )
        tags.append(s["tag"])
        cid += 1

    # same set on both SoCs: duplicate for the 910 backend so each SoC keeps a
    # file whose cases are all runnable on that device family.
    n950 = len(cases)
    for base in list(cases):
        dup = json.loads(json.dumps(base))
        dup["id"] = cid
        dup["inputs"][13]["range_values"] = "ascend910"
        meta = json.loads(dup["aclnn_name"])
        meta["backend"] = "ascend910"
        blob = json.dumps(meta, separators=(",", ":"))
        dup["aclnn_name"] = blob
        dup["save_name"] = blob
        cases.append(dup)
        cid += 1

    path950 = os.path.join(OUT, "all_FlashAttnBwdSeqused_Boundary_950.json")
    path910 = os.path.join(OUT, "all_FlashAttnBwdSeqused_Boundary_910.json")
    with open(path950, "w", encoding="utf-8") as f:
        json.dump(cases[:n950], f, ensure_ascii=False, indent=1)
    with open(path910, "w", encoding="utf-8") as f:
        json.dump(cases[n950:], f, ensure_ascii=False, indent=1)

    print("boundary cases: %d per SoC, %d unique axes"
          % (n950, len(set(tags))))
    print("  ->", path950)
    print("  ->", path910)
    return 0


if __name__ == "__main__":
    sys.exit(main())
