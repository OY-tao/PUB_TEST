#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Negative / exceptional cases for FA v3 TND backward seqused_q/seqused_k.

IMPORTANT — how "abnormal" is expressed in ATK
----------------------------------------------
ATK has no assertion mechanism for "this case must raise". The repo-wide
``expected_error_msg`` field is null in all 235k cases and is not consumed by
``common/op_testcase.py``; ``TEST_MODE`` only decides whether a Failed acc
result is fatal. So a case that *intends* to raise can never be reported as a
pass by the runner.

What this file does instead: every negative case carries

  * an ``expected_error_msg`` substring (documented intent, greppable), and
  * a ``name`` suffix ``_NEG_<TAG>`` plus seq_meta key ``negative``.

and the guard whose message it should trip is a REAL check that already exists
in ``function_FlashAttnBwdSeqused`` / ``seqused_bwd_golden``. That means the
suite is runnable in two ways:

  1. As-is on the CPU node: the CPU path calls the same golden, so every case
     raises the documented ValueError before touching the device. A run that
     shows these errors is a PASS for the guard.
  2. Through ``verify_negative_cases.py``, which asserts that the exact
     expected substring appears -- that script is the pass/fail oracle.

The cases are therefore NOT meant to be dropped into an ACCURACY test set with
strict failure semantics; see the runbook in docs/ for how to wire them.

Guard inventory (all pre-existing, no kernel-side behaviour asserted here)
  region_lt_1        allocated length 0 (sq or sk)          -> golden sum() != packed
  used_gt_region     uq > sq / uk > sk                      -> "seqused ... outside"
  used_zero          uq == 0 / uk == 0                       -> "seqused ... outside"
  used_neg           uq < 0                                  -> "seqused ... outside"
  causal_uq_gt_uk    causal with used_q > used_k             -> function "used_q > used_k"
  sum_mismatch       sum(seqlens) != packed token axis       -> "case meta seqlens sum"
  layout_bad         layout != TND                           -> seq_meta parse -> None
  meta_missing       aclnn_name carries no seq_meta          -> "missing seqused metadata"
  backend_bad        backend not in alias table              -> "unsupported backend"
  scale_invalid      softmax_scale <= 0 or > 1               -> clamped, not an error
                     (documented as a WARNING case, not a negative one)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cases_negative")

# reuse the schema writer so both suites stay byte-compatible
sys.path.insert(0, HERE)
from gen_boundary_cases import (  # noqa: E402
    _attr,
    _tensor,
    case_skeleton,
)


# --------------------------------------------------------------------------
# The function resolves used_q/used_k from seq_meta only, and validates the
# packed token axis against sum(seqlens). To build a *negative* case we must
# write a seq_meta that violates one invariant, while the tensor shapes stay
# whatever the case declares (the guard fires before any real compute).
# --------------------------------------------------------------------------
def neg_case(case_id, tag, guard, expected_substr, sq, sk, uq, uk,
             heads, kv, headdim, dtype, causal, backend="ascend950",
             mode="both", packed_override=None, layout="TND",
             meta_override=None, scale=None):
    total_q = int(sum(sq)) if packed_override is None else packed_override[0]
    total_k = int(sum(sk)) if packed_override is None else packed_override[1]
    batch = max(1, len(sq))
    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    c = case_skeleton(case_id, backend)
    c["inputs"] = [
        _tensor("q", dtype, [total_q, heads, headdim]),
        _tensor("k", dtype, [total_k, kv, headdim]),
        _tensor("v", dtype, [total_k, kv, headdim]),
        _tensor("cu_seqlens_q", "int32", [batch + 1], [0, 1]),
        _tensor("cu_seqlens_k", "int32", [batch + 1], [0, 1]),
        _attr("max_seqlen_q", "int", int(max(sq)) if sq else 0),
        _attr("max_seqlen_k", "int", int(max(sk)) if sk else 0),
        _attr("layout", "string", layout),
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

    if meta_override is not None:
        meta = meta_override
    else:
        meta = json.dumps(
            {
                "layout": layout,
                "backend": backend,
                "mode": mode,
                "sq": [int(x) for x in sq],
                "sk": [int(x) for x in sk],
                "uq": [int(x) for x in uq],
                "uk": [int(x) for x in uk],
                "pass_q": mode in ("both", "q_only"),
                "pass_k": mode in ("both", "k_only"),
                "negative": tag,
            },
            separators=(",", ":"),
        )
    c["aclnn_name"] = meta
    c["save_name"] = meta
    c["name"] = "flash_attn_npu_3._flash_attn_backward:NEG_%s" % tag
    c["expected_error_msg"] = expected_substr
    c["is_boundary"] = False
    c["strategy"] = {"guard": guard, "tag": tag}
    return c


def build_cases():
    N = []
    add = N.append
    IDX = [0]  # sequential id inside the negative suite

    def push(tag, guard, substr, sq, sk, uq, uk, heads, kv, hd,
             dtype="fp16", causal=False, **kw):
        IDX[0] += 1
        add(neg_case(IDX[0], tag, guard, substr, sq, sk, uq, uk,
                     heads, kv, hd, dtype, causal, **kw))

    # ---- used prefix outside the allocated region ----
    push("USED_GT_REGION_Q", "used_gt_region", "outside allocated region",
         [256], [256], [257], [128], 4, 4, 64)
    push("USED_GT_REGION_K", "used_gt_region", "outside allocated region",
         [256], [256], [128], [257], 4, 4, 64)
    push("USED_GT_REGION_BOTH", "used_gt_region", "outside allocated region",
         [512], [512], [999], [999], 4, 4, 64)
    # off-by-one on the last token of a multi-region batch
    push("USED_GT_REGION_LAST", "used_gt_region", "outside allocated region",
         [128, 256], [128, 256], [128, 257], [128, 200], 4, 4, 64)

    # ---- used prefix == 0 (empty attention on that side) ----
    push("USED_ZERO_Q", "used_zero", "outside allocated region",
         [256], [256], [0], [128], 4, 4, 64)
    push("USED_ZERO_K", "used_zero", "outside allocated region",
         [256], [256], [128], [0], 4, 4, 64)
    push("USED_ZERO_BOTH", "used_zero", "outside allocated region",
         [256], [256], [0], [0], 4, 4, 64)

    # ---- negative used prefix ----
    push("USED_NEG_Q", "used_neg", "outside allocated region",
         [256], [256], [-1], [128], 4, 4, 64)
    push("USED_NEG_K", "used_neg", "outside allocated region",
         [256], [256], [128], [-8], 4, 4, 64)

    # ---- causal contract: used_q must not exceed used_k ----
    push("CAUSAL_UQ_GT_UK", "causal_uq_gt_uk", "used_q > used_k",
         [512], [512], [400], [200], 4, 4, 64, causal=True)
    push("CAUSAL_UQ_GT_UK_D1", "causal_uq_gt_uk", "used_q > used_k",
         [512], [512], [257], [256], 4, 4, 64, causal=True)
    # bypassed when only q is passed: used_k falls back to the full region,
    # so this must NOT raise -- kept as a control case.
    push("CAUSAL_Q_ONLY_CONTROL", "none", "<control: must run>",
         [512], [512], [400], [200], 4, 4, 64, causal=True, mode="q_only")

    # ---- packed token axis != sum(seqlens) ----
    push("SUM_MISMATCH_Q", "sum_mismatch", "case meta seqlens sum",
         [256], [256], [128], [128], 4, 4, 64, packed_override=(300, 256))
    push("SUM_MISMATCH_K", "sum_mismatch", "case meta seqlens sum",
         [256], [256], [128], [128], 4, 4, 64, packed_override=(256, 300))
    push("SUM_MISMATCH_BOTH", "sum_mismatch", "case meta seqlens sum",
         [128, 128], [128, 128], [64, 64], [64, 64], 4, 4, 64,
         packed_override=(100, 256))

    # ---- layout other than TND ----
    # _try_parse_seq_meta returns None for a non-TND layout, so the function
    # reports "missing seqused metadata" rather than a layout error.
    push("LAYOUT_BSND", "layout_bad", "missing seqused metadata",
         [256], [256], [128], [128], 4, 4, 64, layout="BSND")
    push("LAYOUT_BNSD", "layout_bad", "missing seqused metadata",
         [256], [256], [128], [128], 4, 4, 64, layout="BNSD")

    # ---- no seq_meta at all ----
    push("META_EMPTY", "meta_missing", "missing seqused metadata",
         [256], [256], [128], [128], 4, 4, 64, meta_override="")
    push("META_NOT_JSON", "meta_missing", "missing seqused metadata",
         [256], [256], [128], [128], 4, 4, 64, meta_override="TND-seqused")

    # ---- unsupported backend ----
    push("BACKEND_BAD", "backend_bad", "unsupported backend",
         [256], [256], [128], [128], 4, 4, 64,
         meta_override=json.dumps(
             {"layout": "TND", "backend": "ascend999", "mode": "both",
              "sq": [256], "sk": [256], "uq": [128], "uk": [128],
              "pass_q": True, "pass_k": True, "negative": "BACKEND_BAD"},
             separators=(",", ":")))

    # ---- shape-contract negatives (asserted by _assert_grad_layout) ----
    # q rank 2 instead of 3: the very first check in __call__.
    IDX[0] += 1
    c = neg_case(IDX[0], "Q_RANK2", "q_rank", "expects TND q rank 3",
                 [256], [256], [128], [128], 4, 4, 64, "fp16", False)
    c["inputs"][0]["shape"] = [256, 64]  # drop the head axis
    N.append(c)

    # kv_heads that do not divide num_heads: broadcast_kv_single computes
    # factor = nheads//nheads_k and indexes kv[:, j] with j up to nheads-1,
    # so it raises an IndexError from the tensor slice, NOT a clean
    # "not divisible" ValueError. Assert the real message.
    push("GQA_NOT_DIVISIBLE", "gqa_bad", "out of range",
         [256], [256], [128], [128], 4, 3, 64)

    # ---- documented NON-error: out-of-range softmax_scale is clamped ----
    # _finalize/_resolve_softmax_scale maps <=0 or >1 to a legal value, so
    # these must RUN, not raise. They are controls that prove the negative
    # suite does not simply "everything raises".
    push("SCALE_ZERO_CONTROL", "none", "<control: must run>",
         [256], [256], [128], [128], 4, 4, 64, scale=0.0)
    push("SCALE_GT1_CONTROL", "none", "<control: must run>",
         [256], [256], [128], [128], 4, 4, 64, scale=4.0)

    return N


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = build_cases()
    path = os.path.join(OUT, "all_FlashAttnBwdSeqused_Negative.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)

    neg = [c for c in cases if c["strategy"]["guard"] != "none"]
    ctl = len(cases) - len(neg)
    guards = sorted({c["strategy"]["guard"] for c in neg})
    print("negative cases: %d total (%d guarded, %d control-must-run)"
          % (len(cases), len(neg), ctl))
    print("  guards: %s" % ", ".join(guards))
    print("  ->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
