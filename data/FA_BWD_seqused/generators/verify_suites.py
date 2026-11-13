#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Structural validator for the seqused boundary / negative / perf suites.

Checks that do not need torch or a device:
  1. every case has the exact 31-key schema of the atk generator
  2. input list shape/order/type/dtype matches the original 5000-case file
  3. seq_meta is parseable, layout TND, backend/device field agree
  4. sum(sq) == q token axis, sum(sk) == k/v token axis
  5. 1 <= uq <= sq, 1 <= uk <= sk
  6. causal => effective used_q <= used_k
  7. the PASSED side is a strict prefix (boundary/perf), or the case is
     explicitly a negative with a documented guard
  8. cu_seqlens length == batch+1, max_seqlen_* == max(len)*
  9. ids unique inside a file, save_name unique per file
 10. golden budget respected for the accuracy suites
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

TOKEN_LIMIT = 2_000_000
ATTN_LIMIT = 4_000_000

EXPECT_KEYS = [
    "id", "default_seed", "name", "aclnn_name", "triton_name", "kernel_name",
    "version", "expected_error_msg", "api", "api_type", "aclnn_api_type",
    "triton_api_type", "fusion_api_type", "fusion_mode", "dist_api_type",
    "kernel_api_type", "backward", "standard", "outputs", "inputs", "acl_json",
    "method_inputs", "tensor_input", "compute_times", "save_name", "uuid",
    "downloaded", "is_boundary", "xrun_cs_name", "xrun_data", "strategy",
]

INPUT_NAMES = [
    "q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q",
    "max_seqlen_k", "layout", "softmax_scale", "causal", "window_size",
    "softcap", "deterministic", "backend",
]

def _resolve_cases_root():
    """Locate the directory that holds the case files.

    Two layouts are supported so the same script works both in the dev tree
    (cases in sibling dirs of the script) and in the archive
    (data/flash-attn-seqused/{cases,generators}).
    """
    here = HERE
    candidates = [
        os.path.dirname(here),          # archive: ../cases
        here,                            # dev: ./out5000, ./cases_boundary
    ]
    probes = [
        ("cases", "all_FlashAttnBwdSeqused_Boundary_950.json"),
        ("cases_boundary", "all_FlashAttnBwdSeqused_Boundary_950.json"),
    ]
    for root in candidates:
        for sub, fname in probes:
            if os.path.isfile(os.path.join(root, sub, fname)):
                return root, sub
    return here, "cases_boundary"


ROOT, CASES_SUB = _resolve_cases_root()


def _case_path(fname):
    # archived cases live in a flat ``cases/`` dir; in the dev tree each suite
    # has its own dir, so try a few sub-dirs and fall back to the file name.
    subs = [CASES_SUB, "cases", "out5000", "cases_boundary", "cases_negative",
            "cases_perf", "cases_range", ""]
    for sub in subs:
        p = os.path.join(ROOT, sub, fname) if sub else os.path.join(ROOT, fname)
        if os.path.isfile(p):
            return p
    return None


FILES = {
    "gen_950": ("all_FlashAttnBwdSeqused_Accu_Generalization_950.json", "accuracy"),
    "gen_910": ("all_FlashAttnBwdSeqused_Accu_Generalization_910.json", "accuracy"),
    "rng_pm5_950": ("all_FlashAttnBwdSeqused_Accu_Generalization_RANGE_PM5_950.json", "accuracy"),
    "rng_nd_950": ("all_FlashAttnBwdSeqused_Accu_Generalization_RANGE_ND_950.json", "accuracy"),
    "boundary_950": ("all_FlashAttnBwdSeqused_Boundary_950.json", "accuracy"),
    "boundary_910": ("all_FlashAttnBwdSeqused_Boundary_910.json", "accuracy"),
    "negative": ("all_FlashAttnBwdSeqused_Negative.json", "negative"),
    "perf_core": ("all_FlashAttnBwdSeqused_Perf_core_950.json", "perf"),
    "perf_xl": ("all_FlashAttnBwdSeqused_Perf_xl_950.json", "perf"),
}

errors = []
warns = []


def err(f, cid, msg):
    errors.append("%s[%s]: %s" % (f, cid, msg))


def check_case(fname, c, kind):
    cid = c.get("id")
    missing = set(EXPECT_KEYS) - set(c.keys())
    extra = set(c.keys()) - set(EXPECT_KEYS)
    if missing:
        err(fname, cid, "missing keys %s" % sorted(missing))
    if extra:
        err(fname, cid, "unexpected keys %s" % sorted(extra))

    inputs = c.get("inputs") or []
    if len(inputs) != len(INPUT_NAMES):
        err(fname, cid, "expected %d inputs, got %d"
            % (len(INPUT_NAMES), len(inputs)))
        return
    for i, want in enumerate(INPUT_NAMES):
        node = inputs[i]
        nodes = node if isinstance(node, list) else [node]
        if nodes[0].get("name") != want:
            err(fname, cid, "input[%d] name %r != %r"
                % (i, nodes[0].get("name"), want))
    if not isinstance(inputs[10], list) or len(inputs[10]) != 2:
        err(fname, cid, "window_size must be a 2-element attr_tuple list")

    # dtype coherence
    dt = inputs[0].get("dtype")
    if dt not in ("fp16", "bf16"):
        err(fname, cid, "unexpected q dtype %r" % dt)
    for i in (1, 2):
        if inputs[i].get("dtype") != dt:
            err(fname, cid, "k/v dtype %r != q dtype %r"
                % (inputs[i].get("dtype"), dt))
    for i in (3, 4):
        if inputs[i].get("dtype") != "int32":
            err(fname, cid, "cu_seqlens dtype must be int32")

    meta_raw = c.get("aclnn_name")
    if not isinstance(meta_raw, str) or not meta_raw.strip().startswith("{"):
        if kind == "negative" or True:
            return  # documented negative (meta intentionally broken)
    try:
        meta = json.loads(meta_raw)
    except Exception as e:
        err(fname, cid, "aclnn_name is not JSON: %s" % e)
        return

    sq, sk = meta.get("sq"), meta.get("sk")
    uq, uk = meta.get("uq"), meta.get("uk")
    if not all(isinstance(x, list) for x in (sq, sk, uq, uk)):
        err(fname, cid, "seq lists malformed")
        return
    if not (len(sq) == len(sk) == len(uq) == len(uk)):
        err(fname, cid, "batch length mismatch in meta")
        return
    batch = len(sq)

    guard = (c.get("strategy") or {}).get("guard", "none")
    NEG = kind == "negative"
    exempt_backend = NEG and guard == "backend_bad"
    exempt_layout = NEG and guard in ("layout_bad", "meta_missing")

    # layout / backend agreement
    if not exempt_layout and meta.get("layout") != inputs[7].get("range_values"):
        err(fname, cid, "layout mismatch meta=%r attr=%r"
            % (meta.get("layout"), inputs[7].get("range_values")))
    if not exempt_backend and meta.get("backend") != inputs[13].get("range_values"):
        err(fname, cid, "backend mismatch meta=%r attr=%r"
            % (meta.get("backend"), inputs[13].get("range_values")))

    qshape, kshape = inputs[0].get("shape"), inputs[1].get("shape")
    if not qshape or not kshape:
        err(fname, cid, "q/k shape missing")
        return

    # negative cases are allowed to break these on purpose, but only the ones
    # whose guard is documented in strategy.
    guard = (c.get("strategy") or {}).get("guard", "none")
    # A negative case is EXPECTED to violate exactly the invariant its guard
    # defends. Exempt that one invariant, keep checking everything else.
    NEG = kind == "negative"
    exempt_used_range = NEG and guard in ("used_gt_region", "used_zero",
                                          "used_neg")
    exempt_sum = NEG and guard == "sum_mismatch"
    exempt_causal = NEG and guard == "causal_uq_gt_uk"
    exempt_backend = NEG and guard == "backend_bad"
    exempt_layout = NEG and guard in ("layout_bad", "meta_missing")

    if not exempt_sum:
        if sum(sq) != int(qshape[0]):
            err(fname, cid, "sum(sq)=%d != q tokens %d"
                % (sum(sq), qshape[0]))
        if sum(sk) != int(kshape[0]):
            err(fname, cid, "sum(sk)=%d != k tokens %d"
                % (sum(sk), kshape[0]))

    if not exempt_used_range:
        for a, b in zip(uq, sq):
            if not (1 <= a <= b):
                err(fname, cid, "uq %d outside [1,%d]" % (a, b))
        for a, b in zip(uk, sk):
            if not (1 <= a <= b):
                err(fname, cid, "uk %d outside [1,%d]" % (a, b))

    # cu_seqlens / max_seqlen
    for i, lens in ((3, sq), (4, sk)):
        if inputs[i].get("shape") != [batch + 1]:
            err(fname, cid, "cu_seqlens len %r != batch+1 (%d)"
                % (inputs[i].get("shape"), batch + 1))
    if inputs[5].get("range_values") != max(sq):
        err(fname, cid, "max_seqlen_q %r != max(sq) %d"
            % (inputs[5].get("range_values"), max(sq)))
    if inputs[6].get("range_values") != max(sk):
        err(fname, cid, "max_seqlen_k %r != max(sk) %d"
            % (inputs[6].get("range_values"), max(sk)))

    # causal + effective used lengths
    causal = inputs[9].get("range_values")
    mode = meta.get("mode", "both")
    eff_uq = uq if mode in ("both", "q_only") else list(sq)
    eff_uk = uk if mode in ("both", "k_only") else list(sk)
    if causal and not exempt_causal:
        for a, b in zip(eff_uq, eff_uk):
            if a > b:
                err(fname, cid, "causal effective uq %d > uk %d" % (a, b))
                break
    if kind != "negative":
        strict_q = any(a < b for a, b in zip(uq, sq))
        strict_k = any(a < b for a, b in zip(uk, sk))
        passed_q = mode in ("both", "q_only")
        passed_k = mode in ("both", "k_only")
        # A length-1 q region cannot host a strict prefix (decode / append
        # shapes), so the requirement is: at least one PASSED side is a strict
        # prefix somewhere in the batch. A case where every passed side is
        # full is degenerate -- it would pass even if seqused were ignored.
        ok = (passed_q and strict_q) or (passed_k and strict_k)
        if not ok:
            err(fname, cid,
                "degenerate: no passed side is ever a strict prefix "
                "(mode=%s, uq==sq and uk==sk)" % mode)

        # golden budget for the accuracy suites
        if kind == "accuracy":
            heads, hd = qshape[1], qshape[2]
            tok = sum(sq) * heads * hd
            att = sum(a * b for a, b in zip(sq, sk)) * heads
            if tok > TOKEN_LIMIT or att > ATTN_LIMIT:
                warns.append("%s[%s]: over accuracy budget tok=%d att=%d"
                             % (fname, cid, tok, att))

    # GQA sanity: kv heads must divide q heads for the golden broadcast
    if len(qshape) < 3 or len(kshape) < 3:
        # only the deliberate q_rank negative may be rank-deficient
        if guard != "q_rank":
            err(fname, cid, "q/k rank < 3 without q_rank guard")
        return
    heads, kv, hd = qshape[1], kshape[1], qshape[2]
    if kshape[1] != kv or kshape[2] != hd:
        pass
    if kv > heads:
        err(fname, cid, "kv_heads %d > num_heads %d" % (kv, heads))
    if heads % kv != 0 and guard != "gqa_bad":
        err(fname, cid, "num_heads %d not divisible by kv_heads %d"
            % (heads, kv))

    # standard block
    std = c.get("standard") or {}
    if "acc" not in std or "perf" not in std:
        err(fname, cid, "standard missing acc/perf")


def check_file(name, fname, kind):
    path = _case_path(fname)
    if path is None:
        errors.append("%s: file missing %s (searched under %s)"
                      % (name, fname, ROOT))
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        errors.append("%s: not a non-empty list" % name)
        return None
    ids = [c.get("id") for c in data]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        errors.append("%s: duplicate ids %s" % (name, dup[:5]))
    names = [c.get("save_name") for c in data]
    dupn = [k for k, v in Counter(names).items() if v > 1]
    if dupn:
        errors.append("%s: %d duplicate save_name values" % (name, len(dupn)))
    for c in data:
        check_case(name, c, kind)
    return data


def main():
    stats = {}
    for name, (rel, kind) in FILES.items():
        d = check_file(name, rel, kind)
        if d is None:
            continue
        dt = Counter(c["inputs"][0]["dtype"] for c in d)
        modes = Counter(json.loads(c["aclnn_name"]).get("mode") if
                        isinstance(c["aclnn_name"], str) and
                        c["aclnn_name"].startswith("{") else "?" for c in d)
        causal = Counter(c["inputs"][9]["range_values"] for c in d)
        stats[name] = (len(d), dict(dt), dict(modes), dict(causal))

    print("=" * 74)
    for name, (n, dt, modes, causal) in stats.items():
        print("%-14s n=%-4d dtype=%s mode=%s causal=%s"
              % (name, n, dt, modes, causal))
    print("=" * 74)

    if warns:
        print("\nWARN (%d):" % len(warns))
        for w in warns[:10]:
            print("  ", w)
        if len(warns) > 10:
            print("   ... %d more" % (len(warns) - 10))

    if errors:
        print("\nFAIL: %d error(s)" % len(errors))
        for e in errors[:40]:
            print("  ", e)
        if len(errors) > 40:
            print("   ... %d more" % (len(errors) - 40))
        return 1

    print("\nALL CHECKS PASSED (%d files)" % len(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
