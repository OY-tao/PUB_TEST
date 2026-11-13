#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Repair the 3 degenerate cases in the 5000-case generalization files.

These were produced before ``_ensure_feasible`` existed. Each one ends up with
seqused == the allocated region on BOTH sides, so it would still pass even if
the kernel ignored seqused completely:

  id=1689  causal, k_only, sq == sk == [473]
           causal forces uk >= uq, and uq falls back to sq, so uk >= 473 == sk.
           The only legal uk is therefore the full region.
  id=4749  q_only, sq == [1]
           a one-token query cannot be reduced; the k side is not passed.
  id=4999  q_only, sq == [1]*6
           same reason, six times.

Repair strategy (keeps the case's intent, changes only what is needed):
  * use q_only/k_only => switch to ``both`` so the other side can be strict
  * grow the one-token query region so a strict prefix exists
  * keep sq/sk uq/uk consistent, rebuild every shape-derived field
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out5000")

TOKEN_LIMIT = 2_000_000
ATTN_LIMIT = 4_000_000

TARGETS = {
    1689: "causal k_only with sq==sk: switch to both and give k a real prefix",
    4749: "q_only with sq==1: grow q region, keep q strict",
    4999: "q_only with sq all 1: grow q regions, keep q strict",
}


def budget_ok(sq, sk, heads, hd):
    tok = sum(sq) * heads * hd
    att = sum(a * b for a, b in zip(sq, sk)) * heads
    return tok <= TOKEN_LIMIT and att <= ATTN_LIMIT


def rebuild(c, sq, sk, uq, uk, mode):
    """Rewrite a case in place: shapes, cu lengths, max_seqlen, meta."""
    heads = c["inputs"][0]["shape"][1]
    hd = c["inputs"][0]["shape"][2]
    kv = c["inputs"][1]["shape"][1]
    causal = bool(c["inputs"][9]["range_values"])
    batch = len(sq)

    # clamp + causal inventory, exactly as the generator does
    uq = [min(max(1, int(u)), int(s)) for u, s in zip(uq, sq)]
    uk = [min(max(1, int(u)), int(s)) for u, s in zip(uk, sk)]
    if causal:
        uq = [min(u, k) for u, k in zip(uq, uk)]
        uk = [max(k, q) if k >= q else k for q, k in zip(uq, sk)]
        uk = [min(u, s) for u, s in zip(uk, sk)]
        uq = [min(u, k) for u, k in zip(uq, uk)]

    c["inputs"][0]["shape"] = [int(sum(sq)), heads, hd]
    c["inputs"][1]["shape"] = [int(sum(sk)), kv, hd]
    c["inputs"][2]["shape"] = [int(sum(sk)), kv, hd]
    c["inputs"][3]["shape"] = [batch + 1]
    c["inputs"][4]["shape"] = [batch + 1]
    c["inputs"][5]["range_values"] = int(max(sq))
    c["inputs"][6]["range_values"] = int(max(sk))

    meta = json.loads(c["aclnn_name"])
    meta.update({
        "mode": mode,
        "sq": [int(x) for x in sq],
        "sk": [int(x) for x in sk],
        "uq": [int(x) for x in uq],
        "uk": [int(x) for x in uk],
        "pass_q": mode in ("both", "q_only"),
        "pass_k": mode in ("both", "k_only"),
    })
    blob = json.dumps(meta, separators=(",", ":"))
    c["aclnn_name"] = blob
    c["save_name"] = blob
    return c


def repair(c):
    """Return (new_sq, new_sk, new_uq, new_uk, new_mode) or None if no-op."""
    m = json.loads(c["aclnn_name"])
    sq, sk = list(m["sq"]), list(m["sk"])
    uq, uk = list(m["uq"]), list(m["uk"])
    mode = m["mode"]
    causal = bool(c["inputs"][9]["range_values"])
    cid = c["id"]

    def strict_q():
        return any(a < b for a, b in zip(uq, sq))

    def strict_k():
        return any(a < b for a, b in zip(uk, sk))

    passed_q = mode in ("both", "q_only")
    passed_k = mode in ("both", "k_only")
    if (passed_q and strict_q()) or (passed_k and strict_k()):
        return None  # already fine

    if cid == 1689:
        # causal + sq==sk: give the k side room by shrinking uk below sk.
        # uq must stay <= uk, so also drop uq.
        new_uk = max(2, sk[0] - 128)
        new_uq = min(uq[0], new_uk)
        return sq, sk, [new_uq], [new_uk], "both"

    if cid in (4749, 4999):
        # q_only with a one-token query: grow each query region to a length
        # that can host a strict prefix, keep k untouched (not passed).
        new_sq = [max(8, s) for s in sq]
        new_uq = [max(1, s // 2) for s in new_sq]
        heads = c["inputs"][0]["shape"][1]
        hd = c["inputs"][0]["shape"][2]
        # the packed q axis grows; make sure the k side still dominates cost
        if not budget_ok(new_sq, sk, heads, hd):
            return None
        return new_sq, sk, new_uq, list(uk), "q_only"

    return None


def main():
    total = 0
    for tag in ("910", "950"):
        path = os.path.join(
            OUT, "all_FlashAttnBwdSeqused_Accu_Generalization_%s.json" % tag
        )
        if not os.path.isfile(path):
            print("MISSING", path)
            return 1
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        fixed = 0
        for c in data:
            if c["id"] not in TARGETS:
                continue
            r = repair(c)
            if r is None:
                print("  %s id=%s: no repair needed / not possible"
                      % (tag, c["id"]))
                continue
            sq, sk, uq, uk, mode = r
            rebuild(c, sq, sk, uq, uk, mode)
            m = json.loads(c["aclnn_name"])
            ok = any(a < b for a, b in zip(m["uq"], m["sq"])) or any(
                a < b for a, b in zip(m["uk"], m["sk"])
            )
            print("  %s id=%-5s -> mode=%s sq=%s uq=%s sk=%s uk=%s strict=%s"
                  % (tag, c["id"], mode, m["sq"][:4], m["uq"][:4],
                     m["sk"][:4], m["uk"][:4], ok))
            fixed += 1

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print("%s: repaired %d case(s) -> %s" % (tag, fixed, path))
        total += fixed

    print("total repaired: %d" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
