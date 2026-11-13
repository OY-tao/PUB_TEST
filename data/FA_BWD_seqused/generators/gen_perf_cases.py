#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""120 performance cases for FA v3 TND backward seqused_q/seqused_k.

Why these shapes
----------------
A perf case is only useful if its shape actually exists in production. The
grid below is derived from what FlashAttention training really runs, not from
a random sampler:

  * prefill      sq long, causal, 1 batch          (the dominant bwd cost)
  * decode/append sq very short vs a long kv cache  (seqused is what makes
                  this possible at all: the cache is allocated to max_len but
                  only used_len is live)
  * varlen batch  many short sequences packed TND   (typical SFT/finetune)
  * GQA           nheads/kv_heads 4 and 8, as used by Llama/Qwen-class models
  * headdim       64 and 128 (the two that map onto Cube tiles)
  * cut ratio     used/allocated swept 1/8 .. 7/8 so the cost of honouring
                  seqused (vs the full region) is measurable

Each case declares test_mode-style intent through the ``perf_profile`` key in
seq_meta, and ``standard.perf = [0.95, 0.95]`` as in the rest of the repo.

Sizing rule: perf cases are ALLOWED to exceed the accuracy golden budget
(_TOKEN_LIMIT / _ATTN_LIMIT) because perf cases only need the device timing
loop; but the CPU golden is still executed for accuracy on the ``cpu`` node.
To keep a perf run tractable the volume is capped at PERF_TOKEN_LIMIT, and the
suite is split into a ``core`` file (runs everywhere) and an ``xl`` file
(needs a real device + long timeout).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cases_perf")

# Perf cases still pay the CPU fp64 golden once (accuracy node). Keep the same
# order of magnitude as the accuracy suite so a perf run is not dominated by
# the reference. The xl file deliberately breaks past this.
PERF_TOKEN_LIMIT = 2_000_000
PERF_ATTN_LIMIT = 4_000_000

sys.path.insert(0, HERE)
from gen_boundary_cases import case_skeleton, build_inputs  # noqa: E402

TARGET = 120


def perf_case(case_id, tag, profile, sq, sk, uq, uk, heads, kv, hd,
              dtype, causal, backend="ascend950", mode="both",
              window=None, softcap=0.0):
    total_q, total_k = int(sum(sq)), int(sum(sk))
    if mode in ("both", "q_only"):
        eff_uq = uq
    else:
        eff_uq = sq
    if mode in ("both", "k_only"):
        eff_uk = uk
    else:
        eff_uk = sk
    for a, b in zip(eff_uq, eff_uk):
        if a < 1 or b < 1:
            raise ValueError("%s: used length must be >= 1" % tag)
        if causal and a > b:
            raise ValueError("%s: causal requires used_q <= used_k" % tag)

    c = case_skeleton(case_id, backend)
    c["inputs"] = build_inputs(sq, sk, uq, uk, heads, kv, hd, dtype, causal,
                               backend)
    if window is not None:
        c["inputs"][10][0]["range_values"] = int(window[0])
        c["inputs"][10][1]["range_values"] = int(window[1])
    if softcap:
        c["inputs"][11]["range_values"] = float(softcap)

    # The seq_meta doubles as save_name, so it MUST carry every axis that
    # makes two cases different (heads, headdim, dtype, window, softcap);
    # otherwise distinct cases collapse onto one result file.
    meta = json.dumps(
        {
            "layout": "TND",
            "backend": backend,
            "mode": mode,
            "dtype": dtype,
            "heads": int(heads),
            "kv_heads": int(kv),
            "head_dim": int(hd),
            "window": [int(window[0]), int(window[1])] if window else [-1, -1],
            "softcap": float(softcap),
            "sq": [int(x) for x in sq],
            "sk": [int(x) for x in sk],
            "uq": [int(x) for x in uq],
            "uk": [int(x) for x in uk],
            "pass_q": mode in ("both", "q_only"),
            "pass_k": mode in ("both", "k_only"),
            "perf_profile": profile,
            "tag": tag,
        },
        separators=(",", ":"),
    )
    c["aclnn_name"] = meta
    c["save_name"] = meta
    c["name"] = "flash_attn_npu_3._flash_attn_backward:PERF_%s" % tag
    c["is_boundary"] = False
    c["strategy"] = {"perf_profile": profile, "tag": tag}
    return c


def build():
    """Return (core, xl) case lists. Ids start at 80000."""
    core, xl = [], []
    cid = [80000]

    def add(bucket, tag, profile, sq, sk, uq, uk, heads, kv, hd, dtype,
            causal, mode="both", window=None, softcap=0.0):
        cid[0] += 1
        c = perf_case(cid[0], tag, profile, sq, sk, uq, uk, heads, kv, hd,
                      dtype, causal, mode=mode, window=window,
                      softcap=softcap)
        bucket.append(c)

    # ---------------- 1. prefill, causal, long q ----------------
    # shape: [B=1, S, H, D] with the full region allocated and a real prefix
    for S, H, D in ((2048, 8, 64), (2048, 8, 128), (4096, 8, 64),
                    (4096, 8, 128), (8192, 8, 64), (8192, 8, 128)):
        add(core, "PREFILL_S%d_H%d_D%d" % (S, H, D), "prefill",
            [S], [S], [S // 2], [S // 2], H, H, D, "bf16", True)
    # same shapes with GQA 8:1
    for S, H, D in ((2048, 8, 128), (4096, 8, 128), (8192, 8, 128)):
        add(core, "PREFILL_GQA8_S%d_H%d_D%d" % (S, H, D), "prefill_gqa",
            [S], [S], [S // 2], [S // 2], H, 1, D, "bf16", True)

    # ---------------- 2. decode / append, kv cache ----------------
    # allocated kv is the max cache; used kv is the live length
    for kvlen, used, H, D in ((4096, 2048, 8, 128), (8192, 4096, 8, 128),
                              (16384, 8192, 8, 128), (8192, 1024, 8, 128),
                              (16384, 2048, 8, 128), (8192, 4096, 32, 128)):
        add(core, "DECODE_KV%d_U%d_H%d_D%d" % (kvlen, used, H, D), "decode",
            [1], [kvlen], [1], [used], H, min(H, 8), D, "bf16", False)
    # append: a few new tokens against a long cache
    for kvlen, used, q, H, D in ((8192, 4096, 16, 8, 128),
                                 (16384, 8192, 32, 8, 128),
                                 (8192, 2048, 64, 8, 64)):
        add(core, "APPEND_KV%d_U%d_Q%d" % (kvlen, used, q), "append",
            [q], [kvlen], [q], [used], H, H, D, "bf16", True)

    # ---------------- 3. varlen packed batch ----------------
    # many short sequences, as in SFT. used < allocated on both sides.
    for n, S, H, D in ((8, 512, 8, 64), (8, 1024, 8, 64), (16, 512, 8, 64),
                       (16, 1024, 8, 128), (32, 256, 8, 64),
                       (32, 512, 8, 128), (64, 128, 8, 64)):
        sq = [S] * n
        add(core, "VARLEN_N%d_S%d_H%d_D%d" % (n, S, H, D), "varlen",
            sq, list(sq), [s * 3 // 4 for s in sq], [s * 3 // 4 for s in sq],
            H, H, D, "bf16", True)
    # ragged lengths
    for n, H, D in ((8, 8, 128), (16, 8, 128)):
        sq = [max(64, (i + 1) * (1024 // n)) for i in range(n)]
        add(core, "RAGGED_N%d_H%d_D%d" % (n, H, D), "varlen_ragged",
            sq, list(sq), [s // 2 for s in sq], [s // 2 for s in sq],
            H, H, D, "bf16", True)

    # ---------------- 4. cut-ratio sweep ----------------
    # the cost of honouring seqused should scale with the live fraction.
    # 8/8 is not representable under causal (uq must stay <= uk) and would
    # also stop being a seqused case, so the sweep tops out just below it.
    base_s, H, D = 2048, 8, 128
    for num, den in ((1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8)):
        used = base_s * num // den
        add(core, "CUT_%dof%d" % (num, den), "cut_ratio", [base_s], [base_s],
            [used], [used], H, H, D, "bf16", True)
    # near-full reference on the k side only. Causal forbids uq > uk, so this
    # reference point has to be non-causal.
    add(core, "CUT_NEARFULL", "cut_ratio", [base_s], [base_s],
        [base_s], [base_s - 1], H, H, D, "bf16", False)
    # same sweep non-causal (kernel path differs)
    for num, den in ((1, 4), (1, 2), (3, 4)):
        used = base_s * num // den
        add(core, "CUT_NC_%dof%d" % (num, den), "cut_ratio_noncausal",
            [base_s], [base_s], [used], [used], H, H, D, "bf16", False)

    # ---------------- 5. GQA / MQA ----------------
    for S, H, KV, D in ((4096, 8, 8, 128), (4096, 8, 2, 128),
                        (4096, 8, 1, 128), (4096, 16, 4, 128),
                        (4096, 32, 8, 128), (8192, 16, 2, 128)):
        add(core, "GQA_S%d_H%d_KV%d_D%d" % (S, H, KV, D), "gqa",
            [S], [S], [S // 2], [S // 2], H, KV, D, "bf16", True)

    # ---------------- 6. headdim sweep ----------------
    for D in (32, 64, 96, 128, 160, 192, 256):
        add(core, "HEADDIM_S%d_D%d" % (2048, D), "headdim",
            [2048], [2048], [1024], [1024], 8, 8, D, "bf16", True)

    # ---------------- 7. dtype coverage ----------------
    for dt in ("fp16", "bf16"):
        for profile, S in (("dtype_prefill", 4096), ("dtype_varlen", 1024)):
            add(core, "DT_%s_%s" % (dt, profile), profile,
                [S], [S], [S // 2], [S // 2], 8, 8, 128, dt, True)

    # ---------------- 8. one-side-only seqused ----------------
    # q_only and k_only exercise different kernel branches.
    add(core, "QONLY_S4096", "q_only", [4096], [4096], [2048], [4096],
        8, 8, 128, "bf16", False, mode="q_only")
    add(core, "KONLY_S4096", "k_only", [4096], [4096], [4096], [2048],
        8, 8, 128, "bf16", False, mode="k_only")
    add(core, "QONLY_GQA_S4096", "q_only_gqa", [4096], [4096], [1024],
        [4096], 8, 1, 128, "bf16", False, mode="q_only")
    add(core, "KONLY_GQA_S4096", "k_only_gqa", [4096], [4096], [4096],
        [1024], 8, 1, 128, "bf16", False, mode="k_only")

    # ---------------- 9. XL: beyond the accuracy golden budget ----------------
    # These need a device with enough HBM and a long per-case timeout. They
    # are kept apart so the core file stays runnable in CI.
    for S, H, D in ((16384, 8, 128), (16384, 16, 128), (32768, 8, 128)):
        add(xl, "XL_PREFILL_S%d_H%d_D%d" % (S, H, D), "xl_prefill",
            [S], [S], [S // 2], [S // 2], H, H, D, "bf16", True)
    for kvlen, used, H, D in ((65536, 32768, 8, 128),
                              (131072, 65536, 8, 128)):
        add(xl, "XL_DECODE_KV%d_U%d" % (kvlen, used), "xl_decode",
            [1], [kvlen], [1], [used], H, min(H, 8), D, "bf16", False)
    for n, S, H, D in ((128, 512, 8, 128), (64, 1024, 8, 128)):
        sq = [S] * n
        add(xl, "XL_VARLEN_N%d_S%d" % (n, S), "xl_varlen",
            sq, list(sq), [s // 2 for s in sq], [s // 2 for s in sq],
            H, H, D, "bf16", True)
    add(xl, "XL_GQA_S16384_H32_KV4", "xl_gqa",
        [16384], [16384], [8192], [8192], 32, 4, 128, "bf16", True)

    # ---------------- 10. dtype x causal x cut matrix ----------------
    # fills the last of the 120: every perf-relevant knobs combination on a
    # mid-size shape, so a regression in any branch shows up.
    for dt in ("fp16", "bf16"):
        for ca in (False, True):
            for num, den in ((1, 4), (1, 2), (3, 4)):
                S = 2048
                used = S * num // den
                add(core, "M_%s_%s_%dof%d" % (dt, "CAU" if ca else "NC", num, den),
                    "matrix", [S], [S], [used], [used], 8, 8, 128, dt, ca)

    # ---------------- 11. batch sweep at fixed total tokens ----------------
    # same FLOPs, different batch: exposes per-batch launch overhead.
    total = 8192
    for n in (1, 2, 4, 8, 16, 32, 64):
        S = total // n
        sq = [S] * n
        add(core, "BATCHTOK_N%d_S%d" % (n, S), "batch_sweep",
            sq, list(sq), [s // 2 for s in sq], [s // 2 for s in sq],
            8, 8, 128, "bf16", True)

    # ---------------- 12. q<<k and q>>k asymmetry ----------------
    for q, k in ((16, 8192), (64, 8192), (128, 4096), (4096, 128),
                 (8192, 64)):
        add(core, "ASYM_Q%d_K%d" % (q, k), "asymmetry",
            [q], [k], [q // 2], [k // 2], 8, 8, 128, "bf16", False)

    # ---------------- 13. GQA with decode (production hot path) ----------
    for H, KV, D in ((32, 8, 128), (32, 4, 128), (16, 8, 64)):
        add(core, "DECODE_GQA_H%d_KV%d_D%d" % (H, KV, D), "decode_gqa",
            [1], [8192], [1], [4096], H, KV, D, "bf16", False)

    # ---------------- 14. sliding-window perf (Mistral/Qwen long ctx) -----
    # window_size != -1 routes to a different sparse mask; it is orthogonal to
    # seqused, so the pair must be measured together.
    for S, W in ((4096, 1024), (4096, 512), (8192, 2048), (8192, 1024),
                 (2048, 256)):
        add(core, "WIN_S%d_W%d" % (S, W), "window", [S], [S], [S // 2],
            [S // 2], 8, 8, 128, "bf16", True, window=(W, W))
    # right-window only (streaming attention)
    for S, W in ((4096, 512), (8192, 512)):
        add(core, "WINR_S%d_W%d" % (S, W), "window_right", [S], [S], [S // 2],
            [S // 2], 8, 8, 128, "bf16", True, window=(-1, W))

    # ---------------- 15. softcap perf (Gemma-2 style) -------------------
    for cap in (30.0, 50.0):
        add(core, "SOFTCAP_%d" % int(cap), "softcap", [2048], [2048],
            [1024], [1024], 8, 8, 128, "bf16", True, softcap=cap)

    # ---------------- 16. one-side-only at long length -------------------
    for S in (8192,):
        add(core, "QONLY_S%d" % S, "q_only_long", [S], [S], [S // 2], [S],
            8, 8, 128, "bf16", False, mode="q_only")
        add(core, "KONLY_S%d" % S, "k_only_long", [S], [S], [S], [S // 2],
            8, 8, 128, "bf16", False, mode="k_only")

    # ---------------- 17. many-batch decode (serving) --------------------
    for n, kv, used in ((8, 4096, 2048), (16, 2048, 1024), (32, 1024, 512)):
        sq = [1] * n
        sk = [kv] * n
        add(core, "DECODE_B%d_KV%d_U%d" % (n, kv, used), "decode_batch",
            sq, sk, [1] * n, [used] * n, 8, 8, 128, "bf16", False)

    # ---------------- 18. tile-exact perf shapes -------------------------
    # used length exactly on 128/256 boundaries: no tail tile to mask, the
    # fastest path; a regression here means the tail handling got pessimised.
    for S in (1024, 2048, 4096):
        add(core, "TILEEXACT_S%d" % S, "tile_exact", [S], [S], [S], [S - 1],
            8, 8, 128, "bf16", False)

    # ---------------- 19. varlen with long tail --------------------------
    # one long sequence dominates the batch, the rest are short: this is the
    # shape that breaks naive load balancing.
    for n, long_s, short_s in ((4, 8192, 128), (8, 4096, 256),
                               (16, 2048, 128)):
        sq = [long_s] + [short_s] * (n - 1)
        add(core, "VARLEN_TAIL_N%d_L%d_S%d" % (n, long_s, short_s),
            "varlen_tail", sq, list(sq),
            [s // 2 for s in sq], [s // 2 for s in sq], 8, 8, 128,
            "bf16", True)

    # ---------------- 20. headdim x GQA cross ---------------------------
    for D, KV in ((64, 1), (128, 1), (256, 2), (192, 4), (96, 8), (160, 8)):
        add(core, "CROSS_S2048_D%d_KV%d" % (D, KV), "cross",
            [2048], [2048], [1024], [1024], 8, KV, D, "bf16", True)

    return core, xl


def main():
    os.makedirs(OUT, exist_ok=True)
    core, xl = build()
    total = len(core) + len(xl)

    p_core = os.path.join(OUT, "all_FlashAttnBwdSeqused_Perf_core_950.json")
    p_xl = os.path.join(OUT, "all_FlashAttnBwdSeqused_Perf_xl_950.json")
    with open(p_core, "w", encoding="utf-8") as f:
        json.dump(core, f, ensure_ascii=False, indent=1)
    with open(p_xl, "w", encoding="utf-8") as f:
        json.dump(xl, f, ensure_ascii=False, indent=1)

    from collections import Counter
    prof = Counter(c["strategy"]["perf_profile"] for c in core + xl)
    print("perf cases: %d total = core %d + xl %d" % (total, len(core), len(xl)))
    print("  profiles:", dict(sorted(prof.items())))
    over = [
        c for c in core
        if sum(c["inputs"][0]["shape"][0:1]) * c["inputs"][0]["shape"][1]
        * c["inputs"][0]["shape"][2] > PERF_TOKEN_LIMIT
    ]
    print("  core cases over accuracy token limit: %d (allowed: perf node only)"
          % len(over))
    print("  ->", p_core)
    print("  ->", p_xl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
