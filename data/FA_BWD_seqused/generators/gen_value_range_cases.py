#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-emit the 5000-case seqused generalization suite with a different
input value range, keeping everything else byte-identical.

Why a post-processor instead of a new atk yaml
----------------------------------------------
The point of the exercise is a CONTROLLED comparison: "same shape, same
seqused prefixes, same tolerance -- only the input magnitude changes".
The atk generator draws shapes from a shared ``random`` state, so flipping
the yaml ``ranges`` would also reshuffle every (batch, seqlens, heads,
headdim) tuple and destroy the pairing. Rewriting only ``range_values`` on
q/k/v keeps:

  * ``id``            -> identical, and ``_make_dout`` seeds on the case id,
                         so dout is bit-identical too
  * shapes / seq_meta -> identical, so used-vs-region ratios are identical
  * ``standard``      -> untouched: max_re_ratio 5 / avg 1.5 / rmse 1.5

That makes a pass/fail flip between two suites attributable to numerics
alone, not to a different shape mix.

Profiles
--------
  pm5  q,k,v ~ U(-5, 5)      the official flash-attn test default
                             (tests/common/test_utils.py: make_random_tensor
                             low=-5.0 high=5.0)
  nd   q,k,v ~ N(0, 1)       the atk "nd" range form used across this repo
                             ({"name":"nd","mean":[0],"std":[1]})

Analytic effect on the logits (q,k independent, scale = 1/sqrt(d)):
  std(logit) = r^2/3 for U(-r,r)  ->  0.33 / 8.33   for pm1 / pm5
  std(logit) = sigma^2  for N(0,s) ->  1.00         for nd
``[-1,1]`` sits far below what real LLMs see (2~6); pm5 is the first
profile in this set that actually exercises a peaked softmax.

Usage
-----
  python gen_value_range_cases.py                     # both profiles, 950
  python gen_value_range_cases.py --backend ascend910
  python gen_value_range_cases.py --profiles pm5
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_dirs():
    """Locate (baseline_dir, output_dir) for both supported layouts.

    dev tree      : <here>/out5000/  holds the baseline, <here>/cases_range/
    archive layout: <here>/../cases/ holds both the baseline and the new files
                    (data/flash-attn-seqused/{cases,generators})
    """
    flat = os.path.join(os.path.dirname(HERE), "cases")
    if os.path.isfile(os.path.join(
            flat, "all_FlashAttnBwdSeqused_Accu_Generalization_950.json")):
        return flat, flat
    return os.path.join(HERE, "out5000"), os.path.join(HERE, "cases_range")


SRC_DIR, OUT_DIR = _resolve_dirs()

BACKEND_DEFAULT = "ascend950"

# q / k / v are the only tensors whose magnitude matters; cu_seqlens_* are
# int32 indices and must keep their [0, 1] placeholder (the real values are
# rebuilt from seq_meta at runtime).
VALUE_TENSOR_INDICES = (0, 1, 2)

PROFILES = {
    "pm5": {
        "range_values": [-5, 5],
        "tag": "RANGE_PM5",
        "note": "q/k/v ~ U(-5,5), official flash-attn test default",
    },
    "nd": {
        "range_values": {"name": "nd", "mean": [0.0], "std": [1.0]},
        "tag": "RANGE_ND",
        "note": "q/k/v ~ N(0,1), atk standard-normal range form",
    },
}


def _src_path(backend):
    return os.path.join(
        SRC_DIR, "all_FlashAttnBwdSeqused_Accu_Generalization_%s.json"
        % ("950" if backend == "ascend950" else "910")
    )


def _out_path(profile, backend):
    return os.path.join(
        OUT_DIR,
        "all_FlashAttnBwdSeqused_Accu_Generalization_%s_%s.json"
        % (PROFILES[profile]["tag"], "950" if backend == "ascend950" else "910"),
    )


def _stamp(case, profile):
    """Record the range profile in both meta blobs.

    ``_try_parse_seq_meta`` only reads sq/sk/uq/uk/pass_*/layout/backend and
    ignores unknown keys, so adding ``range`` is forward-compatible while
    making the result report self-describing. ``id`` and every numeric field
    stay untouched so the case remains pairable with its pm1 twin.
    """
    for key in ("aclnn_name", "save_name"):
        meta = json.loads(case[key])
        meta["range"] = profile
        case[key] = json.dumps(meta, separators=(",", ":"))
    case["name"] = "%s|%s" % (case["name"], PROFILES[profile]["tag"])
    return case


def build_profile(cases, profile):
    spec = PROFILES[profile]
    out = []
    for case in cases:
        c = json.loads(json.dumps(case))  # deep copy, keep the source pristine
        for idx in VALUE_TENSOR_INDICES:
            c["inputs"][idx]["range_values"] = json.loads(
                json.dumps(spec["range_values"])
            )
        out.append(_stamp(c, profile))
    return out


def _verify_paired(src, new, profile):
    """Fail loudly if anything other than the range/meta changed."""
    if len(src) != len(new):
        raise SystemExit("case count changed: %d -> %d" % (len(src), len(new)))
    for a, b in zip(src, new):
        if a["id"] != b["id"]:
            raise SystemExit("id reordered at %s" % a["id"])
        ma = json.loads(a["aclnn_name"])
        mb = json.loads(b["aclnn_name"])
        for k in ("layout", "backend", "mode", "sq", "sk", "uq", "uk",
                  "pass_q", "pass_k"):
            if ma.get(k) != mb.get(k):
                raise SystemExit(
                    "seq_meta drifted for id %s key %s: %r != %r"
                    % (a["id"], k, ma.get(k), mb.get(k))
                )
        if json.dumps(a["standard"], sort_keys=True) != json.dumps(
            b["standard"], sort_keys=True
        ):
            raise SystemExit("tolerance changed for id %s" % a["id"])
        for idx in VALUE_TENSOR_INDICES:
            if a["inputs"][idx]["shape"] != b["inputs"][idx]["shape"]:
                raise SystemExit("shape drifted for id %s" % a["id"])
            if a["inputs"][idx]["dtype"] != b["inputs"][idx]["dtype"]:
                raise SystemExit("dtype drifted for id %s" % a["id"])
            if b["inputs"][idx]["range_values"] != PROFILES[profile]["range_values"]:
                raise SystemExit("range not applied for id %s" % a["id"])
        # every non-q/k/v input must be untouched
        for idx in range(3, len(a["inputs"])):
            if a["inputs"][idx] != b["inputs"][idx]:
                raise SystemExit("input[%d] changed for id %s" % (idx, a["id"]))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default=BACKEND_DEFAULT,
                    choices=["ascend950", "ascend910"])
    ap.add_argument("--profiles", default=",".join(PROFILES),
                    help="comma separated subset of: " + ",".join(PROFILES))
    args = ap.parse_args(argv)

    src_path = _src_path(args.backend)
    if not os.path.isfile(src_path):
        raise SystemExit("missing baseline: %s" % src_path)
    with open(src_path, encoding="utf-8") as f:
        src = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for profile in [p.strip() for p in args.profiles.split(",") if p.strip()]:
        if profile not in PROFILES:
            raise SystemExit("unknown profile %r" % profile)
        cases = build_profile(src, profile)
        _verify_paired(src, cases, profile)
        path = _out_path(profile, args.backend)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cases, f, ensure_ascii=False, separators=(",", ":"))
        written.append((profile, path, len(cases)))

    for profile, path, n in written:
        size = os.path.getsize(path) / 1e6
        print("%-4s %5d cases  %6.2f MB  %s"
              % (profile, n, size, os.path.basename(path)))
        print("      %s" % PROFILES[profile]["note"])
    print("\npaired check: PASS (id/shape/seq_meta/tolerance identical to the "
          "[-1,1] baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
