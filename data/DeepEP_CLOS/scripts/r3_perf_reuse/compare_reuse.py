#!/usr/bin/env python3
"""两套性能补测对比：官方原口径（每 case 独立进程组+Buffer） vs 复用版。

输入：
  ../round3_perf_supp.csv        第 1 套，run_round3.py perf --supp 产出
  ../round3_perf_supp_reuse.csv  第 2 套，run_perf_reuse.py 产出
输出：
  ../round3_perf_reuse_compare.csv  逐 (形状, op) 的逐项对比 + 差值/百分比
  stdout                          汇总（按 EP / op，含 Buffer 初始化次数对比）

连接键 (ep_size, num_tokens, hidden, aiv, num_topk, dtype, op)；两边 gbps_mean
都是「各 rank 的 official_logical_GBps 的算术平均」，口径一致可直接比。
"""
from __future__ import annotations

import csv
import pathlib

HERE = pathlib.Path(__file__).resolve().parent          # .../scripts/r3_perf_reuse
DATA = HERE.parent.parent                                # .../data/DeepEP_CLOS
ISOLATED = DATA / "round3_perf_supp.csv"
REUSE = DATA / "round3_perf_supp_reuse.csv"
OUT = DATA / "round3_perf_reuse_compare.csv"

KEY = ("ep_size", "num_tokens", "hidden", "aiv", "num_topk", "dtype", "op")
OPS = ("dispatch", "cached-dispatch", "combine_unweighted", "combine_weighted")


def key_of(row):
    return tuple(str(row[k]).strip() for k in KEY)


def load(path):
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(path.open()):
        out.setdefault(key_of(r), r)
    return out


def main():
    iso, reu = load(ISOLATED), load(REUSE)
    common = sorted(set(iso) & set(reu))
    only_iso = sorted(set(iso) - set(reu))
    cols = list(KEY) + [
        "gbps_isolated", "gbps_reuse", "delta", "pct",
        "gbps_min_isolated", "gbps_max_isolated",
        "gbps_min_reuse", "gbps_max_reuse",
        "buffer_reused", "buffer_initializations_so_far",
    ]
    rows = []
    for k in common:
        a, b = iso[k], reu[k]
        va, vb = float(a["gbps_mean"]), float(b["gbps_mean"])
        r = dict(zip(KEY, k))
        r.update({
            "gbps_isolated": round(va, 3),
            "gbps_reuse": round(vb, 3),
            "delta": round(vb - va, 3),
            "pct": round((vb - va) / va * 100, 2) if va else "",
            "gbps_min_isolated": a.get("gbps_min", ""),
            "gbps_max_isolated": a.get("gbps_max", ""),
            "gbps_min_reuse": b.get("gbps_min", ""),
            "gbps_max_reuse": b.get("gbps_max", ""),
            "buffer_reused": b.get("buffer_reused", ""),
            "buffer_initializations_so_far": b.get("buffer_initializations_so_far", ""),
        })
        rows.append(r)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print("对比行数=%d（每 EP 每例 4 op）；仅第1套有=%d；仅第2套有=%d" %
          (len(rows), len(only_iso), len(set(reu) - set(iso))))
    print("输出：%s" % OUT.name)
    if not rows:
        return 0
    print("\n按 EP × op 汇总（pct = (复用-原口径)/原口径，负=复用版偏慢）：")
    for ep in ("4", "64"):
        for op in OPS:
            sel = [r for r in rows if r["ep_size"] == ep and r["op"] == op]
            if not sel:
                continue
            ps = [r["pct"] for r in sel if r["pct"] != ""]
            mean_pct = sum(ps) / len(ps) if ps else 0.0
            worst = max(ps, key=lambda x: abs(x)) if ps else 0.0
            print("  EP%-3s %-20s n=%-3d mean_pct=%+7.2f%%  worst_pct=%+7.2f%%" %
                  (ep, op, len(sel), mean_pct, worst))
    n_reused = sum(1 for r in rows if r["buffer_reused"] == "true")
    print("\nBuffer 复用：%d/%d 行标记 reused；" % (n_reused, len(rows)), end="")
    inits = sorted({int(r["buffer_initializations_so_far"]) for r in rows
                    if str(r["buffer_initializations_so_far"]).isdigit()})
    print("第2套 Buffer 初始化计数出现 %s" % (inits or "无"))
    print("口径说明：第2套 num_max_tokens_per_rank 统一取 16384（组内 max），"
          "是「一份 Buffer 覆盖组内 6 例」的必要代价；测量输入仍按各 case 的 num_tokens。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
