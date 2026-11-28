#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R3 重测结果判定 —— 读 round3_perf_retest.csv，按 3 次中位重判达标，生成对照表。

用法（在 data/DeepEP_CLOS 目录下）：
    python scripts/gen_retest_result.py                      # 自动找最新的重测 CSV
    python scripts/gen_retest_result.py --csv round3_perf_retest.csv --date 20261129

判定规则与 R3重测清单_20261128.md §重测后的判定规则 一致：
    3 次中位 Δ ≤ −5%        → 确认真实劣化
    −5% < 中位 < 0           → 边界/噪声（不立项）
    中位 ≥ 0                 → 首测异常（用重测值替换）
    3 次极差 > 5%            → 该格作废，记环境告警

首测值从 round3_perf.csv + round3_perf_supp.csv 取，用于判断「是否复现」。
"""
import argparse
import csv
import io
import os
import re
import statistics
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
import gen_case_compare as G  # noqa: E402

SNAP_TAG = {"dispatch": "d", "cached-dispatch": "c",
            "combine_unweighted": "u", "combine_weighted": "w"}
OP_CN = {"dispatch": "dispatch", "cached-dispatch": "cached",
         "combine_unweighted": "combine_un", "combine_weighted": "combine_weighted"}
DROP = -5.0        # 判定阈值
SPREAD = 5.0       # 极差上限


def load_retest(path):
    """-> {(ep,T,H,aiv): {op: [每次的 gbps_mean]}}，按 run_id 去重（同一 run 的 4 op 行）。"""
    runs = {}
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        k = (int(r["ep_size"]), int(r["num_tokens"]), int(r["hidden"]), int(r["aiv"]))
        rid = r["run_id"]
        m = re.search(r"_r(\d+)$", rid)
        rep = int(m.group(1)) if m else 1
        runs.setdefault(k, {}).setdefault(r["op"], {})[rep] = float(r["gbps_mean"])
    out = {}
    for k, ops in runs.items():
        out[k] = {}
        for op, byrep in ops.items():
            out[k][op] = [byrep[r] for r in sorted(byrep)]
    return out


def find_default_csv():
    cands = [f for f in os.listdir(BASE) if re.match(r"round3_perf_.*retest.*\.csv$", f)]
    if not cands:
        sys.exit("没找到重测 CSV（round3_perf_*retest*.csv）；先用 run_round3.py --csv-tag retest 跑")
    cands.sort(key=lambda f: os.path.getmtime(os.path.join(BASE, f)), reverse=True)
    return os.path.join(BASE, cands[0])


def judge(med_delta, spread, drop=DROP, kind="基线"):
    if spread is not None and spread > SPREAD:
        return "作废", "三次极差 %.1f%% > %.0f%%，环境不稳，需重跑" % (spread, SPREAD)
    if med_delta is None:
        return "无基线", "该档基线缺失，且无同批 AIV=32 对照"
    if med_delta <= drop:
        return "确认劣化", "中位 %+.1f%% ≤ %.0f%%（对%s）" % (med_delta, drop, kind)
    if med_delta < 0:
        return "边界/噪声", "中位 %+.1f%% 在 (%.0f%%, 0)（对%s），不足以立项" % (med_delta, drop, kind)
    return "达标", "中位 %+.1f%% ≥ 0（对%s）" % (med_delta, kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="重测 CSV 路径，默认自动找最新")
    ap.add_argument("--date", default="", help="输出文件名里的日期，默认 20261129")
    ap.add_argument("--out", default="", help="输出 md 路径")
    args = ap.parse_args()

    csv_path = args.csv or find_default_csv()
    RT = load_retest(csv_path)
    if not RT:
        sys.exit("%s 里没有数据" % csv_path)

    M = G.load_round("round3_perf.csv")
    for k, v in G.load_round("round3_perf_supp.csv").items():
        M.setdefault(k, {}).update(v)
    SNAP = G.load_snap()

    date = args.date or "20261129"
    rows = []
    for k in sorted(RT):
        for op, vals in RT[k].items():
            med = statistics.median(vals)
            spread = (max(vals) - min(vals)) / statistics.mean(vals) * 100.0 if len(vals) > 1 else None
            base = SNAP[k][SNAP_TAG[op]] if (k in SNAP and SNAP_TAG.get(op) in SNAP[k]) else None
            first = M.get(k, {}).get(op, {}).get("mean")
            if base:
                d_med = G.pct(med, base)
                d_each = [G.pct(v, base) for v in vals]
                d_first = G.pct(first, base) if first else None
                verdict, why = judge(d_med, spread)
                kind = "基线"
            else:
                # 无基线档（EP4 全档 / EP64 部分格）：用同 EP 同 (T,H) 的 AIV=32 同批重测值作对照，
                # 走 R2 口径（阈值 −10%）。AIV=32 自己就是基准，不判。
                k32 = (k[0], k[1], k[2], 32)
                v32 = RT.get(k32, {}).get(op) if k[3] != 32 else None
                if v32 and len(v32) == len(vals):
                    m32 = statistics.median(v32)
                    d_med = G.pct(med, m32)
                    d_each = [G.pct(a, b) for a, b in zip(vals, v32)]
                    d_first = G.pct(first, M[k32][op]["mean"]) if (first and k32 in M and op in M[k32]) else None
                    verdict, why = judge(d_med, spread, drop=-10.0, kind="同批 AIV=32")
                else:
                    d_med, d_each, d_first = None, [], None
                    if k[3] == 32 and not base:
                        verdict, why = "对照", "AIV=32 为同批基准，只作分母不判达标"
                    else:
                        verdict, why = "无基线", "该档无基线且无同批 AIV=32 重测对照"
                kind = "AIV32"
            rows.append(dict(k=k, op=op, vals=vals, med=med, spread=spread, base=base,
                             d_med=d_med, d_each=d_each, first=first, d_first=d_first,
                             verdict=verdict, why=why, kind=kind))

    # ---------- 控制台 ----------
    print("重测数据：%s" % os.path.basename(csv_path))
    print("%d 格 × 指标 = %d 行\n" % (len(RT), len(rows)))
    cnt = {}
    for r in rows:
        cnt[r["verdict"]] = cnt.get(r["verdict"], 0) + 1
    for v in ("确认劣化", "边界/噪声", "达标", "作废", "无基线", "对照"):
        if v in cnt:
            print("  %-6s %d 行" % (v, cnt[v]))

    # ---------- MD ----------
    L = []
    A = L.append
    A("# R3 性能重测结果（%s）" % date)
    A("")
    A("> **本文由 `scripts/gen_retest_result.py` 机械生成**，源数据 `%s`。" % os.path.basename(csv_path))
    A("> 判定规则见 `R3重测清单_20261128.md` §重测后的判定规则：")
    A("> 3 次中位 ≤ −5% → **确认劣化**；( −5%, 0 ) → 边界/噪声；≥ 0 → 达标/首测异常；极差 > 5% → 作废。")
    A("> 基线仍用 `baseline/V2测试Case_20261128快照.txt`（对外交付口径）；`dispatch`/`cached` 比带权重列。")
    A("")
    A("## 判定汇总")
    A("")
    A("| 定性 | 行数 | 含义 |")
    A("|---|---|---|")
    meaning = {"确认劣化": "重测后仍显著低于基线，可立项/进入二分",
               "边界/噪声": "首测偏低但不可复现，不立项",
               "达标": "重测回到基线以上，首测为异常值",
               "作废": "三次离散过大，数据不可用",
               "无基线": "该档基线缺失且无同批对照",
               "对照": "AIV=32 基准格，只作分母"}
    for v in ("确认劣化", "边界/噪声", "达标", "作废", "无基线", "对照"):
        if v in cnt:
            A("| **%s** | %d | %s |" % (v, cnt[v], meaning[v]))
    A("")

    for verdict in ("确认劣化", "边界/噪声", "达标", "作废", "无基线", "对照"):
        sub = [r for r in rows if r["verdict"] == verdict]
        if not sub:
            continue
        A("## %s（%d 行）" % (verdict, len(sub)))
        A("")
        A("| EP | Tokens | Hidden | AIV | 指标 | 首测 Δ | 重测中位 | 三次 Δ | 极差 | 判据 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(sub, key=lambda r: (r["k"], r["op"])):
            k = r["k"]
            A("| %d | %d | %d | %d | %s | %s | %s | %s | %s | %s |" % (
                k[0], k[1], k[2], k[3], OP_CN.get(r["op"], r["op"]),
                ("%+.1f%%" % r["d_first"]) if r["d_first"] is not None else "—",
                ("%+.1f%%" % r["d_med"]) if r["d_med"] is not None else "—",
                " / ".join("%+.1f%%" % x for x in r["d_each"]) if r["d_each"] else "—",
                ("%.1f%%" % r["spread"]) if r["spread"] is not None else "—",
                r["why"]))
        A("")

    A("## 与首测的差异（只列 |Δ首测 − Δ重测| > 3 个百分点的行）")
    A("")
    drift = [r for r in rows if r["d_first"] is not None and r["d_med"] is not None
             and abs(r["d_med"] - r["d_first"]) > 3.0]
    if drift:
        A("| EP | Tokens | Hidden | AIV | 指标 | 首测 Δ | 重测 Δ | 差 |")
        A("|---|---|---|---|---|---|---|---|")
        for r in sorted(drift, key=lambda r: -abs(r["d_med"] - r["d_first"])):
            k = r["k"]
            A("| %d | %d | %d | %d | %s | %+.1f%% | %+.1f%% | %+.1f pp |" % (
                k[0], k[1], k[2], k[3], OP_CN.get(r["op"], r["op"]),
                r["d_first"], r["d_med"], r["d_med"] - r["d_first"]))
    else:
        A("无：重测与首测的偏差都在 3 个百分点以内 ⇒ 首测数据可信。")
    A("")

    out = args.out or os.path.join(BASE, "R3重测结果_%s.md" % date)
    with io.open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n已写：%s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
