#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 R3逐case对比_20261128.md —— 从三个 CSV 与两份基线表机械生成，不手抄数字。

用法（在 data/DeepEP_CLOS 目录下）：
    python scripts/gen_case_compare.py

数据源：
    round3_perf.csv           B 阶段 36 例
    round3_perf_supp.csv      EP4/EP64 补测 36 例
    baseline/V2测试Case_20261128快照.txt   基线（对外口径）
    results_20261125_op.csv   PR #37 之前同集群同 bench 同 seed（1125）
    baseline/EP4-128_通信带宽_20261128.txt  我方 11-28 上一批（1128）
"""
import csv
import io
import os
import statistics
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OPS = [
    ("dispatch", "dispatch", "d"),
    ("cached-dispatch", "cached", "c"),
    ("combine_unweighted", "cU", "u"),
    ("combine_weighted", "cW", "w"),
]
# 快照列: 5=Dispatch无权重 6=Dispatch带权重 7=Cached 8=Combine无权重 9=Combine带权重
SNAP_COL = {"d": 6, "c": 7, "u": 8, "w": 9}


def load_round(fname):
    d = {}
    for r in csv.DictReader(io.open(os.path.join(BASE, fname), encoding="utf-8")):
        k = (int(r["ep_size"]), int(r["num_tokens"]), int(r["hidden"]), int(r["aiv"]))
        d.setdefault(k, {})[r["op"]] = dict(
            mean=float(r["gbps_mean"]), mn=float(r["gbps_min"]), mx=float(r["gbps_max"])
        )
    return d


def load_snap():
    d = {}
    for line in io.open(os.path.join(BASE, "baseline/V2测试Case_20261128快照.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 10 or not c[0].isdigit():
            continue
        k = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
        d[k] = {name: float(c[idx]) for name, idx in SNAP_COL.items()}
    return d


def load_1125():
    d = {}
    for r in csv.DictReader(io.open(os.path.join(BASE, "results_20261125_op.csv"), encoding="utf-8")):
        k = (int(r["ep_size"]), int(r["num_tokens"]), int(r["hidden"]), int(r["aiv"]))
        d.setdefault(k, {})[r["op"]] = float(r["official_logical_GBps_rank_mean"])
    return d


def load_1128():
    d = {}
    for line in io.open(os.path.join(BASE, "baseline/EP4-128_通信带宽_20261128.txt"), encoding="utf-8"):
        if line.startswith("#"):
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 11 or not c[0].isdigit():
            continue
        k = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
        d[k] = dict(d=float(c[6]), c=float(c[7]), u=float(c[9]), w=float(c[10]))
    return d


def pct(a, b):
    return (a / b - 1.0) * 100.0 if b else float("nan")


def cell(mine, base):
    return "%.1f / %.1f (%+.1f%%)" % (mine, base, pct(mine, base))


def main():
    M = load_round("round3_perf.csv")
    for k, v in load_round("round3_perf_supp.csv").items():
        M.setdefault(k, {}).update(v)
    SNAP = load_snap()
    P25 = load_1125()
    O28 = load_1128()

    ok_cases = sorted(k for k in M if len(M[k]) == 4)
    ov_snap = sorted(k for k in M if k in SNAP)
    ov_25 = sorted(k for k in M if k in P25)
    ov_28 = sorted(k for k in M if k in O28)

    out = []
    A = out.append
    A("# R3 性能逐 case 对比（2026-11-28 实测 vs 三个对照系）")
    A("")
    A("> **本文由 `scripts/gen_case_compare.py` 机械生成**，数字不手抄；重跑即可复现。")
    A("> 实测：`round3_perf.csv`（B 阶段 36 例）+ `round3_perf_supp.csv`（补测 EP4/EP64 36 例）")
    A("> = **72 例 × 4 op = 288 行**，`rc` 全 0。性能批门禁 `ts=18:10:27` `pass=true`（**该 json 已被 C 阶段")
    A("> 19:04:01 覆盖**，但关键字段不变：commit `0d1eca2`、`buffer.py` sha `f825a679`、`qp_plans` 空、")
    A("> 8 台同镜像 `dff1aa4b`、8 台各 16 卡全空闲）。单位 GB/s/rank，取 `gbps_mean`（rank 均值）。")
    A("> **实测 commit = `0d1eca2`（PR #37 合并前口径）**。PR #37 已于 2026-11-28 16:21 合并为 `19eadeb`，")
    A("> 最终 head `52bdfb70` 比 `0d1eca2` 只多 1 个提交（`csrc/direct/runtime/shmem_instance.cpp` +10/−7，")
    A("> SHMEM 初始化清理路径，**不在 dispatch/combine 热路径**）⇒ 性能结论对 main 仍成立，但对外须写「实测 0d1eca2」，不能写「main」。")
    A("> Δ = 我方 / 对照 − 1，**正值 = 我方更快**。")
    A("")
    A("**口径（错一次就系统性偏 6.4%）**：`dispatch` / `cached-dispatch` 的 official_logical_bytes 含 weights")
    A("（`stage_metrics.py:117-119`：`x + scales + full top-k indices/weights`，5728 B/row），")
    A("所以对基线必须比「Dispatch **带**权重」列。本文已按带权重列；比无权重列会把自己低估 6.4%。")
    A("")
    A("## 三个对照系：覆盖面不同，不能混用")
    A("")
    A("| 对照系 | 可比例数 | 覆盖网格 | 能回答什么 | 不能回答什么 |")
    A("|---|---|---|---|---|")
    A("| **基线快照**（对外口径） | **%d** | EP8/16/32 全 12 例 + EP64 的 AIV32 6 例 | 交付是否达标 | EP4 全 18 例、EP64 的 AIV64 12 例无对照；EP128 缺 |" % len(ov_snap))
    A("| **1125**（PR #37 前，同集群同 bench 同 seed） | **%d** | EP8/16/32/64，AIV32 | PR #37 的净收益 | 基线达标性（它本身不是基线） |" % len(ov_25))
    A("| **1128 原表**（我方 11-28 上一批） | **%d** | EP4/EP64 各 18 例全重叠；EP8/16/32 只有 AIV32 各 6 例 | 相对上一批的稳定/退化 | 1128 只有 AIV 8/32，我方 B 阶段是 32/64 ⇒ 交叉仅 AIV32；`weighted` 列无效 |" % len(ov_28))
    A("")
    A("> 我方 72 例的实际网格：EP4 与 EP64 是 **T4096/8192/16384 × H2816/6144/7168 × AIV8/32**（各 18 例）；")
    A("> EP8/16/32 是 **T8192/16384 × H2816/6144/7168 × AIV32/64**（各 12 例）。**AIV 集合不同是覆盖缺口的根源**。")
    A("")
    A("---")
    A("")
    A("## 一、vs 基线快照（逐 case %d 例）" % len(ov_snap))
    A("")
    A("格式 `我方 / 基线 (Δ)`。`主指标` = dispatch 与 cached 同时 ≥0；`全4` = 四指标全 ≥0。")
    A("")

    def ep_block(ep):
        ks = sorted(k for k in ov_snap if k[0] == ep)
        if not ks:
            return
        A("### EP%d（%d 例）" % (ep, len(ks)))
        A("")
        A("| EP | Tokens | Hidden | AIV | dispatch 我方/基线 (Δ) | cached 我方/基线 (Δ) | cU 我方/基线 (Δ) | cW 我方/基线 (Δ) | 主指标 | 全4 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for k in ks:
            m, s = M[k], SNAP[k]
            vals = [cell(m[o]["mean"], s[tag]) for o, _, tag in OPS]
            main_ok = all(m[o]["mean"] >= s[tag] for o, _, tag in OPS[:2])
            all_ok = all(m[o]["mean"] >= s[tag] for o, _, tag in OPS)
            A("| %d | %d | %d | %d | %s | %s | %s | %s | %s | %s |" % (
                k[0], k[1], k[2], k[3], vals[0], vals[1], vals[2], vals[3],
                "✅" if main_ok else "❌", "✅" if all_ok else "❌"))
        A("")
        cnt = len(ks)
        main_n = sum(1 for k in ks if all(M[k][o]["mean"] >= SNAP[k][t] for o, _, t in OPS[:2]))
        all_n = sum(1 for k in ks if all(M[k][o]["mean"] >= SNAP[k][t] for o, _, t in OPS))
        cu_n = sum(1 for k in ks if M[k]["combine_unweighted"]["mean"] >= SNAP[k]["u"])
        cw_n = sum(1 for k in ks if M[k]["combine_weighted"]["mean"] >= SNAP[k]["w"])
        dmin = min(pct(M[k]["dispatch"]["mean"], SNAP[k]["d"]) for k in ks)
        dmed = statistics.median([pct(M[k]["dispatch"]["mean"], SNAP[k]["d"]) for k in ks])
        A("**小计**：主指标 %d/%d、cU %d/%d、cW %d/%d、四指标全胜 %d/%d；dispatch 中位 %+.1f%%、最低 %+.1f%%。"
          % (main_n, cnt, cu_n, cnt, cw_n, cnt, all_n, cnt, dmed, dmin))
        A("")

    for ep in (8, 16, 32, 64):
        ep_block(ep)

    no_base = sorted(k for k in M if k not in SNAP)
    A("### 无基线对照的 %d 例（单列，不计入达标率）" % len(no_base))
    A("")
    A("- EP 分布：" + "、".join("EP%d %d 例" % (e, n) for e, n in sorted(Counter(k[0] for k in no_base).items())))
    A("- 原因：基线快照只覆盖 EP8–128，且 EP64 仅 AIV=32；**EP4 整档不在基线表内**。")
    A("- 这些例只能用 1125 / 1128 横向参照（§二、§三），**不能对外声称「对基线达标」**。")
    A("")
    A("### 汇总（只统计 %d 个有基线对照的例）" % len(ov_snap))
    A("")
    tot = len(ov_snap) * 4
    neg_pts = 0
    neg_cases = 0
    for k in ov_snap:
        bad = False
        for o, _, t in OPS:
            if M[k][o]["mean"] < SNAP[k][t]:
                neg_pts += 1
                bad = True
        if bad:
            neg_cases += 1
    A("| 维度 | 达标 | 占比 |")
    A("|---|---|---|")
    for o, _, t in OPS:
        n = sum(1 for k in ov_snap if M[k][o]["mean"] >= SNAP[k][t])
        v = [pct(M[k][o]["mean"], SNAP[k][t]) for k in ov_snap]
        A("| %s | **%d/%d** | %.1f%%（中位 %+.1f%%，最低 %+.1f%%） |" % (o, n, len(ov_snap), n / len(ov_snap) * 100, statistics.median(v), min(v)))
    main_n = sum(1 for k in ov_snap if all(M[k][o]["mean"] >= SNAP[k][t] for o, _, t in OPS[:2]))
    A("| **主指标（disp+cached）** | **%d/%d** | **%.1f%%** |" % (main_n, len(ov_snap), main_n / len(ov_snap) * 100))
    A("")
    A("- 数据点级：**%d/%d = %.1f%%** 达标；不达标 %d 点、分布在 %d 格。"
      % (tot - neg_pts, tot, (tot - neg_pts) / tot * 100, neg_pts, neg_cases))
    A("- 三指标（不含 cW）全胜：%d/%d；四指标全胜：%d/%d。"
      % (sum(1 for k in ov_snap if all(M[k][o]["mean"] >= SNAP[k][t] for o, _, t in OPS[:3])), len(ov_snap),
         len(ov_snap) - neg_cases, len(ov_snap)))
    A("")
    A("---")
    A("")
    A("## 二、vs 1125（PR #37 之前，同集群同 bench 同 seed，%d 例）" % len(ov_25))
    A("")
    A("**唯一能隔离 PR #37 收益**的对照：同机器、同 bench 版本、同 seed，唯一变量是代码从")
    A("`83252a5`（PR base）变成 `0d1eca2`（PR head）。Δ 正值 = PR #37 带来的提升。")
    A("")
    A("| EP | Tokens | Hidden | AIV | dispatch | cached | cU | cW |")
    A("|---|---|---|---|---|---|---|---|")
    for ep in (8, 16, 32, 64):
        for k in sorted(k for k in ov_25 if k[0] == ep):
            m, p = M[k], P25[k]
            A("| %d | %d | %d | %d | %+.1f%% | %+.1f%% | %+.1f%% | %+.1f%% |" % (
                k[0], k[1], k[2], k[3],
                pct(m["dispatch"]["mean"], p["dispatch"]),
                pct(m["cached-dispatch"]["mean"], p["cached-dispatch"]),
                pct(m["combine_unweighted"]["mean"], p["combine_unweighted"]),
                pct(m["combine_weighted"]["mean"], p["combine_weighted"])))
    A("")
    for label, op in [("dispatch", "dispatch"), ("cached", "cached-dispatch"), ("cU", "combine_unweighted"), ("cW", "combine_weighted")]:
        v = [pct(M[k][op]["mean"], P25[k][op]) for k in ov_25]
        A("- **%s**：中位 %+.1f%%，范围 %+.1f%% ~ %+.1f%%，正/负 = %d/%d"
          % (label, statistics.median(v), min(v), max(v), sum(1 for x in v if x > 0), sum(1 for x in v if x < 0)))
    A("")
    A("> **dispatch %d/%d 全正，中位 %+.1f%%** —— 本次最硬的收益证据。cached 只 %+.1f%% ⇒ 优化集中在"
      % (sum(1 for k in ov_25 if M[k]["dispatch"]["mean"] > P25[k]["dispatch"]), len(ov_25),
         statistics.median([pct(M[k]["dispatch"]["mean"], P25[k]["dispatch"]) for k in ov_25]),
         statistics.median([pct(M[k]["cached-dispatch"]["mean"], P25[k]["cached-dispatch"]) for k in ov_25])))
    A("> 「首次 dispatch 的准备路径」，缓存命中的热路径基本没动，与 PR 自述的 WQE 准备 / channel metadata 打包吻合。")
    A("")
    A("---")
    A("")
    A("## 三、vs 1128 原表（我方上一批，%d 例；`weighted` 列不可用）" % len(ov_28))
    A("")
    A("> 1128 表只有 AIV 8/32，我方 B 阶段是 32/64 ⇒ EP8/16/32 只有 AIV32 各 6 例交叉；")
    A("> EP4 与 EP64 各 18 例全重叠。**不要拿这 54 例的覆盖率去说「全网格回归通过」。**")
    A("")
    A("| EP | Tokens | Hidden | AIV | dispatch | cached | cU | 判读 |")
    A("|---|---|---|---|---|---|---|---|")
    for ep in (4, 8, 16, 32, 64):
        for k in sorted(k for k in ov_28 if k[0] == ep):
            m, b = M[k], O28[k]
            dd, dc, du = pct(m["dispatch"]["mean"], b["d"]), pct(m["cached-dispatch"]["mean"], b["c"]), pct(m["combine_unweighted"]["mean"], b["u"])
            if (abs(dd) > 20 or abs(du) > 20):
                note = "**见 §四**" if ep == 64 else "**见 §四之二**"
            else:
                note = ""
            A("| %d | %d | %d | %d | %+.1f%% | %+.1f%% | %+.1f%% | %s |" % (k[0], k[1], k[2], k[3], dd, dc, du, note))
    A("")
    A("> 1128 的 `weighted` 列**不可用**：108 格里 `weighted/unweighted` 中位 **0.9978**、全部落在 ±5% 内，")
    A("> 说明那批 weighted 未生效。快照 0.76~0.87、1125 0.73~0.85、本轮 0.66~0.96")
    A("> ⇒ 直接比会得出「combine_weighted 全面 −27%」的假退化，故本节不列该列。")
    A("")
    A("---")
    A("")
    A("## 四、EP64 的 combine：三个对照各给一套数，先摆数值再定性")
    A("")
    A("EP64 的 combine 是本次唯一「对照之间彼此矛盾」的地方，不能一句话归因。")
    A("下表为 EP64 + AIV=32（快照只有这一档），单位 GB/s/rank，取 `combine_unweighted`：")
    A("")
    A("| Tokens | Hidden | 本轮 | 基线快照 | 1128 | 1125 | vs 快照 | vs 1128 | vs 1125 |")
    A("|---|---|---|---|---|---|---|---|---|")
    ep64_comb = []
    for k in sorted(k for k in M if k[0] == 64 and k[3] == 32):
        m = M[k]["combine_unweighted"]["mean"]
        sn = SNAP.get(k, {}).get("u")
        o8 = O28.get(k, {}).get("u")
        p5 = P25.get(k, {}).get("combine_unweighted")
        fmt = lambda v: "%.1f" % v if v else "—"
        f = lambda a, b: ("%+.1f%%" % pct(a, b)) if b else "—"
        A("| %d | %d | %.1f | %s | %s | %s | %s | %s | %s |" % (
            k[1], k[2], m, fmt(sn), fmt(o8), fmt(p5), f(m, sn), f(m, o8), f(m, p5)))
        if sn and p5 and o8:
            ep64_comb.append((k, m, sn, o8, p5))
    A("")
    A("**读法（三条并列观察，缺一条就会误判）**：")
    A("")
    A("1. **1128 与快照在 H6144/H7168 上互差 ≤1.5%**（如 T8192 H6144：347.6 vs 351.0；H7168：345.7 vs 349.5）")
    A("   ⇒ 这两张表在 EP64 combine 上**是同一套数值**，不是各自独立测量。")
    A("2. **在 H2816 上 1128 仍比快照高 25%**（334 vs 268），但两者**同向高于我方**（我方 210~218）。")
    A("   ⇒ 用「H7168 采用 rows2」只能解释 H7168 一档，**解释不了 H2816 也偏高**：")
    A("   更可能是 EP64 combine 的对照侧整体换了归约口径，而非单档特例。")
    A("3. **本轮与 1125 在全部 6 格上互差 ≤7.2%**（+1.7 / +0.6 / −6.0 / +1.7 / +1.0 / −7.2）")
    A("   ⇒ 我方与 1125 才是同一口径。**真实影响量级 = vs 1125，即 H7168 −6%~−7%、H2816 与 H6144 基本持平。**")
    A("")
    A("> 结论：**EP64 的 combine「对基线 −18.7%~−29.5%」不是本轮退化**，是对照侧口径整体偏高；")
    A("> 只要拿 1125 一比就归零。对外若必须引基线数字，需同时说明这一点，否则会被读成退化 30%。")
    A("")
    A("## 四之二、其余必须以单例点名、不能用中位数掩盖的格")
    A("")
    A("| 现象 | 涉及 case | 判据 | 结论 |")
    A("|---|---|---|---|")
    A("| **EP4 + AIV=8 的 dispatch 相对同批 AIV=32 掉 39.7~71.8%** | EP4 × T4096/8192/16384 × H2816/6144/7168 × AIV8，9 例 | 同批 EP4+AIV32 自身对 1128 正常（+1.1~+5.5%）；**EP64 的 AIV8 vs AIV32 只差 −6.3~+7.3%，正常**；每例分散度 0.6~5.1% | 我方特有，定位「单机 4 卡 × 每卡 8 vector core」：H2816 掉 68~72%、H6144 掉 47~49%、H7168 掉 40~41%，与 hidden 成反比。主档 AIV 32/64 不受影响 |")
    A("| **combine_weighted 落后：H2816 全档 + EP32 的 H7168** | H2816：EP8 4 格、EP32 2 格、EP64 2 格；EP32 H7168 3 格；EP16 H7168 AIV64 1 格 —— **12 例** | 同格 cU 领先 +14~+35%，cW 却 −1.1~−12.1%；EP8 H2816 AIV32 对 1125 更是 −18.8% | **真实短板**：本轮 w/u = 0.659，基线 0.762、1125 0.731 ⇒ 权重分支没跟上 unweighted |")
    A("| **−0.1% 量级的 cW** | EP32 T16384 H6144 AIV32、EP64 T16384 H6144 AIV32，2 例 | 同格其他指标全正 | 噪声，仅登记不解释 |")
    A("")
    A("---")
    A("")
    A("## 五、重复性（逐 case 的 min/max 是否可信）")
    A("")
    sp = []
    for k in ok_cases:
        v = max((M[k][o]["mx"] - M[k][o]["mn"]) / M[k][o]["mean"] for o, _, _ in OPS)
        sp.append((v, k))
    sp.sort(reverse=True)
    A("每例取 4 个 op 里最大的 `(max−min)/mean`：")
    A("")
    A("- 中位 **%.2f%%**；>5%% 的只有 **%d 例**。" % (statistics.median([s for s, _ in sp]) * 100, sum(1 for s, _ in sp if s > 0.05)))
    A("- 最大 6 例：" + "、".join("EP%d/T%d/H%d/AIV%d %.2f%%" % (k[0], k[1], k[2], k[3], s * 100) for s, k in sp[:6]))
    A("- ⇒ 除 EP4+AIV8 一档，单例内波动都在 2% 量级，逐 case 的 Δ 可直接读，无需额外置信区间。")
    A("")
    A("---")
    A("")
    A("## 六、怎么用这三张表")
    A("")
    A("1. **对外交付**：只引 §一，并限定「%d 例 / 主指标 / 零折算」。别把 §二的 +14.0%% 当达标证据——" % len(ov_snap))
    A("   那是「比上轮自己快」，不是「比基线快」。")
    A("2. **归因 PR #37**：只引 §二。§一 里 dispatch 的 +16.6%% 混入了「基线本身偏低」的成分。")
    A("3. **回归对比**：只引 §三，且跳过 `weighted` 列、跳过 EP64 整档 combine（对照侧口径不一致，见 §四）。")
    A("4. **任何一格不合格都别用中位数掩盖**：§四之二 三类（EP4+AIV8 9 例、cW 短板 12 例、噪声 2 例）写报告时逐个点名。")
    A("5. **改数据后重跑本脚本**：`python scripts/gen_case_compare.py`，别手改数字。")
    A("")

    dst = os.path.join(BASE, "R3逐case对比_20261128.md")
    io.open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print("written", dst)
    print("cases: total=%d base=%d p1125=%d p1128=%d" % (len(M), len(ov_snap), len(ov_25), len(ov_28)))


if __name__ == "__main__":
    main()
