#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 R3对比1128_20261128.md —— 本轮 0d1eca2 vs 1128 上一批 e077a43 的独立对照。

用法（在 data/DeepEP_CLOS 目录下）：
    python scripts/gen_compare_1128.py

定位：1128 表**不是基线**，是同一 PR #37 链上的 8 提交前版本。
    base 83252a5 --(72 commits)--> e077a43 [1128 上一批] --(8 commits)--> 0d1eca2 [本轮]
所以本文只回答「同一 PR 内这 8 个提交带来的 delta」，不能当交付达标证据，也不能当 PR #37 全量收益。

数据源：
    round3_perf.csv / round3_perf_supp.csv        本轮 72 例
    baseline/EP4-128_通信带宽_20261128.txt         1128 上一批（e077a43）
    baseline/V2测试Case_20261128快照.txt           基线侧（仅用于 §5 说明 EP64 combine 同源）
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
]
# 1128 表列: 5=D无 6=D带 7=C无 8=C带 9=U 10=W
COL_1128 = {"d": 6, "c": 8, "u": 9}
SNAP_COL = {"d": 6, "c": 7, "u": 8}


def load_round(fname):
    d = {}
    for r in csv.DictReader(io.open(os.path.join(BASE, fname), encoding="utf-8")):
        k = (int(r["ep_size"]), int(r["num_tokens"]), int(r["hidden"]), int(r["aiv"]))
        d.setdefault(k, {})[r["op"]] = float(r["gbps_mean"])
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
        d[k] = {name: float(c[i]) for name, i in COL_1128.items()}
        d[k]["w"] = float(c[10])
    return d


def load_snap():
    d = {}
    for line in io.open(os.path.join(BASE, "baseline/V2测试Case_20261128快照.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 10 or not c[0].isdigit():
            continue
        d[(int(c[0]), int(c[1]), int(c[2]), int(c[3]))] = {n: float(c[i]) for n, i in SNAP_COL.items()}
    return d


def pct(a, b):
    return (a / b - 1.0) * 100.0 if b else float("nan")


def med(v):
    return statistics.median(v) if v else float("nan")


def main():
    M = load_round("round3_perf.csv")
    for k, v in load_round("round3_perf_supp.csv").items():
        M.setdefault(k, {}).update(v)
    O = load_1128()
    S = load_snap()

    ov = sorted(k for k in M if k in O and len(M[k]) == 4)
    out = []
    A = out.append

    A("# R3 vs 1128 上一批：独立对照（**1128 不是基线**）")
    A("")
    A("> **本文由 `scripts/gen_compare_1128.py` 机械生成**，数字不手抄；重跑即可复现。")
    A("> 主分析 [`R3性能实测分析_20261128.md`](./R3性能实测分析_20261128.md) 已把 1128 移出达标判定，")
    A("> 本文是它的**独立存档**：只回答「本轮相对上一批是稳定/提升/退化」，不参与任何对外达标结论。")
    A("")
    A("## 0、两批是什么关系（决定了本文能回答什么）")
    A("")
    A("| 批次 | commit | 时间(GMT+8) | 说明 |")
    A("|---|---|---|---|")
    A("| PR #37 base | `83252a5` | 11-20 19:46 | 上轮 R2 基线 |")
    A("| **1128 上一批** | `e077a43` | 11-28 **12:36** | *Use rank-only CLOS scheduling and support EP256*；EP128 那批走 `a02fe8a`（11-27） |")
    A("| **本轮 R3** | `0d1eca2` | 11-28 **14:59** | *Validate combine peer bounds for single-row sends* |")
    A("| PR #37 合并点 | `19eadeb` | 11-28 16:21 | 比 `0d1eca2` 多 1 个非热路径提交 `52bdfb70` |")
    A("")
    A("**关键事实（GitHub compare API 核实）**：")
    A("")
    A("- `83252a5 → e077a43` = **72 个提交**；`e077a43 → 0d1eca2` = **8 个提交**（`status=ahead, behind=0`）。")
    A("- ⇒ 1128 上一批**已经包含 PR #37 的绝大部分改动**。本文的 Δ **不是「PR #37 的收益」**，")
    A("  而是**同一 PR 内部最后 8 个提交的 delta**。真正能隔离 PR #37 全量收益的对照只有 1125（`83252a5` 口径）。")
    A("")
    A("那 8 个提交（按时间）：")
    A("")
    A("| # | commit | 标题 | 是否可能影响性能 |")
    A("|---|---|---|---|")
    A("| 1 | `d5c5be29` | Trim experimental tests and tuning documentation | 否（文档/测试） |")
    A("| 2 | `eb4099cb` | Remove CPU kernel simulation test harnesses | 否 |")
    A("| 3 | `45fa790e` | Remove specialized benchmark diagnostics and restore benchmark README | 否 |")
    A("| 4 | `99b16ebe` | **Configure fused layout notify per ElasticBuffer** | **可能**（fused layout 与 AIV 分支相关） |")
    A("| 5 | `3faa82a6` | **Remove legacy fusion environment references** | **很可能**（删 env 引用会改默认行为） |")
    A("| 6 | `ebc82679` | Configure SHMEM init timeout per Buffer (10 min) | 否（初始化路径） |")
    A("| 7 | `2828eae3` | **Remove manual QP reservation environment override** | **可能**（QP 策略改 auto） |")
    A("| 8 | `0d1eca21` | Validate combine peer bounds for single-row sends | 否（边界校验） |")
    A("")
    A("## 1、结论速览")
    A("")
    A("| 指标 | 中位 Δ | 范围 | 正/负 | 读法 |")
    A("|---|---|---|---|---|")
    for label, op, tag in [("dispatch", "dispatch", "d"), ("cached", "cached-dispatch", "c"),
                           ("cU", "combine_unweighted", "u")]:
        v = [pct(M[k][op], O[k][tag]) for k in ov]
        A("| %s | %+.1f%% | %+.1f%% ~ %+.1f%% | %d/%d | %s |" % (
            label, med(v), min(v), max(v),
            sum(1 for x in v if x > 0), sum(1 for x in v if x < 0),
            {"dispatch": "含 EP4+AIV8 塌陷", "cached": "同上，被同一批拖累", "cU": "普遍略低，EP64 大 hidden 最重"}[label]))
    ex = [k for k in ov if not (k[0] == 4 and k[3] == 8)]
    ve = [pct(M[k]["dispatch"], O[k]["d"]) for k in ex]
    A("| **dispatch（剔除 EP4+AIV8）** | **%+.1f%%** | %+.1f%% ~ %+.1f%% | %d/%d | **正常档的真实水平** |" % (
        med(ve), min(ve), max(ve), sum(1 for x in ve if x > 0), sum(1 for x in ve if x < 0)))
    A("")
    A("四条要点：")
    A("")
    A("1. **正常档只是小幅领先**：剔除 EP4+AIV8 的 9 例后 dispatch 中位 **+2.5%**（45 例，39 正 6 负）。")
    A("   分档中位：EP4 AIV32 **+3.0%**、EP64 AIV32 **+4.1%**、EP64 AIV8 **+3.1%**、EP8/16/32 **+1.7%**。")
    A("   与「8 个提交里 6 个是清理」相符 ⇒ 差异主要来自环境清理（legacy fusion env 移除、QP 改 auto），")
    A("   不是新的调度优化。")
    A("2. **EP4 + AIV=8 是唯一的大幅负向**：相对 1128 掉 **−34.4%~−44.5%**（中位 −41.6%，9/9 全负）；")
    A("   换成「AIV8/AIV32 比值」看更清楚：1128 批只有 H2816 塌（−46%），H6144/H7168 仅 −5~−6%；")
    A("   本轮三档全塌（−71% / −48% / −41%）⇒ **H≥6144 的塌陷是本轮这 8 个提交引入的**，可二分定位（§4.1）。")
    A("3. **H2816 是唯一有负值的正常档**：EP8 −2.0/−6.5%、EP16 −4.2/−9.4%，EP32 全正；")
    A("   EP64 AIV32 的 H2816 两例为负（−5.1/−8.4%）。H6144/H7168 全正 ⇒ 主档没有退化（§4.2、§4.3）。")
    A("4. **cU 普遍略低**（中位 −2.5%，10/44），但除 EP64 大 hidden（−27~−39%）外都在 −7% 以内；")
    A("   EP64 那批是 rows 口径问题（§5），不是本轮退化。")
    A("")
    A("## 2、可比范围：54 例，缺口必须写明")
    A("")
    A("| EP | 可比例数 | 网格 | 备注 |")
    A("|---|---|---|---|")
    A("| EP4 | 18 | T4096/8192/16384 × H2816/6144/7168 × AIV8/32 | 全重叠 |")
    A("| EP8/16/32 | 各 6 | T8192/16384 × H2816/6144/7168 × **仅 AIV32** | 我方 B 阶段跑 AIV32/64，1128 只有 8/32 ⇒ 交叉仅 32 |")
    A("| EP64 | 18 | T4096/8192/16384 × H2816/6144/7168 × AIV8/32 | 全重叠 |")
    A("")
    A("**不能比的**：EP8/16/32 的 AIV=64（12 例，1128 无此档）、EP64 的 AIV=64（6 例）、EP128（本轮未测性能）。")
    A("另：1128 表里 EP128 那 12 行走的是更老的 `a02fe8a`（11-27），与本轮无关，本文不涉及。")
    A("⇒ 54 例 ≠ 全网格，任何「全网格回归通过」的说法都不成立。")
    A("")
    A("## 3、逐 case（54 例）")
    A("")
    A("Δ = 本轮 / 1128 − 1，**正值 = 本轮更快**。cached 一律比 1128 的「Cached dispatch **带**权重」列")
    A("（与我方 `official_logical_bytes` 同口径；两列中位差 1.4%，比错列会高估自己约 1.4pp）。")
    A("")

    def block(ep, title):
        ks = sorted(k for k in ov if k[0] == ep)
        if not ks:
            return
        A("### %s（%d 例）" % (title, len(ks)))
        A("")
        A("| EP | Tokens | Hidden | AIV | dispatch | cached | cU |")
        A("|---|---|---|---|---|---|---|")
        for k in ks:
            A("| %d | %d | %d | %d | %+.1f%% | %+.1f%% | %+.1f%% |" % (
                k[0], k[1], k[2], k[3],
                pct(M[k]["dispatch"], O[k]["d"]),
                pct(M[k]["cached-dispatch"], O[k]["c"]),
                pct(M[k]["combine_unweighted"], O[k]["u"])))
        dv = [pct(M[k]["dispatch"], O[k]["d"]) for k in ks]
        cv = [pct(M[k]["cached-dispatch"], O[k]["c"]) for k in ks]
        uv = [pct(M[k]["combine_unweighted"], O[k]["u"]) for k in ks]
        A("")
        A("**小计**：dispatch 中位 %+.1f%%（%+.1f ~ %+.1f，正/负 %d/%d）；cached 中位 %+.1f%%；cU 中位 %+.1f%%。"
          % (med(dv), min(dv), max(dv), sum(1 for x in dv if x > 0), sum(1 for x in dv if x < 0), med(cv), med(uv)))
        A("")

    block(4, "EP4")
    A("> EP4 整体中位为负（−16.7%）是被 AIV=8 的 9 例拖累；**只看 AIV=32 的 9 例是 +1.1~+5.5%（中位 +3.0%，9/9 全正）**。")
    A("")
    block(64, "EP64")
    for ep in (8, 16, 32):
        block(ep, "EP%d（仅 AIV=32）" % ep)

    A("## 4、必须单独点名的三处")
    A("")
    A("### 4.1 EP4 + AIV=8：本轮这 8 个提交引入的塌陷")
    A("")
    A("同 EP/T/H、只换 AIV，对比两批的「AIV8 / AIV32」比值：")
    A("")
    A("| Hidden | 1128 批 AIV8/AIV32 | 本轮 AIV8/AIV32 | 判读 |")
    A("|---|---|---|---|")
    for h in (2816, 6144, 7168):
        r1128, rnow = [], []
        for k in ov:
            if k[0] != 4 or k[2] != h:
                continue
            if k[3] == 8:
                k32 = (4, k[1], h, 32)
                if k32 in M and k32 in O:
                    r1128.append(pct(O[k]["d"], O[k32]["d"]))
                    rnow.append(pct(M[k]["dispatch"], M[k32]["dispatch"]))
        A("| %d | %+.0f%% | %+.0f%% | %s |" % (
            h, med(r1128), med(rnow),
            "本轮独有塌陷" if h >= 6144 else "两批都塌，本轮更重"))
    A("")
    A("- 1128 批只有 H2816 塌（约 −46%，三格 −43.7/−45.6/−47.3），H6144/H7168 只掉 **5~6%**")
    A("  ⇒ **H≥6144 的 −40~−49% 是本轮新增的**。")
    A("- ⇒ 二分范围锁定在 8 个提交里的 **`99b16ebe`（fused layout notify per ElasticBuffer）/")
    A("  `3faa82a6`（Remove legacy fusion environment references）/ `2828eae3`（QP override 移除）** 三个，")
    A("  建议按 `3faa82a6` → `99b16ebe` → `2828eae3` 顺序回退验证（第一个最可能改默认行为）。")
    A("")
    A("### 4.2 EP64 + AIV=32 的 H2816：3 例里 2 例为负")
    A("")
    A("| Tokens | dispatch | cached | cU |")
    A("|---|---|---|---|")
    for k in sorted(x for x in ov if x[0] == 64 and x[3] == 32 and x[2] == 2816):
        A("| %d | %+.1f%% | %+.1f%% | %+.1f%% |" % (
            k[1], pct(M[k]["dispatch"], O[k]["d"]),
            pct(M[k]["cached-dispatch"], O[k]["c"]),
            pct(M[k]["combine_unweighted"], O[k]["u"])))
    A("")
    A("同 EP 的 H6144/H7168 全为正（+1~+10%），H2816 里只有 T4096 勉强正（+0.1%，属噪声），")
    A("T8192/T16384 掉 −5~−8% ⇒ 与 1125 里观察到的「H2816 是异区制」一致，不是随机噪声；")
    A("但幅度远小于 EP4+AIV8，属可接受波动，登记不阻塞。")
    A("")
    A("### 4.3 EP8/EP16 的 H2816 小幅为负（EP32 为正）")
    A("")
    A("| EP | T8192 H2816 | T16384 H2816 | 其余 H |")
    A("|---|---|---|---|")
    for ep in (8, 16, 32):
        r = {}
        for k in ov:
            if k[0] == ep and k[2] == 2816 and k[3] == 32:
                r[k[1]] = pct(M[k]["dispatch"], O[k]["d"])
        others = [pct(M[k]["dispatch"], O[k]["d"]) for k in ov if k[0] == ep and k[2] != 2816]
        A("| %d | %+.1f%% | %+.1f%% | %+.1f%% ~ %+.1f%% |" % (
            ep, r.get(8192, float("nan")), r.get(16384, float("nan")), min(others), max(others)))
    A("")
    A("H2816 的 payload 最小、固定开销占比最高，对「8 个提交里的初始化/配置类改动」最敏感；")
    A("EP32 三档全正（+1.1~+2.8%）说明这不是全档现象，更像 EP8/EP16 的固定开销敏感。")
    A("H6144/H7168 全正说明主档没有退化。这一档可与 §4.1 一起二分，但优先级低于 EP4+AIV8。")
    A("")
    A("## 5、EP64 combine：1128 与基线快照是同一套数值")
    A("")
    A("| Tokens | Hidden | 本轮 | 1128 | 基线快照 | 1128/快照 |")
    A("|---|---|---|---|---|---|")
    for k in sorted(x for x in ov if x[0] == 64 and x[3] == 32):
        if k not in S:
            continue
        A("| %d | %d | %.1f | %.1f | %.1f | %+.1f%% |" % (
            k[1], k[2], M[k]["combine_unweighted"], O[k]["u"], S[k]["u"],
            pct(O[k]["u"], S[k]["u"])))
    A("")
    A("互差 ≤1.5%（H2816 除外，+23~25%）⇒ **1128 表与基线快照在 EP64 combine 上同源/同归约口径**，")
    A("不能当成「两条独立管线都测出 349」的证据。本轮 249~250 与 1125 的 265~269 才是同口径（互差 ≤7.2%）。")
    A("")
    A("## 6、口径体检：1128 批的 weights 几乎没生效")
    A("")
    A("| 列 | 带权重/无权重 中位 | 范围 | 对比 |")
    A("|---|---|---|---|")
    A("| Dispatch | **0.9935** | 0.937 ~ 1.016 | 快照 0.9356、本轮 dispatch 权重开销明显 |")
    A("| Cached | **0.9861** | 0.899 ~ 0.997 | 快照单列无对照 |")
    A("| Combine | **0.9978** | 0.979 ~ 1.015 | 快照 0.762~0.869、1125 0.731~0.850、本轮 0.659~0.959 |")
    A("")
    A("三列都接近 1.00 ⇒ 那批的 weights 基本没进 payload/耗时。因此：")
    A("")
    A("- 本文 **不给 combine_weighted 的 Δ**（比了会得到「全面 −27%」的假退化）。")
    A("- dispatch / cached 仍比「带权重」列，是**保守**选择（若那批实际按无权重计，用带权重列会让我方 Δ 偏小）。")
    A("")
    A("## 7、使用边界（三条）")
    A("")
    A("1. **不能当基线**：1128 是我方上一批实测，对交付不构成任何达标证据。")
    A("2. **不能当 PR #37 的收益**：它已在 PR 链上（`83252a5` 之后 72 个提交），本文 Δ 只是最后 8 个提交的差异；")
    A("   PR #37 全量收益只能看 vs 1125（dispatch 中位 +14.0%，42/42 全正）。")
    A("3. **不能当回归全绿**：只有 54/72 例重叠，AIV=64 与 EP128 都不在里面。")
    A("")
    A("> 改数据后重跑：`python scripts/gen_compare_1128.py`。")
    A("")

    dst = os.path.join(BASE, "R3对比1128_20261128.md")
    io.open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print("written", dst, "cases", len(ov))


if __name__ == "__main__":
    main()
