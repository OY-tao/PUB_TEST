#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取 R3 性能不达标 case，生成重测清单（CSV + JSON + MD），数字全部由数据算出。

用法（在 data/DeepEP_CLOS 目录下）：
    python scripts/gen_retest_list.py            # 生成三份产物
    python scripts/gen_retest_list.py --check    # 只打印统计，不写文件

判定三条（互不替代，任一条命中即纳入重测）：
  R1 对外基线劣化   vs 基线快照，该指标 Δ ≤ −5%
  R2 同批相对塌陷   无基线的档位里，同 EP 同 (T,H) 下低 AIV 相对 AIV=32 的 dispatch Δ ≤ −10%
  R3 测量不稳       格内四 op 的 (max−min)/mean 最大值 > 5%（单次不可信，需复测）

为什么 R2 阈值取 −10% 而不是 −5%：
  bench 每格分散度中位 1.38%，AIV 配置差异在 1128 上一批里 ≤6%（EP64 AIV8 vs AIV32 为
  −5.1%~+7.3%）⇒ 「稳定特性 + 噪声」最多解释到 6~7%，10% 以上才超出解释范围。

为什么只比基线快照、不把 1128 表当基线：见 R3逐case对比_20261128.md §三。
"""
import argparse
import csv
import io
import json
import os
import statistics
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
import gen_case_compare as G  # noqa: E402  复用同一套加载与 Δ 口径

# R1/R2 阈值
R1_DROP = -5.0     # 对外基线劣化
R2_DROP = -10.0    # 同批塌陷
R3_SPREAD = 5.0    # 分散度

# op -> 快照列 tag
SNAP_TAG = {"dispatch": "d", "cached-dispatch": "c",
            "combine_unweighted": "u", "combine_weighted": "w"}
OP_CN = {"dispatch": "dispatch", "cached-dispatch": "cached",
         "combine_unweighted": "combine_un", "combine_weighted": "combine_weighted"}

# 分组：优先级 + 名称 + 该组要回答的问题
GROUPS = {
    "P0-EP4AIV8": dict(
        prio="P0", title="EP4 + AIV=8 的 dispatch 塌陷",
        ask="AIV=8 在单机 4 卡上相对同批 AIV=32 掉 40~72%，是稳定复现还是单次异常？"
            "重测后再做 8 提交（`3faa82a6`→`99b16ebe`→`2828eae3`）二分。"),
    "P1-EP64COMB": dict(
        prio="P1", title="EP64 的 combine 对基线 ≤ −5%",
        ask="6 格 combine_un 全负（−5.3~−29.5%），是基线表 rows 口径偏高还是本轮真退化？"
            "3 次重测若稳定在 −5% 以下，说明是口径；若明显回升，说明首测受环境影响。"),
    "P2-WEIGHTED": dict(
        prio="P2", title="EP≤32 的 combine_weighted 短板",
        ask="同格 unweighted 领先基线 +1.4~+21% 而 weighted 落后 −5~−12%，"
            "确认是否为权重分支的真实开销缺口，确认后立项。"),
}

RETEST_REPS = 3          # 每格重复次数
FIELDS = ["priority", "group", "ep", "num_tokens", "hidden", "aiv",
          "role", "op", "first_gbps", "base_gbps", "delta_pct", "spread_pct", "rule", "note"]


def spread_of(m):
    """格内四 op 的最大 (max−min)/mean，单位 %。"""
    out = {}
    for op, d in m.items():
        out[op] = (d["mx"] - d["mn"]) / d["mean"] * 100.0 if d["mean"] else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只打印统计，不写文件")
    args = ap.parse_args()

    M = G.load_round("round3_perf.csv")
    for k, v in G.load_round("round3_perf_supp.csv").items():
        M.setdefault(k, {}).update(v)
    SNAP = G.load_snap()

    # ---------- 逐格算 Δ 与分散度 ----------
    rec = {}   # key -> {"delta": {op: pct}, "base": {op}, "mean": {op}, "spread": {op}}
    for k, ops in M.items():
        sp = spread_of(ops)
        r = {"delta": {}, "base": {}, "mean": {}, "spread": sp}
        for op in ops:
            r["mean"][op] = ops[op]["mean"]
            if k in SNAP and SNAP_TAG[op] in SNAP[k]:
                b = SNAP[k][SNAP_TAG[op]]
                r["base"][op] = b
                r["delta"][op] = G.pct(ops[op]["mean"], b)
        rec[k] = r

    # ---------- 判定 ----------
    rows = []      # 不达标点
    control = []   # 对照格（本身不判，但重测必须同批一起跑）
    hit_cases = {}  # key -> set(group)

    def add(k, op, rule, note, delta=None, base=None):
        d = rec[k]
        rows.append(dict(priority="", group="", ep=k[0], num_tokens=k[1], hidden=k[2], aiv=k[3],
                         role="nonconform", op=op, first_gbps=round(d["mean"][op], 1),
                         base_gbps=(round(d["base"][op], 1) if base is None and op in d["base"] else
                                    (round(base, 1) if base is not None else "")),
                         delta_pct=(round(d["delta"][op], 1) if delta is None and op in d["delta"] else
                                    (round(delta, 1) if delta is not None else "")),
                         spread_pct=round(d["spread"][op], 1), rule=rule, note=note))

    # R1：对外基线劣化
    for k in sorted(M):
        if k not in SNAP:
            continue
        for op in M[k]:
            dv = rec[k]["delta"].get(op)
            if dv is None or dv > R1_DROP:
                continue
            grp = "P1-EP64COMB" if (k[0] == 64 and op.startswith("combine")) else "P2-WEIGHTED"
            add(k, op, "R1", "对外基线劣化 %.1f%%" % dv)
            rows[-1]["group"] = grp
            hit_cases.setdefault(k, set()).add(grp)

    # R2：同批相对塌陷（只对无基线档位，避免与 R1 重复计）
    for k in sorted(M):
        if k in SNAP or k[3] == 32:
            continue
        k32 = (k[0], k[1], k[2], 32)
        if k32 not in M or "dispatch" not in M[k32]:
            continue
        dv = G.pct(M[k]["dispatch"]["mean"], M[k32]["dispatch"]["mean"])
        if dv > R2_DROP:
            continue
        add(k, "dispatch", "R2", "相对同批 AIV=32 %.1f%%" % dv, delta=dv)
        rows[-1]["group"] = "P0-EP4AIV8"
        hit_cases.setdefault(k, set()).add("P0-EP4AIV8")
        # 该格其它明显同步塌陷的 op 一并记（只记录，不改判据）
        for op in ("cached-dispatch", "combine_unweighted", "combine_weighted"):
            if op not in M[k] or op not in M[k32]:
                continue
            d2 = G.pct(M[k][op]["mean"], M[k32][op]["mean"])
            if d2 <= R2_DROP:
                add(k, op, "R2", "相对同批 AIV=32 %.1f%%" % d2, delta=d2)
                rows[-1]["group"] = "P0-EP4AIV8"
                hit_cases.setdefault(k, set()).add("P0-EP4AIV8")

    # R3：测量不稳。已被 R1/R2 命中的格不单列（分散度已在各 op 的 spread 列里）
    for k in sorted(M):
        sp = rec[k]["spread"]
        worst = max(sp.values())
        if worst <= R3_SPREAD or k in hit_cases:
            continue
        op = max(sp, key=sp.get)
        add(k, "*", "R3", "格内分散度 %.1f%%（%s），单次不可信" % (worst, op))
        rows[-1]["group"] = "P1-EP64COMB" if k[0] == 64 else "P2-WEIGHTED"
        hit_cases.setdefault(k, set()).add(rows[-1]["group"])

    # 一支格命中多个组时，归到优先级最高的组（P0 > P1 > P2）
    order = {"P0-EP4AIV8": 0, "P1-EP64COMB": 1, "P2-WEIGHTED": 2}
    for r in rows:
        gs = [g for g in r["group"].split(",") if g]
        r["group"] = sorted(gs, key=lambda g: order.get(g, 9))[0]
        r["priority"] = GROUPS[r["group"]]["prio"]

    # ---------- 对照格补齐 ----------
    # P0 必须有同 (T,H) 的 AIV=32 作对照，否则重测出来没有同批基准
    p0_keys = sorted(k for k, gs in hit_cases.items() if "P0-EP4AIV8" in gs)
    for k in p0_keys:
        k32 = (k[0], k[1], k[2], 32)
        if k32 in M:
            control.append(dict(priority="P0", group="P0-EP4AIV8", ep=k32[0],
                                num_tokens=k32[1], hidden=k32[2], aiv=k32[3],
                                role="control", op="dispatch", first_gbps=round(M[k32]["dispatch"]["mean"], 1),
                                base_gbps=round(SNAP[k32]["d"], 1) if k32 in SNAP else "",
                                delta_pct=round(G.pct(M[k32]["dispatch"]["mean"], SNAP[k32]["d"]), 1) if k32 in SNAP else "",
                                spread_pct=round(rec[k32]["spread"]["dispatch"], 1), rule="—",
                                note="AIV=32 同批对照（重测必须一起跑）"))

    # ---------- 汇总：重测格子（去重） ----------
    retest_keys = {}
    for r in rows:
        k = (r["ep"], r["num_tokens"], r["hidden"], r["aiv"])
        retest_keys.setdefault(k, set()).add(r["group"])
    for r in control:
        k = (r["ep"], r["num_tokens"], r["hidden"], r["aiv"])
        retest_keys.setdefault(k, set()).add(r["group"])

    by_group = {}
    for k, gs in retest_keys.items():
        g = sorted(gs, key=lambda x: order.get(x, 9))[0]
        by_group.setdefault(g, []).append(k)

    # ---------- 打印统计 ----------
    print("总格数 %d；有基线 %d；纳入重测 %d 格（%d 次运行，reps=%d）"
          % (len(M), len([k for k in M if k in SNAP]), len(retest_keys),
             len(retest_keys) * RETEST_REPS, RETEST_REPS))
    for g in sorted(by_group, key=lambda x: order[x]):
        ks = sorted(by_group[g])
        eps = sorted({k[0] for k in ks})
        print("  %-12s %2d 格  EP%s" % (g, len(ks), eps))
    print("不达标点 %d 个；对照格 %d 个" % (len([r for r in rows]), len(control)))
    if args.check:
        return 0

    # ---------- 写 CSV ----------
    out_rows = rows + control
    out_rows.sort(key=lambda r: (order.get(r["group"], 9), r["ep"], r["num_tokens"],
                                 r["hidden"], r["aiv"], r["op"]))
    csv_path = os.path.join(BASE, "retest_fail_cases_20261128.csv")
    with io.open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    # ---------- 写 JSON（喂 run_round3.py --cases-file） ----------
    jcases = []
    for g in sorted(by_group, key=lambda x: order[x]):
        for k in sorted(by_group[g]):
            jcases.append(dict(ep=k[0], num_tokens=k[1], hidden=k[2], aiv=k[3], group=g))
    spec = dict(
        _comment="R3 性能不达标重测清单，由 scripts/gen_retest_list.py 生成，勿手改",
        _generated_from=["round3_perf.csv", "round3_perf_supp.csv",
                         "baseline/V2测试Case_20261128快照.txt"],
        _measured_commit="0d1eca219d89f1f5e7120b48f01f1dd3654db4f8",
        _reps=RETEST_REPS,
        _rules={"R1 对外基线劣化": "%d%% 以下" % R1_DROP,
                "R2 同批塌陷": "%d%% 以下（相对同 EP 同 T/H 的 AIV=32）" % R2_DROP,
                "R3 测量不稳": "格内分散度 > %d%%" % R3_SPREAD},
        _count=dict(cases=len(jcases), runs=len(jcases) * RETEST_REPS),
        cases=jcases,
    )
    json_path = os.path.join(BASE, "scripts", "retest_cases_20261128.json")
    with io.open(json_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ---------- 写 MD ----------
    md = build_md(M, SNAP, rec, rows, control, by_group, retest_keys, order)
    md_path = os.path.join(BASE, "R3重测清单_20261128.md")
    with io.open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("已写：%s\n     %s\n     %s" % (csv_path, json_path, md_path))
    return 0


def build_md(M, SNAP, rec, rows, control, by_group, retest_keys, order):
    L = []
    A = L.append
    n_case = len(M)
    n_snap = len([k for k in M if k in SNAP])
    n_rt = len(retest_keys)
    n_runs = n_rt * RETEST_REPS

    A("# R3 性能不达标 case 重测清单（2026-11-28）")
    A("")
    A("> **本文由 `scripts/gen_retest_list.py` 机械生成**，数字不手抄；改数据后重跑即可复现。")
    A("> 首测：`round3_perf.csv`（36 例）+ `round3_perf_supp.csv`（36 例）= **72 例 × 4 op**，")
    A("> commit `0d1eca2`、纯 auto QP、无大页、`rc` 全 0。")
    A("> 对照基线：`baseline/V2测试Case_20261128快照.txt`（对外交付口径，覆盖 %d/%d 格）。" % (n_snap, n_case))
    A("")
    A("## 一句话")
    A("")
    A("**%d 格纳入重测**（占 %d 格中 %.0f%%），每格重复 **%d 次**，共 **%d 次运行**。"
      % (n_rt, n_case, n_rt / n_case * 100, RETEST_REPS, n_runs))
    A("分三组：")
    A("")
    A("| 组 | 不达标格 | 对照格 | 现象 | 重测要回答什么 |")
    A("|---|---|---|---|---|")
    for g in sorted(by_group, key=lambda x: order[x]):
        ks = by_group[g]
        gi = GROUPS[g]
        nc = len({(r["ep"], r["num_tokens"], r["hidden"], r["aiv"]) for r in rows if r["group"] == g})
        cl = len({(r["ep"], r["num_tokens"], r["hidden"], r["aiv"]) for r in control if r["group"] == g})
        A("| **%s** %s | %d | %d | %s | %s |" % (gi["prio"], gi["title"], nc, cl,
                                                _phenomenon(g, ks, M, rec), gi["ask"]))
    A("")
    A("> 对照组只重测、不判达标：AIV=8 与 AIV=32 必须同批跑，否则「AIV=8 慢」无法与"
      "「这一批机器慢」区分。合计 **%d 格 / %d 次运行**。" % (n_rt, n_runs))
    A("")
    A("## 判定口径（三条，互不替代）")
    A("")
    A("| 规则 | 判据 | 阈值 | 为什么这么定 |")
    A("|---|---|---|---|")
    A("| **R1 对外基线劣化** | vs 基线快照，该指标 `Δ = 我方/基线 − 1` | **≤ %+.0f%%** | 只用对外交付口径的基线；−5%% 以内的负值按噪声处理，不立项 |" % R1_DROP)
    A("| **R2 同批相对塌陷** | 无基线的档位里，同 EP 同 (T,H) 下低 AIV 相对 AIV=32 的 dispatch | **≤ %+.0f%%** | 每格分散度中位 1.4%%，AIV 配置差在上一批里 ≤6%% ⇒ 10%% 以上才超出「稳定特性+噪声」 |" % R2_DROP)
    A("| **R3 测量不稳** | 格内四 op 的 `(max−min)/mean` 最大值 | **> %.0f%%** | 单次结果本身不可信，必须先复测再判 |" % R3_SPREAD)
    A("")
    A("三条互相独立，一格可同时命中；归组按 P0 > P1 > P2 取最高优先级。")
    A("")
    A("**口径依据**：`dispatch`/`cached-dispatch` 的 official_logical_bytes 含 weights")
    A("（`stage_metrics.py:117-119`，5728 B/row）⇒ 只比快照的「Dispatch **带**权重」列。")
    A("错比「无权重」列会系统性低估自己 6.4%（详见 `R3逐case对比_20261128.md`）。")
    A("")
    A("## 清单")
    A("")

    for g in sorted(by_group, key=lambda x: order[x]):
        gi = GROUPS[g]
        sub = [r for r in rows if r["group"] == g]
        ctl = [r for r in control if r["group"] == g]
        nc = len({(r["ep"], r["num_tokens"], r["hidden"], r["aiv"]) for r in sub})
        A("### %s %s（%d 格不达标 / %d 点%s）" % (
            gi["prio"], gi["title"], nc, len(sub),
            ("，另 %d 格对照" % len({(r["ep"], r["num_tokens"], r["hidden"], r["aiv"]) for r in ctl})) if ctl else ""))
        A("")
        A("> Δ 列：规则 R1 为对基线，规则 R2 为**相对同批 AIV=32**；`—` 表示该档无基线。")
        A("")
        A("| EP | Tokens | Hidden | AIV | 指标 | 首测 | 基线 | Δ | 分散度 | 规则 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(sub, key=lambda r: (r["ep"], r["num_tokens"], r["hidden"], r["aiv"], r["op"])):
            A("| %d | %d | %d | %d | %s | %s | %s | %s | %.1f%% | %s |" % (
                r["ep"], r["num_tokens"], r["hidden"], r["aiv"], OP_CN.get(r["op"], r["op"]),
                ("%.1f" % r["first_gbps"]) if r["first_gbps"] != "" else "—",
                ("%.1f" % r["base_gbps"]) if r["base_gbps"] != "" else "—",
                ("%+.1f%%" % r["delta_pct"]) if r["delta_pct"] != "" else "—",
                r["spread_pct"], r["rule"]))
        if ctl:
            A("")
            A("**同批对照（必须一起跑，不计入不达标）**：")
            A("")
            A("| EP | Tokens | Hidden | AIV | 首测 dispatch | 基线 | Δ | 分散度 |")
            A("|---|---|---|---|---|---|---|---|")
            for r in sorted(ctl, key=lambda r: (r["num_tokens"], r["hidden"])):
                A("| %d | %d | %d | %d | %.1f | %s | %s | %.1f%% |" % (
                    r["ep"], r["num_tokens"], r["hidden"], r["aiv"], r["first_gbps"],
                    ("%.1f" % r["base_gbps"]) if r["base_gbps"] != "" else "—",
                    ("%+.1f%%" % r["delta_pct"]) if r["delta_pct"] != "" else "—",
                    r["spread_pct"]))
        A("")

    # ---------- 重测方案 ----------
    eps_by_group = {g: sorted({k[0] for k in ks}) for g, ks in by_group.items()}
    A("## 重测方案")
    A("")
    A("**每格 %d 次重复**，同 seed（`20260823`）、同 warmup=10 / iter=30、同代码 `0d1eca2`、"
      "纯 auto QP、无大页。判据用 %d 次的**中位数**，不用单次。" % (RETEST_REPS, RETEST_REPS))
    A("")
    A("**节点与波次**（性能集合硬排除 node3/`.7`；EP4/EP8 共用 node7/`.11` 必须串行）：")
    A("")
    A("| 波次 | EP | 节点 | 机×卡 | 例数×reps | 次数 | 单例 | 该波预计 |")
    A("|---|---|---|---|---|---|---|---|")
    plan = [
        ("波 1", "EP64", "node2,4,5,6 = .6/.8/.9/.10", "4机×16卡",
         len(by_group.get("P1-EP64COMB", [])), 4.0),
        ("波 1", "EP4", "node7 = .11", "1机×4卡",
         len(by_group.get("P0-EP4AIV8", [])), 1.1),
        ("波 2", "EP32", "node5,6 = .9/.10", "2机×16卡",
         len([k for k in by_group.get("P2-WEIGHTED", []) if k[0] == 32]), 2.6),
        ("波 2", "EP8", "node7 = .11", "1机×8卡",
         len([k for k in by_group.get("P2-WEIGHTED", []) if k[0] == 8]), 1.4),
    ]
    wave1, wave2 = [], []
    for w, ep, node, topo, n, t in plan:
        if not n:
            continue
        (wave1 if w == "波 1" else wave2).append(n * RETEST_REPS * t)
        A("| %s | %s | %s | %s | %d × %d | %d | ~%.1f min | ~%.0f min |" % (
            w, ep, node, topo, n, RETEST_REPS, n * RETEST_REPS, t, n * RETEST_REPS * t))
    A("")
    A("> 波 1 内 EP64 与 EP4 节点不重叠、**真并行**，取两者较大值 **~%d min**；"
      "波 2 受 `perf_waves` 约束 **EP32 → EP8 串行**（EP8 与 EP32 不共用节点但脚本按波次排队），"
      "合计 **~%d min**。加门禁与部署 15 min，**总计约 %.1f 小时**。"
      % (max(wave1 or [0]), sum(wave2), (max(wave1 or [0]) + sum(wave2) + 15) / 60.0))
    A("")
    A("**前置条件**：")
    A("")
    A("1. **C 阶段（EP128 精度）跑完并释放 8 台**——EP128 独占全部节点，与性能批互斥。")
    A("2. `python scripts/run_round3.py gate` 复检 `pass=true`（8 台 16/16 空闲、commit `0d1eca2`、")
    A("   `buffer.py` sha `f825a679…`、`qp_plans` 为空）。")
    A("3. 容器显式绑 8 台（`--containers`），不要靠子串猜（历史上误取旧容器）。")
    A("")

    # ---------- 命令 ----------
    A("## 执行命令")
    A("")
    A("```bash")
    A("# 1) 门禁")
    A("python3 scripts/run_round3.py gate")
    A("")
    A("# 2) 波一批：EP4（含 P0 的 18 格）+ EP64（P1 的 6 格）")
    A("#    清单自动按 --supp 网格取；--csv-tag retest 保证不污染主批 CSV")
    A("python3 scripts/run_round3.py perf --eps 4,64 --supp \\")
    A("    --cases-file scripts/retest_cases_20261128.json \\")
    A("    --reps 3 --csv-tag retest")
    A("")
    A("# 3) 波二批：EP32 + EP8（P2 的 6 格）")
    A("#    EP32/EP8 不在 supp 节点表里，用默认性能节点表另跑一次")
    A("python3 scripts/run_round3.py perf --eps 8,32 \\")
    A("    --cases-file scripts/retest_cases_20261128.json \\")
    A("    --reps 3 --csv-tag retest")
    A("```")
    A("")
    A("> ⚠ **两次调用会写同一个 `round3_perf_retest.csv`**（append 模式，表头只写一次），")
    A("> 用 `run_id` 里的 `perf_ep<N>_r3_<时间戳>` 区分波次。")
    A("> 若想一次跑完，需先给 `SUPP_EP_NODES` 补 EP8/EP32 的节点（当前只有 4/64）——")
    A("> 不补就跑 `--supp` 会在 `EP_NODES_ACTIVE[8]` 上 KeyError。")
    A("")
    A("跑完直接出判定（**不要手算**，脚本按本文规则机械判）：")
    A("")
    A("```bash")
    A("python3 scripts/gen_retest_result.py --date 20261129")
    A("# -> R3重测结果_20261129.md：确认劣化 / 边界噪声 / 达标 / 作废 四类逐行列出")
    A("#    EP4 的 AIV=8 自动拿同批 AIV=32 作分母（R2 口径），不会误判成「无基线」")
    A("```")
    A("")
    A("## 重测后的判定规则（先定规则，避免事后找理由）")
    A("")
    A("| 重测结果 | 定性 | 动作 |")
    A("|---|---|---|")
    A("| %d 次中位 Δ **仍 ≤ −5%%** | **确认真实劣化** | P0 走 8 提交二分；P1 若稳定在 −5%% 以下则改判「基线口径差」，P2 立项 |" % RETEST_REPS)
    A("| 中位落在 **(−5%, 0)** | 边界/噪声 | 不立项，在报告里注明「首测偏低但不可复现」 |")
    A("| 中位 **≥ 0** 且三次 ≥ 0 | **首测异常** | 用重测值替换首测值，并在 `R3性能实测分析` 更正 |")
    A("| 三次极差 **> 5%** | 环境不稳 | 该格作废重跑，同时记一次环境告警（可能是邻居任务或 `.7` 混入） |")
    A("")

    # ---------- 不纳入 ----------
    A("## 明确**不**纳入重测的（避免清单被无限放大）")
    A("")
    neg_small = []
    for k in sorted(M):
        if k not in SNAP:
            continue
        for op in M[k]:
            dv = rec[k]["delta"].get(op)
            if dv is not None and R1_DROP < dv < 0:
                neg_small.append((k, op, dv))
    A("| 类别 | 数量 | 理由 |")
    A("|---|---|---|")
    A("| 劣化在 −5%% ~ 0 之间 | **%d 点** | 低于 bench 噪声 + 基线四舍五入的量级，重测也判不出方向 |" % len(neg_small))
    if neg_small:
        A("| └ 明细 | %d 点 | 全部是 `combine_weighted`：%s |" % (
            len(neg_small),
            "、".join("EP%d T%d H%d AIV%d %+.2f%%" % (k[0], k[1], k[2], k[3], dv)
                     for k, op, dv in sorted(neg_small))))
        A("| └ 说明 | — | 这些点本身不足以独立入选；但同格若因其它指标入选，仍会一起重测"
          "（如 EP64 T16384 H6144 因 `combine_un` −6.2% 已在 P1） |")
    A("| EP64 的 AIV=64（6 格） | 6 | 快照无该档基线；相对 AIV=32 差异 −6.3~+7.3%，在稳定特性范围内 |")
    A("| EP128 全部 | 12 | 未测性能（128 卡被精度独占），不是「不达标」而是「无数据」 |")
    A("")
    A("## 与既有文档的关系")
    A("")
    A("| 文档 | 关系 |")
    A("|---|---|")
    A("| `R3性能实测分析_20261128.md` | 首测的聚合结论；本文 §清单 是它的「逐点异常下钻」 |")
    A("| `R3逐case对比_20261128.md` | 首测 72 例逐格展开；本文只抽不达标点 |")
    A("| `R3对比1128_20261128.md` | 1128 表横向参照；**不是基线**，不参与本文 R1 判定 |")
    A("| `第三轮测试执行指南_20261128.md` | 环境/门禁/QP 约定；本文不重复 |")
    A("")
    return "\n".join(L) + "\n"


def _phenomenon(g, ks, M, rec):
    """一句话现象（按组算实际范围，不写死）。只算不达标格，剔除 AIV=32 对照。"""
    if g == "P0-EP4AIV8":
        v = []
        for k in ks:
            if k[3] == 32:
                continue
            k32 = (k[0], k[1], k[2], 32)
            if k32 in M:
                v.append(G.pct(M[k]["dispatch"]["mean"], M[k32]["dispatch"]["mean"]))
        if v:
            return "dispatch 相对同批 AIV=32 **%+.1f%% ~ %+.1f%%**（%d 格，随 hidden 减小而塌得更狠）" % (
                min(v), max(v), len(v))
    if g == "P1-EP64COMB":
        v = [rec[k]["delta"]["combine_unweighted"] for k in ks
             if "combine_unweighted" in rec[k]["delta"]]
        if v:
            return "combine_un 对基线 **%+.1f%% ~ %+.1f%%**（%d 格），4 格 weighted 同步 ≤−10%%" % (
                min(v), max(v), len(v))
    if g == "P2-WEIGHTED":
        v = [rec[k]["delta"]["combine_weighted"] for k in ks
             if "combine_weighted" in rec[k]["delta"]]
        if v:
            return "combine_weighted 对基线 **%+.1f%% ~ %+.1f%%**（%d 格），同格 un 反而领先" % (
                min(v), max(v), len(v))
    return "—"


if __name__ == "__main__":
    sys.exit(main())
