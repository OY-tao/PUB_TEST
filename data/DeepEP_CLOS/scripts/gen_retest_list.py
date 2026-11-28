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
    "P1-EP64COMB": dict(
        prio="P1", title="EP64 的 combine 劣化（原始 12 例中的 AIV=32 六格）",
        ask="6 格 combine_un 对基线 −5.3~−29.5%（4 格 weighted 同步 ≤−10%），"
            "是基线表 rows 口径偏高还是本轮真退化？3 次稳定在 −5% 以下 ⇒ 口径；明显回升 ⇒ 首测受环境影响。"),
    "P2-WEIGHTED": dict(
        prio="P2", title="EP≤32 的 combine_weighted 短板",
        ask="同格 unweighted 领先基线 +1.4~+21% 而 weighted 落后 −5~−12%，"
            "确认是否为权重分支的真实开销缺口，确认后立项。"),
}

RETEST_REPS = 3          # 每格重复次数

# ---- 本轮范围（用户 2026-11-29 定）----
# 1) 不看 EP4：18 格（AIV8 9 + AIV32 9）整体排除，含原 P0 组；
# 2) EP64 按「原始 12 例」的口径取，但**只跑其中已确认性能劣化的格**：
#    12 例 = T 8192/16384 × H 2816/6144/7168 × AIV 32/64（与基线快照同口径）；
#    AIV=32 的 6 格首测 combine_un 全部 ≤ −5%（−5.3~−29.5%）⇒ 纳入；
#    AIV=64 的 6 格**从未测过**（R3 只跑过 supp 网格的 AIV 8/32）⇒ 无实测值，
#    谈不上「劣化」，本轮不跑；若要判它是否劣化，需先补首测（把 EP64_FULL_GRID 置 True）。
SCOPE_EPS = {8, 16, 32, 64}
EP64_FULL_GRID = False   # True = 连 AIV=64 的 6 格一起补齐（首测）；False = 只跑已劣化的格
EXCLUDED_NOTE = {
    4: "EP4 全 18 格（AIV8 9 + AIV32 9）：用户 2026-11-29 指示本轮不看 EP4",
}


def ep64_grid():
    """原始 12 例：从主网格规格（perf_cases_v2.json）取 EP64 的 (T,H,AIV) 全组合。"""
    with io.open(os.path.join(BASE, "scripts", "perf_cases_v2.json"), encoding="utf-8") as f:
        ax = json.load(f)["axes"]
    return [(64, t, h, a) for t in ax["num_tokens"] for h in ax["hidden"] for a in ax["aiv"]]
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
        if k[0] not in SCOPE_EPS or k not in SNAP:
            continue
        for op in M[k]:
            dv = rec[k]["delta"].get(op)
            if dv is None or dv > R1_DROP:
                continue
            grp = "P1-EP64COMB" if k[0] == 64 else "P2-WEIGHTED"
            add(k, op, "R1", "对外基线劣化 %.1f%%" % dv)
            rows[-1]["group"] = grp
            hit_cases.setdefault(k, set()).add(grp)

    # R2：同批相对塌陷（只对无基线档位，避免与 R1 重复计）
    for k in sorted(M):
        if k[0] not in SCOPE_EPS or k in SNAP or k[3] == 32:
            continue
        k32 = (k[0], k[1], k[2], 32)
        if k32 not in M or "dispatch" not in M[k32]:
            continue
        dv = G.pct(M[k]["dispatch"]["mean"], M[k32]["dispatch"]["mean"])
        if dv > R2_DROP:
            continue
        add(k, "dispatch", "R2", "相对同批 AIV=32 %.1f%%" % dv, delta=dv)
        rows[-1]["group"] = "P2-WEIGHTED" if k[0] != 64 else "P1-EP64COMB"
        hit_cases.setdefault(k, set()).add(rows[-1]["group"])
        # 该格其它明显同步塌陷的 op 一并记（只记录，不改判据）
        for op in ("cached-dispatch", "combine_unweighted", "combine_weighted"):
            if op not in M[k] or op not in M[k32]:
                continue
            d2 = G.pct(M[k][op]["mean"], M[k32][op]["mean"])
            if d2 <= R2_DROP:
                add(k, op, "R2", "相对同批 AIV=32 %.1f%%" % d2, delta=d2)
                rows[-1]["group"] = "P2-WEIGHTED" if k[0] != 64 else "P1-EP64COMB"
                hit_cases.setdefault(k, set()).add(rows[-1]["group"])

    # R3：测量不稳。已被 R1/R2 命中的格不单列（分散度已在各 op 的 spread 列里）
    for k in sorted(M):
        if k[0] not in SCOPE_EPS:
            continue
        sp = rec[k]["spread"]
        worst = max(sp.values())
        if worst <= R3_SPREAD or k in hit_cases:
            continue
        op = max(sp, key=sp.get)
        add(k, "*", "R3", "格内分散度 %.1f%%（%s），单次不可信" % (worst, op))
        rows[-1]["group"] = "P1-EP64COMB" if k[0] == 64 else "P2-WEIGHTED"
        hit_cases.setdefault(k, set()).add(rows[-1]["group"])

    # 一支格命中多个组时，归到优先级最高的组（P1 > P2）
    order = {"P1-EP64COMB": 0, "P2-WEIGHTED": 1}
    for r in rows:
        gs = [g for g in r["group"].split(",") if g]
        r["group"] = sorted(gs, key=lambda g: order.get(g, 9))[0]
        r["priority"] = GROUPS[r["group"]]["prio"]

    # ---------- 重测格子（去重） ----------
    retest_keys = {}
    for r in rows:
        k = (r["ep"], r["num_tokens"], r["hidden"], r["aiv"])
        retest_keys.setdefault(k, set()).add(r["group"])

    # EP64 可选展开到原始 12 例（EP64_FULL_GRID=True）：命中的 6 格（AIV=32）是子集，
    # AIV=64 的 6 格从未测过 —— 展开即「顺带补首测」，但与「只跑劣化的」口径不符，默认关闭。
    if EP64_FULL_GRID:
        retest_ep64 = set(retest_keys)
        retest_keys = {k: v for k, v in retest_keys.items() if k[0] != 64}
        for k in ep64_grid():
            retest_keys.setdefault(k, set()).add("P1-EP64COMB")

    # 记录每格在重测里的角色：nonconform=首测已不达标、first=从未测过、grid=其余回归格
    role_of = {}
    hit_set = {(r["ep"], r["num_tokens"], r["hidden"], r["aiv"]) for r in rows}
    for k in retest_keys:
        if k in hit_set:
            role_of[k] = "nonconform"
        elif k not in M:
            role_of[k] = "first"
        else:
            role_of[k] = "grid"

    by_group = {}
    for k, gs in retest_keys.items():
        g = sorted(gs, key=lambda x: order.get(x, 9))[0]
        by_group.setdefault(g, []).append(k)

    # ---------- 打印统计 ----------
    print("总格数 %d；有基线 %d；本轮范围 EP%s；纳入重测 %d 格（%d 次运行，reps=%d）"
          % (len(M), len([k for k in M if k in SNAP]), sorted(SCOPE_EPS), len(retest_keys),
             len(retest_keys) * RETEST_REPS, RETEST_REPS))
    for g in sorted(by_group, key=lambda x: order[x]):
        ks = sorted(by_group[g])
        eps = sorted({k[0] for k in ks})
        ro = {}
        for k in ks:
            ro[role_of[k]] = ro.get(role_of[k], 0) + 1
        print("  %-12s %2d 格  EP%s  %s" % (g, len(ks), eps, ro))
    print("不达标点 %d 个（范围外 EP4 的 %d 格整档不计）"
          % (len([r for r in rows]), len([k for k in M if k[0] not in SCOPE_EPS])))
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
            jcases.append(dict(ep=k[0], num_tokens=k[1], hidden=k[2], aiv=k[3],
                               group=g, role=role_of[k]))
    spec = dict(
        _comment="R3 性能不达标重测清单，由 scripts/gen_retest_list.py 生成，勿手改",
        _scope=dict(eps=sorted(SCOPE_EPS),
                    excluded_eps={str(k): v for k, v in EXCLUDED_NOTE.items()},
                    ep64_full_grid=EP64_FULL_GRID),
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
    md = build_md(M, SNAP, rec, rows, control, by_group, retest_keys, order, role_of)
    md_path = os.path.join(BASE, "R3重测清单_20261128.md")
    with io.open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("已写：%s\n     %s\n     %s" % (csv_path, json_path, md_path))
    return 0


def build_md(M, SNAP, rec, rows, control, by_group, retest_keys, order, role_of):
    L = []
    A = L.append
    n_case = len(M)
    n_snap = len([k for k in M if k in SNAP])
    n_rt = len(retest_keys)
    n_runs = n_rt * RETEST_REPS

    A("# R3 性能不达标 case 重测清单（2026-11-28 首测 / 11-29 定范围）")
    A("")
    A("> **本文由 `scripts/gen_retest_list.py` 机械生成**，数字不手抄；改数据后重跑即可复现。")
    A("> 首测：`round3_perf.csv`（36 例）+ `round3_perf_supp.csv`（36 例）= **72 例 × 4 op**，")
    A("> commit `0d1eca2`、纯 auto QP、无大页、`rc` 全 0。")
    A("> 对照基线：`baseline/V2测试Case_20261128快照.txt`（对外交付口径，覆盖 %d/%d 格）。" % (n_snap, n_case))
    A("")
    A("## 本轮范围（2026-11-29 定，与首测口径不同）")
    A("")
    A("| 项 | 首测口径 | 本轮重测口径 |")
    A("|---|---|---|")
    A("| EP4 | 18 格（AIV8 9 + AIV32 9），含原 P0 的 AIV=8 塌陷 | **整档不看**（用户指示）；其 9 格 AIV=8 的 −39.7~−71.8% 记录保留在首测报告，不进本轮结论 |")
    A("| EP64 | 只跑过 supp 网格（T4096/8192/16384 × AIV **8/32**），与基线不可比 | **按原始 12 例的口径取，但只跑已劣化的 6 格**：T 8192/16384 × H 2816/6144/7168 × AIV **32**（combine_un 全部 ≤ −5%） |")
    A("| EP8/EP32 | combine_weighted 6 格不达标 | 不变，仍纳入 |")
    A("")
    A("> EP64 原始 12 例 = 上面的 6 格（AIV=32）+ 另 6 格（AIV=64）。**AIV=64 的 6 格本轮不跑**：")
    A("> R3 从未测过 EP64×AIV=64（supp 网格只有 AIV 8/32），没有实测值就谈不上「劣化」；")
    A("> 基线快照里这 6 格的对照值是齐的，缺的只是我方一侧 —— 若后续要判它，用")
    A("> `EP64_FULL_GRID=True` 重跑本脚本即展开成 12 格。")
    A("")
    A("## 一句话")
    A("")
    A("**%d 格纳入重测**，每格重复 **%d 次**，共 **%d 次运行**（首测 %d 格，范围外 %d 格）。"
      % (n_rt, RETEST_REPS, n_runs, n_case, len([k for k in M if k[0] not in SCOPE_EPS])))
    A("")
    A("| 组 | 格数 | 不达标 | 现象 | 重测要回答什么 |")
    A("|---|---|---|---|---|")
    for g in sorted(by_group, key=lambda x: order[x]):
        ks = by_group[g]
        gi = GROUPS[g]
        nc = len([k for k in ks if role_of[k] == "nonconform"])
        A("| **%s** %s | %d | %d | %s | %s |" % (
            gi["prio"], gi["title"], len(ks), nc, _phenomenon(g, ks, M, rec), gi["ask"]))
    A("")
    A("合计 **%d 格 / %d 次运行**。" % (n_rt, n_runs))
    A("")
    A("## 判定口径（三条，互不替代）")
    A("")
    A("| 规则 | 判据 | 阈值 | 为什么这么定 |")
    A("|---|---|---|---|")
    A("| **R1 对外基线劣化** | vs 基线快照，该指标 `Δ = 我方/基线 − 1` | **≤ %+.0f%%** | 只用对外交付口径的基线；−5%% 以内的负值按噪声处理，不立项 |" % R1_DROP)
    A("| **R2 同批相对塌陷** | 无基线的档位里，同 EP 同 (T,H) 下低 AIV 相对 AIV=32 的 dispatch | **≤ %+.0f%%** | 每格分散度中位 1.4%%，AIV 配置差在上一批里 ≤6%% ⇒ 10%% 以上才超出「稳定特性+噪声」 |" % R2_DROP)
    A("| **R3 测量不稳** | 格内四 op 的 `(max−min)/mean` 最大值 | **> %.0f%%** | 单次结果本身不可信，必须先复测再判 |" % R3_SPREAD)
    A("")
    A("三条互相独立，一格可同时命中；归组按 P1 > P2 取最高优先级。")
    A("**R2 本轮无命中**（EP4 排除后，范围内已无「无基线且低 AIV 塌陷」的档），保留规则以便下批复用。")
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
        A("### %s %s —— 重测 %d 格（其中 %d 格首测不达标 / %d 点）" % (
            gi["prio"], gi["title"], len(by_group[g]), nc, len(sub)))
        A("")
        if g == "P1-EP64COMB":
            A("**取自 EP64 原始 12 例中已实测且劣化的 6 格**（AIV=32；与 `baseline/V2测试Case_20261128快照.txt`")
            A("的 EP64 行同口径）。另 6 格 AIV=64 从未测过，无实测值可判劣化，本轮不跑（见 §明确不纳入）。")
            A("")
        A("**不达标点明细**（%d 点）：" % len(sub))
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
    A("**节点与波次**：三组节点互不重叠，**7 台一次打满、单波并行**（性能集合硬排除 node3/`.7`）：")
    A("")
    A("| 组 | EP | 节点 | 机×卡 | 例数×reps | 次数 | 单例（实测） | 该组预计 |")
    A("|---|---|---|---|---|---|---|---|")
    plan = [
        ("P1-EP64COMB", "EP64", "node2,4,5,6 = .6/.8/.9/.10", "4机×16卡", 4.3),
        ("P2-WEIGHTED", "EP32", "node0,1 = .4/.5", "2机×16卡", 2.6),
        ("P2-WEIGHTED", "EP8", "node7 = .11", "1机×8卡", 1.4),
    ]
    times = []
    for g, ep, node, topo, t in plan:
        eps = 64 if ep == "EP64" else int(ep[2:])
        n = len([k for k in by_group.get(g, []) if k[0] == eps])
        if not n:
            continue
        times.append(n * RETEST_REPS * t)
        A("| %s | %s | %s | %s | %d × %d | %d | ~%.1f min | **~%.0f min** |" % (
            g.split("-")[0], ep, node, topo, n, RETEST_REPS, n * RETEST_REPS, t,
            n * RETEST_REPS * t))
    A("")
    A("> 单波并行 ⇒ 总时长取最慢的一组（EP64）**~%d min**；加门禁与部署 15 min，"
      "**总计约 %.1f 小时**。" % (max(times or [0]), (max(times or [0]) + 15) / 60.0))
    A("> EP64 单例按 4.3 min 是 11-28 补测批的实测均值（18 例 / 72 min）。")
    A("> 12 格都是**首测已不达标的格**（无首测补全项）⇒ 每格 3 次是判「是否稳定复现」的最小样本，")
    A("> 不建议再压到 1 次。")
    A("")
    A("**前置条件**：")
    A("")
    A("1. **C 阶段（EP128 精度）跑完并释放 8 台**——EP128 独占全部节点，与性能批互斥。")
    A("2. `python scripts/run_round3.py gate` 复检 `pass=true`（8 台 16/16 空闲、commit `0d1eca2`、")
    A("   `buffer.py` sha `f825a679…`、`qp_plans` 为空）。")
    A("3. 三组要 **7 台同时空闲**（`.4/.5` 若被别人占则整批起不来；届时不改代码、把 EP32 排到")
    A("   EP64 之后串行（复用 node5,6），或改 `RETEST_EP_NODES` 换一组与 EP64 不重叠的机器）。")
    A("4. 容器显式绑 8 台（`--containers`），不要靠子串猜（历史上误取旧容器）。")
    A("")

    # ---------- 命令 ----------
    A("## 执行命令")
    A("")
    A("```bash")
    A("# 1) 门禁")
    A("python3 scripts/run_round3.py gate")
    A("")
    A("# 2) 一次跑完 12 格（EP64 6 + EP32 3 + EP8 3），单波并行")
    A("#    走主网格（perf_cases_v2.json，AIV 32/64），不是 --supp 的 1128 网格")
    A("python3 scripts/run_round3.py perf --eps 8,32,64 \\")
    A("    --cases-file scripts/retest_cases_20261128.json \\")
    A("    --reps 3 --csv-tag retest")
    A("")
    A("# 3) 出判定（按 3 次中位机械判，不手算）")
    A("python3 scripts/gen_retest_result.py --csv round3_perf_retest.csv --date 20261129")
    A("```")
    A("")
    A("> 带 `--cases-file` 时脚本自动切到 `RETEST_EP_NODES`（EP64=2,4,5,6 / EP32=0,1 / EP8=7），")
    A("> 并在启动前断言三组节点不重叠。")
    A("> `--csv-tag retest` 写 `round3_perf_retest.csv`，不污染主批 `round3_perf.csv`。")
    A("")
    A("> 判定表由 `scripts/gen_retest_result.py` 按本文规则机械生成，四类：确认劣化 / 边界噪声 /")
    A("> 达标 / 作废。12 格**都有基线**（全在快照覆盖内），一律走 R1，本批不会出现「无基线」行。")
    A("")
    A("## 重测后的判定规则（先定规则，避免事后找理由）")
    A("")
    A("| 重测结果 | 定性 | 动作 |")
    A("|---|---|---|")
    A("| 3 次中位 Δ **仍 ≤ −5%** | **确认真实劣化** | EP64 若稳定在 −5% 以下则改判「基线 rows 口径差」并在对外口径写明；P2 立项 |")
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
    A("| **EP4 全档** | **18 格** | 用户 2026-11-29 指示本轮不看 EP4。首测里 EP4+AIV8 相对同批 AIV32 掉 −39.7~−71.8%（9 格）记录仍在 `R3性能实测分析` §5.1，**不进本轮结论** |")
    A("| **EP64 的 AIV=64** | **6 格** | 原始 12 例里的一半，但 R3 从未测过（supp 网格只有 AIV 8/32）⇒ **无实测值，判不出劣化**。基线侧有对照、缺的是我方数据；需要时 `EP64_FULL_GRID=True` 展开 |")
    A("| EP64 的 AIV=8 | 9 格 | 不在原始 12 例里（属 supp 网格的 1128 口径），与基线不可比；AIV=32 已覆盖 EP64 结论 |")
    A("| 劣化在 −5%% ~ 0 之间 | **%d 点** | 低于 bench 噪声 + 基线四舍五入的量级，重测也判不出方向 |" % len(neg_small))
    if neg_small:
        A("| └ 明细 | %d 点 | 全部是 `combine_weighted`：%s |" % (
            len(neg_small),
            "、".join("EP%d T%d H%d AIV%d %+.2f%%" % (k[0], k[1], k[2], k[3], dv)
                     for k, op, dv in sorted(neg_small))))
        A("| └ 说明 | — | 这些点本身不足以独立入选；但同格若因其它指标入选，仍会一起重测"
          "（如 EP64 T16384 H6144 因 `combine_un` −6.2% 已在 P1） |")
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
    """一句话现象（按组算实际范围，不写死）。"""
    if g == "P1-EP64COMB":
        v = [rec[k]["delta"]["combine_unweighted"] for k in ks
             if k in rec and "combine_unweighted" in rec[k]["delta"]]
        w = [rec[k]["delta"]["combine_weighted"] for k in ks
             if k in rec and "combine_weighted" in rec[k]["delta"]]
        if v:
            return ("combine_un 对基线 **%+.1f%% ~ %+.1f%%**（%d 格，均 ≤−5%%）；"
                    "其中 %d 格 weighted 同步 ≤−10%%" % (
                        min(v), max(v), len(v), len([x for x in w if x <= R2_DROP])))
    if g == "P2-WEIGHTED":
        v = [rec[k]["delta"]["combine_weighted"] for k in ks
             if "combine_weighted" in rec[k]["delta"]]
        if v:
            return "combine_weighted 对基线 **%+.1f%% ~ %+.1f%%**（%d 格），同格 un 反而领先" % (
                min(v), max(v), len(v))
    return "—"


if __name__ == "__main__":
    sys.exit(main())
