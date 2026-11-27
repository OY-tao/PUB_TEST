"""生成 2-16P 泛化精度用例清单（每个 EP 50 条）。

按 C:/Users/OY/Desktop/AITest/DeepEP_CLOS_Test.txt 第 1 条要求：
2、4、8、16P 各 50 条泛化精度 case。

只产出清单（CSV + Markdown），不包含执行逻辑——执行复用设备上已有的
bench_v1_precision.py / bench_v2_precision.py（带 --precision-only）。

用法
----
    python gen_precision_cases.py --check          # 校验配比
    python gen_precision_cases.py --out .          # 生成清单
    python gen_precision_cases.py --dry-run        # 打印前几条
"""

import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DEFAULT = os.path.join(HERE, "precision_cases_2to16p.json")

VALID_EP = (2, 4, 8, 16, 32, 64, 128, 256)
VALID_AIV = tuple(range(2, 65, 2))


class Case:
    __slots__ = ("ep", "idx", "group", "params", "tag")

    def __init__(self, ep, idx, group, params, tag):
        self.ep, self.idx, self.group, self.params, self.tag = ep, idx, group, params, tag

    @property
    def nnodes(self):
        return self.params.get("nnodes", 1)

    def bench_args(self, api):
        """生成命令行参数串（单机/多机都适用）。"""
        out = []
        for k in ("ep-size", "nnodes", "node-rank", "master-addr", "master-port",
                  "run-id", "first-device", "num-tokens", "hidden", "num-topk",
                  "experts-per-rank", "dtypes", "expand", "aiv", "warmup",
                  "iterations", "seed", "route-mode", "route-skew",
                  "route-hotspot", "route-group-size", "route-local-fraction",
                  "route-mask-rate"):
            if k in self.params:
                v = self.params[k]
                if k in ("nnodes",) and v == 1:
                    continue          # 单机不传，避免干扰
                out.append(f"--{k}={v}")
        return " ".join(out)


def topk_cap(ep, epr):
    """bench 硬校验：num_topk <= min(128, ep*epr)。

    注意是 128 而非 32——此前版本误写 32，与 bench_v2_bandwidth.py:140 不一致，
    已按源码统一。V2 库文档允许 topk<=32，但 bench 层的上限是 128，
    所以 min(128, ...) 才是实际可执行的边界。
    """
    return min(128, ep * epr)


def build(spec, ep):
    """为单个 EP 展开 50 条。"""
    g = spec["groups"]
    cases = []

    def add(group, params, tag):
        p = {"ep-size": ep, "nnodes": max(1, ep // 16)}
        p.update(params)
        # 裁剪 topk 到合法上限
        cap = topk_cap(ep, p.get("experts-per-rank", 32))
        if p.get("num-topk", 0) > cap:
            p["num-topk"] = cap
        cases.append(Case(ep, len(cases) + 1, group, p, tag))

    # A 形状主干（20）：4 tokens x 4 hidden 全网格 = 16，再补 4 条换 dtype/expand
    A = g["A"]
    toks, hids = A["axes"]["num_tokens"], A["axes"]["hidden"]
    n = 0
    for t in toks:
        for h in hids:
            if n >= 16:
                break
            # 用 n 决定 dtype/expand，保证两者在网格上均匀分布
            dt = A["axes"]["dtypes"][n % 2]
            ex = A["axes"]["expand"][(n // 2) % 2]
            n += 1
            add("A", {"num-tokens": t, "hidden": h,
                      "num-topk": A["fixed"]["num_topk"],
                      "experts-per-rank": A["fixed"]["experts_per_rank"],
                      "dtypes": dt, "expand": ex}, f"A_t{t}_h{h}_{dt}_e{ex}")
    # 补 4 条：四角换另一种 dtype+expand（覆盖同一形状的交叉组合）
    extra = [(toks[0], hids[0], "fp8", 1), (toks[0], hids[-1], "fp8", 1),
             (toks[-1], hids[0], "fp8", 1), (toks[-1], hids[-1], "fp8", 1)]
    for t, h, dt, ex in extra:
        add("A", {"num-tokens": t, "hidden": h,
                  "num-topk": A["fixed"]["num_topk"],
                  "experts-per-rank": A["fixed"]["experts_per_rank"],
                  "dtypes": dt, "expand": ex}, f"A_t{t}_h{h}_{dt}_e{ex}_x")

    # B 专家/topk（12）
    B = g["B"]
    pairs = [(1, 1), (2, 1), (4, 4), (8, 4), (16, 8), (16, 16),
             (32, 16), (1, 64), (4, 64), (8, 32), (2, 16), (16, 64)]
    for j, (k, epr) in enumerate(pairs[:B["count"]]):
        add("B", {"num-tokens": B["fixed"]["num_tokens"], "hidden": B["fixed"]["hidden"],
                  "num-topk": k, "experts-per-rank": epr,
                  "dtypes": "bf16", "expand": j % 2}, f"B_k{k}_epr{epr}_e{j%2}")

    # C AIV（6）
    C = g["C"]
    for j, (aiv, dt) in enumerate([(2, "bf16"), (2, "fp8"), (16, "bf16"), (16, "fp8"),
                                   (32, "bf16"), (32, "fp8")]):
        add("C", {"num-tokens": C["fixed"]["num_tokens"], "hidden": C["fixed"]["hidden"],
                  "num-topk": C["fixed"]["num_topk"],
                  "experts-per-rank": C["fixed"]["experts_per_rank"],
                  "aiv": aiv, "dtypes": dt, "expand": 0}, f"C_aiv{aiv}_{dt}")

    # D 路由与 mask（8）
    D = g["D"]
    base = {"num-tokens": D["fixed"]["num_tokens"], "hidden": D["fixed"]["hidden"],
            "num-topk": D["fixed"]["num_topk"],
            "experts-per-rank": D["fixed"]["experts_per_rank"], "dtypes": "bf16"}
    for ex in (0, 1):
        add("D", {**base, "route-mode": "uniform", "expand": ex}, f"D_uniform_e{ex}")
    for skew in (2, 8):
        add("D", {**base, "route-mode": "rank_hotspot", "route-skew": skew, "expand": 0},
            f"D_hotspot_skew{skew}")
    for fr in (0.5, 0.95):
        add("D", {**base, "route-mode": "local_group", "route-local-fraction": fr, "expand": 0},
            f"D_local_{fr}")
    for mr in (0.3, 0.9):
        add("D", {**base, "route-mode": "uniform", "route-mask-rate": mr, "expand": 0},
            f"D_mask{mr}")

    # E FP8 专项（4）
    E = g["E"]
    for j, h in enumerate(E["axes"]["hidden"]):
        add("E", {"num-tokens": E["fixed"]["num_tokens"], "hidden": h,
                  "num-topk": E["fixed"]["num_topk"],
                  "experts-per-rank": E["fixed"]["experts_per_rank"],
                  "dtypes": "fp8", "expand": j % 2}, f"E_fp8_h{h}_e{j%2}")

    return cases


def check(spec, built):
    rows = []
    ok = True
    total = 0
    for ep, cases in built.items():
        good = len(cases) == spec["per_ep"]
        ok = ok and good
        rows.append((ep, spec["per_ep"], len(cases), good))
        total += len(cases)
    return ok, rows, total


def validate(built, spec):
    errs = []
    for ep, cases in built.items():
        for c in cases:
            p = c.params
            if p["ep-size"] not in VALID_EP:
                errs.append(f"EP{ep} {c.tag}: ep-size 非法")
            if p.get("aiv", 32) not in VALID_AIV:
                errs.append(f"EP{ep} {c.tag}: aiv={p.get('aiv')} 非法")
            if p["hidden"] % 128:
                errs.append(f"EP{ep} {c.tag}: hidden={p['hidden']} 不是 128 倍数")
            cap = topk_cap(ep, p.get("experts-per-rank", 32))
            if p.get("num-topk", 1) > cap:
                errs.append(f"EP{ep} {c.tag}: topk={p['num-topk']} > 上限 {cap}")
            if p.get("num-topk", 1) < 1:
                errs.append(f"EP{ep} {c.tag}: topk 必须 >=1")
            # 每节点 1..16 卡
            nn = p.get("nnodes", 1)
            if ep % nn:
                errs.append(f"EP{ep} {c.tag}: ep % nnodes != 0")
            local = ep // nn
            if not 1 <= local <= 16:
                errs.append(f"EP{ep} {c.tag}: 每节点 {local} 卡超 1..16")
            # V1 形状上限
            if p["hidden"] > 7168:
                errs.append(f"EP{ep} {c.tag}: hidden>7168 V1 不支持")
            if p["num-tokens"] > 16384:
                errs.append(f"EP{ep} {c.tag}: tokens>16384 V1 不支持")
    return errs


def write_csv(built, path):
    cols = ["ep", "nnodes", "idx", "group", "tag", "num_tokens", "hidden",
            "num_topk", "experts_per_rank", "aiv", "dtypes", "expand",
            "route_mode", "route_skew", "route_hotspot", "route_group_size",
            "route_local_fraction", "route_mask_rate", "bench_args"]
    lines = [",".join(cols)]
    for ep in sorted(built):
        for c in built[ep]:
            p = c.params
            row = [
                str(ep), str(p.get("nnodes", 1)), str(c.idx), c.group, c.tag,
                str(p["num-tokens"]), str(p["hidden"]), str(p.get("num-topk", 8)),
                str(p.get("experts-per-rank", 32)), str(p.get("aiv", 32)),
                p.get("dtypes", "bf16"), str(p.get("expand", 0)),
                p.get("route-mode", "uniform"),
                str(p.get("route-skew", "")), str(p.get("route-hotspot", "")),
                str(p.get("route-group-size", "")),
                str(p.get("route-local-fraction", "")),
                str(p.get("route-mask-rate", "")),
                '"' + c.bench_args("v2") + '"',
            ]
            lines.append(",".join(row))
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def write_md(built, spec, path):
    L = []
    L.append("# 2-16P 泛化精度用例清单")
    L.append("")
    L.append("> 每个 EP **50 条**，合计 **200 条**。仅清单，不含执行逻辑。")
    L.append("> 执行复用设备上的 `bench_v1_precision.py` / `bench_v2_precision.py`（`--precision-only`）。")
    L.append("")
    L.append("## 配比")
    L.append("")
    L.append("| 组 | 名称 | 条数 | 维度 |")
    L.append("|---|---|---|---|")
    desc = {
        "A": "4 tokens × 4 hidden 网格取四角+对角+交叉，配 dtype/expand",
        "B": "topk × experts_per_rank 组合，覆盖 1 与上限",
        "C": "AIV 2/16/32 × bf16/fp8",
        "D": "uniform / hotspot / local_group / mask 四种路由",
        "E": "FP8 专项，hidden 128~7168（128 为最小合法值）",
    }
    for k, v in spec["groups"].items():
        L.append(f"| {k} | {v['name']} | {v['count']} | {desc[k]} |")
    L.append("")
    L.append("## 重要约束")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| V1 形状上限 | `H≤7168`（BF16 16 倍数 / FP8 **128** 倍数）、`T≤16384`、`K≤16`、`E≤1024` |")
    L.append("| V2 形状上限 | `H>0`、`K≤32`、`E≤1024`、E 可被 P 整除 |")
    L.append("| topk 上限 | `min(128, ep × epr)`（bench 硬校验，注意是 128 不是 32），本清单已按此裁剪 |")
    L.append("| 每节点卡数 | 1~16，故 EP≤16 均单机 |")
    L.append("| FP8 hidden | **必须 128 倍数**，最小合法 128 |")
    L.append("")
    L.append("## 逐条清单")
    L.append("")
    for ep in sorted(built):
        cases = built[ep]
        L.append(f"### EP{ep}（{len(cases)} 条，{cases[0].nnodes} 机 × {ep // cases[0].nnodes} 卡）")
        L.append("")
        L.append("| # | 组 | 形状 T/H | topk | epr | aiv | dtype | expand | 路由 | tag |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in cases:
            p = c.params
            route = p.get("route-mode", "uniform")
            for extra in ("route-skew", "route-local-fraction", "route-mask-rate"):
                if extra in p:
                    route += f" {extra.split('-')[-1]}={p[extra]}"
            L.append("| %d | %s | %d/%d | %d | %d | %d | %s | %d | %s | `%s` |" % (
                c.idx, c.group, p["num-tokens"], p["hidden"],
                p.get("num-topk", 8), p.get("experts-per-rank", 32),
                p.get("aiv", 32), p.get("dtypes", "bf16"), p.get("expand", 0),
                route, c.tag))
        L.append("")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="生成 2-16P 泛化精度用例清单")
    ap.add_argument("--spec", default=SPEC_DEFAULT)
    ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    spec = json.load(open(args.spec, encoding="utf-8"))
    built = {ep: build(spec, ep) for ep in spec["ep_list"]}

    errs = validate(built, spec)
    if errs:
        print("参数校验失败：")
        for e in errs[:20]:
            print("  " + e)
        return 1

    ok, rows, total = check(spec, built)
    print("=" * 60)
    print("2-16P 泛化精度用例配比")
    print("=" * 60)
    print("%-6s %8s %8s %6s %-14s" % ("EP", "目标", "实际", "结果", "拓扑"))
    print("-" * 60)
    for ep, exp, got, good in rows:
        nn = built[ep][0].nnodes
        print("%-6d %8d %8d %6s %-14s" % (ep, exp, got, "OK" if good else "FAIL",
                                          f"{nn}机x{ep//nn}卡"))
    print("-" * 60)
    print("%-6s %8s %8d %6s" % ("合计", spec["_expected_total"], total,
                                "OK" if total == spec["_expected_total"] else "FAIL"))
    print()

    if not ok:
        return 1

    if args.dry_run:
        allc = [c for cs in built.values() for c in cs]
        n = args.limit or len(allc)
        for c in allc[:n]:
            print(f"[EP{c.ep} #{c.idx}] {c.tag}")
            print("  python bench_v2_precision.py --precision-only " + c.bench_args("v2"))
        if n < len(allc):
            print(f"... 还有 {len(allc)-n} 条")
        return 0

    if args.check or not args.out:
        if not args.out:
            print("（未指定 --out，仅校验）")
        return 0

    os.makedirs(args.out, exist_ok=True)
    write_csv(built, os.path.join(args.out, "precision_cases_2to16p.csv"))
    write_md(built, spec, os.path.join(args.out, "精度用例清单_2to16P.md"))
    print(f"已生成: {args.out}/precision_cases_2to16p.csv")
    print(f"已生成: {args.out}/精度用例清单_2to16P.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
