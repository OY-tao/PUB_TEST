"""按 cases_1148.json 生成 DeepEP CLOS 泛化用例的 bench 命令行批次。

背景
----
上游 bench 一次调用只能跑一组形状参数，且每个 case 固定产出 4 个 op
（dispatch / cached-dispatch / combine_unweighted / combine_weighted）。
1148 例需要 1148 次独立 torchrun，手工敲不现实，本脚本负责展开。

设计原则
--------
1. **配置与代码分离**：1148 例的规格全在 cases_1148.json，改范围只改 JSON。
2. **参数名对齐上游**：所有参数名对照 bench_v2_bandwidth.py 与
   route_workloads.py 核实过，未核实的参数一律不生成（见 _verified 字段）。
3. **可校验**：--check 会逐组核对例数，与规格里的 _expected 对不上就报错。
4. **可分批执行**：输出 shell 脚本，按组切分，便于中断续跑。

用法
----
    # 只校验配比（不写文件、不碰设备）
    python gen_cases.py --check

    # 生成全部（v1 + v2 各 1148 条）
    python gen_cases.py --out batches/

    # 只生成某一组
    python gen_cases.py --out batches/ --groups A B

    # 只生成 V2
    python gen_cases.py --out batches/ --api v2

    # 生成后不执行，只看前几条
    python gen_cases.py --dry-run --limit 5

产物
----
    batches/
    ├── v2_A.sh ... v2_H.sh     每组一个脚本
    ├── v1_A.sh ... v1_H.sh
    ├── manifest_v2.csv         逐条 case 明细（组/编号/参数/命令行）
    ├── summary.md              配比核对表
    └── index.sh                串行执行入口
"""

import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DEFAULT = os.path.join(HERE, "cases_1148.json")

# bench 支持的合法取值（对照 argparse choices 核实）
VALID_EP = (2, 4, 8, 16, 32, 64, 128, 256)
VALID_NNODES = (1, 2, 4, 8, 16, 32)
VALID_AIV = tuple(range(2, 65, 2))
VALID_DTYPES = ("bf16", "fp8")
VALID_EXPAND = (0, 1)
VALID_ROUTE_MODE = ("uniform", "rank_hotspot", "local_group")

# bench / route_workloads 真实接受的参数名白名单。
# 生成时只放行这些键，避免规格文件里的内部字段（如 _id/desc/count）
# 被误当成命令行参数——argparse 的前缀匹配会让它静默通过，是个隐患。
ALLOWED_KEYS = frozenset({
    "ep-size", "nnodes", "node-rank", "master-addr", "master-port",
    "run-id", "first-device", "aiv", "dtypes", "expand",
    "num-tokens", "hidden", "num-topk", "experts-per-rank",
    "warmup", "iterations", "seed", "case-timeout",
    "huge-pages", "combine-bias", "output", "report-only",
    "route-mode", "route-skew", "route-hotspot",
    "route-group-size", "route-local-fraction", "route-mask-rate",
})


class Case:
    """一条 bench 调用。params 只放与 fixed 不同的项。"""

    __slots__ = ("group", "idx", "params", "tag", "level", "needs_nodes")

    def __init__(self, group, idx, params, tag="", level="性能"):
        self.group = group
        self.idx = idx
        self.params = params
        self.tag = tag
        self.level = level
        # 多机 case 需要 --master-addr / --run-id，执行时外部注入
        self.needs_nodes = params.get("nnodes", 1) and params.get("nnodes", 1) > 1

    def command(self, bench, output_root, run_id_prefix):
        args = [f"--ep-size={self.params.get('ep-size', 8)}"]
        for key in (
            "nnodes", "node-rank", "first-device",
            "num-tokens", "hidden", "num-topk", "experts-per-rank",
            "dtypes", "expand", "aiv", "warmup", "iterations", "seed",
            "route-mode", "route-skew", "route-hotspot",
            "route-group-size", "route-local-fraction", "route-mask-rate",
            "combine-bias",
        ):
            if key not in ALLOWED_KEYS or key not in self.params:
                continue
            val = self.params[key]
            if isinstance(val, bool):
                continue
            args.append(f"--{key}={val}")
        if self.params.get("huge_pages"):
            args.append("--huge-pages")

        run_id = f"{run_id_prefix}_{self.group}{self.idx:04d}"
        # 统一用 POSIX 分隔符：本脚本生成的是 bash 脚本，在 Linux 设备上执行，
        # 不能受 Windows 的 os.path.join 影响
        out = output_root.rstrip("/") + "/" + run_id
        args.append(f"--run-id={run_id}")
        args.append(f"--output={out}")
        joined = " ".join(args)
        if self.needs_nodes:
            joined += "  # 多机：需各节点分别执行并注入 --master-addr/--node-rank"
        bench_posix = bench.replace("\\", "/")
        return f"python {bench_posix} {joined}"


def norm(params):
    """把 snake_case 键转成 bench 的 kebab-case 参数名，并按白名单过滤。

    过滤掉两类键：
    1. 以 _ 开头的内部字段（_id / _note / _expected）
    2. 不在 ALLOWED_KEYS 里的字段（desc / count 等规格专用键）
    """
    out = {}
    for k, v in params.items():
        if k.startswith("_"):
            continue
        if k in ("nnodes", "combine_bias", "huge_pages", "node_rank", "first_device"):
            key = k.replace("_", "-")
        elif k == "dtypes":
            key = "dtypes"
        elif k == "ep_size":
            key = "ep-size"
        elif k in ("expand", "aiv", "num_tokens", "hidden", "num_topk",
                   "experts_per_rank", "warmup", "iterations", "seed"):
            key = k.replace("_", "-")
        elif k.startswith("route_"):
            key = k.replace("_", "-")
        else:
            key = k.replace("_", "-")
        if key in ALLOWED_KEYS:
            out[key] = v
    return out


def expand_axes(spec_fixed, axes, group, level, tag_prefix=""):
    """对 axes 做笛卡尔积。"""
    keys = list(axes.keys())
    lists = [axes[k] if isinstance(axes[k], list) else [axes[k]] for k in keys]
    out = []
    for i, combo in enumerate(itertools.product(*lists)):
        p = dict(spec_fixed)
        p.update(dict(zip(keys, combo)))
        # 固定项里若与轴同名，轴优先
        tag = f"{tag_prefix}{'+'.join(str(x) for x in combo)}"
        out.append(Case(group, i + 1, norm(p), tag, level))
    return out


def build_group_A(g, fixed, level):
    axes = g["axes"]
    keys = list(axes.keys())
    lists = [axes[k] for k in keys]
    out = []
    for i, combo in enumerate(itertools.product(*lists)):
        p = dict(fixed)
        p.update(dict(zip(keys, combo)))
        out.append(Case("A", i + 1, norm(p), f"t{p['num_tokens']}_h{p['hidden']}_{p['dtypes']}_e{p['expand']}", level))
    return out


def build_group_B(g, fixed, level):
    axes = g["axes"]
    keys = list(axes.keys())
    lists = [axes[k] for k in keys]
    out = []
    for i, combo in enumerate(itertools.product(*lists)):
        p = dict(fixed)
        p.update(dict(zip(keys, combo)))
        out.append(Case("B", i + 1, norm(p), f"k{p['num_topk']}_epr{p['experts_per_rank']}_{p['dtypes']}_e{p['expand']}", level))
    return out


def build_group_C(g, fixed, level):
    """route_mode 只有 3 种；mask 用 route-mask-rate 叠加。"""
    combos = []
    for v in g["variants"]:
        mode = v["route_mode"]
        axes = {k: val for k, val in v.items()
                if k not in ("route_mode", "count") and isinstance(val, list)}
        n_scalar = v.get("count")
        if axes:
            keys = list(axes.keys())
            for combo in itertools.product(*[axes[k] for k in keys]):
                base = {"route-mode": mode}
                base.update(dict(zip([k.replace("_", "-") for k in keys], combo)))
                combos.append(base)
        elif n_scalar:
            for _ in range(n_scalar):
                combos.append({"route-mode": mode})

    out = []
    i = 0
    for base in combos:
        for dt, ex in itertools.product(g["combinatorics"]["dtypes"],
                                        g["combinatorics"]["expand"]):
            i += 1
            p = dict(fixed)
            p.update(base)
            p["dtypes"] = dt
            p["expand"] = ex
            # mask 叠加参数时标出，避免和纯 uniform 混淆
            mode = base["route-mode"]
            if "route-mask-rate" in base:
                desc = f"mask{base['route-mask-rate']}"
            else:
                desc = mode
                for k, v in base.items():
                    if k != "route-mode":
                        desc += f"_{v}"
            out.append(Case("C", i, norm(p), f"{desc}_{dt}_e{ex}", level))
    return out


def build_group_D(g, fixed, level):
    out = []
    i = 0
    for c in g["cases"]:
        for dt in g["combinatorics"]["dtypes"]:
            i += 1
            p = dict(fixed)
            p["ep_size"] = c["ep_size"]
            p["nnodes"] = c["nnodes"]
            p["dtypes"] = dt
            out.append(Case("D", i, norm(p), f"ep{c['ep_size']}_n{c['nnodes']}_{dt}", level))
    md = g["multi_device"]
    for fd in md["first_device"]:
        for dt in g["combinatorics"]["dtypes"]:
            i += 1
            p = dict(fixed)
            p["ep_size"] = md["ep_size"]
            p["nnodes"] = md["nnodes"]
            p["first_device"] = fd
            p["dtypes"] = dt
            out.append(Case("D", i, norm(p), f"ep{md['ep_size']}_dev{fd}_{dt}", level))
    return out


def build_group_E(g, fixed, level):
    axes = g["axes"]
    keys = list(axes.keys())
    out = []
    i = 0
    for combo in itertools.product(*[axes[k] for k in keys]):
        i += 1
        p = dict(fixed)
        p.update(dict(zip(keys, combo)))
        p.update(g.get("fixed_extra", {}))
        out.append(Case("E", i, norm(p), f"aiv{p['aiv']}_{p['dtypes']}", level))
    for extra in g.get("_extra", []):
        i += 1
        p = dict(fixed)
        p.update({k: v for k, v in extra.items() if not k.startswith("_")})
        out.append(Case("E", i, norm(p), f"aiv{p['aiv']}_base_{p['dtypes']}", level))
    return out


def build_group_F(g, fixed, level):
    out = []
    i = 0
    for c in g["cases"]:
        patch = {k: v for k, v in c.items() if not k.startswith("_")}
        for dt, ex in itertools.product(g["combinatorics"]["dtypes"],
                                        g["combinatorics"]["expand"]):
            i += 1
            p = dict(fixed)
            p.update({k: v for k, v in patch.items()})
            p["dtypes"] = dt
            p["expand"] = ex
            # topk=E 依赖 ep_size * experts_per_rank
            if p.get("num_topk") == "E":
                epr = p.get("experts_per_rank", 32)
                p["num_topk"] = p.get("ep_size", 8) * epr
            tag = f"{c['_id']}_{c['desc']}_{dt}_e{ex}"
            out.append(Case("F", i, norm(p), tag, level))
    return out


def build_group_G(g, fixed, level):
    out = []
    i = 0
    for c in g["cases"]:
        patch = {k: v for k, v in c.items() if not k.startswith("_")}
        for dt, ex in itertools.product(g["combinatorics"]["dtypes"],
                                        g["combinatorics"]["expand"]):
            i += 1
            p = dict(fixed)
            p.update(patch)
            p["dtypes"] = dt
            p["expand"] = ex
            out.append(Case("G", i, norm(p), f"{c['_id']}_{dt}_e{ex}", level))
    return out


def build_group_H(g, fixed, level):
    out = []
    base = dict(fixed)
    base.update(g["case"])
    for i in range(g["repeat"]):
        out.append(Case("H", i + 1, norm(dict(base)), f"repeat{i+1:02d}", level))
    return out


BUILDERS = {
    "A": build_group_A, "B": build_group_B, "C": build_group_C,
    "D": build_group_D, "E": build_group_E, "F": build_group_F,
    "G": build_group_G, "H": build_group_H,
}


def build_all(spec, groups=None):
    fixed = spec["fixed"]
    result = {}
    for key, g in spec["groups"].items():
        if groups and key not in groups:
            continue
        level = g.get("level", "性能")
        cases = BUILDERS[key](g, fixed, level)
        result[key] = cases
    return result


def check(spec, built):
    """逐组核对例数，返回 (ok, rows)。"""
    ok = True
    rows = []
    total = 0
    for key in spec["groups"]:
        if key not in built:
            continue
        exp = spec["groups"][key]["_expected"]
        got = len(built[key])
        good = (exp == got)
        ok = ok and good
        rows.append((key, spec["groups"][key]["name"], exp, got, good))
        total += got
    return ok, rows, total


def validate_params(cases):
    """校验生成的参数值是否在 bench 的 choices 范围内。"""
    errs = []
    for c in cases:
        p = c.params
        if "ep-size" in p and p["ep-size"] not in (2, 4, 8, 16, 32, 64, 128, 256):
            errs.append(f"{c.group}{c.idx}: ep-size={p['ep-size']} 非法")
        if "nnodes" in p and p["nnodes"] not in (1, 2, 4, 8, 16, 32):
            errs.append(f"{c.group}{c.idx}: nnodes={p['nnodes']} 非法")
        if "aiv" in p and p["aiv"] not in range(2, 65, 2):
            errs.append(f"{c.group}{c.idx}: aiv={p['aiv']} 非法")
        if "dtypes" in p and p["dtypes"] not in ("bf16", "fp8"):
            errs.append(f"{c.group}{c.idx}: dtypes={p['dtypes']} 非法")
        if "expand" in p and p["expand"] not in (0, 1):
            errs.append(f"{c.group}{c.idx}: expand={p['expand']} 非法")
        if "route-mode" in p and p["route-mode"] not in ("uniform", "rank_hotspot", "local_group"):
            errs.append(f"{c.group}{c.idx}: route-mode={p['route-mode']} 非法")
        if p.get("hidden", 512) % 128 != 0:
            errs.append(f"{c.group}{c.idx}: hidden={p['hidden']} 不是 128 倍数")
        if p.get("num-tokens", 1) < 1:
            errs.append(f"{c.group}{c.idx}: num-tokens 必须 >=1")
    return errs


def write_outputs(built, spec, outdir, api, bench_path, groups_order):
    os.makedirs(outdir, exist_ok=True)
    manifest = ["组,编号,层级,标签,参数,命令"]
    index_lines = ["#!/bin/bash", "set -u", "# 由 gen_cases.py 自动生成", ""]
    for key in groups_order:
        if key not in built:
            continue
        cases = built[key]
        sh = os.path.join(outdir, f"{api}_{key}.sh")
        with open(sh, "w", encoding="utf-8", newline="\n") as f:
            f.write("#!/bin/bash\n")
            f.write("set -u\n")
            f.write(f"# {api.upper()} 组 {key}（{spec['groups'][key]['name']}）：{len(cases)} 例\n")
            f.write("# 用法: bash %s [起始序号]\n" % os.path.basename(sh))
            f.write('START=${1:-1}\n')
            f.write('set -e\n\n')
            for c in cases:
                # 输出根用 POSIX 相对路径，避免 Windows 反斜杠进到 bash 脚本里
                cmd = c.command(bench_path, "runs", api)
                f.write(f'if [ {c.idx} -ge $START ]; then\n')
                f.write(f'  echo "[{key}{c.idx:04d}/{len(cases)}] {c.tag}"\n')
                f.write(f'  {cmd}\n')
                f.write("fi\n")
        os.chmod(sh, 0o755)
        index_lines.append(f'bash "{os.path.basename(sh)}"')
        for c in cases:
            params = " ".join(f"{k}={v}" for k, v in c.params.items())
            manifest.append(f'{key},{c.idx},{c.level},"{c.tag}","{params}","{c.command(bench_path, "runs", api)}"')
    with open(os.path.join(outdir, f"manifest_{api}.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(manifest) + "\n")
    with open(os.path.join(outdir, "index.sh"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(index_lines) + "\n")
    os.chmod(os.path.join(outdir, "index.sh"), 0o755)


def write_summary(rows, total, path, expected_total):
    lines = ["# 配比核对", "",
             "| 组 | 名称 | 期望 | 实际 | 结果 |", "|---|---|---|---|---|"]
    for key, name, exp, got, good in rows:
        lines.append(f"| {key} | {name} | {exp} | {got} | {'OK' if good else '**不一致**'} |")
    lines.append(f"| **合计** | | **{expected_total}** | **{total}** | "
                 f"{'OK' if total == expected_total else '**不一致**'} |")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="生成 DeepEP CLOS 泛化用例 bench 命令行批次")
    ap.add_argument("--spec", default=SPEC_DEFAULT, help="规格文件路径")
    ap.add_argument("--out", default="", help="输出目录（留空则只校验）")
    ap.add_argument("--check", action="store_true", help="只校验配比，不写文件")
    ap.add_argument("--dry-run", action="store_true", help="打印命令，不写文件")
    ap.add_argument("--limit", type=int, default=0, help="dry-run 时只打印前 N 条")
    ap.add_argument("--groups", nargs="+", default=None, help="只生成指定组，如 A B C")
    ap.add_argument("--api", choices=("v1", "v2", "both"), default="both")
    ap.add_argument("--bench-dir", default="benchmarks/clos",
                    help="bench 脚本所在目录（写进命令行，默认相对仓库根）")
    args = ap.parse_args()

    if not os.path.isfile(args.spec):
        print(f"规格文件不存在: {args.spec}")
        return 2

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    built = build_all(spec, args.groups)
    if not built:
        print("没有匹配的组")
        return 2

    # 参数合法性
    all_cases = [c for cs in built.values() for c in cs]
    errs = validate_params(all_cases)
    if errs:
        print("参数校验失败：")
        for e in errs[:20]:
            print("  " + e)
        if len(errs) > 20:
            print(f"  ... 共 {len(errs)} 条")
        return 1

    # 配比核对
    ok, rows, total = check(spec, built)
    expected_total = sum(spec["groups"][k]["_expected"] for k in built)
    print("=" * 72)
    print("配比核对")
    print("=" * 72)
    print("%-4s %-12s %8s %8s %6s" % ("组", "名称", "期望", "实际", "结果"))
    print("-" * 72)
    for key, name, exp, got, good in rows:
        print("%-4s %-12s %8d %8d %6s" % (key, name, exp, got, "OK" if good else "FAIL"))
    print("-" * 72)
    print("%-4s %-12s %8d %8d %6s" % ("合计", "", expected_total, total,
                                      "OK" if total == expected_total else "FAIL"))
    print()

    if not ok:
        print(f"配比与规格不符，请检查 {os.path.basename(args.spec)}")
        return 1

    if args.dry_run:
        n = args.limit or len(all_cases)
        bench = os.path.join(args.bench_dir, "bench_v2_bandwidth.py")
        for c in all_cases[:n]:
            print(f"[{c.group}{c.idx:04d}] {c.tag}")
            print("  " + c.command(bench, "runs", "v2"))
        if n < len(all_cases):
            print(f"... 还有 {len(all_cases) - n} 条")
        return 0

    if args.check or not args.out:
        if not args.out:
            print("（未指定 --out，仅校验）")
        return 0

    apis = ("v1", "v2") if args.api == "both" else (args.api,)
    for api in apis:
        bench = os.path.join(args.bench_dir, f"bench_{api}_bandwidth.py")
        # v1 的 aiv 取法不同，但命令行参数一致
        write_outputs(built, spec, args.out, api, bench, list(spec["groups"].keys()))
        print(f"已生成 {api.upper()}: {args.out}/{api}_*.sh")
    write_summary(rows, total, os.path.join(args.out, "summary.md"), expected_total)
    print(f"已生成核对表: {args.out}/summary.md")
    print(f"已生成明细:   {args.out}/manifest_*.csv")
    print(f"已生成入口:   {args.out}/index.sh")
    print()
    print("执行示例：")
    print(f"  bash {args.out}/v2_A.sh")
    print(f"  bash {args.out}/v2_A.sh 128     # 从第 128 条续跑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
