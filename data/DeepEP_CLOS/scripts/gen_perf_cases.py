"""按 perf_cases_v2.json 生成 DeepEP CLOS V2 性能用例的 bench 命令行。

参照 OY-tao/Ascned_Test 的 `V2测试Case.txt`：那张表是 EP8~128 的带宽结果汇总
（60 行 × 5 个 op），本脚本把它的形状维度抽成可执行的 bench 命令。

与 gen_cases.py 的分工
----------------------
gen_cases.py      覆盖 1148 例泛化矩阵（8 个组，含边界/路由/AIV/稳定性）
gen_perf_cases.py 只做性能主线：EP × tokens × hidden × AIV，60 例
                  形状与参考表严格一致，便于和历史数据逐行对比

用法
----
    # 校验 + 打印配比
    python gen_perf_cases.py --check

    # 生成全部（按 EP 分组，每组一个脚本）
    python gen_perf_cases.py --out perf_batches/

    # 只生成某个 EP
    python gen_perf_cases.py --out perf_batches/ --ep 8 16

    # 看命令
    python gen_perf_cases.py --dry-run --limit 4

    # 导入参考表，核对形状是否一致
    python gen_perf_cases.py --verify-against /path/to/V2测试Case.txt
"""

import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DEFAULT = os.path.join(HERE, "perf_cases_v2.json")

VALID_EP = (2, 4, 8, 16, 32, 64, 128, 256)
VALID_AIV = tuple(range(2, 65, 2))

# 参考表的 5 个 op，与 bench 的 4 个 op 对应关系
OP_MAP = [
    ("Dispatch 无权重", "dispatch", "topk_weights=None"),
    ("Dispatch 带权重", "dispatch", "topk_weights 非空"),
    ("Cached dispatch", "cached-dispatch", "复用 handle"),
    ("Combine 无权重", "combine_unweighted", "topk_weights=None"),
    ("Combine 带权重", "combine_weighted", "topk_weights 非空"),
]


class PerfCase:
    __slots__ = ("ep", "idx", "params", "tag")

    def __init__(self, ep, idx, params, tag):
        self.ep = ep
        self.idx = idx
        self.params = params
        self.tag = tag

    @property
    def needs_nodes(self):
        return self.params.get("nnodes", 1) > 1

    def command(self, bench, output_root, run_id):
        """生成单机命令；多机时输出带节点占位符的模板。"""
        args = []
        nnodes = self.params.get("nnodes", 1)
        if nnodes > 1:
            args.append("--node-rank=<NODE_RANK>")
        for key in ("ep-size", "nnodes", "num-tokens", "hidden", "num-topk",
                    "experts-per-rank", "dtypes", "expand", "aiv",
                    "warmup", "iterations", "seed", "route-mode"):
            if key not in self.params:
                continue
            val = self.params[key]
            args.append(f"--{key}={val}")
        if nnodes > 1:
            args.append("--master-addr=<MASTER_ADDR>")
            args.append("--master-port=29500")
        args.append(f"--run-id={run_id}")
        out = output_root.rstrip("/") + "/" + run_id
        args.append(f"--output={out}")
        bench_posix = bench.replace("\\", "/")
        return f"python {bench_posix} " + " ".join(args)


def build(spec):
    """展开 EP × tokens × hidden × AIV，Experts 由 experts_map 决定。"""
    fixed = spec["fixed"]
    ax = spec["axes"]
    experts = spec["experts_map"]
    topo = spec["topology"]

    out = {}
    for ep in ax["ep_size"]:
        cases = []
        i = 0
        for tokens, hidden, aiv in itertools.product(
            ax["num_tokens"], ax["hidden"], ax["aiv"]
        ):
            i += 1
            # fixed 里的键用 snake_case（与 gen_cases.py 的规格风格一致），
            # 这里统一转成 bench 的 kebab-case 参数名
            p = {k.replace("_", "-"): v for k, v in fixed.items()}
            p["ep-size"] = ep
            p["num-tokens"] = tokens
            p["hidden"] = hidden
            p["aiv"] = aiv
            # Experts 是全局总数，转成 per_rank
            tot = experts[str(ep)]
            if tot % ep:
                raise ValueError(f"EP={ep} 的 Experts={tot} 不能被 EP 整除")
            p["experts-per-rank"] = tot // ep
            p["dtypes"] = fixed["dtypes"][0]      # 单值，bench 接受 nargs=+
            p["expand"] = fixed["expand"][0]
            # 拓扑
            if str(ep) in topo:
                p["nnodes"] = topo[str(ep)]["nnodes"]
            tag = f"ep{ep}_t{tokens}_h{hidden}_aiv{aiv}"
            cases.append(PerfCase(ep, i, p, tag))
        out[ep] = cases
    return out


def check(spec, built):
    rows = []
    total = 0
    ok = True
    for ep, cases in built.items():
        exp = len(spec["axes"]["num_tokens"]) * len(spec["axes"]["hidden"]) * len(spec["axes"]["aiv"])
        got = len(cases)
        good = exp == got
        ok = ok and good
        rows.append((ep, exp, got, good))
        total += got
    exp_total = spec["_expected"]
    ok = ok and total == exp_total
    return ok, rows, total, exp_total


def verify_against(spec, path):
    """解析参考表，核对形状是否与本规格完全一致。"""
    shapes = set()
    with open(path, encoding="utf-8-sig") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if len(parts) != 10:
                continue
            try:
                ep, tokens, hidden, aiv, experts = (int(x) for x in parts[:5])
                float(parts[5])
            except ValueError:
                continue
            shapes.add((ep, tokens, hidden, aiv, experts))

    ax = spec["axes"]
    experts = spec["experts_map"]
    mine = set()
    for ep, tokens, hidden, aiv in itertools.product(
        ax["ep_size"], ax["num_tokens"], ax["hidden"], ax["aiv"]
    ):
        mine.add((ep, tokens, hidden, aiv, experts[str(ep)]))

    print("=" * 72)
    print("与参考表核对")
    print("=" * 72)
    print(f"参考表形状数: {len(shapes)}")
    print(f"本规格形状数: {len(mine)}")
    only_ref = shapes - mine
    only_mine = mine - shapes
    if only_ref:
        print(f"\n仅参考表有 ({len(only_ref)}):")
        for s in sorted(only_ref)[:10]:
            print(f"  EP={s[0]} t={s[1]} h={s[2]} aiv={s[3]} experts={s[4]}")
    if only_mine:
        print(f"\n仅本规格有 ({len(only_mine)}):")
        for s in sorted(only_mine)[:10]:
            print(f"  EP={s[0]} t={s[1]} h={s[2]} aiv={s[3]} experts={s[4]}")
    if not only_ref and not only_mine:
        print("\n完全一致：60 个形状逐项匹配")
        return True
    return False


def validate(built):
    errs = []
    for ep, cases in built.items():
        for c in cases:
            p = c.params
            if p["ep-size"] not in VALID_EP:
                errs.append(f"{c.tag}: ep-size 非法")
            if p["aiv"] not in VALID_AIV:
                errs.append(f"{c.tag}: aiv={p['aiv']} 非法")
            if p["hidden"] % 128 != 0:
                errs.append(f"{c.tag}: hidden={p['hidden']} 不是 128 倍数")
            # bench 校验: num_topk <= min(128, ep*epr)
            cap = min(128, p["ep-size"] * p["experts-per-rank"])
            if p["num-topk"] > cap:
                errs.append(f"{c.tag}: num-topk={p['num-topk']} > 上限 {cap}")
            # 拓扑约束
            nn = p.get("nnodes", 1)
            if p["ep-size"] % nn:
                errs.append(f"{c.tag}: ep_size %% nnodes != 0")
            local = p["ep-size"] // nn
            if not 1 <= local <= 16:
                errs.append(f"{c.tag}: 每节点 {local} 卡，超出 1..16")
    return errs


def write_outputs(spec, built, outdir, bench):
    os.makedirs(outdir, exist_ok=True)
    manifest = ["EP,编号,标签,拓扑,参数,命令"]
    index = ["#!/bin/bash", "set -u", "# 由 gen_perf_cases.py 自动生成", ""]
    for ep in spec["axes"]["ep_size"]:
        cases = built[ep]
        nn = cases[0].params.get("nnodes", 1)
        sh = os.path.join(outdir, f"perf_ep{ep}.sh")
        with open(sh, "w", encoding="utf-8", newline="\n") as f:
            f.write("#!/bin/bash\nset -u\n")
            f.write(f"# EP={ep}，{len(cases)} 例，{nn} 机 × {ep // nn} 卡\n")
            if nn > 1:
                f.write("# ⚠️ 多机：每台都要跑本脚本，用环境变量区分节点\n")
                f.write("#   NODE_RANK=0 bash perf_ep32.sh   (node 0)\n")
                f.write("#   NODE_RANK=1 bash perf_ep32.sh   (node 1)\n")
                f.write("#   MASTER_ADDR 默认 30.104.128.4\n")
                f.write('NODE_RANK=${NODE_RANK:-0}\n')
                f.write('MASTER_ADDR=${MASTER_ADDR:-30.104.128.4}\n')
            f.write("START=${1:-1}\nset -e\n\n")
            for c in cases:
                cmd = c.command(bench, "runs", f"perf_ep{ep}_{c.idx:03d}")
                if nn > 1:
                    cmd = cmd.replace("<NODE_RANK>", "${NODE_RANK}").replace(
                        "<MASTER_ADDR>", "${MASTER_ADDR}")
                f.write(f'if [ {c.idx} -ge $START ]; then\n')
                f.write(f'  echo "[ep{ep} {c.idx:02d}/{len(cases)}] {c.tag}"\n')
                f.write(f'  {cmd}\n')
                f.write("fi\n")
        os.chmod(sh, 0o755)
        index.append(f'bash "perf_ep{ep}.sh"')
        for c in cases:
            params = " ".join(f"{k}={v}" for k, v in c.params.items())
            topo = f"{c.params.get('nnodes',1)}机x{c.params['ep-size']//c.params.get('nnodes',1)}卡"
            manifest.append(f'{ep},{c.idx},"{c.tag}",{topo},"{params}","{c.command(bench, "runs", f"perf_ep{ep}_{c.idx:03d}")}"')
    with open(os.path.join(outdir, "manifest.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(manifest) + "\n")
    with open(os.path.join(outdir, "index.sh"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(index) + "\n")
    os.chmod(os.path.join(outdir, "index.sh"), 0o755)


def main():
    ap = argparse.ArgumentParser(description="生成 DeepEP CLOS V2 性能用例 bench 命令")
    ap.add_argument("--spec", default=SPEC_DEFAULT)
    ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ep", nargs="+", type=int, default=None, help="只生成指定 EP")
    ap.add_argument("--bench-dir", default="benchmarks/clos")
    ap.add_argument("--verify-against", default="", help="参考表路径，核对形状一致性")
    args = ap.parse_args()

    if not os.path.isfile(args.spec):
        print(f"规格文件不存在: {args.spec}")
        return 2
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    built = build(spec)
    if args.ep:
        built = {k: v for k, v in built.items() if k in args.ep}

    # 参数校验
    errs = validate(built)
    if errs:
        print("参数校验失败：")
        for e in errs[:20]:
            print("  " + e)
        return 1

    # 形状核对
    all_cases = [c for cs in built.values() for c in cs]
    print("=" * 72)
    print("性能用例配比核对")
    print("=" * 72)
    print("%-6s %8s %8s %6s %-14s" % ("EP", "期望", "实际", "结果", "拓扑"))
    print("-" * 72)
    total = 0
    for ep in sorted(built):
        cases = built[ep]
        nn = cases[0].params.get("nnodes", 1)
        exp = len(spec["axes"]["num_tokens"]) * len(spec["axes"]["hidden"]) * len(spec["axes"]["aiv"])
        print("%-6d %8d %8d %6s %-14s" % (
            ep, exp, len(cases), "OK" if exp == len(cases) else "FAIL",
            f"{nn}机x{ep//nn}卡"))
        total += len(cases)
    print("-" * 72)
    print("%-6s %8s %8d %6s" % ("合计", spec["_expected"], total,
                                "OK" if total == spec["_expected"] else "FAIL"))
    print()

    # 与参考表核对
    if args.verify_against:
        if not os.path.isfile(args.verify_against):
            print(f"参考表不存在: {args.verify_against}")
            return 2
        if not verify_against(spec, args.verify_against):
            return 1

    if args.dry_run:
        n = args.limit or len(all_cases)
        bench = os.path.join(args.bench_dir, "bench_v2_bandwidth.py")
        for c in all_cases[:n]:
            print(f"[{c.tag}]")
            print("  " + c.command(bench, "runs", f"perf_ep{c.ep}_{c.idx:03d}"))
        if n < len(all_cases):
            print(f"... 还有 {len(all_cases) - n} 条")
        return 0

    if args.check or not args.out:
        if not args.out:
            print("（未指定 --out，仅校验）")
        return 0

    bench = os.path.join(args.bench_dir, "bench_v2_bandwidth.py")
    write_outputs(spec, built, args.out, bench)
    print(f"已生成: {args.out}/perf_ep*.sh")
    print(f"已生成: {args.out}/manifest.csv")
    print(f"已生成: {args.out}/index.sh")
    print()
    print("单机执行：")
    print(f"  bash {args.out}/perf_ep8.sh")
    print("多机执行（如 EP=32）：")
    print(f"  NODE_RANK=0 bash {args.out}/perf_ep32.sh   # 在 30.104.128.4")
    print(f"  NODE_RANK=1 bash {args.out}/perf_ep32.sh   # 在 30.104.128.6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
