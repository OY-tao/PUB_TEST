"""从 V1 bench 结果里提取 layout / expand / aggregate 三个阶段耗时。

背景：上游 bench_v1_bandwidth.py 只输出 4 个 op（dispatch / cached-dispatch /
combine_unweighted / combine_weighted），但 V1 的 5 个通信算子里
get_dispatch_layout、expand、aggregate 的耗时其实已经算进了 samples，
只是 summarize() 没单独成列。

本脚本零侵入地补出这三项（对应设计文档的 V1-2 / V1-4 / V1-5）。

用法：
    python extract_v1_stages.py ./run_v1_ep8
    python extract_v1_stages.py ./run_v1_ep8 --csv stages.csv
    python extract_v1_stages.py ./run_v1_ep8 --json stages.json

产物：
    - 终端表格：每 rank 每阶段的 median / min / max / 占 device_sum 比例
    - job 级汇总：max-rank 锁步（与 stage_metrics.summarize 同口径）
    - 可选 CSV / JSON
"""

import argparse
import json
import os
import statistics
import sys

# samples 里可能出现的时间字段 -> 对应的被测项
STAGE_FIELDS = [
    ("layout_us", "V1-2 get_dispatch_layout"),
    ("notify_us", "notify（非被测项，参考）"),
    ("comm_us", "V1-3 dispatch（communication）"),
    ("expand_us", "V1-4 expand"),
    ("aggregate_us", "V1-5 aggregate"),
    ("reduce_us", "reduce（V1 为 null）"),
    ("fused_combine_us", "V1-6 combine（fused）"),
    ("other_us", "other"),
    ("device_sum_us", "device_sum（参照）"),
    ("device_span_us", "device_span（参照）"),
]

# 每个阶段在哪个 op 的样本里
STAGE_OWNER = {
    "layout_us": "dispatch",
    "expand_us": "dispatch",
    "aggregate_us": "combine_unweighted",
    "fused_combine_us": "combine_unweighted",
    "comm_us": "dispatch",
}


def load_ranks(root):
    """读 root 下所有 rankN/result.json。

    支持两种目录形状：
      1. <root>/rank0/result.json                  —— 单个 case 目录
      2. <root>/<case_name>/rank0/result.json       —— bench 的 --output 根目录
         （bench_v2_bandwidth.py:1192 `path = args.output / case["name"]`）
    多 case 时按 case 分组返回，避免不同 case 的样本混在一起。
    """
    def scan(base):
        found = []
        for name in sorted(os.listdir(base)):
            if not name.startswith("rank"):
                continue
            p = os.path.join(base, name, "result.json")
            if not os.path.isfile(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    found.append((name, json.load(f)))
            except Exception as e:
                print("[warn] 读取 %s 失败: %s" % (p, e))
        return found

    groups = {}

    # 形状 1：根目录直接有 rankN
    direct = scan(root)
    if direct:
        groups["(root)"] = direct

    # 形状 2：每个 case 子目录
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub) or name.startswith("rank"):
            continue
        got = scan(sub)
        if got:
            groups[name] = got

    return groups


def collect(ranks, op_filter=None):
    """返回 {op: {stage: {rank: [values]}}}"""
    table = {}
    for rank_name, data in ranks:
        for op, payload in (data.get("operators") or {}).items():
            if op_filter and op != op_filter:
                continue
            samples = ((payload or {}).get("profile") or {}).get("samples") or []
            if not samples:
                continue
            for stage, _label in STAGE_FIELDS:
                vals = [s.get(stage) for s in samples if s.get(stage) is not None]
                # 全 0 说明该阶段在这个 op 里不存在（如 cached-dispatch 无 layout），跳过
                if not vals or all(float(v) == 0 for v in vals):
                    continue
                table.setdefault(op, {}).setdefault(stage, {})[rank_name] = [
                    float(v) for v in vals
                ]
    return table


def fmt(v, width=10, prec=2):
    if v is None:
        return "-".rjust(width)
    return ("%.*f" % (prec, v)).rjust(width)


def report_rank_level(table):
    print("=" * 104)
    print("按 rank 统计（每个阶段在各 rank 的 median / min / max / 占 device_sum 比例）")
    print("=" * 104)

    for op in sorted(table):
        print("\n### op = %s" % op)
        print("%-32s %-10s %10s %10s %10s %10s"
              % ("阶段", "rank", "median_us", "min_us", "max_us", "占sum%"))
        print("-" * 104)

        dsum_by_rank = {}
        for rk, vals in (table[op].get("device_sum_us") or {}).items():
            dsum_by_rank[rk] = statistics.median(vals)

        for stage, label in STAGE_FIELDS:
            per_rank = table[op].get(stage)
            if not per_rank:
                continue
            for rk in sorted(per_rank, key=lambda x: int(x.replace("rank", "") or 0)):
                vals = per_rank[rk]
                med = statistics.median(vals)
                ds = dsum_by_rank.get(rk)
                pct = (med / ds * 100) if ds else None
                print("%-32s %-10s %10.2f %10.2f %10.2f %10s"
                      % (label[:32], rk, med, min(vals), max(vals),
                         "%.1f" % pct if pct is not None else "-"))
        print()


def report_job_level(table):
    """job 级：每阶段取各 rank 的 max，再求中位（与 summarize 的 max-rank 锁步一致）。"""
    print("=" * 104)
    print("job 级汇总（max-rank 锁步：先取每样本各 rank 最大值，再求平均）")
    print("=" * 104)
    print("%-32s %-20s %12s %12s" % ("阶段", "所属 op", "job_median_us", "样本数"))
    print("-" * 104)

    rows = []
    for stage, label in STAGE_FIELDS:
        # 找到该 stage 出现在哪个 op
        owners = [op for op in table if table[op].get(stage)]
        if not owners:
            continue
        op = owners[0]
        per_rank = table[op][stage]
        n = min(len(v) for v in per_rank.values())
        if n == 0:
            continue
        lockstep = []
        for i in range(n):
            lockstep.append(max(per_rank[rk][i] for rk in per_rank))
        job_med = statistics.median(lockstep)
        print("%-32s %-20s %12.2f %12d" % (label[:32], op, job_med, n))
        rows.append(dict(stage=stage, label=label, op=op, job_us=job_med, samples=n))
    print()
    return rows


def report_coverage(table):
    """检查 V1 的 5 个通信算子是否真的取到数据。

    V1 共 7 项，其中 2 项（Buffer 构造、destroy）是生命周期，不在这里检查。
    """
    print("=" * 104)
    print("V1 通信算子覆盖检查（7 项里的 5 个通信/元数据算子）")
    print("=" * 104)

    checks = [
        ("V1-2 get_dispatch_layout", "dispatch", "layout_us"),
        ("V1-3 dispatch", "dispatch", "comm_us"),
        ("V1-4 expand", "dispatch", "expand_us"),
        ("V1-5 aggregate", "combine_unweighted", "aggregate_us"),
        ("V1-6 combine", "combine_unweighted", "fused_combine_us"),
    ]
    ok = 0
    for name, op, stage in checks:
        has = bool(table.get(op, {}).get(stage))
        print("  %-30s %-8s -> %s" % (name, "有数据" if has else "缺数据",
                                      "OK" if has else "MISSING"))
        if has:
            ok += 1
    print("  %-30s %-8s -> 生命周期项，用 lifecycle_test.py 测"
          % ("V1-1 Buffer / V1-7 destroy", "N/A"))
    print("\n  通信算子覆盖 %d/5" % ok)
    if ok < 5:
        print("  提示：expand/aggregate 需 --expand 1；若缺失请确认该 case 走了 expanded 路径。")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="bench 输出目录（含 rankN/result.json）")
    ap.add_argument("--csv", default="", help="导出 job 级汇总到 CSV")
    ap.add_argument("--json", default="", help="导出完整明细到 JSON")
    ap.add_argument("--op", default="", help="只看某个 op")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print("目录不存在: %s" % args.root)
        return 1

    groups = load_ranks(args.root)
    if not groups:
        print("在 %s 下没找到 rankN/result.json" % args.root)
        print("提示：确认这是 bench 的 --output 目录（含 <case_name>/rankN/ 子目录），")
        print("      且 case 已跑完（有 result.json 而不是只有 process.json）")
        return 1

    total_ranks = sum(len(v) for v in groups.values())
    print("发现 %d 个 case，共 %d 个 rank：\n" % (len(groups), total_ranks))
    for name, ranks in groups.items():
        print("  %-32s %d 个 rank: %s"
              % (name, len(ranks), ", ".join(r[0] for r in ranks)))
    print()

    all_rows = []
    for case_name, ranks in groups.items():
        print("\n" + "#" * 104)
        print("# case = %s" % case_name)
        print("#" * 104)

        table = collect(ranks, op_filter=args.op or None)
        if not table:
            print("  samples 为空，可能 case 未跑完或 profiler 解析失败，跳过")
            continue

        report_rank_level(table)
        rows = report_job_level(table)
        report_coverage(table)
        for r in rows:
            r["case"] = case_name
        all_rows.extend(rows)

    if args.csv and all_rows:
        import csv
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "stage", "label", "op", "job_us", "samples"])
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print("CSV -> %s" % args.csv)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(job_level=all_rows,
                           cases={k: len(v) for k, v in groups.items()}),
                      f, ensure_ascii=False, indent=2)
        print("JSON -> %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
