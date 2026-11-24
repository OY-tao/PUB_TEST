"""复算 DeepEP 带宽基准的字节账本，验证设计文档里的数字。

用途：
  1. 给定 ep/tokens/hidden/topk/dtype/route，算出 6 种字节口径的实际值
  2. 验证「combine 恒为 BF16」「mask 不影响 algorithm_bytes」「热点会撞 slot 上限」
  3. 更换形状时快速估算带宽分母，避免手算出错

依赖：upstream/ 下的 route_workloads.py 与 stage_metrics.py（未改动的上游副本）

用法：
    python verify_ledger.py                      # 默认形状全表
    python verify_ledger.py --tokens 16384 --hidden 8192 --topk 4
    python verify_ledger.py --sweep-topk         # topk 扫描，看 algorithm 与 payload 分离
    python verify_ledger.py --sweep-mask         # mask 扫描，看 algorithm 不变
    python verify_ledger.py --sweep-skew         # 热点扫描，看 slot 饱和
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "upstream"))

import route_workloads as rw                      # noqa: E402
from stage_metrics import byte_ledger, route_counts  # noqa: E402

MB = 1e6


def build_routes(world, per_rank, tokens, topk, seed=20260823, **kw):
    return [rw.generate(r, tokens, world, per_rank, topk, seed=seed, **kw) for r in range(world)]


def show_ledger(world, per_rank, tokens, hidden, topk, mode="uniform", mask=0.0, rank=0):
    kw = dict(route_mode=mode, route_mask_rate=mask)
    routes = build_routes(world, per_rank, tokens, topk, **kw)
    counts = route_counts(routes, per_rank)
    st = rw.statistics(routes, per_rank, 1)

    print("=" * 100)
    print("ep=%d experts/rank=%d tokens=%d hidden=%d topk=%d mode=%s mask=%.1f"
          % (world, per_rank, tokens, hidden, topk, mode, mask))
    print("路由: recv_max_over_mean=%.3f  single_contrib=%.3f  valid_hits=%d"
          % (st["recv_max_over_mean"] or 0,
             st["single_contribution_fraction"] or 0,
             st["valid_expert_hits"]))
    print("-" * 100)
    print("%-6s %-20s %11s %11s %13s %13s %11s"
          % ("dtype", "op", "payload", "application", "official_logic", "algorithm", "valid_algo"))
    print("-" * 100)

    for dtype in ("bf16", "fp8"):
        for op in ("dispatch", "cached-dispatch", "combine_unweighted", "combine_weighted"):
            b = byte_ledger(counts, rank, tokens=tokens, hidden=hidden, topk=topk,
                            dtype=dtype, idx_bytes=8, op=op)
            print("%-6s %-20s %10.1fM %10.1fM %12.1fM %12.1fM %10.1fM"
                  % (dtype, op,
                     b["remote_payload_bytes"] / MB, b["remote_application_bytes"] / MB,
                     b["official_logical_bytes"] / MB, b["algorithm_bytes"] / MB,
                     b["valid_algorithm_bytes"] / MB))
    print()


def sweep_topk(world, per_rank, tokens, hidden):
    print("=" * 100)
    print("topk 扫描：algorithm_bytes 线性增长，remote_payload 跟去重行数走")
    print("-" * 100)
    print("%6s %13s %13s %10s" % ("topk", "algorithm", "remote_payload", "payload/algo"))
    for topk in (1, 2, 4, 8, 16, 32):
        if topk > min(128, world * per_rank):
            continue
        routes = build_routes(world, per_rank, tokens, topk)
        counts = route_counts(routes, per_rank)
        b = byte_ledger(counts, 0, tokens=tokens, hidden=hidden, topk=topk,
                        dtype="bf16", idx_bytes=8, op="dispatch")
        print("%6d %12.1fM %12.1fM %9.1f%%"
              % (topk, b["algorithm_bytes"] / MB, b["remote_payload_bytes"] / MB,
                 100.0 * b["remote_payload_bytes"] / b["algorithm_bytes"]))
    print()


def sweep_mask(world, per_rank, tokens, hidden, topk):
    print("=" * 100)
    print("mask 扫描：algorithm_bytes 恒定，remote_payload 随 mask 线性下降")
    print("（mask 用例用 algorithm_bytes 算带宽会虚高）")
    print("-" * 100)
    print("%10s %13s %13s %13s %11s" % ("mask", "algorithm", "valid_algo", "remote_payload", "真实占比"))
    for mask in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
        routes = build_routes(world, per_rank, tokens, topk, route_mask_rate=mask)
        counts = route_counts(routes, per_rank)
        b = byte_ledger(counts, 0, tokens=tokens, hidden=hidden, topk=topk,
                        dtype="bf16", idx_bytes=8, op="dispatch")
        print("%10.1f %12.1fM %12.1fM %12.1fM %10.1f%%"
              % (mask, b["algorithm_bytes"] / MB, b["valid_algorithm_bytes"] / MB,
                 b["remote_payload_bytes"] / MB,
                 100.0 * b["remote_payload_bytes"] / b["algorithm_bytes"]))
    print()


def sweep_skew(world, per_rank, tokens, hidden, topk):
    print("=" * 100)
    print("热点扫描：热点 rank 收行在 skew>=8 时撞到 tokens*topk 上限")
    print("单 rank slot 上限 = tokens*topk = %d" % (tokens * topk))
    print("-" * 100)
    print("%6s %12s %14s %14s" % ("skew", "max/mean", "hot_recv", "cold_recv_avg"))
    for skew in (1.0, 1.5, 2, 4, 6, 8, 16, 32):
        try:
            routes = build_routes(world, per_rank, tokens, topk,
                                  route_mode="rank_hotspot", route_skew=skew, route_hotspot=0)
            st = rw.statistics(routes, per_rank, 1)
            recv = st["recv_rows"]
            print("%6.1f %12.3f %14d %14.0f"
                  % (skew, st["recv_max_over_mean"],
                     max(recv), sum(recv[1:]) / max(len(recv) - 1, 1)))
        except Exception as e:
            print("%6.1f  FAIL %s" % (skew, str(e)[:70]))
    print()
    print("注：skew=1.0 走 uniform 分支，不是热点。")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-size", type=int, default=8)
    ap.add_argument("--experts-per-rank", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--sweep-topk", action="store_true")
    ap.add_argument("--sweep-mask", action="store_true")
    ap.add_argument("--sweep-skew", action="store_true")
    ap.add_argument("--all", action="store_true", help="跑全部扫描")
    args = ap.parse_args()

    w, p, t, h, k = (args.ep_size, args.experts_per_rank,
                     args.tokens, args.hidden, args.topk)

    if not (args.sweep_topk or args.sweep_mask or args.sweep_skew or args.all):
        show_ledger(w, p, t, h, k)
        return 0

    if args.sweep_topk or args.all:
        sweep_topk(w, p, t, h)
    if args.sweep_mask or args.all:
        sweep_mask(w, p, t, h, k)
    if args.sweep_skew or args.all:
        sweep_skew(w, p, t, h, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
