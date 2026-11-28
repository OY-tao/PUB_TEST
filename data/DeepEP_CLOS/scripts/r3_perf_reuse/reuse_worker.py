#!/usr/bin/env python3
"""R3 性能补测「复用统一 Buffer」版持久 worker（容器内运行，8 机 torchrun 拉起）。

与官方 `bench_v2_bandwidth.launch()` 的区别：
  * 官方：每 case 起独立 torchrun 进程组 -> worker() 内 init_process_group +
    构造 ElasticBuffer -> 测量 -> buffer.destroy() + destroy_process_group。
  * 本 worker：**一个持久进程组**，按 (hidden, num_topk, dtype, huge_pages)
    分组复用 ElasticBuffer（容量取组内 max tokens）；组内每个 case 仍调用官方
    `worker()` 的测量逻辑（profiling / warmup / iterations / 精度校验），
    只把 Buffer 构造与进程组生命周期提到组级。

口径保持：official_logical_GBps = official_logical_bytes / comm_us / 1000，
comm_us 来自官方 profiler（samples 均值）；每 case 4 op。

monkeypatch（仅本进程内生效，不改任何源码文件）：
  1. dist.init_process_group / destroy_process_group -> no-op（持久组只建一次）；
  2. deep_ep.backends.global_clos.ElasticBuffer -> 按组缓存实例的工厂
     （官方 worker() 里的函数内 import 会拿到这个工厂）；
  3. 工厂返回 _NoDestroy 代理，吞掉官方 worker() 末尾的 buffer.destroy()，
     真实 Buffer 由组末统一 destroy。

rank0 每 case 用 dist.all_gather_object 汇总各 rank 的 GBps，写
<output>/summary_reuse.csv（逐 case 逐 op 的 rank mean/min/max + 复用标记）。
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pathlib
import time
import traceback

import bench_v2_bandwidth as B


class _NoDestroy:
    """吞掉官方 worker() 末尾的 buffer.destroy()，真实 destroy 由组末统一做。"""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def destroy(self):
        return None


class _BufferPool:
    """按 (hidden, num_topk, dtype, huge_pages) 分组复用；组边界 destroy_all。

    total_initializations 跨组累计，用于对比「复用后共初始化几次 Buffer」。
    """

    def __init__(self, real_cls, max_tokens, world_group):
        self._real_cls = real_cls
        self.max_tokens = max_tokens
        self.world_group = world_group
        self.pool = {}
        self.total_initializations = 0

    @staticmethod
    def key_of(kwargs):
        return (int(kwargs["hidden"]), int(kwargs["num_topk"]),
                bool(kwargs.get("use_fp8_dispatch", False)),
                bool(kwargs.get("use_1g_huge_pages", False)))

    def factory(self, *args, **kwargs):
        key = self.key_of(kwargs)
        if key not in self.pool:
            kw = dict(kwargs)
            kw["num_max_tokens_per_rank"] = self.max_tokens
            self.pool[key] = self._real_cls(self.world_group, **kw)
            self.total_initializations += 1
        return _NoDestroy(self.pool[key])

    def destroy_all(self):
        for buf in self.pool.values():
            try:
                buf.destroy()
            except Exception:
                pass
        self.pool.clear()


def patch_runtime(pool, dist, gc):
    orig = (gc.ElasticBuffer, dist.init_process_group, dist.destroy_process_group)
    dist.init_process_group = lambda *a, **k: None
    dist.destroy_process_group = lambda *a, **k: None
    gc.ElasticBuffer = pool.factory
    return orig


def restore_runtime(dist, gc, orig):
    gc.ElasticBuffer, dist.init_process_group, dist.destroy_process_group = orig


def build_argv(case, out_root, run_id, ep, nnodes, node_rank, master_addr, master_port):
    p = case["params"]
    return [
        "--ep-size=%d" % ep, "--nnodes=%d" % nnodes,
        "--node-rank=%d" % node_rank, "--first-device=0",
        "--master-addr=" + master_addr, "--master-port=%d" % master_port,
        "--run-id=" + run_id, "--output=" + out_root + "/" + case["variant"],
        "--num-tokens=%d" % p["num_tokens"], "--hidden=%d" % p["hidden"],
        "--num-topk=%d" % p["num_topk"], "--experts-per-rank=%d" % p["experts_per_rank"],
        "--dtypes=" + case["dtype"], "--expand=%d" % case["expand"],
        "--aiv=%d" % p["aiv"], "--warmup=%d" % p["warmup"],
        "--iterations=%d" % p["iterations"], "--seed=%d" % p["seed"],
        "--route-mode=" + p["route_mode"],
    ]


def gbps_of_result(res):
    """按官方 summarize() 的 per-rank 公式算 4 个 op 的 official_logical_GBps。

    comm_us = mean(profile.samples[i]['comm_us'])；
    official_logical_GBps = official_logical_bytes / comm_us / 1000。
    """
    out = {}
    for op, data in res["operators"].items():
        samples = data["profile"]["samples"]
        comm_us = sum(float(s["comm_us"]) for s in samples) / len(samples)
        official_bytes = float(data["bytes"]["official_logical_bytes"])
        out[op] = official_bytes / comm_us / 1000.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--ep-size", type=int, required=True)
    # 名字与 torchrun 的 --node-rank 区分，避免被上层解析器消费
    ap.add_argument("--worker-node-rank", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--master-addr", default="30.104.128.4")
    ap.add_argument("--master-port", type=int, default=29870)
    o = ap.parse_args()
    node_rank = o.worker_node_rank

    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    import deep_ep.backends.global_clos as gc

    plan = json.loads(o.plan.read_text(encoding="utf-8"))
    cases = plan["cases"]
    ep = o.ep_size
    nnodes = plan["nnodes"]
    per_node = ep // nnodes

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != ep or int(os.environ["LOCAL_WORLD_SIZE"]) != per_node:
        raise RuntimeError("world size mismatch: ep=%d nnodes=%d" % (ep, nnodes))
    if rank != node_rank * per_node + local_rank:
        raise RuntimeError("rank mapping mismatch")

    torch.set_num_threads(1)
    torch.npu.set_device(local_rank)
    o.output.mkdir(parents=True, exist_ok=True)

    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=900))
    dist.barrier()

    pool = _BufferPool(gc.ElasticBuffer, plan["max_tokens"], dist.group.WORLD)
    orig = patch_runtime(pool, dist, gc)
    summary_path = o.output / "summary_reuse.csv"
    rows = []
    started = time.time()
    record = {"rank": rank, "ep_size": ep, "node_rank": node_rank,
              "process_group_initializations": 1, "buffer_initializations": 0,
              "cases": [], "state": "running"}
    try:
        prev_key = None
        for case in cases:
            variant = case["variant"]
            key = (int(case["params"]["hidden"]), int(case["params"]["num_topk"]),
                   case["dtype"] == "fp8", False)
            # 组边界：上一个 Buffer 组统一 destroy（所有 rank 同步），避免多组
            # Buffer 同时常驻撑爆 HBM，也与精度批次的 compatibility_key 语义一致。
            if prev_key is not None and key != prev_key:
                dist.barrier()
                pool.destroy_all()
                dist.barrier()
            prev_key = key
            argv = build_argv(case, str(o.output), o.run_id, ep, nnodes,
                              node_rank, o.master_addr, o.master_port)
            args = B.arguments(argv)
            before = pool.total_initializations
            t0 = time.time()
            B.worker(args)
            dt = time.time() - t0
            reused = pool.total_initializations == before
            # 本 rank 的 4 op GBps
            res = json.loads((o.output / variant / ("rank%d" % rank) / "result.json").read_text())
            mine = gbps_of_result(res)
            gathered = [None] * ep
            dist.all_gather_object(gathered, mine)
            dist.barrier()
            if rank == 0:
                for op in mine:
                    vals = [g[op] for g in gathered]
                    rows.append({
                        "variant": variant, "ep_size": ep,
                        "hidden": case["params"]["hidden"],
                        "num_topk": case["params"]["num_topk"],
                        "dtype": case["dtype"], "aiv": case["params"]["aiv"],
                        "num_tokens": case["params"]["num_tokens"],
                        "op": op,
                        "gbps_mean": round(sum(vals) / len(vals), 3),
                        "gbps_min": round(min(vals), 3),
                        "gbps_max": round(max(vals), 3),
                        "buffer_reused": "true" if reused else "false",
                        "buffer_initializations_so_far": pool.total_initializations,
                        "elapsed_s": round(dt, 2),
                    })
                with summary_path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
                    w.writeheader()
                    w.writerows(rows)
            record["cases"].append({"variant": variant, "reused": reused,
                                    "elapsed_s": round(dt, 2)})
            print("REUSE_CASE rank=%d variant=%s reused=%s %.1fs" %
                  (rank, variant, reused, dt), flush=True)
        dist.barrier()
        pool.destroy_all()
        record["buffer_initializations"] = pool.total_initializations
        record["state"] = "passed"
        record["total_s"] = round(time.time() - started, 2)
        dist.barrier()
    except BaseException:
        record["state"] = "failed"
        record["traceback"] = traceback.format_exc()
        pool.destroy_all()
        raise
    finally:
        restore_runtime(dist, gc, orig)
        (o.output / ("reuse_rank%d.json" % rank)).write_text(
            json.dumps(record, indent=2) + "\n")
    print("REUSE_PASSED rank=%d buffer_inits=%d cases=%d" %
          (rank, pool.total_initializations, len(cases)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
