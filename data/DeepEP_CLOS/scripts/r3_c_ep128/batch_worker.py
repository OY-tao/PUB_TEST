import argparse
import gc
import json
import os
import pathlib
import time
import traceback


# 8 机 128 rank 必须显式传 shmem_ip_port。shmem_ip_port=None 走 UID bootstrap
# (_prepare_shmem_bootstrap -> rank0 get_shmem_unique_id + gloo all_gather)，
# 在 128 rank 下 aclshmemx_init_attr(...) 返回 status=-7。今早在本容器成功跑过
# 128 rank 的 bench(bench_fast_sweep/dispatch_stages/rank_sweep) 都传
# shmem_ip_port=f"tcp://{MASTER_ADDR}:{shmem_port}"，此时 bootstrap 直接走 TCP
# endpoint。端口 8 机保持一致，避开 29910/29911(本任务)、26620/26740(别人的任务)。
SHMEM_PORT = 29920


def shmem_endpoint():
    return "tcp://%s:%d" % (os.environ.get("MASTER_ADDR", "30.104.128.4"), SHMEM_PORT)


def compatibility_key(case):
    # EP32-calibrated grouping: V1 shares one Buffer across all cases; V2 grouped
    # by hidden/topk/dtype (FP8 mode and hidden/topk are immutable constructor args).
    return tuple(case["buffer_config"]["key"])


def make_args(case, node_rank, output, run_id):
    import bench_v2_precision as shared
    p = case["parameters"]
    argv = ["--ep-size=128", "--nnodes=8", "--node-rank=" + str(node_rank),
            "--first-device=0", "--master-addr=30.104.128.4", "--master-port=29910",
            "--run-id=" + run_id, "--output=" + str(output),
            "--dtypes=" + case["dtype"], "--expand=" + str(case["expand"]), "--precision-only"]
    for key, value in p.items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key.replace("_", "-"))
        else:
            argv.append("--" + key.replace("_", "-") + "=" + str(value))
    return shared.arguments(argv)


def execute(cases, output, node_rank, run_id, torch, dist, factories, checkers):
    rank = int(os.environ["RANK"])
    buffer = None
    key = None
    initialization = 0
    pending = []
    retained_check = None
    reuse_log = []

    def close_group():
        nonlocal buffer, pending, retained_check
        torch.npu.synchronize()
        dist.barrier()
        buffer.destroy()
        torch.npu.synchronize()
        if retained_check is not None:
            retained_check()
        retained_check = None
        for variant in pending:
            path = output / variant / ("rank" + str(rank)) / "result.json"
            data = json.loads(path.read_text())
            data["destroyed"] = True
            data["teardown_scope"] = "buffer_group"
            path.write_text(json.dumps(data, indent=2))
        reuse_log.append({"event": "destroy", "initialization": initialization, "variants": pending[:]})
        print("BUFFER_DESTROYED rank=" + str(rank) + " initialization=" + str(initialization), flush=True)
        buffer = None
        pending = []
        gc.collect()
        dist.barrier()

    for case in cases:
        new_key = compatibility_key(case)
        if new_key != key:
            if buffer is not None:
                close_group()
            initialization += 1
            key = new_key
            buffer = factories[case["api"]](case)
            reuse_log.append({"event": "initialize", "initialization": initialization, "compatibility_key": list(key)})
            print("BUFFER_INITIALIZED rank=" + str(rank) + " initialization=" + str(initialization)
                  + " api=" + case["api"] + " dtype=" + case["dtype"], flush=True)
        else:
            retained_check = None
            gc.collect()
            print("BUFFER_REUSED rank=" + str(rank) + " initialization=" + str(initialization)
                  + " variant=" + case["variant"], flush=True)
        variant = case["variant"]
        args = make_args(case, node_rank, output / variant, run_id)
        torch.npu.synchronize()
        dist.barrier()
        retained_check = checkers[case["api"]](args, buffer)
        path = output / variant / ("rank" + str(rank)) / "result.json"
        result = json.loads(path.read_text())
        if not result.get("precision_passed") or not result.get("precision_only") or result.get("operators") != {}:
            raise RuntimeError("Invalid precision-only result: " + variant)
        result["buffer_initialization"] = initialization
        result["buffer_group_key"] = list(key)
        result["buffer_config"] = case["buffer_config"]
        result["source_id"] = case["source_id"]
        result["group"] = case["group"]
        result["teardown_scope"] = "pending_buffer_group"
        path.write_text(json.dumps(result, indent=2))
        pending.append(variant)
        (output / ("progress_rank" + str(rank) + ".json")).write_text(json.dumps({
            "rank": rank,
            "case_completed": len([e for e in reuse_log if e["event"] == "case"]) + 1,
            "total_cases": len(cases), "current": variant,
            "buffer_initializations": initialization,
            "buffer_workspace_bytes": case["buffer_config"]["workspace_bytes"],
            "state": "running"}, indent=2))
        reuse_log.append({"event": "case", "initialization": initialization, "variant": variant})
        torch.npu.synchronize()
        dist.barrier()
        print("BATCH_CASE_PASSED rank=" + str(rank) + " variant=" + variant
              + " initialization=" + str(initialization), flush=True)
    if buffer is not None:
        close_group()
    return {"rank": rank, "code": 0, "process_group_initializations": 1,
            "buffer_initializations": initialization, "reuse_log": reuse_log,
            "completed": [c["variant"] for c in cases], "ended": time.time()}


def main():
    parser = argparse.ArgumentParser(description="Persistent EP128 eight-node precision-only worker; no profiler or performance loops")
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=tuple(range(8)), required=True)
    parser.add_argument("--run-id", required=True)
    options = parser.parse_args()
    import datetime
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep.backends.global_clos import Buffer, ElasticBuffer
    from checks_v1 import check_case as check_v1
    from checks_v2 import check_case as check_v2

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 128 or int(os.environ["LOCAL_WORLD_SIZE"]) != 16 \
            or rank != options.node_rank * 16 + local_rank:
        raise RuntimeError("Expected eight nodes, 16 local ranks and 128 global ranks")
    torch.set_num_threads(1)
    torch.npu.set_device(local_rank)
    options.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(options.plan.read_text())["remaining"]
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=900))
    dist.barrier()

    def new_v1(case):
        p = case["parameters"]
        capacity = case["buffer_config"]["workspace_bytes"]
        return Buffer(dist.group.WORLD, num_nvl_bytes=capacity, explicitly_destroy=True,
                      transport="udma", use_1g_huge_pages=p["huge_pages"],
                      shmem_ip_port=shmem_endpoint())

    def new_v2(case):
        p = case["parameters"]
        return ElasticBuffer(dist.group.WORLD,
                             num_max_tokens_per_rank=case["buffer_config"]["max_tokens"],
                             hidden=p["hidden"], num_topk=p["num_topk"],
                             use_fp8_dispatch=case["dtype"] == "fp8",
                             num_bytes=case["buffer_config"]["workspace_bytes"],
                             explicitly_destroy=True, transport="udma",
                             use_1g_huge_pages=p["huge_pages"],
                             shmem_ip_port=shmem_endpoint())

    try:
        record = execute(cases, options.output, options.node_rank, options.run_id, torch, dist,
                         {"v1": new_v1, "v2": new_v2}, {"v1": check_v1, "v2": check_v2})
        dist.barrier()
        dist.destroy_process_group()
        (options.output / ("batch_rank" + str(rank) + ".json")).write_text(json.dumps(record, indent=2))
        print("BATCH_PRECISION_PASSED rank=" + str(rank)
              + " buffer_initializations=" + str(record["buffer_initializations"]), flush=True)
        return 0
    except BaseException:
        (options.output / ("batch_rank" + str(rank) + ".json")).write_text(
            json.dumps({"rank": rank, "code": 1, "traceback": traceback.format_exc()}, indent=2))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
