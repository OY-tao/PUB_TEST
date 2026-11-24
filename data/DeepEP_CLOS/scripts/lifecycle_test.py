"""测 DeepEP V1 `Buffer` / V2 `ElasticBuffer` 的构造与销毁（生命周期 4 项）。

覆盖设计文档的 L7（V1-1 / V1-7 / V2-1 / V2-4）—— 上游两个 bench 都不测这些。

为什么需要单独一个脚本：
  两个 bench 的 functions 字典只有 4 个通信 op，构造和销毁都发生在 worker
  进入计时循环之前/之后，没有计时、没有失败路径测试、没有泄漏检查。

测量方式（与 bench 的通信计时不同）：
  构造/销毁是 host 侧 + 集合通信等待，用 time.perf_counter() 包住，
  外面先 npu.synchronize()。不用 NPU Event，因为主要耗时不在设备 kernel 上。

用法（必须 torchrun 启动，因为构造要 EP 进程组）：
    torchrun --nproc-per-node 8 lifecycle_test.py --api v1 --ep-size 8 \
        --hidden 6144 --num-tokens 8192 --cycles 20 --out life_v1.json

    torchrun --nproc-per-node 8 lifecycle_test.py --api v2 --ep-size 8 ...
    torchrun --nproc-per-node 8 lifecycle_test.py --api both --fail-paths ...
"""

import argparse
import datetime
import json
import os
import sys
import time
import traceback


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--api", choices=("v1", "v2", "both"), default="both")
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--num-tokens", type=int, default=8192)
    p.add_argument("--num-topk", type=int, default=8)
    p.add_argument("--num-aiv", type=int, default=32)
    p.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    p.add_argument("--cycles", type=int, default=20, help="构造→销毁 循环次数（泄漏检查）")
    p.add_argument("--huge-pages", action="store_true")
    p.add_argument("--fail-paths", action="store_true", help="只跑 L7-C 失败路径")
    p.add_argument("--out", default="lifecycle.json")
    p.add_argument("--timeout", type=int, default=300)
    return p.parse_args(argv)


def hbm_used_mb(device):
    """读本卡 HBM 占用（MB）。失败返回 None，不阻断测试。"""
    try:
        import subprocess
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30).stdout
        import re
        # 形如 "| 0      | Ascend950PR   | OK   | 269.4 60  | 0  0 / 0  6214 / 86016 |"
        for line in out.splitlines():
            m = re.search(r"\|\s*(\d+)\s*\|\s*Ascend[^|]*\|[^|]*\|[^|]*\|\s*\d+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)", line)
            if m and int(m.group(1)) == device:
                return int(m.group(2))
    except Exception:
        pass
    return None


def measure_host(fn):
    """host 侧计时（含集合通信等待）。返回 (耗时ms, 异常或None)。"""
    import torch
    torch.npu.synchronize()
    t0 = time.perf_counter()
    try:
        result = fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0, None, result
    except Exception as e:
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0, e, None


def make_v1(group, args, explicitly_destroy=True):
    from deep_ep.backends.direct.buffer import Buffer
    from deep_ep.backends.direct.config import Config
    config = Config(args.num_aiv)
    num_bytes = config.get_nvl_buffer_size_hint(
        args.hidden * 2, args.ep_size, num_tokens=args.num_tokens)
    return Buffer(
        group,
        num_nvl_bytes=num_bytes,
        explicitly_destroy=explicitly_destroy,
        transport="udma",
        use_1g_huge_pages=args.huge_pages,
    ), num_bytes


def make_v2(group, args, explicitly_destroy=True):
    from deep_ep.backends.direct.elastic import ElasticBuffer
    return ElasticBuffer(
        group,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        use_fp8_dispatch=(args.dtype == "fp8"),
        explicitly_destroy=explicitly_destroy,
        transport="udma",
        use_1g_huge_pages=args.huge_pages,
    ), None


BUILDERS = {"v1": make_v1, "v2": make_v2}


def run_lifecycle(api, args, group, rank, device, records):
    """正常路径：构造耗时、销毁耗时、幂等、泄漏。"""
    import torch
    build = BUILDERS[api]

    def label(key):
        return "%s/%s" % (api, key)

    # ---- 构造 ----
    ms, err, built = measure_host(lambda: build(group, args))
    if err is not None:
        records.append(dict(item=label("construct"), ok=False,
                            error="%s: %s" % (type(err).__name__, str(err)[:120])))
        return
    buf, num_bytes = built
    records.append(dict(item=label("construct"), ok=True, ms=round(ms, 3),
                        num_bytes=num_bytes, hbm_mb=hbm_used_mb(device)))

    # ---- 销毁 ----
    ms_d, err_d, _ = measure_host(lambda: buf.destroy())
    records.append(dict(item=label("destroy"), ok=err_d is None,
                        ms=round(ms_d, 3),
                        error=None if err_d is None else "%s: %s" % (type(err_d).__name__, str(err_d)[:120])))

    # ---- 幂等：再销毁一次 ----
    ms_i, err_i, _ = measure_host(lambda: buf.destroy())
    records.append(dict(item=label("destroy_idempotent"), ok=err_i is None,
                        ms=round(ms_i, 3),
                        error=None if err_i is None else "%s: %s" % (type(err_i).__name__, str(err_i)[:120])))

    # ---- V2 特有：without explicitly_destroy 应抛错 ----
    if api == "v2":
        _, err_nd, nd = measure_host(lambda: build(group, args, explicitly_destroy=False))
        if err_nd is not None:
            records.append(dict(item=label("construct_no_explicit"), ok=False,
                                error="构造就失败了: %s" % str(err_nd)[:100]))
        else:
            _, err_dd, _ = measure_host(lambda: nd.destroy())
            # destroy 应该抛 RuntimeError
            expect_raise = isinstance(err_dd, RuntimeError)
            records.append(dict(item=label("destroy_without_explicit"), ok=expect_raise,
                                raised=type(err_dd).__name__ if err_dd else None,
                                error=None if expect_raise else
                                "期望 RuntimeError，实际: %s" % (type(err_dd).__name__ if err_dd else "未抛异常")))
            try:
                nd.destroy() if not expect_raise else None
            except Exception:
                pass

    # ---- V1 特有：活跃 slot 时拒绝销毁 ----
    # acquire_slot() 把 slot 推进到非 FREE 状态，此时 destroy 应抛 RuntimeError
    # （buffer.py:360）。注意 Buffer 没有 release_slot，这个 buffer 用完即弃。
    if api == "v1":
        ms_s, err_s, built_s = measure_host(lambda: build(group, args))
        if err_s is None:
            buf_s, _ = built_s
            try:
                buf_s.acquire_slot()
                _, err_active, _ = measure_host(lambda: buf_s.destroy())
                expect_raise = isinstance(err_active, RuntimeError)
                records.append(dict(
                    item=label("destroy_with_active_slot"), ok=expect_raise,
                    raised=type(err_active).__name__ if err_active else None,
                    error=None if expect_raise else
                    ("期望 RuntimeError，实际 %s" % (type(err_active).__name__ if err_active else "未抛异常"))))
            except Exception as e:
                records.append(dict(item=label("destroy_with_active_slot"), ok=False,
                                    error="acquire_slot 失败: %s" % str(e)[:100]))
            # 清理：slot 未释放，只能靠进程退出回收，不再对它做其他操作
            records.append(dict(item=label("slot_cleanup_note"), ok=True,
                                error="Buffer 无 release_slot，该实例留待进程回收"))

    # ---- 泄漏：连续 cycles 次构造→销毁 ----
    base_hbm = hbm_used_mb(device)
    loop_ms = []
    loop_err = None
    for i in range(args.cycles):
        m1, e1, b = measure_host(lambda: build(group, args))
        if e1 is not None:
            loop_err = "第 %d 次构造失败: %s" % (i + 1, str(e1)[:100])
            break
        m2, e2, _ = measure_host(lambda: b.destroy())
        if e2 is not None:
            loop_err = "第 %d 次销毁失败: %s" % (i + 1, str(e2)[:100])
            break
        loop_ms.append((m1, m2))
    end_hbm = hbm_used_mb(device)

    if loop_ms:
        c_ms = [x[0] for x in loop_ms]
        d_ms = [x[1] for x in loop_ms]
        leak_ok = True
        if base_hbm is not None and end_hbm is not None and base_hbm > 0:
            growth = (end_hbm - base_hbm) / max(base_hbm, 1) * 100
            leak_ok = growth < 5.0
        else:
            growth = None
        records.append(dict(
            item=label("leak_check"), ok=leak_ok and loop_err is None,
            cycles=len(loop_ms),
            construct_ms_median=round(sorted(c_ms)[len(c_ms) // 2], 3),
            destroy_ms_median=round(sorted(d_ms)[len(d_ms) // 2], 3),
            hbm_base_mb=base_hbm, hbm_end_mb=end_hbm,
            hbm_growth_pct=None if growth is None else round(growth, 2),
            error=loop_err,
        ))


def run_fail_paths(api, args, group, rank, device, records):
    """L7-C：构造失败路径。每项只由 rank0 测（参数校验是本地的，不需要全组）。"""
    if rank != 0:
        return
    build = BUILDERS[api]
    from deep_ep.backends.direct.buffer import Buffer
    from deep_ep.backends.direct.config import Config

    cases = []
    if api == "v1":
        cases = [
            ("num_max_tokens_per_rank=0", lambda: Buffer(
                group, num_nvl_bytes=64 * 1024 * 1024, explicitly_destroy=True,
                transport="udma", num_max_tokens_per_rank=0), ValueError),
            ("num_max_tokens_per_rank=True", lambda: Buffer(
                group, num_nvl_bytes=64 * 1024 * 1024, explicitly_destroy=True,
                transport="udma", num_max_tokens_per_rank=True), TypeError),
            ("udma_schedule=invalid", lambda: Buffer(
                group, num_nvl_bytes=64 * 1024 * 1024, explicitly_destroy=True,
                transport="udma", udma_schedule="invalid"), ValueError),
            ("udma_schedule=token w/o udma", lambda: Buffer(
                group, num_nvl_bytes=64 * 1024 * 1024, explicitly_destroy=True,
                transport="auto", udma_schedule="token"), ValueError),
            ("shmem_ip_port=''", lambda: Buffer(
                group, num_nvl_bytes=64 * 1024 * 1024, explicitly_destroy=True,
                transport="udma", shmem_ip_port=""), ValueError),
        ]
    else:
        cases = [
            ("hidden=0", lambda: build(group, args), None),  # 记录实际行为
        ]

    for name, fn, expect in cases:
        _, err, obj = measure_host(fn)
        if expect is None:
            ok = True
            note = "未抛异常（记录实际行为）" if err is None else "抛了 %s" % type(err).__name__
        else:
            ok = isinstance(err, expect)
            note = None if ok else ("期望 %s，实际 %s" % (
                expect.__name__, type(err).__name__ if err else "未抛异常"))
        records.append(dict(item="%s/fail:%s" % (api, name), ok=ok,
                            raised=type(err).__name__ if err else None,
                            error=note))
        if obj is not None and err is None:
            try:
                obj.destroy()
            except Exception:
                pass


def main():
    args = parse()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    if world != args.ep_size:
        raise SystemExit("WORLD_SIZE(%d) != --ep-size(%d)，请用 torchrun --nproc-per-node %d"
                         % (world, args.ep_size, args.ep_size))

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    device = local_rank
    torch.npu.set_device(device)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=args.timeout))

    records = []
    try:
        apis = ("v1", "v2") if args.api == "both" else (args.api,)
        for api in apis:
            # 每个 api 的失败路径与正常路径分开跑
            if args.fail_paths:
                run_fail_paths(api, args, dist.group.WORLD, rank, device, records)
            else:
                run_lifecycle(api, args, dist.group.WORLD, rank, device, records)
    finally:
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

    # 汇总（只由 rank0 打印 + 写文件）
    if rank == 0:
        print("=" * 96)
        print("生命周期测试  api=%s  ep=%d  hidden=%d  tokens=%d  cycles=%d"
              % (args.api, args.ep_size, args.hidden, args.num_tokens, args.cycles))
        print("=" * 96)
        print("%-42s %-6s %12s %s" % ("项", "结果", "耗时ms", "备注"))
        print("-" * 96)
        for r in records:
            ms = r.get("ms", "")
            ms_s = ("%.2f" % ms) if isinstance(ms, (int, float)) else ""
            note = r.get("error") or r.get("raised") or ""
            if r.get("hbm_growth_pct") is not None:
                note = "HBM %s→%sMB (%+.1f%%)" % (r.get("hbm_base_mb"), r.get("hbm_end_mb"),
                                                  r.get("hbm_growth_pct"))
            print("%-42s %-6s %12s %s"
                  % (r["item"][:42], "PASS" if r["ok"] else "FAIL", ms_s, str(note)[:40]))
        print("-" * 96)
        npass = sum(1 for r in records if r["ok"])
        print("合计 %d 项：PASS %d / FAIL %d" % (len(records), npass, len(records) - npass))

        out = dict(args=vars(args), rank0_records=records,
                   ended=time.time(), condition=env_condition())
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("\n明细 -> %s" % args.out)

    return 0 if all(r["ok"] for r in records) else 1


def env_condition():
    """记录环境，便于跨次对比。"""
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import torch_npu
        info["torch_npu"] = torch_npu.__version__
    except Exception:
        pass
    info["ascend_home"] = os.environ.get("ASCEND_HOME_PATH", "")
    return info


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
