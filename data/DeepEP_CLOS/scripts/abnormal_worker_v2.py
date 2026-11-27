"""E1-E9 abnormal-scenario worker, v2 (prepared, NOT executed).

Corrections vs v1, per 2to16P异常场景结果_20261127.md §8.5 / §8.6
(the "latest abnormal-scenario notes"):

  1. E5 uses the REAL over-capacity input from the original spec:
     every rank posts x with num_tokens > num_max_tokens_per_rank, while
     topk_idx VALUES stay legal (0..num_experts-1). Expected: a Python-layer
     ValueError "local token count exceeds num_max_tokens_per_rank"
     (V1, buffer_v1.py:509) / "x.size(0) exceeds num_max_tokens_per_rank"
     (V2, elastic_v2.py:1378), and round 1 must still combine correctly.
     V1 Buffer is now constructed WITH num_max_tokens_per_rank so that check
     is actually active (v1 omitted it, so the check never fired).
  2. E4/E9 build the second "same-layout expert output" as a PURE TENSOR
     from the golden reference, without a second dispatch (v1 used
     dispatch_cached, which added an unnecessary communication round).
  3. E7/E8/E9 notes explicitly state expanded=True / via expand+aggregate.
  4. E5 judgement catches Exception, not BaseException.
  5. Prepares for full EP2/4/8/16 coverage and --repeat (controller side).

Still precision-only: no profiler, timing, bandwidth, GBps or CV.
Safe scenarios share one process group and one reused Buffer per API;
E5/E6 remain isolated (they can leave the NPU in an unrecoverable state).
"""
import argparse
import copy
import datetime
import json
import os
import pathlib
import sys
import time
import traceback

SKIP_ARGS = ("huge_pages", "seed", "warmup", "iterations", "flush_l2_cache",
             "combine_bias", "rendezvous_timeout")
CAPACITY_ERROR_MARKERS = ("exceeds num_max_tokens_per_rank",
                          "local token count exceeds")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


class Harness:
    def __init__(self, opts, torch, dist, golden, profiling, shared, buffer, config,
                 V1Operations):
        self.o = opts
        self.torch = torch
        self.dist = dist
        self.g = golden
        self.profiling = profiling
        self.shared = shared
        self.buffer = buffer
        self.config = config
        self.V1Operations = V1Operations
        self.rank = int(os.environ["RANK"])
        self.ep = opts.ep
        self.seed_base = opts.seed_base
        self.capacity = opts.capacity

    def args_for(self, seed):
        a = copy.copy(self.base_args)
        a.seed = seed
        a._topk_reference_cache = {}
        a._dispatch_reference_cache = None
        return a

    def wait(self, result):
        self.profiling._consume_profile_result(result)
        return result

    def dispatch(self, seed, delta, expanded):
        a = self.args_for(seed)
        routes = [self.g._make_topk_idx(p, a) for p in range(self.ep)]
        idx = routes[self.rank].to("npu")
        weights = self.g._make_weights(self.rank, a).to("npu")
        x = self.g._to_npu(self.g._make_dispatch_x(self.rank, a, delta=delta))
        if self.o.api == "v1":
            ops = self.V1Operations(self.buffer, self.config, idx, weights,
                                    self.ep * a.experts_per_rank, expanded)
            res = self.wait(ops.dispatch(x))
            dispatched, expansion = res[0], res[1]
            handle = (dispatched[4], expansion[2] if expanded else None)
            recv = expansion[0] if expanded else dispatched[0]
            recv_weights = expansion[1] if expanded else dispatched[2]
        else:
            ops = None
            res = self.wait(self.buffer.dispatch(
                x, topk_idx=idx, topk_weights=weights, handle=None,
                num_experts=self.ep * a.experts_per_rank, num_sms=a.num_aiv,
                do_expand=expanded, expert_alignment=1,
                async_with_compute_stream=True))
            recv, _, recv_weights, handle, _ = res
        owned = self.g._dequantize_x(recv, self.torch.bfloat16).clone()
        self.torch.npu.synchronize()
        return {"args": a, "ops": ops, "handle": handle, "owned": owned,
                "recv_weights": recv_weights, "recv": recv, "idx": idx,
                "weights": weights, "x": x, "delta": delta, "expanded": expanded}

    def expert_output_alt(self, rnd, delta):
        """Same-layout expert output with different values, NO second dispatch.

        The original spec says "换一份同布局的专家输出": a pure tensor is enough.
        Building it from the golden reference keeps the layout identical to the
        dispatch result while avoiding dispatch_cached (v1's extra round).
        """
        a = rnd["args"]
        ref = self.g._dispatch_reference(self.rank, a, delta=delta)
        if rnd["expanded"]:
            exp = self.g._expand_reference(ref["recv_x"], ref["recv_topk_idx"],
                                           ref["recv_topk_weights"], ref["expert_counts"])
            recv = exp["expanded_x"]
        else:
            recv = ref["recv_x"]
        return self.g._dequantize_x(recv, self.torch.bfloat16).to("npu")

    def dispatch_over_capacity(self, seed, bad_tokens):
        """All ranks post an over-capacity x; topk_idx VALUES remain legal.

        This is the E5 input from the original text ("所有 rank 上传超容量非法输入").
        It is deliberately NOT the out-of-range expert id used by v1.
        """
        a = self.args_for(seed)
        ab = copy.copy(a)
        ab.num_tokens = bad_tokens
        ab._topk_reference_cache = {}
        ab._dispatch_reference_cache = None
        routes = [self.g._make_topk_idx(p, ab) for p in range(self.ep)]
        idx = routes[self.rank].to("npu")
        weights = self.g._make_weights(self.rank, ab).to("npu")
        x = self.g._to_npu(self.g._make_dispatch_x(self.rank, ab, delta=3.0))
        if self.o.api == "v1":
            ops = self.V1Operations(self.buffer, self.config, idx, weights,
                                    self.ep * a.experts_per_rank, False)
            return self.wait(ops.dispatch(x))
        return self.wait(self.buffer.dispatch(
            x, topk_idx=idx, topk_weights=weights, handle=None,
            num_experts=self.ep * a.experts_per_rank, num_sms=a.num_aiv,
            do_expand=False, expert_alignment=1, async_with_compute_stream=True))

    def combine(self, rnd, owned=None, weighted=False):
        x = rnd["owned"] if owned is None else owned
        if self.o.api == "v1":
            res = self.wait(rnd["ops"].combine(
                x, rnd["handle"], rnd["recv_weights"] if weighted else None, None))
            combined = res[1]
            return combined[0], combined[1]
        res = self.wait(self.buffer.combine(
            x, rnd["handle"], topk_weights=rnd["recv_weights"] if weighted else None,
            bias=None, num_sms=rnd["args"].num_aiv, async_with_compute_stream=True))
        return res[0], res[1]

    def expected_combine(self, rnd, delta):
        a = rnd["args"]
        g = self.g
        if not rnd["expanded"]:
            ref, _ = g._combine_reference(self.rank, a, delta=delta)
            return ref
        source = g._dequantize_x(g._make_dispatch_x(self.rank, a, delta=delta),
                                 self.torch.bfloat16)
        routes = g._make_topk_idx(self.rank, a)
        acc = self.torch.zeros_like(source, dtype=self.torch.float32)
        for peer in range(self.ep):
            mask = (routes >= peer * a.experts_per_rank) & \
                   (routes < (peer + 1) * a.experts_per_rank)
            partial = self.torch.zeros_like(source, dtype=self.torch.float32)
            for k in range(a.num_topk):
                partial[mask[:, k]] += source[mask[:, k]].float()
            acc += partial.to(self.torch.bfloat16).float()
        return acc.to(self.torch.bfloat16)

    def check_combine(self, rnd, delta, label, owned=None):
        actual, _ = self.combine(rnd, owned=owned)
        expected = self.expected_combine(rnd, delta)
        self.g._assert_equal(label, actual, expected)
        return True

    def check_dispatch_payload(self, rnd, label):
        ref = self.g._dispatch_reference(self.rank, rnd["args"], delta=rnd["delta"])
        if rnd["expanded"]:
            exp = self.g._expand_reference(ref["recv_x"], ref["recv_topk_idx"],
                                           ref["recv_topk_weights"], ref["expert_counts"])
            self.g._assert_equal(label, rnd["recv"], exp["expanded_x"])
        else:
            self.g._assert_equal(label, rnd["recv"], ref["recv_x"])
        return True


def pad_rows(t, rows, torch):
    if t.size(0) >= rows:
        return t[:rows]
    pad = t[-1:].expand(rows - t.size(0), -1).contiguous()
    return torch.cat([t, pad], dim=0)


def scenario(h, sid):
    """Return (passed, cls, checks, notes). cls: passed / defect_suspected /
    device_crash / failed_runtime."""
    S = h.seed_base
    notes = []
    checks = {}
    expanded = sid in ("E7", "E8", "E9")
    if expanded:
        notes.append("expanded=True via buffer.expand + buffer.aggregate (V1 only)")

    if sid in ("E1", "E2", "E3", "E7", "E8"):
        r1 = h.dispatch(S + 1, 0.0, expanded)
        r2 = h.dispatch(S + 2, 1.0, expanded)
        checks["round1_payload"] = h.check_dispatch_payload(r1, sid + " dispatch round1")
        checks["round2_payload"] = h.check_dispatch_payload(r2, sid + " dispatch round2")
        if sid in ("E1", "E7"):
            h.check_combine(r2, 1.0, sid + " combine round2 only")
            checks["combine_round2"] = True
            r3 = h.dispatch(S + 3, 2.0, expanded)
            h.check_combine(r3, 2.0, sid + " next round still works")
            checks["next_round"] = True
            notes.append("only round2 combined; a further round still succeeded")
        elif sid in ("E2", "E8"):
            h.check_combine(r1, 0.0, sid + " combine round1 after round2")
            checks["combine_round1"] = True
            notes.append("round2 used a different route seed and payload; round1 result still correct")
        else:
            h.check_combine(r2, 1.0, sid + " combine round2 first")
            h.check_combine(r1, 0.0, sid + " combine round1 second")
            checks["combine_order_2_then_1"] = True
            notes.append("combined in 2 -> 1 order; both results correct")
        return True, "passed", checks, notes

    if sid in ("E4", "E9"):
        r = h.dispatch(S + 1, 0.0, expanded)
        checks["dispatch_payload"] = h.check_dispatch_payload(r, sid + " dispatch")
        h.check_combine(r, 0.0, sid + " combine #1")
        checks["combine_1"] = True
        owned2 = h.expert_output_alt(r, 1.0)      # pure tensor, no second dispatch
        h.check_combine(r, 1.0, sid + " combine #2 (same-layout expert output)",
                        owned=owned2)
        checks["combine_2_same_handle"] = True
        notes.append("same handle combined twice; second expert output is a pure "
                     "tensor (golden reference, delta=1), no second dispatch")
        return True, "passed", checks, notes

    if sid == "E5":
        r1 = h.dispatch(S + 1, 0.0, False)
        checks["round1_payload"] = h.check_dispatch_payload(r1, "E5 dispatch round1")
        bad_tokens = h.capacity + 1
        notes.append("round2 over-capacity: num_tokens=%d > capacity=%d on ALL ranks; "
                     "topk_idx values stay legal (0..num_experts-1)" %
                     (bad_tokens, h.capacity))
        rejected = False
        err = None
        try:
            h.dispatch_over_capacity(S + 4, bad_tokens)
            notes.append("round2 over-capacity input did NOT raise")
        except Exception as exc:                  # not BaseException (§8.6.5)
            err = str(exc)
            rejected = any(k in err for k in CAPACITY_ERROR_MARKERS)
            notes.append("round2 rejected: %s: %s" % (type(exc).__name__, err[:200]))
        checks["round2_rejected_by_capacity_check"] = rejected
        try:
            h.check_combine(r1, 0.0, "E5 combine round1 after rejection")
            checks["round1_recovers"] = True
        except Exception as exc:
            checks["round1_recovers"] = False
            notes.append("round1 combine failed: " + str(exc)[:200])
            return False, "failed_runtime", checks, notes
        ok = rejected and checks["round1_recovers"]
        cls = "passed" if ok else "defect_suspected"
        if not rejected:
            notes.append("expected Python-layer ValueError from buffer_v1.py:509 "
                         "/ elastic_v2.py:1378")
        return ok, cls, checks, notes

    if sid == "E6":
        r1 = h.dispatch(S + 1, 0.0, False)
        r2 = h.dispatch(S + 2, 1.0, False)
        checks["round1_payload"] = h.check_dispatch_payload(r1, "E6 dispatch round1")
        checks["round2_payload"] = h.check_dispatch_payload(r2, "E6 dispatch round2")
        n1, n2 = r1["owned"].size(0), r2["owned"].size(0)
        notes.append("round1 recv rows=%d, round2 recv rows=%d" % (n1, n2))
        owned_a = r2["owned"] if n2 != n1 else r2["owned"][:-1]
        rejected_a = False
        try:
            h.combine(r1, owned=owned_a)
            notes.append("group A (different row count) did NOT reject")
        except Exception as exc:
            rejected_a = True
            notes.append("group A rejected: " + type(exc).__name__ + ": " + str(exc)[:160])
        checks["group_a_rejected"] = rejected_a
        owned_b = r2["owned"][:n1] if n2 >= n1 else pad_rows(r2["owned"], n1, h.torch)
        detected_b = False
        try:
            actual, _ = h.combine(r1, owned=owned_b)
            expected, _ = h.g._combine_reference(h.rank, r1["args"], delta=0.0)
            try:
                h.g._assert_equal("E6 group B", actual, expected)
                notes.append("group B matched round1 reference (no misplacement exposed)")
            except Exception:
                detected_b = True
                notes.append("group B silently returned a WRONG result; detected by per-token comparison")
        except Exception as exc:
            detected_b = True
            notes.append("group B rejected the mismatched handle: " + type(exc).__name__)
        checks["group_b_silent_error_detected"] = detected_b
        h.check_combine(r1, 0.0, "E6 restore with correct handle")
        checks["restore_correct"] = True
        ok = checks["restore_correct"] and (rejected_a or detected_b)
        return ok, ("passed" if ok else "defect_suspected"), checks, notes

    raise RuntimeError("unknown scenario " + sid)


def run_scenarios(h, sids, output):
    for sid in sids:
        dest = output / sid / ("rank" + str(h.rank))
        dest.mkdir(parents=True, exist_ok=False)
        write_json(dest / "process.json",
                   dict(pid=os.getpid(), rank=h.rank, device=int(os.environ["LOCAL_RANK"]),
                        api=h.o.api, scenario=sid, started=time.time()))
        t0 = time.time()
        h.torch.npu.synchronize()
        h.dist.barrier()
        try:
            passed, cls, checks, notes = scenario(h, sid)
            h.torch.npu.synchronize()
            elapsed = time.time() - t0
            write_json(dest / "precision.json",
                       dict(passed=passed, classification=cls, checks=checks,
                            notes=notes, elapsed_s=round(elapsed, 2)))
            write_json(dest / "result.json",
                       dict(rank=h.rank, api=h.o.api, scenario=sid,
                            precision_passed=passed, precision_only=True, operators={},
                            buffer_reused=True, classification=cls, checks=checks,
                            notes=notes, elapsed_s=round(elapsed, 2)))
            write_json(dest / "exit.json", dict(code=0, ended=time.time()))
            print("ABNORMAL_V2 %s %s rank=%d -> %s %.2fs" %
                  (h.o.api, sid, h.rank, cls, elapsed), flush=True)
        except BaseException as exc:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            device_crash = any(k in tb for k in ("EE9999", "vector core", "Inner error",
                                                 "device error", "rtDeviceSynchronize",
                                                 "npuSynchronizeDevice"))
            cls = "device_crash" if device_crash else "failed_runtime"
            err = str(exc)[:400]
            write_json(dest / "precision.json",
                       dict(passed=False, classification=cls, error=err,
                            elapsed_s=round(elapsed, 2)))
            write_json(dest / "result.json",
                       dict(rank=h.rank, api=h.o.api, scenario=sid,
                            precision_passed=False, precision_only=True, operators={},
                            buffer_reused=True, classification=cls, error=err,
                            elapsed_s=round(elapsed, 2)))
            write_json(dest / "exit.json", dict(code=1, ended=time.time(), traceback=tb))
            print("ABNORMAL_V2 %s %s rank=%d -> %s %s" %
                  (h.o.api, sid, h.rank, cls, str(exc)[:120]), flush=True)
        finally:
            try:
                h.torch.npu.synchronize()
                h.dist.barrier()
            except BaseException:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--api", choices=("v1", "v2"), required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--num-tokens", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--experts-per-rank", type=int, default=32)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--aiv", type=int, default=32)
    ap.add_argument("--workspace-bytes", type=int, required=True)
    ap.add_argument("--capacity", type=int, default=16384,
                    help="num_max_tokens_per_rank; E5 posts capacity+1 tokens")
    ap.add_argument("--seed-base", type=int, default=20260823)
    o = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    import bench_v2_precision as shared
    from deep_ep.backends.global_clos import Buffer, ElasticBuffer
    from deep_ep.backends.direct import Config
    from bench_v1_precision import V1Operations
    import profiling
    sys.path.insert(0, str(shared.REPO / "tests/clos"))
    import common as golden

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != o.ep or int(os.environ["LOCAL_WORLD_SIZE"]) != o.ep \
            or rank != local_rank:
        raise RuntimeError("expected single node WORLD_SIZE=LOCAL_WORLD_SIZE=EP=%d" % o.ep)
    torch.set_num_threads(1)
    torch.npu.set_device(local_rank)
    if rank == 0:
        o.output.mkdir(parents=True, exist_ok=False)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=900))
    dist.barrier()

    params = dict(num_tokens=o.num_tokens, hidden=o.hidden, num_topk=o.num_topk,
                  experts_per_rank=o.experts_per_rank, aiv=o.aiv,
                  route_mode="uniform", huge_pages=False, seed=o.seed_base,
                  warmup=10, iterations=30, flush_l2_cache=True,
                  combine_bias="none", rendezvous_timeout=900)
    argv = ["--ep-size=%d" % o.ep, "--nnodes=1", "--node-rank=0", "--first-device=0",
            "--run-id=" + o.run_id, "--output=" + str(o.output),
            "--dtypes=" + o.dtype, "--expand=0", "--precision-only"]
    for k, v in params.items():
        if k in SKIP_ARGS:
            continue
        if isinstance(v, bool):
            if v:
                argv.append("--" + k.replace("_", "-"))
        elif v not in (None, ""):
            argv.append("--" + k.replace("_", "-") + "=" + str(v))
    base = shared.arguments(argv)
    base.seed = o.seed_base
    base.num_aiv = o.aiv
    base.dtype = o.dtype
    base.expert_alignment = 1
    base.combine_notify = True
    os.environ[shared.COMBINE_NOTIFY_ENV] = "0"

    if o.api == "v1":
        # v1 omitted num_max_tokens_per_rank, so buffer_v1.py:509 never fired.
        buffer = Buffer(dist.group.WORLD, num_nvl_bytes=o.workspace_bytes,
                        num_max_tokens_per_rank=o.capacity,
                        explicitly_destroy=True, transport="udma",
                        use_1g_huge_pages=False)
        config = Config(o.aiv)
    else:
        buffer = ElasticBuffer(dist.group.WORLD, num_max_tokens_per_rank=o.capacity,
                               hidden=o.hidden, num_topk=o.num_topk,
                               use_fp8_dispatch=(o.dtype == "fp8"),
                               num_bytes=o.workspace_bytes, explicitly_destroy=True,
                               transport="udma", use_1g_huge_pages=False)
        config = None
    print("BUFFER_INITIALIZED_V2 rank=%d api=%s ep=%d capacity=%d" %
          (rank, o.api, o.ep, o.capacity), flush=True)

    h = Harness(o, torch, dist, golden, profiling, shared, buffer, config, V1Operations)
    h.base_args = base
    sids = [s for s in o.scenarios.split(",") if s]
    try:
        run_scenarios(h, sids, o.output)
        dist.barrier()
        try:
            buffer.destroy()
        except BaseException:
            pass
        dist.destroy_process_group()
        print("ABNORMAL_V2_GROUP_DONE rank=%d api=%s" % (rank, o.api), flush=True)
        return 0
    except BaseException:
        write_json(o.output / ("group_rank%d_failure.json" % rank),
                   dict(rank=rank, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
