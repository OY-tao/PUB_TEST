"""E1-E9 abnormal-scenario worker for EP8 / EP16 (single node).

One ``dist.init_process_group`` and one Buffer per API group; every scenario of
that API runs in sequence reusing the Buffer (no per-case re-init). Each
scenario writes its own ``<output>/<sid>/rank<rank>/{result,precision,exit}.json``
so a failed/suspected scenario cannot hide the others. Precision-only: no
profiler, timing, bandwidth, GBps or CV collection.

Scenarios follow 边界与异常用例清单_2to16P.md section 4:
  E1-E4  multi-round dispatch / handle isolation / handle reuse
  E5     over-capacity round-2 input on ALL ranks, then combine round 1
  E6     round-2 expert output with round-1 handle (different/same row count)
  E7-E9  V1-only expand/aggregate variants of E1/E2/E4
E5/E6 are executed last in each API group because they may disturb the Buffer.

R2（2026-11-27）按原文口径修正两处，旧实现保留为 E5X 以便对照：

  E4  原文「一次 dispatch，用同一个 handle combine 两次，第二次换一份同布局的
      专家输出」。旧实现用 dispatch_cached 又跑了一次 dispatch（多了一次通信，
      也可能改动 handle），属于等价但更重的偏离。现在第二次的专家输出直接由
      golden 路由参考生成（同 seed、同行数、不同取值），真正只 dispatch 一次。

  E5  原文「第 2 轮在所有 rank 上传超容量输入」。旧实现传的是越界 expert id
      （≥ num_experts），打的是"取值未校验"这一另一个缺口，实测直接设备级崩溃。
      现在按字面构造：num_tokens = num_max_tokens_per_rank + 1，期望在两版
      kernel 之前就被 ValueError 拒绝（V1 buffer_v1.py:508 / V2 elastic_v2.py:1378）。
      越界 expert id 的旧场景改名为 E5X 保留，仍可复现此前的崩溃结论。
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


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


class ExpectedReject(Exception):
    """Scenario correctly rejected an illegal operation."""


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
        """Fresh dispatch using a distinct route seed and payload delta."""
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

    def dispatch_cached(self, rnd, delta):
        """Same handle, new payload (delta). Layout/row count unchanged."""
        a = rnd["args"]
        cached_x = self.g._to_npu(self.g._make_dispatch_x(self.rank, a, delta=delta))
        if self.o.api == "v1":
            res = self.wait(rnd["ops"].dispatch(cached_x, handles=rnd["handle"]))
            dispatched, expansion = res[0], res[1]
            recv = expansion[0] if rnd["expanded"] else dispatched[0]
        else:
            res = self.wait(self.buffer.dispatch(
                cached_x, handle=rnd["handle"],
                num_experts=self.ep * a.experts_per_rank, num_sms=a.num_aiv,
                do_expand=rnd["expanded"], expert_alignment=1,
                async_with_compute_stream=True))
            recv = res[0]
        owned = self.g._dequantize_x(recv, self.torch.bfloat16).clone()
        self.torch.npu.synchronize()
        return owned

    def dispatch_illegal(self, seed):
        """All ranks post an out-of-range topk_idx (>= num_experts).  [E5X, legacy]"""
        from deep_ep import topk_idx_t
        a = self.args_for(seed)
        num_experts = self.ep * a.experts_per_rank
        bad = self.torch.full((a.num_tokens, a.num_topk), num_experts,
                              dtype=topk_idx_t).to("npu")
        x = self.g._to_npu(self.g._make_dispatch_x(self.rank, a, delta=3.0))
        weights = self.g._make_weights(self.rank, a).to("npu")
        if self.o.api == "v1":
            ops = self.V1Operations(self.buffer, self.config, bad, weights,
                                    num_experts, False)
            return self.wait(ops.dispatch(x))
        return self.wait(self.buffer.dispatch(
            x, topk_idx=bad, topk_weights=weights, handle=None,
            num_experts=num_experts, num_sms=a.num_aiv, do_expand=False,
            expert_alignment=1, async_with_compute_stream=True))

    def dispatch_over_capacity(self):
        """E5 严格口径：payload 超容量（tokens > num_max_tokens_per_rank）。

        与越界 expert id 是两个不同维度：这里路由本身合法，只是本 rank 的 token
        行数超过了 Buffer 声明的容量，应在 kernel 之前被 Python 层 ValueError 拒绝。

        校验位置（快照 scripts/upstream/）：
          V1 buffer_v1.py:508-509  "local token count exceeds num_max_tokens_per_rank"
          V2 elastic_v2.py:1378    "x.size(0) exceeds num_max_tokens_per_rank"
        """
        a = self.args_for(self.seed_base + 4)
        a.num_tokens = int(self.o.max_tokens) + 1
        routes = [self.g._make_topk_idx(p, a) for p in range(self.ep)]
        idx = routes[self.rank].to("npu")
        weights = self.g._make_weights(self.rank, a).to("npu")
        x = self.g._to_npu(self.g._make_dispatch_x(self.rank, a, delta=3.0))
        if self.o.api == "v1":
            ops = self.V1Operations(self.buffer, self.config, idx, weights,
                                    self.ep * a.experts_per_rank, False)
            return self.wait(ops.dispatch(x))
        return self.wait(self.buffer.dispatch(
            x, topk_idx=idx, topk_weights=weights, handle=None,
            num_experts=self.ep * a.experts_per_rank, num_sms=a.num_aiv,
            do_expand=False, expert_alignment=1, async_with_compute_stream=True))

    def expert_output_from_reference(self, rnd, delta):
        """同布局、不同取值的专家输出（不跑第二次 dispatch）。

        E4 原文只要求"换一份同布局的专家输出"；用 golden 的路由参考按同一 seed
        生成另一组行，行数与顺序完全一致、内容不同，比 dispatch_cached 更贴近
        原文——后者会真的再通信一次，可能顺带改动 handle 状态。
        """
        ref = self.g._dispatch_reference(self.rank, rnd["args"], delta=delta)
        if rnd["expanded"]:
            exp = self.g._expand_reference(ref["recv_x"], ref["recv_topk_idx"],
                                           ref["recv_topk_weights"], ref["expert_counts"])
            cpu = exp["expanded_x"]
        else:
            cpu = ref["recv_x"]
        own = self.g._dequantize_x(cpu, self.torch.bfloat16)
        if getattr(own, "device", None) is not None and own.device.type == "cpu":
            own = self.g._to_npu(own)
        return own

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
        """Reference for combine; expanded mode accumulates per topk slot.

        Non-expanded uses golden._combine_reference (one accumulation per token
        routed to a rank). Expanded uses the per-slot accumulation implemented
        in checks_v1.py, including the bf16 rounding step, so the assertion is
        bit-exact against the kernel result.
        """
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
        """Guard: the dispatch round itself delivered the expected expert input."""
        ref = self.g._dispatch_reference(self.rank, rnd["args"], delta=rnd["delta"])
        if rnd["expanded"]:
            exp = self.g._expand_reference(ref["recv_x"], ref["recv_topk_idx"],
                                           ref["recv_topk_weights"], ref["expert_counts"])
            self.g._assert_equal(label, rnd["recv"], exp["expanded_x"])
        else:
            self.g._assert_equal(label, rnd["recv"], ref["recv_x"])
        return True


def scenario(h, sid):
    """Return (passed, cls, checks, notes). cls in passed / defect_suspected /
    failed_runtime; raises on unexpected runtime error."""
    S = h.seed_base
    notes = []
    checks = {}
    expanded = sid in ("E7", "E8", "E9")

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
        else:  # E3
            h.check_combine(r2, 1.0, sid + " combine round2 first")
            h.check_combine(r1, 0.0, sid + " combine round1 second")
            checks["combine_order_2_then_1"] = True
            notes.append("combined in 2 -> 1 order; both results correct")
        return True, "passed", checks, notes

    if sid in ("E4", "E9"):
        # 原文：一次 dispatch，同一个 handle combine 两次；第二次换一份同布局的
        # 专家输出，两次都要成功且各自算对。**只 dispatch 一次**。
        r = h.dispatch(S + 1, 0.0, expanded)
        checks["dispatch_payload"] = h.check_dispatch_payload(r, sid + " dispatch")
        h.check_combine(r, 0.0, sid + " combine #1")
        checks["combine_1"] = True
        owned2 = h.expert_output_from_reference(r, 1.0)
        checks["second_payload_same_layout"] = (
            tuple(owned2.shape) == tuple(r["owned"].shape))
        h.check_combine(r, 1.0, sid + " combine #2 (same handle, same-layout expert output)",
                        owned=owned2)
        checks["combine_2_same_handle"] = True
        notes.append("single dispatch; same handle combined twice, second payload derived from "
                     "the golden route reference (same seed/shape, delta=1) -- no extra dispatch")
        return True, "passed", checks, notes

    if sid == "E5":
        # 原文：第 1 轮 dispatch 成功；第 2 轮在所有 rank 上传**超容量输入**；
        # 再 combine 第 1 轮 —— 第 2 轮明确报错，第 1 轮仍能算对。
        capacity = int(h.o.max_tokens)
        r1 = h.dispatch(S + 1, 0.0, False)
        checks["round1_payload"] = h.check_dispatch_payload(r1, "E5 dispatch round1")
        notes.append("round2 posts %d tokens on every rank, capacity=%d"
                     % (capacity + 1, capacity))
        rejected, err, msg_ok = False, None, False
        try:
            h.dispatch_over_capacity()
            notes.append("round2 over-capacity input was ACCEPTED (no ValueError)")
        except ValueError as exc:
            rejected = True
            err = str(exc)[:240]
            msg_ok = "num_max_tokens_per_rank" in err
            notes.append("round2 rejected with ValueError: " + err)
        except BaseException as exc:
            rejected = True
            err = type(exc).__name__ + ": " + str(exc)[:240]
            notes.append("round2 rejected with a NON-ValueError (capacity check did not fire "
                         "first): " + err)
        checks["round2_rejected"] = rejected
        checks["capacity_message"] = msg_ok
        try:
            h.check_combine(r1, 0.0, "E5 combine round1 after over-capacity round2")
            checks["round1_recovers"] = True
        except BaseException as exc:
            checks["round1_recovers"] = False
            notes.append("round1 combine failed after over-capacity round2: " + str(exc)[:240])
            return False, "failed_runtime", checks, notes
        ok = rejected and msg_ok and checks["round1_recovers"]
        cls = "passed" if ok else "defect_suspected"
        if not ok:
            notes.append("expected: round2 raises ValueError mentioning num_max_tokens_per_rank "
                         "AND round1 still computes correctly")
        return ok, cls, checks, notes

    if sid == "E5X":
        # 旧口径（越界 expert id），保留用于对照上一轮的 device_crash 结论。
        r1 = h.dispatch(S + 1, 0.0, False)
        checks["round1_payload"] = h.check_dispatch_payload(r1, "E5X dispatch round1")
        rejected = False
        err = None
        try:
            h.dispatch_illegal(S + 4)
            notes.append("round2 illegal input did NOT raise (out-of-range topk_idx accepted)")
        except ExpectedReject:
            rejected = True
        except BaseException as exc:
            rejected = True
            err = type(exc).__name__ + ": " + str(exc)[:240]
            notes.append("round2 illegal input rejected: " + err)
        checks["round2_rejected"] = rejected
        try:
            h.check_combine(r1, 0.0, "E5X combine round1 after illegal round2")
            checks["round1_recovers"] = True
        except BaseException as exc:
            checks["round1_recovers"] = False
            notes.append("round1 combine failed after illegal round2: " + str(exc)[:240])
            return False, "failed_runtime", checks, notes
        cls = "passed" if rejected and checks["round1_recovers"] else "defect_suspected"
        if cls == "defect_suspected":
            notes.append("expected: round2 rejects AND round1 still computes correctly")
        return rejected and checks["round1_recovers"], cls, checks, notes

    if sid == "E6":
        r1 = h.dispatch(S + 1, 0.0, False)
        r2 = h.dispatch(S + 2, 1.0, False)
        checks["round1_payload"] = h.check_dispatch_payload(r1, "E6 dispatch round1")
        checks["round2_payload"] = h.check_dispatch_payload(r2, "E6 dispatch round2")
        n1 = r1["owned"].size(0)
        n2 = r2["owned"].size(0)
        notes.append("round1 recv rows=%d, round2 recv rows=%d" % (n1, n2))
        # Group A: different row count -> expect a rejection.
        owned_a = r2["owned"] if n2 != n1 else r2["owned"][:-1]
        rejected_a = False
        try:
            h.combine(r1, owned=owned_a)
            notes.append("group A (different row count) did NOT reject")
        except BaseException as exc:
            rejected_a = True
            notes.append("group A rejected: " + type(exc).__name__ + ": " + str(exc)[:160])
        checks["group_a_rejected"] = rejected_a
        # Group B: same row count, content from round2 -> silent misplacement.
        # Compare against the correct round1 reference; a mismatch must be detected.
        owned_b = r2["owned"][:n1] if n2 >= n1 else pad_rows(r2["owned"], n1, h.torch)
        detected_b = False
        try:
            actual, _ = h.combine(r1, owned=owned_b)
            expected, _ = h.g._combine_reference(h.rank, r1["args"], delta=0.0)
            try:
                h.g._assert_equal("E6 group B", actual, expected)
                notes.append("group B returned a result that MATCHED round1 reference (no misplacement exposed)")
            except BaseException:
                detected_b = True
                notes.append("group B silently returned a WRONG result; detected by per-token comparison")
        except BaseException as exc:
            detected_b = True
            notes.append("group B rejected the mismatched handle: " + type(exc).__name__)
        checks["group_b_silent_error_detected"] = detected_b
        # Restore: correct handle / correct expert output must still be right.
        h.check_combine(r1, 0.0, "E6 restore with correct handle")
        checks["restore_correct"] = True
        ok = checks["restore_correct"] and (rejected_a or detected_b)
        cls = "passed" if ok else "defect_suspected"
        return ok, cls, checks, notes

    raise RuntimeError("unknown scenario " + sid)


def pad_rows(t, rows, torch):
    """Pad a tensor to `rows` by repeating its last row (for same-row-count group)."""
    if t.size(0) >= rows:
        return t[:rows]
    pad = t[-1:].expand(rows - t.size(0), -1).contiguous()
    return torch.cat([t, pad], dim=0)


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
            print("ABNORMAL %s %s rank=%d -> %s %.1fs" %
                  (h.o.api, sid, h.rank, cls, elapsed), flush=True)
        except BaseException as exc:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            # An illegal input that surfaces as an asynchronous NPU device error
            # (vector core exception / EE9999) is a finding, not a retryable
            # transient failure: the process cannot continue and prior rounds
            # are not recoverable.
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
            print("ABNORMAL %s %s rank=%d -> %s %s" %
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
    ap.add_argument("--scenarios", required=True, help="comma separated, e.g. E1,E2")
    ap.add_argument("--num-tokens", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--experts-per-rank", type=int, default=32)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--aiv", type=int, default=32)
    ap.add_argument("--workspace-bytes", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--v1-capacity", choices=("declare", "omit"), default="declare",
                    help="V1 是否声明 num_max_tokens_per_rank；E5 需要 declare 才有校验")
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
        # V1 的容量校验只在显式声明 num_max_tokens_per_rank 时生效
        # （buffer_v1.py:508）。bench_v1_bandwidth 不传这个参数，等于关掉了
        # 校验；E5 要测"超容量被拒"，必须按文档把容量声明出来。
        capacity = o.max_tokens if o.v1_capacity == "declare" else None
        buffer = Buffer(dist.group.WORLD, num_nvl_bytes=o.workspace_bytes,
                        num_max_tokens_per_rank=capacity,
                        explicitly_destroy=True, transport="udma", use_1g_huge_pages=False)
        config = Config(o.aiv)
    else:
        buffer = ElasticBuffer(dist.group.WORLD, num_max_tokens_per_rank=o.max_tokens,
                               hidden=o.hidden, num_topk=o.num_topk,
                               use_fp8_dispatch=(o.dtype == "fp8"),
                               num_bytes=o.workspace_bytes, explicitly_destroy=True,
                               transport="udma", use_1g_huge_pages=False)
        config = None
    print("BUFFER_INITIALIZED rank=%d api=%s ep=%d" % (rank, o.api, o.ep), flush=True)

    h = Harness(o, torch, dist, golden, profiling, shared, buffer, config, V1Operations)
    h.base_args = base
    sids = [s for s in o.scenarios.split(",") if s]
    try:
        run_scenarios(h, sids, o.output)
        dist.barrier()
        # Destruction failures must not erase already-written scenario results.
        try:
            buffer.destroy()
        except BaseException:
            pass
        dist.destroy_process_group()
        print("ABNORMAL_GROUP_DONE rank=%d api=%s" % (rank, o.api), flush=True)
        return 0
    except BaseException:
        write_json(o.output / ("group_rank%d_failure.json" % rank),
                   dict(rank=rank, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
