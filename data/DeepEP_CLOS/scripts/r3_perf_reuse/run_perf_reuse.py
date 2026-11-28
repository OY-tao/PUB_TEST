#!/usr/bin/env python3
"""R3 性能补测「复用统一 Buffer」控制器 + 自动接续守护。

两套对比中的第 2 套：原口径（每 case 独立进程组 + 独立 Buffer）由
`auto_start_supp.py` 跑完 36 例后，本脚本用**持久进程组 + 分组复用 Buffer**
重跑同样 36 例，结果写 `round3_perf_supp_reuse.csv`。

测量逻辑复用官方 `bench_v2_bandwidth.worker()`（monkeypatch，口径零改动）：
  reuse_worker.py 把 dist.init/destroy_process_group 与 ElasticBuffer 构造
  改为「组内复用」，4 op 的 profiling / warmup / iterations / 精度校验全部
  沿用官方实现。GBps 用官方 summarize 公式在 rank0 汇总。

分组：按 (hidden, num_topk, dtype, huge_pages) 分组，组内 6 例复用一份
Buffer（容量 num_max_tokens_per_rank = 组内 max tokens = 16384）；组边界
destroy。EP4 / EP64 各 18 例 -> 各 3 次 Buffer 初始化（原口径各 18 次）。

节点（沿用补测，避开 .4/.5 与 .7）：
  EP4  -> [7] = .11 单机 4 卡
  EP64 -> [2,4,5,6] = .6/.8/.9/.10 四机 64 rank
两组节点不重叠；按 EP 顺序执行（先 EP4 后 EP64），master_port 分开
（29870 / 29880）避免 rdzv-id 与端口冲突；每轮 attempt 输出目录独立。

触发：等 `round3_perf_supp.csv` 出现 36 个 run_id（第 1 套完成）+
EP4/EP64 所需 5 台全空闲 + 无其它补测/复用控制器在跑。
被 kill / 出错：按 checkpoint（按 EP）重启，直到 36 例写完。

用法： python3 -u run_perf_reuse.py [--interval 60] [--max-wait 7200]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent          # .../scripts/r3_perf_reuse
SCRIPTS = HERE.parent                                   # .../scripts
DATA = SCRIPTS.parent                                   # .../data/DeepEP_CLOS
STATUS = DATA / "round3_reuse_status.json"
SUPP_CSV = DATA / "round3_perf_supp.csv"       # 第 1 套（原口径）完成判据
REUSE_CSV = DATA / "round3_perf_supp_reuse.csv"  # 第 2 套结果
REUSE_LOG = "/tmp/r3_reuse_4_64.log"
SPEC = SCRIPTS / "perf_cases_supp.json"

HOSTS = ["30.104.128.%d" % i for i in range(4, 12)]
CONTAINER = "deepep_ep128_full_matrix_20261128_073617"
REMOTE = "/work/r3_perf_reuse"
SSH = ["ssh", "-T", "-p", "56000", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]

# 性能集合（避开 .4/.5 被占、.7 性能禁区）
EP_NODES = {4: [7], 64: [2, 4, 5, 6]}
NEEDED_HOSTS = ["30.104.128.%d" % i for i in (6, 8, 9, 10, 11)]
MASTER_PORTS = {4: 29870, 64: 29880}
MAX_TOKENS = 16384
LOCAL_RANKS = 16

# 官方 bench / golden 的 PYTHONPATH（新版 stage_metrics，不能混精度补丁的旧版）
PY_PATHS = [
    REMOTE,
    "/work/DeepEP_lueelu/benchmarks/clos",
    "/work/DeepEP_lueelu/tests/clos",
    "/work/env/lib/python3.13/site-packages",
]
LD_PATHS = [
    "/work/env/lib/python3.13/site-packages/deep_ep/.libs",
    "/opt/conda/envs/hunyuan4_tenpu/lib/python3.13/site-packages/torch/lib",
    "/opt/conda/envs/hunyuan4_tenpu/lib/python3.13/site-packages/torch_npu/lib",
]


def log(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def idle_count(host):
    try:
        p = subprocess.run(
            SSH + ["root@" + host, "npu-smi info 2>/dev/null | grep -c 'No running processes found in NPU'"],
            capture_output=True, text=True, timeout=45)
        out = (p.stdout or "").strip()
        return int(out) if out.isdigit() else -1
    except Exception:
        return -1


def all_needed_idle():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        return dict(zip(NEEDED_HOSTS, pool.map(idle_count, NEEDED_HOSTS)))


def controllers():
    out = []
    me = os.getpid()
    for p in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            pid = int(p.name)
            if pid == me:
                continue
            s = b" ".join((p / "cmdline").read_bytes().split(bytes([0])))
            if s.startswith((b"python3", b"python", b"/opt/conda", b"/usr/bin/python")) and (
                    b"run_round3.py" in s or b"run_perf_reuse.py" in s):
                out.append(s.decode("utf-8", "replace"))
        except (OSError, ValueError, ProcessLookupError):
            pass
    return out


def other_controllers():
    return [c for c in controllers()
            if "run_perf_reuse.py" not in c or "--worker-controller" in c]


def supp_done():
    if not SUPP_CSV.exists():
        return 0
    try:
        rows = list(csv.DictReader(SUPP_CSV.open()))
        return len({r["run_id"] for r in rows})
    except Exception:
        return 0


def write_status(state, **kw):
    d = {"state": state, "ts": time.strftime("%F %T")}
    d.update(kw)
    STATUS.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# plan 生成：按 compatibility_key 分组排序，让同组 case 连续
# --------------------------------------------------------------------------- #
def build_plan():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    ax, fx, experts = spec["axes"], spec["fixed"], spec["experts"]
    plans = {}
    for ep in (4, 64):
        cases = []
        for tokens in ax["num_tokens"]:
            for hidden in ax["hidden"]:
                for aiv in ax["aiv"]:
                    p = {
                        "num_tokens": tokens, "hidden": hidden,
                        "num_topk": fx["num_topk"], "experts_per_rank": experts // ep,
                        "aiv": aiv, "warmup": fx["warmup"], "iterations": fx["iterations"],
                        "seed": fx["seed"], "route_mode": fx["route_mode"],
                    }
                    variant = "r_ep%d_t%d_h%d_a%d" % (ep, tokens, hidden, aiv)
                    cases.append({"variant": variant, "dtype": fx["dtypes"][0],
                                  "expand": fx["expand"][0], "params": p})
        # 按 (hidden, num_topk, dtype) 分组排序，组内复用同一 Buffer
        cases.sort(key=lambda c: (c["params"]["hidden"], c["params"]["num_topk"],
                                  c["dtype"], c["params"]["num_tokens"], c["params"]["aiv"]))
        plans[ep] = {"ep_size": ep, "nnodes": len(EP_NODES[ep]),
                     "max_tokens": MAX_TOKENS, "cases": cases}
        assert len(cases) == 18
    return plans


def deploy(plans):
    """把 reuse_worker + 两个 plan 部署到 8 机（本机 -> 宿主机 tmp -> 容器）。

    生成的 plan 写到 /tmp/r3_reuse_deploy，不落在仓库里（避免污染 git 工作区）。
    plans 可以是「按 checkpoint 过滤后」的子集，用于崩溃后只补未完成的 case。
    """
    tmp = pathlib.Path("/tmp/r3_reuse_deploy")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "plan_ep4.json").write_text(json.dumps(plans[4], indent=2) + "\n")
    (tmp / "plan_ep64.json").write_text(json.dumps(plans[64], indent=2) + "\n")
    files = [str(HERE / "reuse_worker.py"),
             str(tmp / "plan_ep4.json"), str(tmp / "plan_ep64.json")]

    def one(host):
        subprocess.run(["scp", "-q", "-P", "56000", "-o", "BatchMode=yes"] + files +
                       ["root@" + host + ":/tmp/"], check=True, timeout=180)
        inner = ("mkdir -p %s && " % REMOTE +
                 " && ".join("docker cp /tmp/%s %s:%s/%s" % (n, CONTAINER, REMOTE, n)
                             for n in ("reuse_worker.py", "plan_ep4.json", "plan_ep64.json")) +
                 " && echo OK")
        p = subprocess.run(SSH + ["root@" + host, "bash -lc " + shlex.quote(inner)],
                           capture_output=True, text=True, timeout=180)
        return host, p.returncode, (p.stdout or "").strip()[-40:], (p.stderr or "").strip()[-120:]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for h, rc, o, e in pool.map(one, HOSTS):
            log("deploy %s rc=%s %s %s" % (h, rc, o, e))
            if rc != 0:
                raise RuntimeError("deploy failed on " + h)


def torchrun_cmd(ep, node_idx, run_id):
    """一个 EP 一个持久进程组；ep 内所有节点同时起 torchrun（静态 rdzv）。
    EP4 单机 nnodes=1、nproc=4；EP64 四机 nnodes=4、nproc=16。
    """
    idxs = EP_NODES[ep]
    nnodes = len(idxs)
    nproc = ep if nnodes == 1 else LOCAL_RANKS
    master = HOSTS[idxs[0]]
    port = MASTER_PORTS[ep]
    plan = "plan_ep4.json" if ep == 4 else "plan_ep64.json"
    # 每轮 attempt 独立输出目录：官方 worker() 对 destination 用 mkdir(exist_ok=False)，
    # 复用同一目录会在「补跑同一 variant」时撞 FileExistsError。
    out = "%s/out/%s/ep%d" % (REMOTE, run_id, ep)
    argv = [
        "/work/env/bin/python", "-m", "torch.distributed.run",
        "--max-restarts=0", "--nnodes=%d" % nnodes,
        "--nproc-per-node=%d" % nproc,
        "--node-rank=%d" % node_idx,
        "--rdzv-backend=static", "--rdzv-id=%s_ep%d" % (run_id, ep),
        "--master-addr=" + master, "--master-port=%d" % (port + 1),
        "--rdzv-conf=timeout=900",
        REMOTE + "/reuse_worker.py",
        "--plan", REMOTE + "/" + plan,
        "--output", out,
        "--ep-size=%d" % ep,
        "--worker-node-rank=%d" % node_idx,
        "--run-id", run_id + "_ep%d" % ep,
        "--master-addr", master,
        "--master-port=%d" % port,
    ]
    shell = ("source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
             "unset SHMEM_UID_SESSION_ID ASCEND_RT_VISIBLE_DEVICES DEEPEP_CLOS_QP_PLANS && "
             "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && "
             "export LD_LIBRARY_PATH=" + ":".join(LD_PATHS) + ":$LD_LIBRARY_PATH && "
             "export PYTHONPATH=" + ":".join(PY_PATHS) + ":$PYTHONPATH && "
             "env -u DEEPEP_CLOS_QP_PLANS " + shlex.join(argv))
    return "docker exec %s bash -lc %s" % (CONTAINER, shlex.quote(shell))


def run_ep(ep, run_id, logf):
    """并行启动一个 EP 的所有节点 torchrun，等全部结束，返回 {node_idx: rc}。"""
    idxs = EP_NODES[ep]
    cmds = {i: torchrun_cmd(ep, i, run_id) for i in range(len(idxs))}

    def one(i):
        host = HOSTS[idxs[i]]
        # 按 attempt 分文件，续跑时保留上一轮日志便于定位失败原因
        lp = open("%s.%s.ep%d.node%d.log" % (logf[:-4], run_id, ep, i), "w")
        try:
            p = subprocess.run(SSH + ["root@" + host, cmds[i]], stdout=lp,
                               stderr=subprocess.STDOUT, timeout=8 * 3600)
            return i, p.returncode
        finally:
            lp.close()

    rcs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(idxs)) as pool:
        futs = [pool.submit(one, i) for i in range(len(idxs))]
        for f in concurrent.futures.as_completed(futs):
            i, rc = f.result()
            rcs[i] = rc
    return rcs


def collect_ep(ep, run_id):
    """rank0 的 summary_reuse.csv 写在 EP_NODES[ep][0] 那台机器的容器本地盘。"""
    host = HOSTS[EP_NODES[ep][0]]
    path = "%s/out/%s/ep%d/summary_reuse.csv" % (REMOTE, run_id, ep)
    code = ("import pathlib,sys,json;"
            "p=pathlib.Path('%s');"
            "print(p.read_text() if p.exists() else '')" % path)
    r = subprocess.run(SSH + ["root@" + host,
                              "docker exec %s /work/env/bin/python -c %s" % (CONTAINER, shlex.quote(code))],
                       capture_output=True, text=True, timeout=120)
    if r.returncode or not r.stdout.strip():
        return []
    return list(csv.DictReader(__import__("io").StringIO(r.stdout)))


def append_csv(rows):
    cols = ["variant", "ep_size", "hidden", "num_topk", "dtype", "aiv", "num_tokens", "op",
            "gbps_mean", "gbps_min", "gbps_max", "buffer_reused",
            "buffer_initializations_so_far", "elapsed_s", "mode"]
    new = not REUSE_CSV.exists()
    with REUSE_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        if new:
            w.writeheader()
        for r in rows:
            r["mode"] = "reuse"
            w.writerow({k: r.get(k, "") for k in cols})


def completed_variants():
    if not REUSE_CSV.exists():
        return set()
    try:
        return {r["variant"] for r in csv.DictReader(REUSE_CSV.open())}
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-wait", type=int, default=7200)
    ap.add_argument("--max-attempts", type=int, default=6)
    o = ap.parse_args()
    deadline = time.time() + o.max_wait
    plans = build_plan()
    run_id = "r3_reuse_%s" % time.strftime("%Y%m%d_%H%M%S")
    log("复用版控制器启动：等第1套补测 36 例完成 + 5 台空闲；EP4/EP64 各 18 例，持久进程组复用 Buffer")
    write_status("waiting", started=time.strftime("%F %T"), mode="EP4/EP64 各18例, 按hidden分组复用Buffer",
                 note="测量逻辑沿用官方 worker(); 容量 max_tokens=16384")

    while time.time() < deadline:
        try:
            sd = supp_done()
            idle = all_needed_idle()
            n_idle = sum(1 for v in idle.values() if v >= 16)
            others = other_controllers()
            ready = (sd >= 36 and n_idle == 5 and not others)
            log("第1套=%d/36 空闲=%d/5 其它控制器=%d ready=%s" % (sd, n_idle, len(others), ready))
            if ready:
                break
            write_status("waiting", supp_done=sd, idle_npus=idle, ready=ready)
        except Exception as exc:
            log("轮询异常（继续）: %s" % str(exc)[:200])
        time.sleep(o.interval)
    else:
        write_status("timeout", note="等第1套完成/节点空闲超时")
        return 1

    log("触发条件满足，开始跑复用版")
    write_status("deployed", run_id=run_id)

    # 逐轮 attempt：每轮只跑「本 EP 尚未写入 REUSE_CSV 的 variant」，被 kill /
    # 出错后按 checkpoint 续跑；每轮用独立 rdzv-id 与独立输出目录，避免与上一轮
    # 残留的 rdzv / mkdir(exist_ok=False) 目录冲突。
    for attempt in range(1, o.max_attempts + 1):
        att_run_id = "%s_a%d" % (run_id, attempt)
        for ep in (4, 64):
            done = completed_variants()
            remaining = [c for c in plans[ep]["cases"] if c["variant"] not in done]
            if not remaining:
                log("EP%d 已完成，跳过" % ep)
                continue
            att_plans = dict(plans)
            att_plans[ep] = dict(plans[ep], cases=remaining)
            deploy(att_plans)
            log("attempt=%d EP%d 启动：%d 例（补跑），节点=%s" %
                (attempt, ep, len(remaining), EP_NODES[ep]))
            write_status("running", attempt=attempt, ep=ep, run_id=att_run_id,
                         remaining=len(remaining))
            rcs = run_ep(ep, att_run_id, REUSE_LOG)
            log("attempt=%d EP%d rcs=%s" % (attempt, ep, rcs))
            rows = collect_ep(ep, att_run_id)
            # 只写本 EP 未写过的 variant（checkpoint 幂等；旧 summary 残留也不会重复）
            fresh = [r for r in rows if r["variant"] not in done]
            if fresh:
                append_csv(fresh)
            log("attempt=%d EP%d 收集 %d 行，新增 %d 行（4 op/例）" %
                (attempt, ep, len(rows), len(fresh)))
            write_status("running", attempt=attempt, ep=ep, collected=len(fresh), rcs=rcs)
        total = len(completed_variants())
        if total >= 36:
            write_status("completed", variants=total, attempts=attempt,
                         note="复用版 36 例齐全")
            log("DONE variants=%d/36 attempts=%d" % (total, attempt))
            return 0
        log("attempt=%d 结束仍缺 %d 例，%ds 后按 checkpoint 补跑" %
            (attempt, 36 - total, o.interval))
        write_status("retry", variants=total, attempt=attempt,
                     note="未达 36 例，按 checkpoint 重启补跑")
        time.sleep(o.interval)

    total = len(completed_variants())
    write_status("retry", variants=total,
                 note="attempts 用尽仍缺 %d 例，需人工重启控制器" % (36 - total))
    log("DONE variants=%d/36 attempts=%d (incomplete)" % (total, o.max_attempts))
    return 1


if __name__ == "__main__":
    sys.exit(main())
