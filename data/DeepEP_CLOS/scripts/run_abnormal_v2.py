"""Controller for E1-E9 abnormal scenarios v2 (prepared, NOT executed).

DEFAULT IS PLAN-ONLY: without --run this script only writes
`abnormal_v2_plan.json` and prints the plan; it never ssh-es to a host,
never deploys, and never starts NPU work. Use --run to actually execute.

Covers the gaps recorded in 2to16P异常场景结果_20261127.md §8.5/§8.6:
  * full EP2/4/8/16 (v1 only ran EP8/EP16)          -> --eps
  * real over-capacity E5 input                     -> abnormal_worker_v2.py
  * repeatability for multi-round handle isolation  -> --repeat
  * E5/E6 isolation in their own process groups     -> build_groups()
  * controller-side elapsed (v1 E5 recorded 0.0)    -> run_group() timer
Results go to NEW files (abnormal_v2_results.csv / status_v2.json); v1
artifacts are never overwritten.
"""
import argparse
import concurrent.futures
import csv
import json
import pathlib
import shlex
import subprocess
import threading
import time
import traceback

BASE = pathlib.Path(__file__).resolve().parent
TAG = "precision_2to16p_abnormal_v2_20261127"
REMOTE = "/workspace/" + TAG
SSH = ["ssh", "-T", "-p", "56000", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]
V1_SCENARIOS = ["E1", "E2", "E3", "E4", "E7", "E8", "E9", "E5", "E6"]
V2_SCENARIOS = ["E1", "E2", "E3", "E4", "E5", "E6"]
CAPACITY = 16384
MAX_RETRY = 2
JOBS = {
    2: {"host": "30.104.128.6", "container": "deepep-perf-v2-67-puVjF7-node0",
        "mount": "/data2/perf_v2_67_20261125_puVjF7", "port": 29930,
        "workspace_bytes": 547356672},
    4: {"host": "30.104.128.6", "container": "deepep-perf-v2-67-puVjF7-node0",
        "mount": "/data2/perf_v2_67_20261125_puVjF7", "port": 29931,
        "workspace_bytes": 1084227584},
    8: {"host": "30.104.128.6", "container": "deepep-perf-v2-67-puVjF7-node0",
        "mount": "/data2/perf_v2_67_20261125_puVjF7", "port": 29932,
        "workspace_bytes": 2157969408},
    16: {"host": "30.104.128.7", "container": "deepep-perf-v2-20261125-Lu2QQN-node1",
         "mount": "/data2/perf_v2_20261125_Lu2QQN", "port": 29933,
         "workspace_bytes": 4305453056},
}
DEVICE_ERROR_MARKERS = ("EE9999", "vector core", "Inner error", "device error",
                        "rtDeviceSynchronize", "npuSynchronizeDevice")
LOCK = threading.Lock()
STATE = {}
STATUS = BASE / "status_v2.json"


def save():
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(STATE, indent=2, ensure_ascii=False))
    tmp.replace(STATUS)


def remote(job, cmd, **kw):
    return subprocess.run(SSH + ["root@" + job["host"], cmd], **kw)


def read_remote(job, cmd, timeout=90):
    r = remote(job, cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(job["host"] + ": " + (r.stderr or "")[-400:] +
                           (r.stdout or "")[-400:])
    return r.stdout


def build_groups(ep, api, sids, repeat_index):
    """Safe scenarios share one Buffer; E5/E6 are isolated (device-crash risk)."""
    safe = [s for s in sids if s not in ("E5", "E6")]
    groups = []
    if safe:
        groups.append({"group_id": "%s_safe" % api, "api": api, "sids": safe})
    for s in sids:
        if s in ("E5", "E6"):
            groups.append({"group_id": "%s_%s" % (api, s), "api": api, "sids": [s]})
    for g in groups:
        g.update({"ep": ep, "repeat_index": repeat_index})
    return groups


def build_plan(eps, only, repeat):
    groups = []
    for ep in eps:
        selected = {"v1": V1_SCENARIOS, "v2": V2_SCENARIOS}
        if only:
            want = set(only.split(","))
            selected = {a: [s for s in v if s in want] for a, v in selected.items()}
        for rep in range(1, repeat + 1):
            for api, sids in selected.items():
                groups.extend(build_groups(ep, api, sids, rep))
    return {
        "generated_at": time.strftime("%F %T"),
        "note": "PREPARED ONLY — not executed. Run with --run to start NPU work.",
        "tag": TAG, "worker": "abnormal_worker_v2.py",
        "eps": list(eps), "only": only or None, "repeat": repeat,
        "capacity": CAPACITY, "num_tokens": 1024, "hidden": 2048,
        "num_topk": 8, "experts_per_rank": 32, "dtype": "bf16", "aiv": 32,
        "scenarios_per_ep": len(V1_SCENARIOS) + len(V2_SCENARIOS),
        "total_groups": len(groups),
        "total_executions": sum(len(g["sids"]) for g in groups),
        "assignment": {str(ep): {"host": JOBS[ep]["host"],
                                 "container": JOBS[ep]["container"],
                                 "port": JOBS[ep]["port"]} for ep in eps},
        "corrections": {
            "E5": "real over-capacity tokens (num_tokens=capacity+1, legal topk_idx values); "
                  "V1 Buffer constructed with num_max_tokens_per_rank so buffer_v1.py:509 fires",
            "E4/E9": "second same-layout expert output is a pure golden tensor, no second dispatch",
            "E7/E8/E9": "notes explicitly state expanded=True / via expand+aggregate",
            "E5_judgement": "except Exception (not BaseException)",
            "EP_range": "full 2/4/8/16 (v1 only ran 8/16)",
            "repeatability": "--repeat N for multi-round handle isolation (E1/E2/E3)",
            "elapsed": "controller-side timer, so crashes no longer record 0.0",
        },
        "groups": groups,
    }


def deploy(job):
    read_remote(job, "mkdir -p " + shlex.quote(job["mount"] + "/" + TAG), timeout=30)
    subprocess.run(["scp", "-q", "-P", "56000", "-o", "BatchMode=yes",
                    str(BASE / "abnormal_worker_v2.py"),
                    "root@" + job["host"] + ":" + job["mount"] + "/" + TAG + "/"], check=True)


def check_idle(job):
    import re
    text = read_remote(job, "npu-smi info", timeout=30)
    idle = {int(n) for n in re.findall(r"No running processes found in NPU\s+(\d+)", text)}
    busy = [n for n in range(16) if n not in idle]
    if busy:
        raise RuntimeError("%s NPUs busy: %s" % (job["host"], busy))


def cleanup(job):
    code = ("import os,signal,pathlib\n"
            "me=os.getpid()\n"
            "tag=%r\n"
            "w=b'abnormal_'+b'worker_v2.py'\n"
            "for p in pathlib.Path('/proc').glob('[0-9]*'):\n"
            "  try:\n"
            "    pid=int(p.name)\n"
            "    if pid==me: continue\n"
            "    s=b' '.join((p/'cmdline').read_bytes().split(bytes([0])))\n"
            "    if tag in s and w in s:\n"
            "      os.kill(pid,signal.SIGKILL)\n"
            "  except (OSError,ValueError,ProcessLookupError): pass\n"
            "print('cleaned')\n") % TAG.encode()
    try:
        read_remote(job, "docker exec " + job["container"] + " python -c " + shlex.quote(code),
                    timeout=60)
    except Exception:
        pass


def collect(job, out_abs):
    code = '''import pathlib,json
p=pathlib.Path(ROOT)
out={}
if p.is_dir():
  for f in p.rglob("*.json"):
    if f.name in ("result.json","precision.json","exit.json","process.json"):
      out[str(f.relative_to(p))]=f.read_text()
print(json.dumps(out))
'''.replace("ROOT", repr(out_abs))
    r = remote(job, "docker exec " + job["container"] + " python -c " + shlex.quote(code),
               capture_output=True, text=True, timeout=180)
    if r.returncode:
        return {}
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def judge(sid, ep, files):
    classes, notes, elapsed = [], [], []
    for g in range(ep):
        prefix = sid + "/rank" + str(g) + "/"
        try:
            res = json.loads(files[prefix + "result.json"])
            prec = json.loads(files[prefix + "precision.json"])
            ex = json.loads(files[prefix + "exit.json"])
        except KeyError:
            return "failed_runtime", ["rank%d missing artifacts" % g], 0.0
        err = str(res.get("error", ""))
        ok = (res.get("precision_passed") is True and res.get("precision_only") is True
              and res.get("operators") == {} and res.get("buffer_reused") is True
              and prec.get("passed") is True and ex.get("code") == 0)
        if not ok:
            if res.get("classification") == "device_crash" or \
                    any(k in err for k in DEVICE_ERROR_MARKERS):
                return "device_crash", ["rank%d device-level crash: %s" % (g, err[:240])], 0.0
            return "failed_runtime", ["rank%d checks failed: %s" % (g, err)], 0.0
        classes.append(res.get("classification", "failed_runtime"))
        elapsed.append(float(res.get("elapsed_s", 0.0)))
        if g == 0:
            notes = res.get("notes", [])
    if len(set(classes)) != 1:
        return "failed_runtime", ["rank classifications disagree: %s" % classes], max(elapsed)
    return classes[0], notes, max(elapsed)


def run_group(job, g, attempt):
    ep, api, sids = g["ep"], g["api"], g["sids"]
    group_id = g["group_id"]
    rep = g["repeat_index"]
    out_abs = REMOTE + "/out/ep%d/r%d/%s/a%d" % (ep, rep, group_id, attempt)
    run_id = "abnv2_ep%d_r%d_%s_a%d" % (ep, rep, group_id, attempt)
    cleanup(job)
    read_remote(job, "docker exec " + job["container"] + " rm -rf " + shlex.quote(out_abs),
                timeout=30)
    argv = [
        "python", "-m", "torch.distributed.run",
        "--nnodes=1", "--nproc-per-node=%d" % ep, "--node-rank=0",
        "--master-addr=127.0.0.1", "--master-port=%d" % job["port"],
        "--max-restarts=0",
        REMOTE + "/abnormal_worker_v2.py",
        "--ep=%d" % ep, "--api=" + api, "--output=" + out_abs,
        "--run-id=" + run_id, "--scenarios=" + ",".join(sids),
        "--num-tokens=1024", "--hidden=2048", "--num-topk=8",
        "--experts-per-rank=32", "--dtype=bf16", "--aiv=32",
        "--workspace-bytes=%d" % job["workspace_bytes"],
        "--capacity=%d" % CAPACITY, "--seed-base=20260823",
    ]
    inner = ("source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
             "unset SHMEM_UID_SESSION_ID ASCEND_RT_VISIBLE_DEVICES && "
             "export SHMEM_UDMA_CONNECT_TIMEOUT_MS=300000 OMP_NUM_THREADS=1 "
             "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && "
             "export PYTHONPATH=" + REMOTE +
             ":/workspace/deepep/benchmarks/clos:/workspace/runtime:$PYTHONPATH && "
             "timeout 2400 " + shlex.join(argv))
    cmd = ("docker exec -e SHMEM_UDMA_CONNECT_TIMEOUT_MS=300000 " + job["container"] +
           " bash -lc " + shlex.quote(inner))
    log = BASE / ("run_ep%d_r%d_%s_a%d.log" % (ep, rep, group_id, attempt))
    t0 = time.time()
    with log.open("w") as f:
        try:
            rc = remote(job, cmd, stdout=f, stderr=subprocess.STDOUT, timeout=2500).returncode
        except Exception as exc:
            rc = "EXC:" + str(exc)[:120]
    controller_elapsed = round(time.time() - t0, 1)
    files = collect(job, out_abs)
    results = {}
    for sid in sids:
        cls, notes, elapsed = judge(sid, ep, files)
        results[sid] = {"classification": cls, "notes": notes,
                        "worker_elapsed_s": elapsed,
                        "controller_elapsed_s": controller_elapsed,
                        "attempt": attempt, "repeat": rep, "output": out_abs}
    return {"rc": rc, "results": results}


def done_count():
    return sum(len(sids) for ep_map in STATE["results"].values()
               for sids in ep_map.values())


def run_host(host, groups):
    """Groups of one host run serially (no NPU contention).

    This host's results are kept in `final` and merged into STATE only once,
    at the end, so progress cannot double-count.
    """
    final = {}
    for g in groups:
        pending = list(g["sids"])
        for attempt in range(1, MAX_RETRY + 2):
            if not pending:
                break
            gg = dict(g, sids=pending)
            rec = run_group(JOBS[g["ep"]], gg, attempt)
            with LOCK:
                for sid, r in rec["results"].items():
                    final.setdefault(str(g["ep"]), {}).setdefault(g["api"], {})[sid] = r
                    STATE["attempts"].append(
                        {"ep": g["ep"], "api": g["api"], "scenario": sid,
                         "repeat": g["repeat_index"], "attempt": attempt,
                         "classification": r["classification"], "rc": rec["rc"]})
                    STATE["done"] = done_count() + sum(
                        len(v) for m in final.values() for v in m.values())
                    save()
                print("ABNORMAL_V2 ep%d r%d %s a%d rc=%s -> %s" %
                      (g["ep"], g["repeat_index"], g["group_id"], attempt, rec["rc"],
                       {k: v["classification"] for k, v in rec["results"].items()}), flush=True)
            pending = [sid for sid in pending
                       if rec["results"][sid]["classification"] == "failed_runtime"]
    with LOCK:
        for ep_str, apis in final.items():
            for api, sids in apis.items():
                STATE["results"].setdefault(ep_str, {}).setdefault(api, {}).update(sids)
        STATE["done"] = done_count()
        save()


def write_summary():
    rows = []
    for ep_str, apis in sorted(STATE["results"].items(), key=lambda x: int(x[0])):
        for api, sids in sorted(apis.items()):
            for sid, r in sorted(sids.items()):
                rows.append({"ep": int(ep_str), "api": api, "scenario": sid,
                             "classification": r["classification"],
                             "repeat": r["repeat"], "attempt": r["attempt"],
                             "notes": "; ".join(r["notes"])[:600],
                             "worker_elapsed_s": r["worker_elapsed_s"],
                             "controller_elapsed_s": r["controller_elapsed_s"],
                             "output": r["output"]})
    with (BASE / "abnormal_v2_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ep", "api", "scenario", "classification",
                                          "repeat", "attempt", "notes",
                                          "worker_elapsed_s", "controller_elapsed_s",
                                          "output"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="2,4,8,16", help="comma separated EP list")
    ap.add_argument("--only", default="", help="comma separated scenarios, e.g. E5")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat every scenario N times (handle-isolation races)")
    ap.add_argument("--run", action="store_true",
                    help="ACTUALLY execute on NPU; default is plan-only (no NPU work)")
    ap.add_argument("--plan-out", default="abnormal_v2_plan.json")
    o = ap.parse_args()
    eps = tuple(int(x) for x in o.eps.split(",") if x)
    assert all(ep in JOBS for ep in eps), "unsupported EP"
    assert o.repeat >= 1
    plan = build_plan(eps, o.only, o.repeat)
    (BASE / o.plan_out).write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print("PLAN groups=%d total_executions=%d eps=%s repeat=%d capacity=%d" %
          (len(plan["groups"]), plan["total_executions"], list(eps), o.repeat, CAPACITY))
    print("PLAN corrections:", json.dumps(plan["corrections"], ensure_ascii=False))
    if not o.run:
        print("PLAN_ONLY: wrote %s; no ssh / deploy / NPU work started." % o.plan_out)
        return
    hosts = {}
    for g in plan["groups"]:
        hosts.setdefault(JOBS[g["ep"]]["host"], []).append(g)
    STATE.update({"state": "running", "started": time.strftime("%F %T"),
                  "mode": "v2-full", "tag": TAG, "eps": list(eps),
                  "repeat": o.repeat, "only": o.only or None,
                  "total": plan["total_executions"], "done": 0,
                  "results": {}, "attempts": [],
                  "retry_policy": "failed_runtime retried; "
                                  "passed/defect_suspected/device_crash terminal"})
    save()
    for host in hosts:
        deploy(JOBS[next(g["ep"] for g in hosts[host])])
        check_idle(JOBS[next(g["ep"] for g in hosts[host])])
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futs = [pool.submit(run_host, h, gs) for h, gs in hosts.items()]
        for f in concurrent.futures.as_completed(futs):
            f.result()
    rows = write_summary()
    STATE["state"] = "completed"
    STATE["ended"] = time.strftime("%F %T")
    import collections
    STATE["counts"] = dict(collections.Counter(r["classification"] for r in rows))
    save()
    print("ABNORMAL_V2_FINISHED rows=%d counts=%s" %
          (len(rows), STATE["counts"]), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        STATE["state"] = "failed"
        STATE["error"] = traceback.format_exc()[-1500:]
        save()
        raise
