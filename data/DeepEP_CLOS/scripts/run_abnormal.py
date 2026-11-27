"""Controller: E1-E9 abnormal scenarios on EP8 (.6) and EP16 (.7), in parallel.

Per EP, the v1 group (9 scenarios) and v2 group (6 scenarios) run serially on
the same host (no NPU contention); the two hosts run in parallel. Each group is
one torchrun process group with one reused Buffer. Scenarios that end in
failed_runtime are retried with a fresh output root (checkpoint style);
passed / defect_suspected are terminal.
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
TAG = BASE.name
REMOTE = "/workspace/" + TAG
SSH = ["ssh", "-T", "-p", "56000", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]
JOBS = [
    {"ep": 8, "host": "30.104.128.6", "container": "deepep-perf-v2-67-puVjF7-node0",
     "port": 29920, "workspace_bytes": 2157969408},
    {"ep": 16, "host": "30.104.128.7", "container": "deepep-perf-v2-20261125-Lu2QQN-node1",
     "port": 29921, "workspace_bytes": 4305453056},
]
V1_SCENARIOS = ["E1", "E2", "E3", "E4", "E7", "E8", "E9", "E5", "E6"]
V2_SCENARIOS = ["E1", "E2", "E3", "E4", "E5", "E6"]
MAX_RETRY = 2
LOCK = threading.Lock()
STATE = {}
STATUS = BASE / "status.json"


def save():
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(STATE, indent=2, ensure_ascii=False))
    tmp.replace(STATUS)


def remote(job, cmd, **kw):
    return subprocess.run(SSH + ["root@" + job["host"], cmd], **kw)


def read_remote(job, cmd, timeout=90):
    r = remote(job, cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(job["host"] + ": " + (r.stderr or "")[-400:] + (r.stdout or "")[-400:])
    return r.stdout


def deploy(job):
    read_remote(job, "mkdir -p " + shlex.quote(job["mount"] + "/" + TAG), timeout=30)
    subprocess.run(["scp", "-q", "-P", "56000", "-o", "BatchMode=yes",
                    str(BASE / "abnormal_worker.py"),
                    "root@" + job["host"] + ":" + job["mount"] + "/" + TAG + "/"], check=True)


def check_idle(job):
    text = read_remote(job, "npu-smi info", timeout=30)
    import re
    idle = {int(n) for n in re.findall(r"No running processes found in NPU\s+(\d+)", text)}
    busy = [n for n in range(16) if n not in idle]
    if busy:
        raise RuntimeError("%s NPUs busy: %s" % (job["host"], busy))


def cleanup(job):
    """Kill only this tag's leftover workers inside the container."""
    code = ("import os,signal,pathlib\n"
            "me=os.getpid()\n"
            "tag=%r\n"
            "w=b'abnormal_'+b'worker.py'\n"
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


DEVICE_ERROR_MARKERS = ("EE9999", "vector core", "Inner error", "device error",
                        "rtDeviceSynchronize", "npuSynchronizeDevice")


def judge(sid, ep, files):
    """All EP ranks must be complete and agree on the classification.

    A device-level crash caused by an illegal input is a terminal finding
    (device_crash), not a retryable transient failure.
    """
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
                return "device_crash", [
                    "rank%d illegal input surfaced as NPU device-level crash: %s" %
                    (g, err[:240])], 0.0
            return "failed_runtime", ["rank%d checks failed: %s" % (g, err)], 0.0
        classes.append(res.get("classification", "failed_runtime"))
        elapsed.append(float(res.get("elapsed_s", 0.0)))
        if g == 0:
            notes = res.get("notes", [])
    if len(set(classes)) != 1:
        return "failed_runtime", ["rank classifications disagree: %s" % classes], max(elapsed)
    return classes[0], notes, max(elapsed)


def run_group(job, api, group_id, sids, attempt):
    ep = job["ep"]
    out_rel = "out/ep%d/%s/a%d" % (ep, group_id, attempt)
    out_abs = REMOTE + "/" + out_rel
    run_id = "abn_ep%d_%s_a%d_%s" % (ep, group_id, attempt, TAG[:8])
    cleanup(job)
    read_remote(job, "docker exec " + job["container"] + " rm -rf " + shlex.quote(out_abs),
                timeout=30)
    argv = [
        "python", "-m", "torch.distributed.run",
        "--nnodes=1", "--nproc-per-node=%d" % ep, "--node-rank=0",
        "--master-addr=127.0.0.1", "--master-port=%d" % job["port"],
        "--max-restarts=0",
        REMOTE + "/abnormal_worker.py",
        "--ep=%d" % ep, "--api=" + api, "--output=" + out_abs,
        "--run-id=" + run_id, "--scenarios=" + ",".join(sids),
        "--num-tokens=1024", "--hidden=2048", "--num-topk=8",
        "--experts-per-rank=32", "--dtype=bf16", "--aiv=32",
        "--workspace-bytes=%d" % job["workspace_bytes"], "--max-tokens=16384",
        "--seed-base=20260823",
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
    log = BASE / ("run_ep%d_%s_a%d.log" % (ep, group_id, attempt))
    t0 = time.time()
    with log.open("w") as f:
        try:
            rc = remote(job, cmd, stdout=f, stderr=subprocess.STDOUT, timeout=2500).returncode
        except Exception as exc:
            rc = "EXC:" + str(exc)[:120]
    files = collect(job, out_abs)
    results = {}
    for sid in sids:
        cls, notes, elapsed = judge(sid, ep, files)
        results[sid] = {"classification": cls, "notes": notes, "elapsed_s": elapsed,
                        "attempt": attempt, "output": out_abs}
    return {"rc": rc, "elapsed_s": round(time.time() - t0, 1), "results": results,
            "out_abs": out_abs}


def build_groups(api, sids):
    """Safe scenarios share one process group / Buffer; E5 and E6 are isolated.

    Both E5 (illegal input) and E6 (handle mismatch) can leave the NPU in an
    unrecoverable device-error state, so each gets its own short-lived process
    group. This is required by the scenario semantics, not a change of test
    semantics; the safe scenarios still reuse a single Buffer.
    """
    safe = [s for s in sids if s not in ("E5", "E6")]
    groups = []
    if safe:
        groups.append((api + "_safe", api, safe))
    for s in sids:
        if s in ("E5", "E6"):
            groups.append((api + "_" + s, api, [s]))
    return groups


def run_job(job, only, smoke):
    ep = job["ep"]
    selected = {"v1": V1_SCENARIOS, "v2": V2_SCENARIOS}
    if smoke:
        # E7 exercises the V1 expanded/aggregate path and its per-slot reference.
        selected = {"v1": ["E1", "E7"], "v2": ["E1"]}
    elif only:
        want = set(only.split(","))
        selected = {a: [s for s in v if s in want] for a, v in selected.items()}
    groups = []
    for api, sids in selected.items():
        groups.extend(build_groups(api, sids))
    final = {}
    for group_id, api, sids in groups:
        pending = list(sids)
        for attempt in range(1, MAX_RETRY + 2):
            if not pending:
                break
            rec = run_group(job, api, group_id, pending, attempt)
            with LOCK:
                for sid, r in rec["results"].items():
                    final.setdefault(api, {})[sid] = r
                    STATE["attempts"].append(
                        {"ep": ep, "api": api, "scenario": sid, "group": group_id,
                         "attempt": attempt, "classification": r["classification"],
                         "rc": rec["rc"]})
                    STATE["done"] = done_count() + sum(len(v) for v in final.values())
                    save()
                print("ABNORMAL ep%d %s a%d rc=%s -> %s" %
                      (ep, group_id, attempt, rec["rc"],
                       {k: v["classification"] for k, v in rec["results"].items()}), flush=True)
            pending = [sid for sid in pending
                       if rec["results"][sid]["classification"] == "failed_runtime"]
        with LOCK:
            # Merge per scenario: `final` is nested {api: {sid: result}}, so a
            # plain dict.update would replace the whole api sub-dict and drop
            # previously completed scenarios (visible in --only merge mode).
            ep_map = STATE["results"].setdefault(str(ep), {})
            for api_key, sids_map in final.items():
                ep_map.setdefault(api_key, {}).update(sids_map)
            STATE["done"] = done_count()
            save()


def done_count():
    return sum(len(sids) for ep_map in STATE["results"].values() for sids in ep_map.values())


def write_summary():
    rows = []
    for ep in (8, 16):
        for api in ("v1", "v2"):
            for sid, r in sorted(STATE["results"].get(str(ep), {}).get(api, {}).items()):
                rows.append({"ep": ep, "api": api, "scenario": sid,
                             "classification": r["classification"],
                             "notes": "; ".join(r["notes"])[:600],
                             "elapsed_s": r["elapsed_s"], "attempt": r["attempt"],
                             "output": r["output"]})
    with (BASE / "abnormal_results_ep8_ep16.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ep", "api", "scenario", "classification",
                                          "notes", "elapsed_s", "attempt", "output"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="EP8 only, E1/E7 + E1")
    ap.add_argument("--run", action="store_true", help="full 2 EPs x 15 scenarios")
    ap.add_argument("--only", default="",
                    help="comma separated scenarios to (re)run, merging into existing results")
    o = ap.parse_args()
    if not (o.smoke or o.run or o.only):
        ap.error("need --smoke / --run / --only")
    if o.smoke and o.only:
        ap.error("--smoke and --only are mutually exclusive")
    for job in JOBS:
        job["mount"] = ("/data2/perf_v2_67_20261125_puVjF7" if job["ep"] == 8
                        else "/data2/perf_v2_20261125_Lu2QQN")
    jobs = JOBS[:1] if o.smoke else JOBS
    if o.smoke:
        per_ep = 3
    elif o.only:
        want = set(o.only.split(","))
        per_ep = sum(len([s for s in v if s in want])
                     for v in (V1_SCENARIOS, V2_SCENARIOS))
    else:
        per_ep = len(V1_SCENARIOS) + len(V2_SCENARIOS)
    # --only merges into the previous full-run results so the summary stays 30 rows.
    prev_results, prev_attempts = {}, []
    if o.only and STATUS.exists():
        try:
            prev = json.loads(STATUS.read_text())
            prev_results = prev.get("results", {})
            prev_attempts = prev.get("attempts", [])
        except Exception:
            pass
    total = 30 if o.only else per_ep * len(jobs)
    STATE.update({"state": "running", "started": time.strftime("%F %T"),
                  "mode": "smoke" if o.smoke else ("only:" + o.only if o.only else "full"),
                  "tag": TAG, "total": total, "done": 0,
                  "assignment": {j["host"]: j["ep"] for j in jobs},
                  "results": prev_results, "attempts": list(prev_attempts),
                  "retry_policy": ("failed_runtime retried; "
                                   "passed/defect_suspected/device_crash terminal")})
    STATE["done"] = done_count()
    save()
    for job in jobs:
        deploy(job)
        check_idle(job)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futs = [pool.submit(run_job, j, o.only, o.smoke) for j in jobs]
        for f in concurrent.futures.as_completed(futs):
            f.result()
    rows = write_summary()
    STATE["state"] = "completed"
    STATE["ended"] = time.strftime("%F %T")
    STATE["counts"] = {}
    for r in rows:
        STATE["counts"][r["classification"]] = STATE["counts"].get(r["classification"], 0) + 1
    save()
    print("ABNORMAL_FINISHED state=%s rows=%d counts=%s" %
          (STATE["state"], len(rows), STATE["counts"]), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        STATE["state"] = "failed"
        STATE["error"] = traceback.format_exc()[-1500:]
        save()
        raise
