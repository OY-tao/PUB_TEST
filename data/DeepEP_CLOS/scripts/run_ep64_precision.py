"""EP64 precision-only projection of the V2 design. Default: prepare/check, never run."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import time

import gen_cases

HERE = pathlib.Path(__file__).resolve().parent
SPEC = HERE / "cases_1148.json"
NODES = [
    ("30.104.128.4", "deepep-main-64card-20261124-node0", False),
    ("30.104.128.5", "deepep-perf-v2-20261125-Lu2QQN-node0", True),
    ("30.104.128.6", "deepep-perf-v2-67-puVjF7-node0", True),
    ("30.104.128.7", "deepep-perf-v2-20261125-Lu2QQN-node1", True),
]
SSH = ["ssh", "-T", "-p", "56000", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]
PRECISION_SHA = {"v1": "4cfc6bb778033fb18853e6af78c10ed3fd3fd730f52649b57e8fb22494997adb", "v2": "2c44f878385919bf07f13779cd9c8c638d05b5d6556862a43181feec843a04af"}


def matrix():
    spec = json.loads(SPEC.read_text())
    built = gen_cases.build_all(spec)
    assert gen_cases.check(spec, built)[0] and sum(map(len, built.values())) == 1148
    rows = []
    for group, cases in built.items():
        for c in cases:
            original = dict(c.params)
            if group == "D" and not (original["ep-size"] == 64 and original["nnodes"] == 4):
                continue
            params = dict(original)
            params.update({"ep-size": 64, "nnodes": 4, "first-device": 0})
            params.setdefault("route-mode", "uniform")
            notes = []
            reason = None
            if group == "F":
                item = spec["groups"]["F"]["cases"][(c.idx - 1) // 4]
                if item.get("num_topk") == "E":
                    params["num-topk"] = 64 * params["experts-per-rank"]
                if item["_id"] == "F8":
                    params["route-group-size"] = 4
                    notes.append("F8 source omits required group size; explicitly use C-group size=4 for extreme-locality precision")
                if item.get("_special") == "uneven_tokens":
                    reason = "F11 requires per-rank token lengths; existing golden/CLI cannot express them"
                if item.get("ep_size") == 2:
                    reason = "F15 tests minimum EP=2; cannot replace it with EP64 without changing its meaning"
            if params["num-topk"] > min(128, 64 * params["experts-per-rank"]):
                reason = "topk=" + str(params["num-topk"]) + " exceeds min(128, EP*epr); do not silently clamp topk=E"
            if group == "G":
                notes.append("precision-only: iterations/warmup/cache settings do not collect performance or validate bandwidth accounting")
            if group == "H":
                notes.append("repeat correctness only; no comm_us samples or performance CV")
            if "expand" not in params:
                notes.append("original CLI defaults to expand=[0,1]; both layouts remain enabled")
            resource_review = params["num-tokens"] >= 32768 or params["hidden"] >= 8192 or params["num-topk"] >= 32 or params["experts-per-rank"] >= 128 or bool(params.get("huge-pages"))
            if resource_review:
                notes.append("EP64 CPU golden/HBM footprint may be large; review available resources before explicit execution")
            rows.append({"source_id": group + str(c.idx).zfill(4), "group": group, "source_tag": c.tag, "params": params, "applicable": reason is None, "not_applicable_reason": reason, "resource_review": resource_review, "notes": notes})
    assert len(rows) == 1122
    return rows


def remote(rank, command, **kwargs):
    return subprocess.run(SSH + ["root@" + NODES[rank][0], command], **kwargs)


def read_remote(rank, command):
    r = remote(rank, command, capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError(NODES[rank][0] + ": " + r.stderr + r.stdout)
    return r.stdout


def command(case, api, rank, prefix, port):
    run_id = prefix + "_" + api + "_" + case["source_id"]
    params = dict(case["params"], **{"node-rank": rank, "master-addr": NODES[0][0], "master-port": port, "run-id": run_id, "output": "/workspace/runs_precision/" + run_id})
    argv = ["python", "/workspace/deepep/benchmarks/clos/bench_" + api + "_precision.py"]
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key)
        else:
            argv.append("--" + key + "=" + str(value))
    argv.append("--precision-only")
    shell = "source /usr/local/Ascend/ascend-toolkit/set_env.sh && unset SHMEM_UID_SESSION_ID ASCEND_RT_VISIBLE_DEVICES && export SHMEM_UDMA_CONNECT_TIMEOUT_MS=300000 && "
    if NODES[rank][2]:
        shell += "export PYTHONPATH=/workspace/runtime:$PYTHONPATH && "
    shell += shlex.join(argv)
    return run_id, params["output"] + "/node" + str(rank), "docker exec -e SHMEM_UDMA_CONNECT_TIMEOUT_MS=300000 " + NODES[rank][1] + " bash -lc " + shlex.quote(shell)


def preflight(rank):
    info = json.loads(read_remote(rank, "docker inspect " + NODES[rank][1]))[0]
    if info["State"]["Status"] != "running" or info["HostConfig"]["NetworkMode"] != "host":
        raise RuntimeError("Container must already be running with host network: " + NODES[rank][1])
    text = read_remote(rank, "npu-smi info")
    idle = {int(n) for n in re.findall(r"No running processes found in NPU\s+(\d+)", text)}
    health = {int(n): h for n, h in re.findall(r"^\|\s*(\d+)\s*\|\s*Ascend[^|]*\|\s*(OK|Warning|Alarm)\s*\|", text, re.M)}
    if any(n not in idle or health.get(n) not in ("OK", "Warning") for n in range(16)):
        raise RuntimeError(NODES[rank][0] + ": busy/unhealthy/unverified NPUs; no new precision workers launched")
    hashes = read_remote(rank, "docker exec " + NODES[rank][1] + " sha256sum /workspace/deepep/benchmarks/clos/bench_v1_precision.py /workspace/deepep/benchmarks/clos/bench_v2_precision.py")
    if [line.split()[0] for line in hashes.splitlines()] != [PRECISION_SHA["v1"], PRECISION_SHA["v2"]]:
        raise RuntimeError("Precision snapshots differ or lack the verified performance-skip branches")
    version = re.search(r"npu-smi\s+(\S+)", text)
    if not version:
        raise RuntimeError("Driver version missing")
    return version.group(1)


def run_node(case, api, rank, args, directory):
    run_id, output, cmd = command(case, api, rank, args.run_prefix, args.master_port)
    dest = directory / run_id / ("node" + str(rank))
    dest.mkdir(parents=True, exist_ok=False)
    with (dest / "run.log").open("w") as log:
        process = remote(rank, cmd, stdout=log, stderr=subprocess.STDOUT)
    code = '''import json,pathlib
p=pathlib.Path(ROOT)
files={f:p.joinpath(f).read_text() for f in ("suite_exit.json","plan.json") if p.joinpath(f).is_file()}
for c in p.iterdir():
 if not c.is_dir() or "_expand" not in c.name:continue
 for rank in range(START,START+16):
  for name in ("precision.json","result.json","exit.json"):
   f=c/("rank"+str(rank))/name
   if f.is_file():files[str(f.relative_to(p))]=f.read_text()
print(json.dumps(files))
'''
    code = "ROOT=" + repr(output) + "; START=" + str(rank * 16) + "\n" + code
    artifacts = json.loads(read_remote(rank, "docker exec " + NODES[rank][1] + " python -c " + shlex.quote(code)))
    for name, text in artifacts.items():
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    suite = json.loads(artifacts.get("suite_exit.json", "{}"))
    expected = [case["params"]["dtypes"] + "_expand" + str(ex) + "_aiv" + str(case["params"]["aiv"]) for ex in ([case["params"]["expand"]] if "expand" in case["params"] else [0, 1])]
    passed = process.returncode == 0 and suite.get("code") == 0 and set(suite.get("completed", [])) == set(expected)
    for variant in expected:
        for global_rank in range(rank * 16, rank * 16 + 16):
            prefix = variant + "/rank" + str(global_rank) + "/"
            precision = json.loads(artifacts.get(prefix + "precision.json", "{}"))
            result = json.loads(artifacts.get(prefix + "result.json", "{}"))
            exited = json.loads(artifacts.get(prefix + "exit.json", "{}"))
            passed = passed and precision.get("passed") and result.get("precision_only") and result.get("precision_passed") and result.get("destroyed") and result.get("rank") == global_rank and result.get("operators") == {} and exited.get("code") == 0
    return {"node_rank": rank, "host": NODES[rank][0], "passed": bool(passed), "exit_code": process.returncode, "source_id": case["source_id"], "log": str(dest / "run.log")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly start tests; default only checks/prepares")
    parser.add_argument("--check", action="store_true", help="check matrix only; incompatible with --run")
    parser.add_argument("--manifest-out", type=pathlib.Path, help="write prepared matrix; does not run tests")
    parser.add_argument("--groups", nargs="+", choices=list("ABCDEFGH"), default=list("ABCDEFGH"))
    parser.add_argument("--api", choices=("v1", "v2", "both"), default="both")
    parser.add_argument("--start", type=int, default=1, help="start at this position in the selected applicable case list")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-large", action="store_true", help="explicitly allow resource-review cases; no footprint guarantee")
    parser.add_argument("--run-prefix", default="precision_ep64_4to7_20261125_r1")
    parser.add_argument("--master-port", type=int, default=29860)
    args = parser.parse_args()
    if args.run and (args.check or args.manifest_out):
        parser.error("preparation flags cannot be combined with --run")
    if args.start < 1 or args.limit < 0 or not 1 <= args.master_port <= 65534 or not re.fullmatch(r"[A-Za-z0-9_-]{1,52}", args.run_prefix):
        parser.error("invalid start/limit/port/run-prefix")
    rows = [c for c in matrix() if c["group"] in args.groups]
    counts = {g: {"candidates": sum(c["group"] == g for c in rows), "applicable": sum(c["group"] == g and c["applicable"] for c in rows)} for g in args.groups}
    print(json.dumps({"groups": counts, "precision_only": True, "executed": False}, ensure_ascii=False))
    if args.manifest_out:
        with args.manifest_out.open("x") as f:
            json.dump({"source_spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(), "projection": "EP64/4 nodes; epr=32 default retained; original source IDs; not original 1148 topology sweep", "environment": {"SHMEM_UDMA_CONNECT_TIMEOUT_MS": "300000"}, "precision_only": True, "counts": counts, "cases": rows}, f, ensure_ascii=False, indent=2)
    if not args.run:
        return 0
    cases = [c for c in rows if c["applicable"]][args.start - 1:]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        parser.error("no selected cases")
    if not args.allow_large and any(c["resource_review"] for c in cases):
        parser.error("selected matrix contains resource-review cases; select a safe subset or explicitly use --allow-large after review")
    directory = HERE.parent / args.run_prefix
    directory.mkdir(exist_ok=False)
    state = {"state": "preflight", "precision_only": True, "cases": [], "not_applicable": [c for c in rows if not c["applicable"]]}
    def save():
        (directory / "status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2))
    save()
    apis = ("v1", "v2") if args.api == "both" else (args.api,)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for api in apis:
                for case in cases:
                    versions = list(pool.map(preflight, range(4)))
                    if len(set(versions)) != 1:
                        raise RuntimeError("Driver versions differ")
                    for rank in range(4):
                        _, output, _ = command(case, api, rank, args.run_prefix, args.master_port)
                        read_remote(rank, "docker exec " + NODES[rank][1] + " test ! -e " + shlex.quote(output))
                    state["state"] = "running"
                    state["current"] = {"api": api, "source_id": case["source_id"]}
                    save()
                    jobs = [pool.submit(run_node, case, api, rank, args, directory) for rank in range(4)]
                    results = [j.result() for j in jobs]
                    state["cases"].append({"api": api, "source_id": case["source_id"], "nodes": results})
                    save()
                    if not all(r["passed"] for r in results):
                        raise RuntimeError("Precision failed; stop all four nodes at this source case")
        state["state"] = "completed"
        save()
        return 0
    except Exception as exc:
        state["state"] = "failed"
        state["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
