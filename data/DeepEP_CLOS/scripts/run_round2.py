#!/usr/bin/env python3
"""R2（第二轮）测试编排：代码门禁 → 性能 48 例 → 精度 200 例 → 异常 E4/E5 重测。

对应需求原文 `../需求原文_R2_20261127.txt` 的四条：

  1. 更新 deepep main 分支代码，重编译安装 dispatch / combine 算子
  2. 新代码下重测性能 EP8-EP64，不开 1G 大页，充分利用 4 机
  3. 新代码下 EP8-EP64 精度，各复测 50 条，充分利用 4 机
  4. 异常场景重测 E4、E5

设计约束
--------
* **门禁优先**：`gate` 不通过，`perf` / `precision` 直接拒绝执行。否则会把
  "版本没换"的结果当成"新代码结果"，这是本轮最容易犯的错。
* **波次只为占满四机，不改任何用例参数**：EP64 独占四机（必须同时启动）；
  EP8/16/32 分居 node3 / node2 / node0+1 并行，四台机器都不空转。
* **不开大页**：命令行显式禁止出现 `--huge-pages`，并在产物里记录 `huge_pages=no`。
* **每 case 一次进程组**（与上轮同口径，便于逐条对比），中断可用 `--start/--limit` 续跑。
* `plan` 子命令完全离线，不 ssh、不碰设备，用于评审用例与拓扑。

拓扑（上轮实测验证过）
----------------------
node0..node3 = 30.104.128.4 / .5 / .6 / .7，各 16 卡；ssh 端口 56000；
容器内源码 `/workspace/deepep`、运行库 `/workspace/runtime`；
bench 自带 torchrun 编排（`bench_*:172` 起拼 `torch.distributed.run`），
所以每节点只需普通 `python bench_*.py --node-rank=N`，**不要外面再套 torchrun**。

用法
----
    python run_round2.py plan                       # 离线看计划（默认）
    python run_round2.py gate                       # 四机门禁：commit/.so/空闲/大页
    python run_round2.py perf  --eps 8,16,32,64     # 48 例性能
    python run_round2.py precision --eps 8,16,32,64 # 200 例精度
    python run_round2.py abnormal --eps 8,16 --api v1,v2
    python run_round2.py perf --only-ep 64 --start 0 --limit 6   # 断点续跑
"""

import argparse
import concurrent.futures
import csv
import io
import json
import os
import pathlib
import shlex
import subprocess
import sys
import threading
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent
LOGDIR = DATA / "round2_logs"
STATE_PATH = DATA / "round2_state.json"

SSH_PORT = 56000
SSH_BASE = ["ssh", "-T", "-p", str(SSH_PORT), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6"]

NODES = [
    {"node": 0, "host": "30.104.128.4"},
    {"node": 1, "host": "30.104.128.5"},
    {"node": 2, "host": "30.104.128.6"},
    {"node": 3, "host": "30.104.128.7"},
]
MASTER = NODES[0]["host"]

REPO = "/workspace/deepep"
BENCH_DIR = REPO + "/benchmarks/clos"
RUNTIME = "/workspace/runtime"
STAGE = "/workspace/round2"
CONTAINER_HINT = "deepep"          # docker ps 名字过滤

# EP -> 节点编号。8/16/32 互不重叠，故可与 64 之外的彼此并行；
# 64 需要四机全占，只能独占一波。
EP_NODES = {8: [3], 16: [2], 32: [0, 1], 64: [0, 1, 2, 3]}
PORTS = {"perf": 29860, "precision": 29880, "abnormal": 29900}

STATE = {}
LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# 基础设施
# --------------------------------------------------------------------------- #
def now():
    return time.strftime("%F %T")


def save_state():
    STATE_PATH.write_text(json.dumps(STATE, indent=2, ensure_ascii=False, default=str) + "\n",
                          encoding="utf-8")


def node_by_idx(i):
    for n in NODES:
        if n["node"] == i:
            return n
    raise KeyError(i)


def run_ssh(host, cmd, timeout=600, log=None):
    argv = SSH_BASE + ["root@" + host, cmd]
    if log is None:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    with open(log, "wb") as f:
        return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)


def container_of(node):
    """自动发现容器名；可用 --containers host=name 覆盖。"""
    key = node["host"]
    override = STATE.get("container_override", {})
    if key in override:
        return override[key]
    cached = STATE.setdefault("containers", {}).get(key)
    if cached:
        return cached
    r = run_ssh(key, "docker ps --format '{{.Names}}'", timeout=60)
    if r.returncode:
        raise RuntimeError("%s: docker ps 失败: %s" % (key, (r.stderr or "")[-200:]))
    names = [ln.strip() for ln in r.stdout.splitlines() if CONTAINER_HINT in ln]
    if not names:
        raise RuntimeError("%s: 未找到含 %r 的容器，请先拉起环境" % (key, CONTAINER_HINT))
    if len(names) > 1:
        names = sorted(names, key=len)          # 取名字最短的，通常是稳定命名
    with LOCK:
        STATE["containers"][key] = names[0]
        save_state()
    return names[0]


def docker_exec(node, inner, timeout=600, log=None):
    """在容器内用 login shell 跑一段命令。"""
    cmd = "docker exec -e SHMEM_UDMA_CONNECT_TIMEOUT_MS=300000 %s bash -lc %s" % (
        container_of(node), shlex.quote(inner))
    return run_ssh(node["host"], cmd, timeout=timeout, log=log)


def env_prefix():
    """与上轮相同的环境基线；PYTHONPATH 追加而不是覆盖。"""
    pp = ":".join([STAGE, BENCH_DIR, RUNTIME, "$PYTHONPATH"])
    return ("source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
            "unset SHMEM_UID_SESSION_ID ASCEND_RT_VISIBLE_DEVICES && "
            "export SHMEM_UDMA_CONNECT_TIMEOUT_MS=300000 OMP_NUM_THREADS=1 "
            "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && "
            "export PYTHONPATH=" + pp + " && "
            "cd " + BENCH_DIR + " && ")


def join_argv(argv):
    try:
        return shlex.join(argv)
    except AttributeError:                    # py<3.8
        return " ".join(shlex.quote(a) for a in argv)


def deploy(node, local_path, name):
    """本地 -> 宿主机 /tmp -> 容器 /workspace/round2，不依赖宿主机挂载路径。"""
    host = node["host"]
    tmp = "/tmp/round2_" + name
    subprocess.run(["scp", "-q", "-P", str(SSH_PORT), "-o", "BatchMode=yes",
                    str(local_path), "root@%s:%s" % (host, tmp)], check=True)
    c = container_of(node)
    run_ssh(host, "docker exec %s mkdir -p %s" % (c, STAGE), timeout=60)
    r = run_ssh(host, "docker cp %s %s:%s/%s" % (tmp, c, STAGE, name), timeout=120)
    if r.returncode:
        raise RuntimeError("docker cp 失败 %s: %s" % (host, (r.stderr or "")[-200:]))


def check_idle(node, need_cards):
    r = run_ssh(node["host"], "npu-smi info", timeout=60)
    import re
    idle = {int(n) for n in re.findall(r"No running processes found in NPU\s+(\d+)", r.stdout)}
    busy = [n for n in range(need_cards) if n not in idle]
    if busy:
        raise RuntimeError("%s 卡 %s 被占用（需要前 %d 张空闲）"
                           % (node["host"], busy, need_cards))
    return True


# --------------------------------------------------------------------------- #
# 用例集
# --------------------------------------------------------------------------- #
def perf_cases(eps):
    """12 例/EP：2 tokens × 3 hidden × 2 aiv，形状对齐 V2测试Case.txt。"""
    spec_path = HERE / "perf_cases_v2.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    ax = spec["axes"]
    fx = spec["fixed"]
    out = []
    for ep in eps:
        experts = spec["experts_map"][str(ep)]
        epr = experts // ep
        idx = 0
        for tokens in ax["num_tokens"]:
            for hidden in ax["hidden"]:
                for aiv in ax["aiv"]:
                    idx += 1
                    out.append({
                        "stage": "perf", "ep": ep, "idx": idx,
                        "num_tokens": tokens, "hidden": hidden, "aiv": aiv,
                        "num_topk": fx["num_topk"], "experts_per_rank": epr,
                        "dtypes": fx["dtypes"][0], "expand": fx["expand"][0],
                        "route_mode": fx["route_mode"],
                        "warmup": fx["warmup"], "iterations": fx["iterations"],
                        "seed": fx["seed"],
                    })
    return out


def precision_cases(eps):
    """复用 2-16P 的 50 条规格，只把 EP 与节点数投影过去（用例内容不变）。"""
    sys.path.insert(0, str(HERE))
    import gen_precision_cases as gpc
    spec = json.loads((HERE / "precision_cases_2to16p.json").read_text(encoding="utf-8"))
    out = []
    for ep in eps:
        for c in gpc.build(spec, ep):
            p = c.params
            out.append({
                "stage": "precision", "ep": ep, "idx": c.idx, "group": c.group,
                "tag": c.tag, "params": p, "nnodes": p.get("nnodes", 1),
            })
    return out


def nnodes_of(ep):
    return len(EP_NODES[ep])


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #
def run_on_nodes(ep, argv_tail, port, tag, timeout, extra_check=None):
    """同一 case 在所有相关节点上同时启动（bench 自己起 torchrun，必须同时到场）。

    argv_tail 为不含 --node-rank 的参数列表；每个节点自动补 node-rank/master。
    返回 {node_idx: rc}。
    """
    idxs = EP_NODES[ep]
    barrier = threading.Barrier(len(idxs))
    rcs = {}

    def one(local_rank, node_idx):
        node = node_by_idx(node_idx)
        argv = list(argv_tail)
        if len(idxs) > 1:
            argv += ["--node-rank=%d" % local_rank,
                     "--master-addr=" + MASTER,
                     "--master-port=%d" % port]
        else:
            argv += ["--node-rank=0"]
        inner = env_prefix() + "timeout %d " % max(60, timeout - 120) + join_argv(argv)
        log = LOGDIR / ("%s_node%d.log" % (tag, node_idx))
        barrier.wait(timeout=300)
        t0 = time.time()
        try:
            r = docker_exec(node, inner, timeout=timeout, log=log)
            rc = r.returncode
        except Exception as exc:
            rc = "EXC:" + str(exc)[:120]
        with LOCK:
            rcs[node_idx] = rc
        print("  [%s] node%d rc=%s %.0fs" % (tag, node_idx, rc, time.time() - t0), flush=True)

    threads = [threading.Thread(target=one, args=(i, n)) for i, n in enumerate(idxs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return rcs


def collect_json(node, root):
    code = r'''
import json, pathlib
root = pathlib.Path(ROOT)
out = {}
if root.is_dir():
    for f in root.rglob("*.json"):
        if f.name in ("result.json", "precision.json", "exit.json", "process.json"):
            try:
                out[str(f.relative_to(root))] = f.read_text()
            except OSError:
                pass
print(json.dumps(out))
'''.replace("ROOT", repr(root))
    r = docker_exec(node, "python -c " + shlex.quote(code), timeout=240)
    if r.returncode:
        return {}
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def read_text(node, path):
    r = docker_exec(node, "cat " + shlex.quote(path), timeout=120)
    return r.stdout if r.returncode == 0 else ""


def perf_argv(case, out_root, run_id):
    a = ["python", "bench_v2_bandwidth.py",
         "--ep-size=%d" % case["ep"], "--nnodes=%d" % nnodes_of(case["ep"]),
         "--num-tokens=%d" % case["num_tokens"], "--hidden=%d" % case["hidden"],
         "--num-topk=%d" % case["num_topk"],
         "--experts-per-rank=%d" % case["experts_per_rank"],
         "--dtypes=" + case["dtypes"], "--expand=%d" % case["expand"],
         "--aiv=%d" % case["aiv"], "--warmup=%d" % case["warmup"],
         "--iterations=%d" % case["iterations"], "--seed=%d" % case["seed"],
         "--route-mode=" + case["route_mode"],
         "--run-id=" + run_id, "--output=" + out_root]
    assert "--huge-pages" not in a, "本轮明令不开 1G 大页"
    return a


def precision_argv(case, out_root, run_id, api):
    bench = "bench_%s_precision.py" % api
    p = case["params"]
    a = ["python", bench, "--precision-only",
         "--ep-size=%d" % case["ep"], "--nnodes=%d" % case["nnodes"]]
    for k in ("num-tokens", "hidden", "num-topk", "experts-per-rank",
              "route-skew", "route-group-size", "route-local-fraction",
              "route-mask-rate"):
        if k in p:
            a.append("--%s=%s" % (k, p[k]))
    a.append("--aiv=%d" % int(p.get("aiv", 32)))
    a += ["--dtypes=" + str(p.get("dtypes", "bf16")),
          "--expand=%d" % int(p.get("expand", 0)),
          "--route-mode=" + str(p.get("route-mode", "uniform")),
          "--run-id=" + run_id, "--output=" + out_root]
    return a


def summarize_pref(text):
    """从 bench 的 summary.csv 取每 op 的 official_logical_GBps 统计。

    bench 的列名以 `official_logical_GBps` 为前缀（含 rank_mean / rank_min /
    rank_max），这里按前缀匹配而不写死全名，避免上游改列名就整表读空。
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return []
    cols = list(rows[0].keys())

    def pick(pred):
        for c in cols:
            if pred(c.lower()):
                return c
        return None

    opc = pick(lambda k: k in ("op", "operator", "op_name", "case_op"))
    mean_c = pick(lambda k: "official_logical_gbps" in k and "rank_mean" in k)
    min_c = pick(lambda k: "official_logical_gbps" in k and "rank_min" in k)
    max_c = pick(lambda k: "official_logical_gbps" in k and "rank_max" in k)
    if not (opc and mean_c):
        return [{"op": "SCHEMA_UNRECOGNIZED", "gbps_mean": "", "gbps_min": "",
                 "gbps_max": "", "columns": ";".join(cols[:12])}]
    out = []
    for r in rows:
        out.append({"op": r.get(opc, ""), "gbps_mean": r.get(mean_c, ""),
                    "gbps_min": r.get(min_c, "") if min_c else "",
                    "gbps_max": r.get(max_c, "") if max_c else ""})
    return out


# --------------------------------------------------------------------------- #
# 阶段：gate
# --------------------------------------------------------------------------- #
GATE_CODE = r'''
import hashlib, json, pathlib
import deep_ep
from deep_ep.backends.global_clos import Buffer, ElasticBuffer
bad = [c.__module__ for c in (Buffer, ElasticBuffer) if "mesh_clos" in c.__module__]
root = pathlib.Path(deep_ep.__file__).resolve().parent
sos = sorted(str(p) for p in root.rglob("*.so"))
print(json.dumps({
    "deep_ep": str(root),
    "backends": [Buffer.__module__, ElasticBuffer.__module__],
    "mesh_clos": bad,
    "sos": {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sos},
}))
'''


def stage_gate(args):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    report = {"ts": now(), "nodes": {}, "checks": {}, "pass": False}
    for node in NODES:
        host = node["host"]
        rec = {}
        try:
            c = container_of(node)
            rec["container"] = c
            r = run_ssh(host, "git -C %s rev-parse HEAD" % REPO, timeout=60)
            rec["commit"] = r.stdout.strip()
            r = run_ssh(host, "git -C %s log -1 --format='%%h %%ci %%s'" % REPO, timeout=60)
            rec["commit_detail"] = r.stdout.strip()
            r = run_ssh(host, "git -C %s status --porcelain | head -20" % REPO, timeout=60)
            rec["dirty"] = [ln for ln in r.stdout.splitlines() if ln.strip()]
            r = docker_exec(node, "python -c " + shlex.quote(GATE_CODE), timeout=180)
            if r.returncode:
                rec["import_error"] = (r.stderr or r.stdout or "")[-500:]
            else:
                rec.update(json.loads(r.stdout.strip().splitlines()[-1]))
            r = run_ssh(host, "grep -i Huge /proc/meminfo", timeout=30)
            rec["hugepages"] = [ln for ln in r.stdout.splitlines() if ln.strip()]
            r = run_ssh(host, "npu-smi info | head -14", timeout=60)
            rec["npu_head"] = r.stdout.splitlines()[:4]
        except Exception as exc:
            rec["error"] = str(exc)[:400]
        report["nodes"][host] = rec

    ok_nodes = [h for h, r in report["nodes"].items() if "error" not in r and r.get("sos")]
    digests = {h: tuple(sorted(report["nodes"][h]["sos"].items())) for h in ok_nodes}
    same = len(set(digests.values())) == 1 if digests else False
    commits = {report["nodes"][h].get("commit") for h in ok_nodes}
    report["checks"] = {
        "all_nodes_reachable": len(ok_nodes) == len(NODES),
        "single_commit": len(commits) == 1,
        "commit": (list(commits)[0] if len(commits) == 1 else sorted(map(str, commits))),
        "so_identical": same,
        "no_mesh_clos": all(not report["nodes"][h].get("mesh_clos") for h in ok_nodes),
        "clean_worktree": all(not report["nodes"][h].get("dirty") for h in ok_nodes),
    }
    report["pass"] = all([report["checks"]["all_nodes_reachable"],
                          report["checks"]["single_commit"],
                          report["checks"]["so_identical"],
                          report["checks"]["no_mesh_clos"]])
    (DATA / "round2_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2, ensure_ascii=False))
    print("GATE %s -> %s" % ("PASS" if report["pass"] else "FAIL",
                             DATA / "round2_gate.json"))
    return 0 if report["pass"] else 2


def require_gate():
    p = DATA / "round2_gate.json"
    if not p.exists():
        sys.exit("没有 round2_gate.json：先跑 `python run_round2.py gate`。"
                 "本轮换了代码，必须先用门禁确认四机同一 commit、同一 .so。")
    g = json.loads(p.read_text(encoding="utf-8"))
    if not g.get("pass"):
        sys.exit("gate 未通过（见 round2_gate.json），拒绝执行性能/精度，"
                 "否则结果不能代表新代码。")
    return g


# --------------------------------------------------------------------------- #
# 阶段：perf
# --------------------------------------------------------------------------- #
PERF_COLS = ["run_id", "ts", "ep_size", "nnodes", "num_tokens", "hidden", "aiv",
             "num_topk", "experts_per_rank", "dtype", "expand", "route_mode",
             "huge_pages", "commit", "op", "gbps_mean", "gbps_min", "gbps_max",
             "rc", "output"]


def append_csv(path, cols, rows):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def stage_perf(args):
    gate = require_gate()
    commit = gate["checks"].get("commit")
    cases = perf_cases(args.eps)
    if args.only_ep:
        cases = [c for c in cases if c["ep"] in args.only_ep]
    cases = cases[args.start:]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        sys.exit("没有可跑的用例")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA / "round2_perf.csv"
    out_root = STAGE + "/out/perf"
    print("perf: %d 例，commit=%s，huge_pages=no" % (len(cases), commit))

    for wave in perf_waves({c["ep"] for c in cases}):
        jobs = [(ep, [c for c in cases if c["ep"] == ep]) for ep in wave]
        jobs = [(ep, cs) for ep, cs in jobs if cs]
        print("\n== 波次 %s ==" % " + ".join("EP%d×%d" % (ep, len(cs)) for ep, cs in jobs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futs = [pool.submit(run_perf_ep, ep, cs, commit, out_root, csv_path, args)
                    for ep, cs in jobs]
            for f in futs:
                f.result()
    print("\n结果: %s" % csv_path)
    return 0


def run_perf_ep(ep, cases, commit, out_root, csv_path, args):
    need = nnodes_of(ep)
    for i in EP_NODES[ep]:
        check_idle(node_by_idx(i), ep // need)
    tag_base = "perf_ep%d_r2_%s" % (ep, time.strftime("%Y%m%d_%H%M%S"))
    print("-- EP%d：%d 例，%d 机 × %d 卡（node%s）" % (
        ep, len(cases), need, ep // need, ",".join(map(str, EP_NODES[ep]))), flush=True)
    for c in cases:
        run_id = "%s_c%02d" % (tag_base, c["idx"])
        argv = perf_argv(c, out_root, run_id)
        t0 = time.time()
        rcs = run_on_nodes(ep, argv, PORTS["perf"], run_id, timeout=args.timeout)
        ok = all(v == 0 for v in rcs.values())
        rows = []
        if ok:
            text = read_text(node_by_idx(EP_NODES[ep][0]),
                             "%s/%s/node0/summary.csv" % (out_root, run_id))
            if not text.strip():
                text = read_text(node_by_idx(EP_NODES[ep][0]),
                                 "%s/%s/summary.csv" % (out_root, run_id))
            for s in summarize_pref(text):
                rows.append(dict(s, run_id=run_id, ts=now(), ep_size=ep,
                                 nnodes=need, num_tokens=c["num_tokens"],
                                 hidden=c["hidden"], aiv=c["aiv"],
                                 num_topk=c["num_topk"],
                                 experts_per_rank=c["experts_per_rank"],
                                 dtype=c["dtypes"], expand=c["expand"],
                                 route_mode=c["route_mode"],
                                 huge_pages="no", commit=commit,
                                 rc=rcs, output=out_root + "/" + run_id))
        append_csv(csv_path, PERF_COLS, rows)
        print("[%s] T=%d H=%d aiv=%d -> %s %.0fs%s" % (
            run_id, c["num_tokens"], c["hidden"], c["aiv"],
            "OK" if ok else "FAIL", time.time() - t0,
            "" if ok else " rc=%s" % rcs), flush=True)
        with LOCK:
            STATE.setdefault("perf", []).append(
                {"run_id": run_id, "ep": ep, "idx": c["idx"], "rc": rcs,
                 "ok": ok, "output": out_root + "/" + run_id})
            save_state()


# --------------------------------------------------------------------------- #
# 阶段：precision
# --------------------------------------------------------------------------- #
PREC_COLS = ["run_id", "ts", "api", "ep_size", "nnodes", "group", "tag",
             "num_tokens", "hidden", "num_topk", "experts_per_rank", "aiv",
             "dtype", "expand", "ranks_expected", "ranks_ok", "digest",
             "state", "commit", "output"]


def rank_in_key(key):
    """从相对路径里取 rank 号；兼容 rank3/... 与 a/rank3/... 两种布局。"""
    import re
    m = re.search(r"rank(\d+)[/\\]", key)
    return int(m.group(1)) if m else -1


def judge_precision(files, ep):
    """逐 rank 判定：precision.json.passed 必须为 true，且 result.json 是纯精度。"""
    ranks, digests, errs = 0, [], []
    for r in range(ep):
        cand = [k for k in files if rank_in_key(k) == r]
        if not cand:
            errs.append("rank%d 无产物" % r)
            continue
        prec = next((files[k] for k in cand if k.endswith("precision.json")), None)
        res = next((files[k] for k in cand if k.endswith("result.json")), None)
        if prec is None or res is None:
            errs.append("rank%d 缺 precision/result" % r)
            continue
        try:
            p = json.loads(prec)
            d = json.loads(res)
        except Exception as exc:
            errs.append("rank%d 解析失败 %s" % (r, exc))
            continue
        if p.get("passed") is not True:
            errs.append("rank%d precision.passed!=true" % r)
            continue
        if d.get("precision_only") is not True or d.get("operators") not in ({}, None):
            errs.append("rank%d 不是纯精度（污染性能口径）" % r)
            continue
        ranks += 1
        digests.append(p.get("digest", d.get("digest", "")))
    uniq = sorted({x for x in digests if x})
    return ranks, (uniq[0] if len(uniq) == 1 else ""), errs


def stage_precision(args):
    gate = require_gate()
    commit = gate["checks"].get("commit")
    cases = precision_cases(args.eps)
    if args.only_ep:
        cases = [c for c in cases if c["ep"] in args.only_ep]
    cases = cases[args.start:]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        sys.exit("没有可跑的用例")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA / "round2_precision.csv"
    out_root = STAGE + "/out/prec"
    apis = args.api
    print("precision: %d 例/API × %s，commit=%s" % (len(cases), apis, commit))

    for wave in perf_waves({c["ep"] for c in cases}):
        jobs = [(ep, [c for c in cases if c["ep"] == ep]) for ep in wave]
        jobs = [(ep, cs) for ep, cs in jobs if cs]
        print("\n== 波次 %s ==" % " + ".join("EP%d×%d" % (ep, len(cs)) for ep, cs in jobs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futs = [pool.submit(run_prec_ep, ep, cs, apis, commit, out_root, csv_path, args)
                    for ep, cs in jobs]
            for f in futs:
                f.result()
    print("\n结果: %s" % csv_path)
    return 0


def run_prec_ep(ep, cases, apis, commit, out_root, csv_path, args):
    need = nnodes_of(ep)
    for i in EP_NODES[ep]:
        check_idle(node_by_idx(i), ep // need)
    tag_base = "prec_ep%d_r2_%s" % (ep, time.strftime("%Y%m%d_%H%M%S"))
    print("-- EP%d：%d 例 × %d API，%d 机 × %d 卡" % (
        ep, len(cases), len(apis), need, ep // need), flush=True)
    for api in apis:
        for c in cases:
            run_id = "%s_%s_c%02d" % (tag_base, api, c["idx"])
            argv = precision_argv(c, out_root, run_id, api)
            rcs = run_on_nodes(ep, argv, PORTS["precision"], run_id, timeout=args.timeout)
            ok_rc = all(v == 0 for v in rcs.values())
            ranks = digest = 0
            state = "failed_runtime"
            errs = []
            if ok_rc:
                files = collect_json(node_by_idx(EP_NODES[ep][0]), out_root + "/" + run_id)
                ranks, digest, errs = judge_precision(files, ep)
                state = "passed" if ranks == ep else "failed_judge"
            append_csv(csv_path, PREC_COLS, [{
                "run_id": run_id, "ts": now(), "api": api, "ep_size": ep,
                "nnodes": need, "group": c.get("group", ""), "tag": c.get("tag", ""),
                "num_tokens": c["params"].get("num-tokens", ""),
                "hidden": c["params"].get("hidden", ""),
                "num_topk": c["params"].get("num-topk", ""),
                "experts_per_rank": c["params"].get("experts-per-rank", ""),
                "aiv": c["params"].get("aiv", 32),
                "dtype": c["params"].get("dtypes", "bf16"),
                "expand": c["params"].get("expand", 0),
                "ranks_expected": ep, "ranks_ok": ranks, "digest": digest,
                "state": state, "commit": commit,
                "output": out_root + "/" + run_id}])
            print("[%s] %s rank %s/%d %s%s" % (
                run_id, c.get("tag", ""), ranks or "-", ep, state,
                "" if not errs else " | " + "; ".join(errs[:3])), flush=True)
            with LOCK:
                STATE.setdefault("precision", []).append(
                    {"run_id": run_id, "ep": ep, "api": api, "state": state,
                     "ranks_ok": ranks, "errs": errs[:6],
                     "output": out_root + "/" + run_id})
                save_state()
    print("\n结果: %s" % csv_path)
    return 0


# --------------------------------------------------------------------------- #
# 阶段：abnormal（E4 / E5 严格重测）
# --------------------------------------------------------------------------- #
def stage_abnormal(args):
    gate = require_gate()
    commit = gate["checks"].get("commit")
    # 用严格口径的 v2 worker（超容量 E5 + 纯张量 E4，见 commit 5cc6d2d）。
    # v1 的 abnormal_worker.py 原样保留，它对应上轮已实测的越界 expert id 口径。
    worker = HERE / "abnormal_worker_v2.py"
    LOGDIR.mkdir(parents=True, exist_ok=True)
    eps = args.eps
    sids = args.scenarios
    jobs = []
    for ep in eps:
        for api in args.api:
            for sid in sids:
                jobs.append({"ep": ep, "api": api, "sid": sid})
    print("abnormal: %d 次执行（%s），commit=%s" % (
        len(jobs), ",".join("%s/%s" % (j["api"], j["sid"]) for j in jobs[:4]) + "…", commit))

    for ep in sorted({j["ep"] for j in eps}):
        need = nnodes_of(ep)
        for i in EP_NODES[ep]:
            check_idle(node_by_idx(i), ep // need)
            deploy(node_by_idx(i), worker, "abnormal_worker_v2.py")

    results = []

    def one(job):
        ep, api, sid = job["ep"], job["api"], job["sid"]
        node = node_by_idx(EP_NODES[ep][0])       # 异常场景沿用上轮的 EP8/EP16 单机
        run_id = "abn_ep%d_%s_%s_r2" % (ep, api, sid)
        out_root = "%s/out/abn/%s" % (STAGE, run_id)
        argv = ["python", "torch.distributed.run",
                "--nnodes=1", "--nproc-per-node=%d" % ep, "--node-rank=0",
                "--master-addr=127.0.0.1", "--master-port=%d" % (PORTS["abnormal"] + ep),
                "--max-restarts=0",
                STAGE + "/abnormal_worker_v2.py",
                "--ep=%d" % ep, "--api=" + api, "--output=" + out_root,
                "--run-id=" + run_id, "--scenarios=" + sid,
                "--num-tokens=1024", "--hidden=2048", "--num-topk=8",
                "--experts-per-rank=32", "--dtype=bf16", "--aiv=32",
                "--capacity=%d" % args.capacity, "--seed-base=20260823",
                # 上轮实测工作区：EP8 约 2.16 GB、EP16 约 4.31 GB，即每 rank 256 MiB
                "--workspace-bytes=%d" % (args.workspace_bytes or 268435456 * ep)]
        inner = env_prefix() + "timeout %d " % max(120, args.timeout - 120) + join_argv(argv)
        log = LOGDIR / ("%s.log" % run_id)
        r = docker_exec(node, inner, timeout=args.timeout, log=log)
        files = collect_json(node, out_root)
        cls, notes, ranks = "failed_runtime", [], 0
        for rk in range(ep):
            key = next((k for k in files if rank_in_key(k) == rk
                        and k.endswith("result.json")), None)
            if not key:
                notes.append("rank%d 无产物" % rk)
                continue
            d = json.loads(files[key])
            if d.get("classification"):
                cls = d["classification"]
            notes.extend(d.get("notes", []))
            ranks += 1
        rec = {"run_id": run_id, "ep": ep, "api": api, "scenario": sid,
               "classification": cls, "ranks": ranks, "rc": r.returncode,
               "notes": "; ".join(notes)[:600], "output": out_root, "ts": now()}
        with LOCK:
            results.append(rec)
            STATE.setdefault("abnormal", []).append(rec)
            save_state()
        print("[%s] %s/%s -> %s (rank=%d, rc=%s)" % (
            run_id, api, sid, cls, ranks, r.returncode), flush=True)
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(one, jobs))

    csv_path = DATA / "round2_abnormal.csv"
    cols = ["run_id", "ts", "ep", "api", "scenario", "classification", "ranks", "rc",
            "notes", "output"]
    if csv_path.exists():
        csv_path.unlink()
    append_csv(csv_path, cols, results)
    print("\n结果: %s" % csv_path)
    return 0


# --------------------------------------------------------------------------- #
# 阶段：plan（离线）
# --------------------------------------------------------------------------- #
def stage_plan(args):
    print("R2 第二轮测试计划（离线，不碰设备）")
    print("=" * 74)
    print("拓扑: " + "  ".join("node%d=%s" % (n["node"], n["host"]) for n in NODES))
    print("      各 16 卡；master=%s；ssh -p %d" % (MASTER, SSH_PORT))
    print()

    perf = perf_cases(args.eps)
    prec = precision_cases(args.eps)
    print("阶段 B 性能：%d 例（每 EP 12 例：2 tokens × 3 hidden × 2 aiv）" % len(perf))
    print("  %-4s %-5s %-6s %s" % ("EP", "例数", "拓扑", "节点分配"))
    for ep in args.eps:
        print("  %-4d %-5d %-6s node%s" % (
            ep, len([c for c in perf if c["ep"] == ep]),
            "%d机x%d卡" % (nnodes_of(ep), ep // nnodes_of(ep)),
            ",".join(map(str, EP_NODES[ep]))))
    print("  大页: 关闭（argv 断言不含 --huge-pages）")
    print("  波次: " + " | ".join(
        " + ".join("EP%d" % e for e in w) for w in perf_waves(args.eps)))
    print()
    print("阶段 C 精度：%d 例（每 EP 50 条，沿用 2-16P 同一套规格）" % len(prec))
    print("  API: %s" % ",".join(args.api))
    print("  波次: " + " | ".join(
        " + ".join("EP%d" % e for e in w) for w in perf_waves(args.eps)))
    print()
    print("阶段 D 异常：E4 / E5 严格重构造")
    print("  E4: 一次 dispatch + 同一 handle 两次 combine，第二次换本地生成的同布局专家输出")
    print("      （旧实现用 dispatch_cached 多跑了一次 dispatch，已改掉）")
    print("  E5: 第二轮传超容量 tokens（num_tokens = capacity+1），期望 ValueError")
    print("      （v1 脚本用越界 expert id，触发设备级崩溃，属另一个缺口）")
    print("  矩阵: EP%s × {%s} × {%s}" % (
        ",".join(map(str, args.eps)), ",".join(args.api), ",".join(args.scenarios)))
    print()
    sample = perf[0]
    ep = sample["ep"]
    rid = "perf_ep%d_r2_preview_c01" % ep
    print("示例命令（node%s，EP%d）：" % (EP_NODES[ep][0], ep))
    print("  " + " ".join(perf_argv(sample, STAGE + "/out/perf", rid)))
    if len(EP_NODES[ep]) > 1:
        print("  ＋ 各节点补 --node-rank=<0..%d> --master-addr=%s --master-port=%d"
              % (len(EP_NODES[ep]) - 1, MASTER, PORTS["perf"]))
    p0 = prec[0]
    print()
    print("示例精度命令（EP%d）：" % p0["ep"])
    print("  " + " ".join(precision_argv(p0, STAGE + "/out/prec", "prec_preview", args.api[0])))
    return 0


def perf_waves(eps):
    """EP64 独占一波；其余并行一波（节点互不重叠）。"""
    eps = sorted(set(eps), reverse=True)
    waves = []
    if 64 in eps:
        waves.append([64])
    rest = [e for e in eps if e != 64]
    if rest:
        waves.append(sorted(rest))
    return waves


# --------------------------------------------------------------------------- #
def parse_eps(text):
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def main():
    ap = argparse.ArgumentParser(description="DeepEP CLOS 第二轮（R2）测试编排")
    ap.add_argument("stage", nargs="?", default="plan",
                    choices=["plan", "gate", "perf", "precision", "abnormal"])
    ap.add_argument("--eps", default="8,16,32,64",
                    help="性能/精度阶段参与编排的 EP，默认全部")
    ap.add_argument("--only-ep", default="", help="只跑指定 EP，如 64")
    ap.add_argument("--api", default="v2", help="精度/异常阶段的 API，如 v2 或 v1,v2")
    ap.add_argument("--scenarios", default="E4,E5", help="异常场景，默认 E4,E5")
    ap.add_argument("--containers", default="",
                    help="覆盖容器发现，格式 host=name,host=name")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=3600, help="单个 case 超时秒")
    ap.add_argument("--jobs", type=int, default=4, help="异常阶段并发数")
    ap.add_argument("--capacity", type=int, default=16384,
                    help="num_max_tokens_per_rank；E5 传 capacity+1 造超容量")
    ap.add_argument("--workspace-bytes", type=int, default=0,
                    help="V1 工作区字节数；默认每 rank 256 MiB")

    args = ap.parse_args()

    args.eps = parse_eps(args.eps)
    args.only_ep = parse_eps(args.only_ep) if args.only_ep else []
    args.api = [x for x in args.api.replace(" ", "").split(",") if x]
    args.scenarios = [x for x in args.scenarios.replace(" ", "").split(",") if x]

    if args.containers:
        STATE["container_override"] = dict(
            kv.split("=", 1) for kv in args.containers.split(",") if "=" in kv)

    LOGDIR.mkdir(parents=True, exist_ok=True)
    STATE.setdefault("runs", []).append({"stage": args.stage, "ts": now(),
                                         "argv": sys.argv[1:]})
    save_state()

    fn = {"plan": stage_plan, "gate": stage_gate, "perf": stage_perf,
          "precision": stage_precision, "abnormal": stage_abnormal}[args.stage]
    try:
        return fn(args)
    except BaseException:
        STATE["last_error"] = traceback.format_exc()[-2000:]
        save_state()
        raise


if __name__ == "__main__":
    sys.exit(main())
