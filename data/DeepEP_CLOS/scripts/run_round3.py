#!/usr/bin/env python3
"""R3（第三轮）测试编排：PR #37 代码门禁 → C(EP128 精度) → B(性能 48) → D(精度 200) → E(异常 E4/E5)。

对应需求原文 `../需求原文_R3_20261128.txt` 的五条 + 用户补充（.7 只做精度、
PR 已修 QP 保持 auto），指南 `../第三轮测试执行指南_20261128.md`。

  1. 代码切到 lueelu/DeepEP PR #37 head `0d1eca2`（未合并），重编译安装
  2. 性能 EP8-EP64 共 48 例，不开 1G 大页，**7 台**（{0,1,2,4,5,6,7}，硬排除 .7）
  3. EP128 泛化精度，**8 台**（含 .7，128 卡），epr 双档（主 4 / 探针 32）
  4. EP8-64 精度各 50 条共 200 例，**8 台**一波打满（EP8 放 .7）
  5. 异常重测 E4 / E5（严格口径，abnormal_worker_v2.py）

关键口径（§A2.5）
------------------
* 代码是 **PR head** 而非 `origin/main`；gate 额外断言 `not_origin_main`、
  `base_ancestor`(83252a5 是祖先)、`same_container_image`、`container_bound`。
* 容器里有两套 `deep_ep`（版本号都报 0.1.0.dev5）。装和跑必须用
  `/work/env/bin/python`，并把 `/work/env/lib/python3.13/site-packages` 放在
  PYTHONPATH **首位**；gate 用 `buffer.py` sha == `f825a679…` 判新旧，
  **不能用 `backends/direct/__init__.py`**（新旧同 sha）。
* **纯 auto**：`env -u DEEPEP_CLOS_QP_PLANS`（PR 的 2828eae 已删该变量读取），
  不传 `udma_*` 手动参数，不调初始化超时（ebc8267 默认 10 min）。

设计约束
--------
* **门禁优先**：gate 不通过，B/C/D/E 一律拒绝执行。
* **两个节点集合**：`PERF_NODES`={0,1,2,4,5,6,7}（7 台），`PREC_NODES`=0..7（8 台）。
  node 号 = IP 末位 − 4。性能波次 EP64‖EP32‖EP16 占满 7 台、EP8 尾波；
  精度 EP128 独占一波、EP8-64 一波打满 8 台。
* **不开大页**：命令行断言不含 `--huge-pages`，产物记录 `huge_pages=no`。
* **多机精度逐节点收集判据**（不回传原始产物）；每 case 一次进程组便于逐条对比。
* `plan` 子命令完全离线，不 ssh、不碰设备。

拓扑（128P 的 8 机容器）
------------------------
node0..node7 = 30.104.128.4 ... .11，各 16 卡；ssh 端口 56000；
容器统一为 `deepep_ep128_full_matrix_20261128_073617`（8 机同名，必须显式绑定）；
容器内源码 `/work/DeepEP_lueelu`（已 checkout PR #37），deep_ep 装在
`/work/env/lib/python3.13/site-packages`；`/work` 是各机本地 xfs、不共享。

精度 bench（`bench_v1_precision.py` / `bench_v2_precision.py`）**不在**
lueelu/DeepEP 仓里，是本目录的本地补丁：部署到各节点 `/work/round3`
后用绝对路径运行，PYTHONPATH 补 `/work/DeepEP_lueelu/tests/clos` 提供 golden。

bench 自带 torchrun 编排，所以每节点只需 `python bench_*.py --node-rank=N`，
**不要外面再套 torchrun**。

用法（执行顺序 gate → C → B → D → E）
----
    python run_round3.py plan                                   # 离线看计划（默认）
    python run_round3.py gate                                   # 八机门禁（PR head/.so/解释器/QP/空闲）
    python run_round3.py precision --eps 128 --epr-tiers main   # C: EP128（8 台）
    python run_round3.py perf --eps 8,16,32,64                 # B: 48 例（7 台，去 .7）
    python run_round3.py precision --eps 8,16,32,64            # D: 200 例（8 台）
    python run_round3.py abnormal --eps 8,16 --api v1,v2       # E: E4/E5
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
LOGDIR = DATA / "round3_logs"
STATE_PATH = DATA / "round3_state.json"

SSH_PORT = 56000
SSH_BASE = ["ssh", "-T", "-p", str(SSH_PORT), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6"]

# node 号 = IP 末位 − 4（不要改序，否则 .7 一跳后面全错位）
NODES = [{"node": i, "host": "30.104.128.%d" % (4 + i)} for i in range(8)]
MASTER = NODES[0]["host"]
# R3 定案（方案 A）：性能硬排除 node3(.7)；精度用全部 8 台（EP128 需 128 卡）
PERF_NODES = {0, 1, 2, 4, 5, 6, 7}
PREC_NODES = set(range(8))
PERF_HOST_EXCLUDED = "30.104.128.7"

PR_HEAD = "0d1eca219d89f1f5e7120b48f01f1dd3654db4f8"
PR_BASE = "83252a54d2363734a706a7dd8368bebc3094af5a"
BUFFER_SHA_EXPECTED = ("f825a67986286d8f8b852e57028090e8762111e508af"
                       "1991351a94e948cfbbcb")
ENV_SITE = "/work/env/lib/python3.13/site-packages"
ENV_PYTHON = "/work/env/bin/python"

REPO = "/work/DeepEP_lueelu"       # 8 机容器内 clone 的 lueelu/DeepEP（已 checkout PR#37）
BENCH_DIR = REPO + "/benchmarks/clos"
GOLDEN_DIR = REPO + "/tests/clos"
RUNTIME = ""                       # 无独立 runtime 目录；site-packages 由 ENV_SITE 前置
STAGE = "/work/round3"
CONTAINER_HINT = "deepep_ep128_full_matrix_20261128_073617"   # 8 机统一容器名（必须显式绑定）
PROXY = "http://star-proxy.oa.com:3128"  # 容器内默认无代理，git fetch 需要

# 精度 bench 不在 lueelu/DeepEP main 仓里，是本目录的本地补丁：部署到 STAGE
# 后以绝对路径运行（避免污染 git 工作区，gate 的 clean_worktree 才不会失败）
PRECISION_BENCH_FILES = ("bench_v1_precision.py", "bench_v2_precision.py")
# 精度补丁依赖的**旧版**辅助模块：PR #37 改写了 stage_metrics.py，
# 新版已无 COMBINE_NOTIFY_ENV / L2CacheFlush / ProfileSamples 等符号，
# 直接用 REPO/benchmarks/clos 会让补丁在 import 阶段就崩。这些是 64P 容器里
# 与补丁配套的快照，部署到 STAGE 后由 PYTHONPATH（STAGE 在前）优先命中。
PRECISION_DEPS_FILES = ("stage_metrics.py", "route_workloads.py",
                        "bandwidth_multinode.py", "profiling.py")

# 性能阶段 EP -> 节点（性能集合硬排除 node3/.7，用户 2026-11-28 确认）：
#   R3 主 48 例：第 1 波 EP64 ‖ EP32 ‖ EP16；第 2 波 EP8 复用 node7(.11)。
#   补测 EP4/EP64（1128 表）：用户指示「换其他空闲机器」——原 EP64=[0,1,2,4]
#   含被别人占的 .4/.5，改为 [2,4,5,6]=.6/.8/.9/.10；EP4 用 [7]=.11 单机 4 卡。
#   两组节点不重叠，可一波并行；均不含 .7 与 .4/.5。
# 精度阶段另用 PREC_EP_NODES（§六：EP8 放 node3/.7，一波打满 8 台）。
EP_NODES = {4: [7], 8: [7], 16: [7], 32: [5, 6], 64: [2, 4, 5, 6], 128: list(range(8))}
PREC_EP_NODES = {8: [3], 16: [7], 32: [5, 6], 64: [0, 1, 2, 4], 128: list(range(8))}
# 补测专用节点（覆盖 EP_NODES 的 4/64）：避开 .4/.5 被占、避开 .7 性能禁区
SUPP_EP_NODES = {4: [7], 64: [2, 4, 5, 6]}
PORTS = {"perf": 29860, "precision": 29880, "abnormal": 29900}

# 当前阶段生效的节点映射：性能用 EP_NODES（7 台、去 .7），精度用 PREC_EP_NODES
# （8 台、EP8 放 .7）。两个集合不同，产物路径带 node 号，绝不混。
EP_NODES_ACTIVE = dict(EP_NODES)


def set_phase(phase):
    global EP_NODES_ACTIVE
    if phase == "supp":
        EP_NODES_ACTIVE = dict(SUPP_EP_NODES)
    else:
        EP_NODES_ACTIVE = dict(EP_NODES if phase == "perf" else PREC_EP_NODES)
    STATE["phase"] = phase

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
    """R3：禁用子串猜测。执行阶段必须经 --containers 显式绑定全部 8 台。

    旧逻辑靠 CONTAINER_HINT 过滤 docker ps，命中多个时取最短名字，可能落到
    上一轮的旧容器（镜像 / .so 全不同却查不出来）。见指南 §三·A0.6。
    """
    override = STATE.get("container_override", {})
    missing = [n["host"] for n in NODES if n["host"] not in override or not override[n["host"]]]
    if missing:
        sys.exit("--containers 必须显式绑定全部 8 台（host=name,...）；缺: %s"
                 % ",".join(missing))
    name = override[node["host"]]
    with LOCK:
        STATE.setdefault("containers", {})[node["host"]] = name
        save_state()
    return name


def docker_exec(node, inner, timeout=600, log=None):
    """在容器内用 login shell 跑一段命令。

    R3：不再注入 SHMEM_UDMA_CONNECT_TIMEOUT_MS——PR #37 的 ebc8267 已把
    初始化超时改为按 Buffer 配置、默认 10 min，指南 §A2.5 明令不传 timeout。
    """
    cmd = "docker exec %s bash -lc %s" % (container_of(node), shlex.quote(inner))
    return run_ssh(node["host"], cmd, timeout=timeout, log=log)


def env_prefix():
    """环境基线（R3）。

    - ENV_SITE 放 PYTHONPATH **首位**：保证 import deep_ep 解析到 /work/env
      新 wheel，而不是 /opt/conda 旧包（§A2.5，两套版本号都是 0.1.0.dev5）。
    - STAGE 放部署的脚本（含精度 bench 补丁）；BENCH_DIR / GOLDEN_DIR 补依赖。
    - `env -u DEEPEP_CLOS_QP_PLANS`：PR #37 的 2828eae 已删掉该变量读取，
      保持纯 auto；env -u 清掉可能从旧 shell 继承的脏值（门禁断言为空）。
    """
    paths = [ENV_SITE, STAGE, BENCH_DIR, GOLDEN_DIR, RUNTIME, "$PYTHONPATH"]
    pp = ":".join(x for x in paths if x)
    return ("source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
            "unset SHMEM_UID_SESSION_ID ASCEND_RT_VISIBLE_DEVICES && "
            "unset DEEPEP_CLOS_QP_PLANS && "
            "export PATH=/work/env/bin:$PATH && "
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && "
            "export PYTHONPATH=" + pp + " && "
            "cd " + BENCH_DIR + " && ")


def join_argv(argv):
    try:
        return shlex.join(argv)
    except AttributeError:                    # py<3.8
        return " ".join(shlex.quote(a) for a in argv)


def deploy(node, local_path, name):
    """本地 -> 宿主机 /tmp -> 容器 /work/round3，不依赖宿主机挂载路径。"""
    host = node["host"]
    tmp = "/tmp/round3_" + name
    subprocess.run(["scp", "-q", "-P", str(SSH_PORT), "-o", "BatchMode=yes",
                    str(local_path), "root@%s:%s" % (host, tmp)], check=True)
    c = container_of(node)
    run_ssh(host, "docker exec %s mkdir -p %s" % (c, STAGE), timeout=60)
    r = run_ssh(host, "docker cp %s %s:%s/%s" % (tmp, c, STAGE, name), timeout=120)
    if r.returncode:
        raise RuntimeError("docker cp 失败 %s: %s" % (host, (r.stderr or "")[-200:]))


def deploy_bench_files(eps):
    """把精度 bench 本地补丁部署到参与节点。

    lueelu/DeepEP main 仓的 benchmarks/clos 里没有 bench_v*_precision.py，
    且不能塞进 REPO（会污染 git 工作区，gate 的 clean_worktree 会失败），
    因此部署到 STAGE 并以绝对路径运行。abnormal_worker_v2.py 也 import 它们。
    """
    idxs = sorted({i for ep in eps for i in EP_NODES_ACTIVE[ep]})
    for name in PRECISION_BENCH_FILES + PRECISION_DEPS_FILES:
        src = HERE / name
        if not src.exists():
            raise RuntimeError("缺少精度补丁/依赖: %s" % src)
        for i in idxs:
            deploy(node_by_idx(i), src, name)
    return idxs


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
def perf_cases(eps, supp=False):
    """性能用例。

    主 B：12 例/EP（2 tokens × 3 hidden × 2 aiv），对齐 baseline/V2测试Case_20261128快照.txt。
    补测 --supp：18 例/EP（3 tokens × 3 hidden × 2 aiv，Experts 恒 256），
      对齐 baseline/EP4-128_通信带宽_20261128.txt（1128 实测原表，两表勿混用）。
    """
    spec_path = HERE / ("perf_cases_supp.json" if supp else "perf_cases_v2.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    ax = spec["axes"]
    fx = spec["fixed"]
    out = []
    for ep in eps:
        experts = spec["experts"] if supp else spec["experts_map"][str(ep)]
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


def precision_cases(eps, epr_tiers=("main",)):
    """复用 2-16P 的 50 条规格，只把 EP 与节点数投影过去（用例内容不变）。

    R3 §五·C2/C3：EP128 走双档，主档 epr=4（E=512，与性能/基线同口径）、
    探针 epr=32（E=4096，沿用 R2 spec）。两档分开报，不混在一张通过率里。
    EP8-64 一律沿用 spec 原样（epr=32），与 R2 逐条可比。
    """
    sys.path.insert(0, str(HERE))
    import gen_precision_cases as gpc
    spec = json.loads((HERE / "precision_cases_2to16p.json").read_text(encoding="utf-8"))
    out = []
    for ep in eps:
        tiers = epr_tiers if ep == 128 else ("spec",)
        for tier in tiers:
            cap = {"main": 4, "probe": 32}.get(tier)
            for c in gpc.build(spec, ep):
                p = dict(c.params)
                tag = c.tag
                if cap is not None:
                    epr = min(int(p.get("experts-per-rank", 32)), cap)
                    p["experts-per-rank"] = epr
                    # 按所选档位重裁 topk 上限（min(128, ep*epr)）
                    k = int(p.get("num-topk", 8))
                    p["num-topk"] = min(k, gpc.topk_cap(ep, epr))
                    tag = "%s_%s" % (c.tag, tier)
                out.append({
                    "stage": "precision", "ep": ep, "idx": c.idx, "group": c.group,
                    "tag": tag, "epr_tier": tier, "params": p,
                    "nnodes": p.get("nnodes", max(1, ep // 16)),
                })
    return out


def nnodes_of(ep):
    return len(EP_NODES_ACTIVE[ep])


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #
def run_on_nodes(ep, argv_tail, port, tag, timeout, extra_check=None):
    """同一 case 在所有相关节点上同时启动（bench 自己起 torchrun，必须同时到场）。

    argv_tail 为不含 --node-rank 的参数列表；每个节点自动补 node-rank/master。
    返回 {node_idx: rc}。
    """
    idxs = EP_NODES_ACTIVE[ep]
    # rendezvous 的 master 必须是本 EP 的 node-rank=0 节点（即 idxs[0]），
    # 不能固定 node0=.4：EP32 用 node5,6 时 master=.4 上没有该 run 的 rank0，
    # bench 会一直 WAIT 到超时。
    ep_master = node_by_idx(idxs[0])["host"]
    barrier = threading.Barrier(len(idxs))
    rcs = {}

    def one(local_rank, node_idx):
        node = node_by_idx(node_idx)
        argv = list(argv_tail)
        if len(idxs) > 1:
            argv += ["--node-rank=%d" % local_rank,
                     "--master-addr=" + ep_master,
                     "--master-port=%d" % port]
        else:
            argv += ["--node-rank=0"]
        # env -u 显式证明「纯 auto 起跑」：PR #37 已不读该变量，残留值只会
        # 让环境不干净；guide §十·五 要求 env -u 起跑并在 gate 断言为空。
        inner = (env_prefix() + "timeout %d env -u DEEPEP_CLOS_QP_PLANS " %
                 max(60, timeout - 120) + join_argv(argv))
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
    # .7 硬排除：性能集合 ∩ {30.104.128.7} 必须为空（指南 §四·B2）
    hosts = {NODES[i]["host"] for i in EP_NODES_ACTIVE[case["ep"]]}
    assert PERF_HOST_EXCLUDED not in hosts, (
        "性能批次混入 .7（node3），整批作废且无法事后剔除: %s" % sorted(hosts))
    a = [ENV_PYTHON, "bench_v2_bandwidth.py",
         "--ep-size=%d" % case["ep"], "--nnodes=%d" % nnodes_of(case["ep"]),
         "--num-tokens=%d" % case["num_tokens"], "--hidden=%d" % case["hidden"],
         "--num-topk=%d" % case["num_topk"],
         "--experts-per-rank=%d" % case["experts_per_rank"],
         "--dtypes=" + case["dtypes"], "--expand=%d" % case["expand"],
         "--aiv=%d" % case["aiv"], "--warmup=%d" % case["warmup"],
         "--iterations=%d" % case["iterations"], "--seed=%d" % case["seed"],
         "--route-mode=" + case["route_mode"],
         "--run-id=" + run_id, "--output=" + out_root + "/" + run_id]
    assert "--huge-pages" not in a, "本轮明令不开 1G 大页"
    return a


def precision_argv(case, out_root, run_id, api):
    # 精度 bench 是部署到 STAGE 的本地补丁（main 仓没有），必须用绝对路径
    # 解释器用 /work/env/bin/python（§A2.5）；QP 变量由 env_prefix 的 env -u 清掉
    bench = STAGE + "/bench_%s_precision.py" % api
    p = case["params"]
    a = [ENV_PYTHON, bench, "--precision-only",
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
          "--run-id=" + run_id, "--output=" + out_root + "/" + run_id]
    return a


def summarize_pref(text):
    """从 bench 的 summary.csv 取每 op 的 official_logical_GBps 统计。

    PR #37 的 bench 改成逐 rank 明细：每行一个 (rank, op)，rank 取 0..EP-1
    外加一行 `all`（whole-EP 聚合）。列名是 `official_logical_GBps`，不再有
    旧版的 `official_logical_GBps_rank_mean`。按指南 §四·B3「只采信
    official_logical_GBps_rank_mean」，这里对**数值 rank**（排除 `all`）按 op
    分组求 mean/min/max；`all` 行只作诊断、不计入主表。
    """
    rows = list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))
    if not rows:
        return []
    cols = list(rows[0].keys())

    def pick(pred):
        return next((c for c in cols if pred(c.lower())), None)

    opc = pick(lambda k: k in ("op", "operator", "op_name", "case_op"))
    gbps_c = pick(lambda k: k == "official_logical_gbps")
    # 兼容旧版：若上游已给好 *_rank_mean/min/max，直接取用
    mean_c = pick(lambda k: "official_logical_gbps" in k and "rank_mean" in k)
    min_c = pick(lambda k: "official_logical_gbps" in k and "rank_min" in k)
    max_c = pick(lambda k: "official_logical_gbps" in k and "rank_max" in k)
    if opc and mean_c:
        return [{"op": r.get(opc, ""), "gbps_mean": r.get(mean_c, ""),
                 "gbps_min": r.get(min_c, "") if min_c else "",
                 "gbps_max": r.get(max_c, "") if max_c else ""} for r in rows]
    if not (opc and gbps_c):
        return [{"op": "SCHEMA_UNRECOGNIZED", "gbps_mean": "", "gbps_min": "",
                 "gbps_max": "", "columns": ";".join(cols[:12])}]

    rankc = pick(lambda k: k in ("rank", "rank_id"))
    by_op = {}
    for r in rows:
        rk = (r.get(rankc) or "").strip() if rankc else ""
        if not rk.isdigit():          # 跳过 `all` 聚合行
            continue
        try:
            v = float(r.get(gbps_c, "") or "")
        except (TypeError, ValueError):
            continue
        by_op.setdefault(r.get(opc, ""), []).append(v)
    out = []
    for op, vals in by_op.items():
        if not vals:
            continue
        out.append({"op": op,
                    "gbps_mean": "%.3f" % (sum(vals) / len(vals)),
                    "gbps_min": "%.3f" % min(vals),
                    "gbps_max": "%.3f" % max(vals)})
    return out


# --------------------------------------------------------------------------- #
# 阶段：gate
# --------------------------------------------------------------------------- #
GATE_CODE = r'''
import hashlib, json, pathlib, os, sys
import deep_ep
from deep_ep.backends.global_clos import Buffer, ElasticBuffer
bad = [c.__module__ for c in (Buffer, ElasticBuffer) if "mesh_clos" in c.__module__]
root = pathlib.Path(deep_ep.__file__).resolve().parent
sos = sorted(str(p) for p in root.rglob("*.so"))
bi = root / "_build_info.json"
bp = root / "backends/direct/buffer.py"
ver = "unknown"
try:
    import importlib.metadata as md
    ver = md.version("ascend-deepep")
except Exception:
    pass
print(json.dumps({
    "interpreter": sys.executable,
    "deep_ep": str(root),
    "version": ver,
    "buffer_sha": hashlib.sha256(bp.read_bytes()).hexdigest() if bp.exists() else "",
    "qp_plans": os.environ.get("DEEPEP_CLOS_QP_PLANS", ""),
    "build_info_sha": hashlib.sha256(bi.read_bytes()).hexdigest() if bi.exists() else "",
    "backends": [Buffer.__module__, ElasticBuffer.__module__],
    "mesh_clos": bad,
    "sos": {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sos},
}))
'''


def git_in(node, args, timeout=90):
    """在**容器内**取 git 信息（容器里的那份才是真正被编译/运行的），失败回退宿主机。

    8 机容器默认无代理，gate 里的 `fetch origin main` 必须显式带代理。
    """
    inner = "export http_proxy=%s https_proxy=%s && git -C %s %s" % (
        PROXY, PROXY, REPO, args)
    try:
        r = docker_exec(node, inner, timeout=timeout)
        if not r.returncode:
            return r.stdout.strip(), "container"
    except Exception:
        pass
    r = run_ssh(node["host"], inner, timeout=timeout)
    return (r.stdout.strip() if not r.returncode else ""), "host"


def stage_gate(args):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    report = {"ts": now(), "nodes": {}, "checks": {}, "pass": False}
    for node in NODES:
        host = node["host"]
        rec = {}
        try:
            c = container_of(node)
            rec["container"] = c
            rec["commit"], rec["git_where"] = git_in(node, "rev-parse HEAD")
            rec["commit_detail"], _ = git_in(node, "log -1 --format='%h %ci %s'")
            rec["origin_main"], _ = git_in(node, "rev-parse origin/main")
            rec["remote"], _ = git_in(node, "remote get-url origin")
            dirty, _ = git_in(node, "status --porcelain")
            rec["dirty"] = [ln for ln in dirty.splitlines() if ln.strip()][:20]
            # origin/main 的本地引用可能过期：fetch 一次再看，避免"同步"是假象
            git_in(node, "fetch --quiet origin main", timeout=180)
            rec["origin_main_fetched"], _ = git_in(node, "rev-parse origin/main")
            # 显式用 /work/env/bin/python + ENV_SITE 首位，保证 gate 校验的
            # 解释器/模块与实测完全一致（§A2.5）
            gate_py = ("export PYTHONPATH=%s:$PYTHONPATH && %s -c " %
                       (ENV_SITE, ENV_PYTHON)) + shlex.quote(GATE_CODE)
            r = docker_exec(node, gate_py, timeout=180)
            if r.returncode:
                rec["import_error"] = (r.stderr or r.stdout or "")[-500:]
            else:
                rec.update(json.loads(r.stdout.strip().splitlines()[-1]))
            rec["base_ancestor"], _ = git_in(
                node, "merge-base --is-ancestor %s HEAD && echo OK || echo FAIL" % PR_BASE)
            rec["image"] = run_ssh(
                host, "docker inspect --format '{{.Image}}' %s" % c,
                timeout=60).stdout.strip()
            rec["container_bound"] = host in STATE.get("container_override", {})
            rec["qp_plans_env"] = docker_exec(
                node, "printenv DEEPEP_CLOS_QP_PLANS || true", timeout=60).stdout.strip()
            r = run_ssh(host, "grep -i Huge /proc/meminfo", timeout=30)
            rec["hugepages"] = [ln for ln in r.stdout.splitlines() if ln.strip()]
            r = run_ssh(host, "npu-smi info", timeout=60)
            import re as _re
            rec["idle_npus"] = len(_re.findall(
                r"No running processes found in NPU\s+(\d+)", r.stdout))
            rec["npu_head"] = r.stdout.splitlines()[:4]
        except Exception as exc:
            rec["error"] = str(exc)[:400]
        report["nodes"][host] = rec

    ok_nodes = [h for h, r in report["nodes"].items() if "error" not in r and r.get("sos")]
    digests = {h: tuple(sorted(report["nodes"][h]["sos"].items())) for h in ok_nodes}
    same = len(set(digests.values())) == 1 if digests else False
    commits = {report["nodes"][h].get("commit") for h in ok_nodes}
    origin = {report["nodes"][h].get("origin_main_fetched")
              or report["nodes"][h].get("origin_main") for h in ok_nodes}
    remotes = {report["nodes"][h].get("remote") for h in ok_nodes}
    infos = {h: report["nodes"][h].get("build_info_sha", "") for h in ok_nodes}

    # 关键算子库必须进包：Host 扩展 _C*.so + Device kernel libdeepep_kernels.so
    kern = {}
    for h in ok_nodes:
        names = [os.path.basename(p) for p in report["nodes"][h]["sos"]]
        kern[h] = {"host_ext": any(n.startswith("_C") for n in names),
                   "device_kernels": any(n == "libdeepep_kernels.so" for n in names)}
    kernels_ok = bool(kern) and all(v["host_ext"] and v["device_kernels"] for v in kern.values())

    def node_vals(key):
        return [report["nodes"][h].get(key) for h in ok_nodes]

    interpreters = set(node_vals("interpreter"))
    images = set(node_vals("image"))
    buf_shas = set(node_vals("buffer_sha"))
    commit = (list(commits)[0] if len(commits) == 1 and "" not in commits
              else sorted(map(str, commits)))
    report["checks"] = {
        "all_nodes_reachable": len(ok_nodes) == len(NODES),
        "remote_repo": bool(remotes) and all(r and "lueelu/DeepEP" in r for r in remotes),
        "remotes": sorted(map(str, remotes)),
        "is_pr_head": commit == PR_HEAD,
        "commit": commit,
        "pr_head_expected": PR_HEAD,
        "not_origin_main": (bool(commit) and isinstance(commit, str)
                            and commit not in origin),
        "origin_main": sorted(map(str, origin)),
        "base_ancestor": bool(ok_nodes) and all(v == "OK" for v in node_vals("base_ancestor")),
        "single_commit": len(commits) == 1 and "" not in commits,
        "same_container_image": len(images) == 1 and "" not in images,
        "image": sorted(map(str, images)),
        "container_bound": bool(ok_nodes) and all(node_vals("container_bound")),
        "same_interpreter": (len(interpreters) == 1
                             and next(iter(interpreters), "") == ENV_PYTHON),
        "interpreters": sorted(map(str, interpreters)),
        "deep_ep_module_path": bool(ok_nodes) and all(
            "/opt/conda/" not in (report["nodes"][h].get("deep_ep") or "")
            for h in ok_nodes),
        "deep_ep_buffer_sha": (len(buf_shas) == 1
                               and next(iter(buf_shas), "") == BUFFER_SHA_EXPECTED),
        "buffer_sha": sorted(map(str, buf_shas)),
        "qp_plans_unset": bool(ok_nodes) and all(
            not (report["nodes"][h].get("qp_plans") or "")
            and not (report["nodes"][h].get("qp_plans_env") or "")
            for h in ok_nodes),
        "so_identical": same,
        "kernels_so_present": kernels_ok,
        "build_info_identical": len(set(infos.values())) == 1,
        "build_info_state": ("present" if all(infos.values())
                             else ("absent" if not any(infos.values()) else "partial")),
        "versions": sorted({report["nodes"][h].get("version", "") for h in ok_nodes}),
        "no_mesh_clos": all(not report["nodes"][h].get("mesh_clos") for h in ok_nodes),
        "clean_worktree": all(not report["nodes"][h].get("dirty") for h in ok_nodes),
        "cards_idle_all8": bool(ok_nodes) and len(ok_nodes) == len(NODES) and all(
            report["nodes"][h].get("idle_npus", -1) >= 16 for h in ok_nodes),
        "idle_npus": {h: report["nodes"][h].get("idle_npus", -1) for h in ok_nodes},
    }
    # 代码/容器/解释器判据（不含卡空闲）：换代码是否真的生效，只看这些。
    code_keys = ["all_nodes_reachable", "remote_repo", "is_pr_head",
                 "not_origin_main", "base_ancestor", "single_commit",
                 "same_container_image", "container_bound", "same_interpreter",
                 "deep_ep_module_path", "deep_ep_buffer_sha", "qp_plans_unset",
                 "so_identical", "kernels_so_present", "build_info_identical",
                 "no_mesh_clos", "clean_worktree"]
    report["code_pass"] = all(report["checks"][k] for k in code_keys)
    report["blocked_nodes"] = sorted(h for h, n in report["checks"]["idle_npus"].items()
                                     if n < 16)
    # full_pass 额外要求 8 台全空闲（C:EP128 需要 128 卡整齐）。如实记录，不撒谎。
    report["pass"] = report["code_pass"] and report["checks"]["cards_idle_all8"]
    (DATA / "round3_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2, ensure_ascii=False))
    print("code_pass=%s  full_pass=%s  blocked_nodes=%s" %
          (report["code_pass"], report["pass"], report["blocked_nodes"]))
    print("GATE %s -> %s" % ("PASS" if report["pass"] else "FAIL",
                             DATA / "round3_gate.json"))
    return 0 if report["pass"] else 2


def require_gate():
    p = DATA / "round3_gate.json"
    if not p.exists():
        sys.exit("没有 round3_gate.json：先跑 `python run_round3.py gate`。"
                 "本轮换了代码，必须先用门禁确认八机同一 PR head、同一 .so、纯 auto。")
    g = json.loads(p.read_text(encoding="utf-8"))
    if not g.get("code_pass"):
        sys.exit("gate 代码判据未全绿（见 round3_gate.json），拒绝执行性能/精度，"
                 "否则结果不能代表新代码。")
    blocked = g.get("blocked_nodes") or []
    if blocked:
        # 不整轮否决：各阶段 run_perf_ep / run_prec_ep / stage_abnormal 开头都有
        # check_idle(参与节点)，会在被占机器上硬挡，绝不与别人的任务抢卡。
        print("GATE NOTE: %d 台被占 %s；各阶段只对参与节点 check_idle 硬挡，"
              "被占机器不会被本任务触碰" % (len(blocked), blocked), flush=True)
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
    supp = getattr(args, "supp", False)
    set_phase("supp" if supp else "perf")   # 补测用空闲节点集；主 B 用 7 台集合
    gate = require_gate()
    commit = gate["checks"].get("commit")
    cases = perf_cases([e for e in args.eps if e != 128], supp=supp)
    if args.only_ep:
        cases = [c for c in cases if c["ep"] in args.only_ep]
    cases = cases[args.start:]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        sys.exit("没有可跑的用例")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA / ("round3_perf_supp.csv" if supp else "round3_perf.csv")
    out_root = STAGE + ("/out/perf_supp" if supp else "/out/perf")
    print("perf%s: %d 例，commit=%s，huge_pages=no，节点=%s" %
          ("(补测)" if supp else "", len(cases), commit,
           {ep: EP_NODES_ACTIVE[ep] for ep in sorted({c["ep"] for c in cases})}))

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
    for i in EP_NODES_ACTIVE[ep]:
        check_idle(node_by_idx(i), ep // need)
    tag_base = "perf_ep%d_r2_%s" % (ep, time.strftime("%Y%m%d_%H%M%S"))
    print("-- EP%d：%d 例，%d 机 × %d 卡（node%s）" % (
        ep, len(cases), need, ep // need, ",".join(map(str, EP_NODES_ACTIVE[ep]))), flush=True)
    for c in cases:
        run_id = "%s_c%02d" % (tag_base, c["idx"])
        argv = perf_argv(c, out_root, run_id)
        t0 = time.time()
        rcs = run_on_nodes(ep, argv, PORTS["perf"], run_id, timeout=args.timeout)
        ok = all(v == 0 for v in rcs.values())
        rows = []
        if ok:
            text = read_text(node_by_idx(EP_NODES_ACTIVE[ep][0]),
                             "%s/%s/node0/summary.csv" % (out_root, run_id))
            if not text.strip():
                text = read_text(node_by_idx(EP_NODES_ACTIVE[ep][0]),
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
PREC_COLS = ["run_id", "ts", "api", "ep_size", "nnodes", "group", "tag", "epr_tier",
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
    set_phase("precision")       # 8 台集合（含 node3/.7），EP8 放 .7
    gate = require_gate()
    commit = gate["checks"].get("commit")
    cases = precision_cases(args.eps, args.epr_tiers)
    if args.only_ep:
        cases = [c for c in cases if c["ep"] in args.only_ep]
    cases = cases[args.start:]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        sys.exit("没有可跑的用例")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    deploy_bench_files({c["ep"] for c in cases})
    csv_path = DATA / "round3_precision.csv"
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
    for i in EP_NODES_ACTIVE[ep]:
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
                # 逐节点收集后合并（8 机 /work 不共享，首节点只有本机的 16 rank）。
                # 仅读取判据所需的 json，不回传原始产物（既定「产物不回传」口径）。
                files = {}
                for i in EP_NODES_ACTIVE[ep]:
                    files.update(collect_json(node_by_idx(i), out_root + "/" + run_id))
                ranks, digest, errs = judge_precision(files, ep)
                state = "passed" if ranks == ep else "failed_judge"
            append_csv(csv_path, PREC_COLS, [{
                "run_id": run_id, "ts": now(), "api": api, "ep_size": ep,
                "nnodes": need, "group": c.get("group", ""), "tag": c.get("tag", ""),
                "epr_tier": c.get("epr_tier", "spec"),
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
    set_phase("precision")       # 异常用 EP8/EP16 单机，沿用精度集合（.7 可用）
    gate = require_gate()
    commit = gate["checks"].get("commit")
    # 用严格口径的 v2 worker（超容量 E5 + 纯张量 E4，见 commit 5cc6d2d）。
    # v1 的 abnormal_worker.py 原样保留，它对应上轮已实测的越界 expert id 口径。
    worker = HERE / "abnormal_worker_v2.py"
    LOGDIR.mkdir(parents=True, exist_ok=True)
    deploy_bench_files(args.eps)   # abnormal_worker_v2.py 也 import bench_v2_precision
    eps = args.eps
    sids = args.scenarios
    jobs = []
    for ep in eps:
        for api in args.api:
            for sid in sids:
                jobs.append({"ep": ep, "api": api, "sid": sid})
    print("abnormal: %d 次执行（%s），commit=%s" % (
        len(jobs), ",".join("%s/%s" % (j["api"], j["sid"]) for j in jobs[:4]) + "…", commit))

    for ep in sorted({j["ep"] for j in jobs}):
        need = nnodes_of(ep)
        for i in EP_NODES_ACTIVE[ep]:
            check_idle(node_by_idx(i), ep // need)
            deploy(node_by_idx(i), worker, "abnormal_worker_v2.py")

    results = []

    def one(job):
        ep, api, sid = job["ep"], job["api"], job["sid"]
        node = node_by_idx(EP_NODES_ACTIVE[ep][0])       # 异常场景沿用上轮的 EP8/EP16 单机
        run_id = "abn_ep%d_%s_%s_r2" % (ep, api, sid)
        out_root = "%s/out/abn/%s" % (STAGE, run_id)
        argv = ["python", "-m", "torch.distributed.run",
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

    csv_path = DATA / "round3_abnormal.csv"
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
    print("R3 第三轮测试计划（离线，不碰设备）")
    print("=" * 74)
    print("代码: PR #37 head %s（未合并；base 83252a5 须为祖先）" % PR_HEAD)
    print("容器: %s（8 机同名，显式绑定）" % CONTAINER_HINT)
    print("解释器: %s；PYTHONPATH 首位 %s" % (ENV_PYTHON, ENV_SITE))
    print("QP: env -u DEEPEP_CLOS_QP_PLANS，纯 auto（不传 udma_*、不调 timeout）")
    print("拓扑: " + "  ".join("node%d=%s" % (n["node"], n["host"]) for n in NODES))
    print("      各 16 卡；master=%s；ssh -p %d" % (MASTER, SSH_PORT))
    print("性能集合 PERF_NODES=%s（硬排除 node3/.7）" % sorted(PERF_NODES))
    print("精度集合 PREC_NODES=%s（全 8 台，含 .7）" % sorted(PREC_NODES))
    print()

    supp = getattr(args, "supp", False)
    plan_eps = [e for e in args.eps if e != 128]
    perf = perf_cases(plan_eps, supp=supp)
    print("%s：%d 例（%s）" %
          ("补测 EP4/EP64（实测原表 18 例/EP，空闲 5 台）" if supp else "阶段 B 性能（7 台）",
           len(perf),
           "每 EP 18 例：3 tokens × 3 hidden × 2 aiv" if supp
           else "每 EP 12 例：2 tokens × 3 hidden × 2 aiv"))
    set_phase("supp" if supp else "perf")
    print("  %-4s %-5s %-6s %s" % ("EP", "例数", "拓扑", "节点分配"))
    for ep in plan_eps:
        print("  %-4d %-5d %-6s node%s" % (
            ep, len([c for c in perf if c["ep"] == ep]),
            "%d机x%d卡" % (nnodes_of(ep), ep // nnodes_of(ep)),
            ",".join(map(str, EP_NODES_ACTIVE[ep]))))
    print("  大页: 关闭（argv 断言不含 --huge-pages）")
    print("  波次: " + " | ".join(
        " + ".join("EP%d" % e for e in w) for w in perf_waves(plan_eps)))
    print("  .7 排除断言: perf_argv 里 host 集合 ∩ {.7} = ∅")
    if supp:
        print("  补测节点: EP64=[2,4,5,6]=.6/.8/.9/.10，EP4=[7]=.11；避开 .4/.5 与 .7")
        print("  结果: round3_perf_supp.csv")
        print()
        if perf:
            sample = perf[0]
            print("示例命令（EP%d）：" % sample["ep"])
            print("  " + " ".join(perf_argv(sample, STAGE + "/out/perf_supp", "supp_preview_c01")))
        return 0
    print()

    set_phase("precision")
    prec = precision_cases(args.eps, args.epr_tiers)
    print("阶段 C/D 精度（8 台）：%d 例（每 EP 50 条 × epr 档位）" % len(prec))
    print("  API: %s；EP128 epr 档位: %s（主 4 / 探针 32，分开报不混算）" %
          (",".join(args.api), ",".join(args.epr_tiers)))
    print("  %-4s %-5s %-6s %s" % ("EP", "例数", "拓扑", "节点分配"))
    for ep in args.eps:
        print("  %-4d %-5d %-6s node%s" % (
            ep, len([c for c in prec if c["ep"] == ep]),
            "%d机x%d卡" % (nnodes_of(ep), ep // nnodes_of(ep)),
            ",".join(map(str, EP_NODES_ACTIVE[ep]))))
    print("  波次: " + " | ".join(
        " + ".join("EP%d" % e for e in w) for w in perf_waves(args.eps)))
    print()

    print("阶段 E 异常：E4 / E5 严格重构造（abnormal_worker_v2.py，2 台）")
    print("  E4: 一次 dispatch + 同一 handle 两次 combine，第二次换本地同布局专家输出")
    print("  E5: 第二轮传超容量 tokens（num_tokens = capacity+1），期望 ValueError")
    print("  矩阵: EP8,EP16 × {%s} × {%s}" % (
        ",".join(args.api), ",".join(args.scenarios)))
    print()

    print("门禁判据(round3_gate.json): is_pr_head / not_origin_main / base_ancestor /")
    print("  same_container_image / container_bound / same_interpreter /")
    print("  deep_ep_module_path / deep_ep_buffer_sha(==f825a679…) / qp_plans_unset /")
    print("  so_identical / kernels_so_present / no_mesh_clos / clean_worktree / cards_idle_all8")
    print("执行顺序: gate → C(EP128,8台) → B(性能,7台) → D(精度200,8台) → E(异常,2台)")
    print()
    if perf:
        set_phase("perf")
        sample = perf[0]
        ep = sample["ep"]
        rid = "perf_ep%d_r3_preview_c01" % ep
        print("示例命令（node%s，EP%d）：" % (EP_NODES_ACTIVE[ep][0], ep))
        print("  " + " ".join(perf_argv(sample, STAGE + "/out/perf", rid)))
        if len(EP_NODES_ACTIVE[ep]) > 1:
            print("  ＋ 各节点补 --node-rank=<0..%d> --master-addr=%s --master-port=%d"
                  % (len(EP_NODES_ACTIVE[ep]) - 1, MASTER, PORTS["perf"]))
    set_phase("precision")
    p0 = prec[0]
    print()
    print("示例精度命令（EP%d，%s）：" % (p0["ep"], p0["epr_tier"]))
    print("  " + " ".join(precision_argv(p0, STAGE + "/out/prec", "prec_preview", args.api[0])))
    return 0


def perf_waves(eps):
    """R3 波次（指南 §四·B2 / §六）。

    性能（7 台）：第 1 波 EP64(0,1,2,4) ‖ EP32(5,6) ‖ EP16(7) 占满 7 台；
                  第 2 波 EP8(7) 尾波（EP8 与 EP16 共用 node7，必须串行）。
                  .7 全程空置。
    精度（8 台）：EP128 独占一波（128 卡全占）；其余 EP64+EP32+EP16+EP8
                  一波打满 8 台（EP8 在 node3/.7）。
    """
    eps = set(eps)
    phase = STATE.get("phase", "perf")
    waves = []
    if 128 in eps:
        waves.append([128])
    rest = eps - {128}
    if phase == "supp":
        # 补测 EP4/EP64：节点不重叠（EP64=.6/.8/.9/.10, EP4=.11），一波并行。
        one = [e for e in (64, 4) if e in rest]
        if one:
            waves.append(one)
    elif phase == "perf":
        first = [e for e in (64, 32, 16) if e in rest]
        if first:
            waves.append(first)
        if 8 in rest:
            waves.append([8])
    else:
        one = [e for e in (64, 32, 16, 8) if e in rest]
        if one:
            waves.append(one)
    return waves


# --------------------------------------------------------------------------- #
def parse_eps(text):
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def main():
    ap = argparse.ArgumentParser(description="DeepEP CLOS 第三轮（R3）测试编排")
    ap.add_argument("stage", nargs="?", default="plan",
                    choices=["plan", "gate", "perf", "precision", "abnormal"])
    ap.add_argument("--eps", default="8,16,32,64",
                    help="性能/精度阶段参与编排的 EP；C 阶段用 --eps 128；补测用 --eps 4,64 --supp")
    ap.add_argument("--supp", action="store_true",
                    help="补测模式：EP4/EP64 用空闲节点(.6/.8/.9/.10/.11)，避开 .4/.5 与 .7，结果写 round3_perf_supp.csv")
    ap.add_argument("--only-ep", default="", help="只跑指定 EP，如 64")
    ap.add_argument("--api", default="v2", help="精度/异常阶段的 API，如 v2 或 v1,v2")
    ap.add_argument("--scenarios", default="E4,E5", help="异常场景，默认 E4,E5")
    ap.add_argument("--epr-tiers", default="main",
                    help="EP128 的 epr 档位：main(4,与基线同口径) / probe(32,大E泛化) / main,probe 双跑")
    ap.add_argument("--containers", default="",
                    help="显式绑定容器，格式 host=name,... 必须 8 台；默认 8 台同名新容器")
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
    args.epr_tiers = tuple(x for x in args.epr_tiers.replace(" ", "").split(",") if x)
    assert all(t in ("main", "probe") for t in args.epr_tiers), \
        "--epr-tiers 只能是 main / probe / main,probe"

    # 8 台显式绑定（不靠子串猜测）。未给 --containers 时用已确认的 8 台同名新容器。
    if args.containers:
        STATE["container_override"] = dict(
            kv.split("=", 1) for kv in args.containers.split(",") if "=" in kv)
    else:
        STATE["container_override"] = {n["host"]: CONTAINER_HINT for n in NODES}

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
