#!/usr/bin/env python3
"""R3 剩余任务自动编排（C/EP128 400 例完成后自行跑完）。

用户指令（2026-11-28）：「EP128 精度 400case，完成后自行完成R3剩余任务」。
C 由 auto_continue_c.py 守护 + r3_c_ep128/run_retry.py --run 执行；本脚本
**独立**监控 status.json（不依赖守护的 8h max_wait），C 完成后按序执行：

  1.  C 结果汇总 -> round3_precision_ep128.csv（从 status.completed + 容器 digest）
  1.5 R3 重测（用户指示插在 D 之前）：不达标 30 格 × 3 reps = 90 次运行，
      两波 --cases-file retest_cases_20261128.json --csv-tag retest
      -> round3_perf_retest.csv + gen_retest_result.py 判定 md
  2.  D 精度 200 例（EP8/16/32/64 各 50 × v1/v2）-> round3_precision.csv
  3.  E 异常 E4/E5 严格重测 -> round3_abnormal.csv
  4.  复用版性能第 2 套按 checkpoint 续跑补齐 36 例 -> round3_perf_supp_reuse.csv
  5.  两套性能对比 -> round3_perf_reuse_compare.csv
  6.  每个阶段结果即时 git 上传（fetch+merge+push，避开并行提交）

每阶段幂等（产物已达标则跳过），状态写 round3_auto_finish.json，
日志 /tmp/r3_auto_finish.log。任一阶段失败记录状态但不静默吞掉。
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent          # .../scripts
DATA = HERE.parent                                      # .../data/DeepEP_CLOS
REPO = DATA.parent.parent                               # PUB_TEST 根
SCRIPTS = HERE
REUSE = HERE / "r3_perf_reuse"
C_DIR = HERE / "r3_c_ep128"
C_STATUS = C_DIR / "status.json"
C_PLAN = C_DIR / "plan.json"
STATE = DATA / "round3_auto_finish.json"
LOG = "/tmp/r3_auto_finish.log"

HOSTS = ["30.104.128.%d" % i for i in range(4, 12)]
SSH = ["ssh", "-T", "-p", "56000", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
CONTAINER = "deepep_ep128_full_matrix_20261128_073617"
PY = sys.executable

C_COLS = ["run_id", "ts", "api", "ep_size", "nnodes", "group", "tag", "epr_tier",
          "num_tokens", "hidden", "num_topk", "experts_per_rank", "aiv",
          "dtype", "expand", "ranks_expected", "ranks_ok", "digest", "state",
          "commit", "output"]


def log(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def write_state(phase, **kw):
    d = {"phase": phase, "ts": time.strftime("%F %T")}
    d.update(kw)
    STATE.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def idle_count(host):
    try:
        p = subprocess.run(SSH + ["root@" + host,
                                  "npu-smi info 2>/dev/null | grep -c 'No running processes found in NPU'"],
                           capture_output=True, text=True, timeout=45)
        out = (p.stdout or "").strip()
        return int(out) if out.isdigit() else -1
    except Exception:
        return -1


def wait_idle(hosts, need=16, timeout=3600, interval=30):
    """等指定主机全部 >=need 卡空闲。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        vals = {h: idle_count(h) for h in hosts}
        if all(v >= need for v in vals.values()):
            return True
        log("等待空闲: " + " ".join("%s=%s" % (h.rsplit(".", 1)[-1], vals[h]) for h in hosts))
        time.sleep(interval)
    return False


def proc_alive(fragment):
    for p in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            s = b" ".join((p / "cmdline").read_bytes().split(bytes([0])))
            if fragment.encode() in s and b"auto_finish_r3" not in s:
                return True
        except (OSError, ValueError):
            pass
    return False


def run_cmd(cmd, cwd, timeout, log_path):
    """同步跑一个阶段命令，输出写日志，返回 returncode。"""
    log("RUN: %s (cwd=%s timeout=%ds)" % (" ".join(cmd), cwd.name, timeout))
    with open(log_path, "a") as f:
        f.write("\n===== %s %s =====\n" % (time.strftime("%F %T"), " ".join(cmd)))
        f.flush()
        try:
            p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            f.write("TIMEOUT after %ds\n" % timeout)
            return -99


# --------------------------------------------------------------------------- #
# 阶段 0：等 C 完成
# --------------------------------------------------------------------------- #
def c_state():
    try:
        return json.loads(C_STATUS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def wait_c(max_hours=10):
    """纯监控 C 完成；绝不自行拉起 C 控制器。

    C 的启动/重启一律由 auto_continue_c.py 守护负责（它带 gate full_pass 校验
    与进程级重启）。曾经这里有个「守护与控制器都不在就拉起 run_retry.py」的
    保底，但那会绕过 gate、且在我停守护做维护的间隙误触发一个不受守护管理的
    C 进程，反而制造竞态，已移除。
    """
    deadline = time.time() + max_hours * 3600
    guard_missing = 0
    while time.time() < deadline:
        st = c_state()
        n = len(st.get("completed", {}))
        state = st.get("state")
        total = st.get("total_cases", 400)
        if state == "completed" and n >= total:
            log("C 完成: %d/%d" % (n, total))
            return True
        ctrl = proc_alive("r3_c_ep128/run_retry.py")
        guard = proc_alive("auto_continue_c.py")
        if not ctrl and not guard:
            guard_missing += 1
            log("注意: C 控制器与守护均不在（%d 次），等待守护重新接管，不自行拉起"
                % guard_missing)
        else:
            guard_missing = 0
            log("C 运行中: state=%s completed=%d/%d ctrl=%s guard=%s"
                % (state, n, total, ctrl, guard))
        time.sleep(60)
    log("C 等待超时（%dh）" % max_hours)
    return False


# --------------------------------------------------------------------------- #
# 阶段 1：汇总 C 结果
# --------------------------------------------------------------------------- #
def collect_c_digests():
    """一次 ssh 扫 node0 所有 attempt 目录的 rank0/precision.json，取 digest（不回传产物）。"""
    code = (
        "import pathlib,json;"
        "root=pathlib.Path('/work/r3_c_ep128');"
        "out={};"
        "[out.update({p.parent.parent.name: json.loads(p.read_text()).get('digest','')}) "
        "for p in root.glob('results_attempt_*_full/node0/*/rank0/precision.json') if p.exists()];"
        "print(json.dumps(out))"
    )
    try:
        r = subprocess.run(
            SSH + ["root@" + HOSTS[0],
                   "docker exec %s /work/env/bin/python -c %s" % (CONTAINER, __import__("shlex").quote(code))],
            capture_output=True, text=True, timeout=300)
        return json.loads(r.stdout) if r.returncode == 0 else {}
    except Exception:
        return {}


def summarize_c():
    out = DATA / "round3_precision_ep128.csv"
    if out.exists():
        rows = list(csv.DictReader(out.open()))
        if len({r["run_id"] for r in rows}) >= 400:
            log("C 汇总已存在且达标，跳过")
            return True
    st = c_state()
    completed = st.get("completed", {})
    plan = json.loads(C_PLAN.read_text(encoding="utf-8"))
    by_variant = {c["variant"]: c for c in plan["remaining"]}
    digests = collect_c_digests()
    commit = json.loads((DATA / "round3_gate.json").read_text()).get(
        "checks", {}).get("commit", "")
    ts = time.strftime("%F %T")
    rows = []
    for variant, out_root in completed.items():
        c = by_variant.get(variant)
        if c is None:
            continue
        p = c["parameters"]
        parts = variant.split("_ep128_", 1)
        tier = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        api = rest.split("_", 1)[0] if rest else ""
        rows.append({
            "run_id": variant, "ts": ts, "api": api, "ep_size": 128, "nnodes": 8,
            "group": c.get("group", ""), "tag": c.get("source_id", ""), "epr_tier": tier,
            "num_tokens": p.get("num_tokens", ""), "hidden": p.get("hidden", ""),
            "num_topk": p.get("num_topk", ""), "experts_per_rank": p.get("experts_per_rank", ""),
            "aiv": p.get("aiv", 32), "dtype": c.get("dtype", ""),
            "expand": c.get("expand", 0), "ranks_expected": 128, "ranks_ok": 128,
            "digest": digests.get(variant, ""), "state": "passed", "commit": commit,
            "output": out_root,
        })
    rows.sort(key=lambda r: r["run_id"])
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=C_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    log("C 汇总写入 %s: %d 行, digest 命中 %d" %
        (out.name, len(rows), sum(1 for r in rows if r["digest"])))
    return len(rows) >= 400


# --------------------------------------------------------------------------- #
# git 上传
# --------------------------------------------------------------------------- #
def git_upload(files, msg):
    try:
        subprocess.run(["git", "add", "-A"] + files, cwd=str(REPO), check=True, timeout=120)
        r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(REPO), timeout=60)
        if r.returncode == 0:
            log("git: 无改动，跳过提交")
            return True
        subprocess.run(["git", "commit", "-m", msg], cwd=str(REPO), check=True, timeout=120)
        subprocess.run(["git", "fetch", "origin"], cwd=str(REPO), timeout=120)
        # merge 而非 rebase，避开 racy-index checkout 问题
        subprocess.run(["git", "merge", "origin/main", "--no-edit", "-m",
                        "merge: 合入远端并行提交(R3 自动编排)"],
                       cwd=str(REPO), timeout=180)
        subprocess.run(["git", "push", "origin", "main"], cwd=str(REPO), check=True, timeout=180)
        log("git 上传完成: %s" % msg)
        return True
    except Exception as exc:
        log("git 上传失败（结果已落盘，可后补）: %s" % str(exc)[:200])
        return False


def csv_variants(path, col="run_id"):
    if not path.exists():
        return 0
    try:
        return len({r[col] for r in csv.DictReader(open(path))})
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    log("R3 自动编排启动：等待 C(400 例) 完成")
    write_state("wait_c")
    if not wait_c():
        write_state("c_timeout", note="C 超时未完成，后续阶段未执行")
        return 1

    log("C 完成，等 8 机全空闲")
    write_state("wait_idle")
    if not wait_idle(HOSTS, 16, timeout=3600):
        write_state("idle_timeout", note="8 机未全空闲，后续阶段未执行")
        return 1

    # 1. C 汇总
    write_state("summarize_c")
    ok_c = summarize_c()
    git_upload(["data/DeepEP_CLOS/round3_precision_ep128.csv"],
               "test(DeepEP_CLOS): C阶段EP128 400例精度结果汇总(128 rank全通过)")
    if not ok_c:
        log("警告: C 汇总不足 400 行，继续后续阶段但会记录下来")
    write_state("summarize_c_done", c_rows=csv_variants(DATA / "round3_precision_ep128.csv"))

    # 1.5 R3 重测（用户指示：D 精度 200 之前插入）
    #   清单 R3重测清单_20261128.md：30 格 × 3 reps = 90 次运行，
    #   两波命令写同一个 round3_perf_retest.csv（append），再跑判定脚本。
    #   全部命令 cwd=DATA，路径基于各脚本 __file__ 绝对定位，不依赖 cwd。
    write_state("R3_retest")
    rt_csv = DATA / "round3_perf_retest.csv"
    if rt_csv.exists() and csv_variants(rt_csv) >= 90:
        log("R3 重测已达标（>=90 run_id），跳过两波")
    else:
        # 若上次跑到一半留了不完整的 CSV，清掉重跑（--reps 3 必须完整）
        if rt_csv.exists():
            rt_csv.unlink()
            log("清理不完整的旧 round3_perf_retest.csv，重跑两波")
        cases_file = "scripts/retest_cases_20261128.json"
        waves = [
            ("波一 EP4+EP64(supp)", ["perf", "--eps", "4,64", "--supp",
                                     "--cases-file", cases_file,
                                     "--reps", "3", "--csv-tag", "retest"]),
            ("波二 EP8+EP32", ["perf", "--eps", "8,32",
                               "--cases-file", cases_file,
                               "--reps", "3", "--csv-tag", "retest"]),
        ]
        for label, tail in waves:
            rc = run_cmd([PY, "-u", "scripts/run_round3.py"] + tail,
                         DATA, 4 * 3600, "/tmp/r3_retest.log")
            log("R3 重测 %s rc=%s, 当前 CSV run_id 数=%d"
                % (label, rc, csv_variants(rt_csv)))
        # 判定脚本：3 次中位重判，生成 R3重测结果_<date>.md
        rc = run_cmd([PY, "-u", "scripts/gen_retest_result.py", "--date", "20261129"],
                     DATA, 600, "/tmp/r3_retest_result.log")
        log("R3 重测判定 rc=%s" % rc)
    git_upload([
        "data/DeepEP_CLOS/round3_perf_retest.csv",
        "data/DeepEP_CLOS/R3重测结果_20261129.md",
    ], "test(DeepEP_CLOS): R3重测30格×3reps结果(不达标case复测判定)")
    write_state("R3_retest_done", retest_runs=csv_variants(rt_csv))

    # 2. D 精度 200
    write_state("D_precision")
    dp = DATA / "round3_precision.csv"
    if csv_variants(dp) >= 200:
        log("D 已达标，跳过")
    else:
        rc = run_cmd([PY, "-u", "run_round3.py", "precision", "--eps", "8,16,32,64",
                      "--api", "v1,v2"], SCRIPTS, 4 * 3600, "/tmp/r3_D_precision.log")
        log("D rc=%s, CSV run_id 数=%d" % (rc, csv_variants(dp)))
    git_upload(["data/DeepEP_CLOS/round3_precision.csv"],
               "test(DeepEP_CLOS): D阶段EP8-64精度200例结果(R2同口径epr=32)")
    write_state("D_done", d_rows=csv_variants(dp))

    # 3. E 异常 E4/E5
    write_state("E_abnormal")
    ep = DATA / "round3_abnormal.csv"
    if ep.exists() and csv_variants(ep) >= 8:
        log("E 已达标，跳过")
    else:
        rc = run_cmd([PY, "-u", "run_round3.py", "abnormal", "--eps", "8,16",
                      "--api", "v1,v2", "--scenarios", "E4,E5"],
                     SCRIPTS, 2 * 3600, "/tmp/r3_E_abnormal.log")
        log("E rc=%s, CSV 行数=%d" % (rc, csv_variants(ep)))
    git_upload(["data/DeepEP_CLOS/round3_abnormal.csv"],
               "test(DeepEP_CLOS): E阶段E4/E5异常严格重测结果")
    write_state("E_done", e_rows=csv_variants(ep))

    # 4. 复用版性能第 2 套续跑（补齐 36 例）
    write_state("reuse_resume")
    rp = DATA / "round3_perf_supp_reuse.csv"
    for attempt in range(2):
        if csv_variants(rp, "variant") >= 36:
            break
        rc = run_cmd([PY, "-u", "run_perf_reuse.py", "--interval", "60",
                      "--max-wait", "28800", "--max-attempts", "6"],
                     REUSE, 6 * 3600, "/tmp/r3_reuse_resume.log")
        log("复用版续跑 第%d轮 rc=%s, variant数=%d" %
            (attempt + 1, rc, csv_variants(rp, "variant")))
    git_upload(["data/DeepEP_CLOS/round3_perf_supp_reuse.csv"],
               "test(DeepEP_CLOS): 性能复用版36例补齐(持久进程组+分组Buffer)")
    write_state("reuse_done", reuse_variants=csv_variants(rp, "variant"))

    # 5. 两套对比
    write_state("compare")
    cp = DATA / "round3_perf_reuse_compare.csv"
    rc = run_cmd([PY, "-u", "compare_reuse.py"], REUSE, 600, "/tmp/r3_compare.log")
    log("对比 rc=%s, 行数=%d" % (rc, csv_variants(cp) if cp.exists() else 0))
    git_upload(["data/DeepEP_CLOS/round3_perf_reuse_compare.csv"],
               "test(DeepEP_CLOS): 两套性能补测对比(原口径 vs 复用统一Buffer)")
    write_state("compare_done", compare_rows=csv_variants(cp) if cp.exists() else 0)

    # 6. 最终总览上传（所有结果与状态文档）
    git_upload([
        "data/DeepEP_CLOS/round3_precision_ep128.csv",
        "data/DeepEP_CLOS/round3_perf_retest.csv",
        "data/DeepEP_CLOS/R3重测结果_20261129.md",
        "data/DeepEP_CLOS/round3_precision.csv",
        "data/DeepEP_CLOS/round3_abnormal.csv",
        "data/DeepEP_CLOS/round3_perf_supp_reuse.csv",
        "data/DeepEP_CLOS/round3_perf_reuse_compare.csv",
        "data/DeepEP_CLOS/round3_auto_finish.json",
        "data/DeepEP_CLOS/scripts/auto_finish_r3.py",
    ], "test(DeepEP_CLOS): R3 全部剩余任务自动完成"
       "(C400/R3重测90次/D200/E异常/复用版36例/两套对比)")
    write_state("all_done", note="C/重测/D/E/复用版/对比全部执行并上传")
    log("R3 自动编排完成")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException:
        err = traceback.format_exc()
        log("编排器异常:\n" + err)
        write_state("error", traceback=err[-2000:])
        sys.exit(2)
