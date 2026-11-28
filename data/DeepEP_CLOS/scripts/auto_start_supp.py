#!/usr/bin/env python3
"""等 B 阶段收尾后自动启动「EP4 + EP64」性能补测（基于 1128 性能表）。

用户 2026-11-28 指令：「基于1128的性能表，补充ep4和ep64性能，node7不测性能，
换其他空闲机器」。经确认 node7 指 .7（node3，性能有问题那台），补测继续排除它；
「换其他空闲机器」= EP64 原分配 [0,1,2,4] 含被别人占的 .4/.5，改用
[2,4,5,6]=.6/.8/.9/.10；EP4 用 [7]=.11 单机 4 卡。

触发条件：
  1. B 控制器（run_round3.py perf --eps 8,16,32）已退出；
  2. 补测所需的 5 台（.6/.8/.9/.10/.11）全部 16/16 空闲；
  3. 没有别的补测控制器在跑（防重复）。
  注意：C 守护等的是「8 台全空闲」，补测只用 5 台、且不占 .4/.5，
  因此不会与 C 守护抢卡；若 .4/.5 释放，C 守护仍可能在补测占用
  .6~.11 时判定 8 台不全空闲而继续等待，属预期（C 需 128 卡整齐）。

补测本身用 run_round3.py perf --eps 4,64 --supp，36 例（各 18 例，来源
EP4-128_通信带宽_20261128.txt 实测原表），一波并行：
EP64 四机与 EP4 单机节点不重叠。被 kill / 出错时按 60s 间隔重启，直到
round3_perf_supp.csv 出现 36 个不同 run_id（每个 case 4 个 op 行）。

用法： python3 -u auto_start_supp.py [--interval 60] [--max-wait 7200]
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent
STATUS = DATA / "round3_supp_status.json"
SUPP_CSV = DATA / "round3_perf_supp.csv"
SUPP_LOG = "/tmp/r3_supp_4_64.log"
HOSTS = ["30.104.128.%d" % i for i in range(4, 12)]
# 补测需要的机器（node2/4/5/6/7 = .6/.8/.9/.10/.11），排除 .4/.5 与 .7
SUPP_HOSTS = ["30.104.128.%d" % i for i in (6, 8, 9, 10, 11)]
SSH = ["ssh", "-T", "-p", "56000", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


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


def supp_idle():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        return dict(zip(SUPP_HOSTS, pool.map(idle_count, SUPP_HOSTS)))


def controllers():
    out = []
    me = os.getpid()
    for p in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            pid = int(p.name)
            if pid == me:
                continue
            s = b" ".join((p / "cmdline").read_bytes().split(bytes([0])))
            if s.startswith((b"python3", b"python", b"/opt/conda")) and b"run_round3.py" in s:
                out.append(s.decode("utf-8", "replace"))
        except (OSError, ValueError, ProcessLookupError):
            pass
    return out


def b_running():
    return any("run_round3.py" in c and " perf" in c and "--eps 8,16,32" in c
               for c in controllers())


def supp_running():
    return any("run_round3.py" in c and " perf" in c and "--supp" in c
               for c in controllers())


def write_status(state, **kw):
    d = {"state": state, "ts": time.strftime("%F %T")}
    d.update(kw)
    STATUS.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def completed_cases():
    if not SUPP_CSV.exists():
        return 0
    import csv
    try:
        rows = list(csv.DictReader(SUPP_CSV.open()))
        return len({r["run_id"] for r in rows})
    except Exception:
        return 0


def launch_supp():
    lf = open(SUPP_LOG, "a")
    cmd = [sys.executable, "-u", str(HERE / "run_round3.py"), "perf",
           "--eps", "4,64", "--supp"]
    p = subprocess.Popen(cmd, cwd=str(HERE), stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    log("补测控制器已启动 pid=%s log=%s cmd=%s" % (p.pid, SUPP_LOG, " ".join(cmd)))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-wait", type=int, default=7200)
    o = ap.parse_args()
    deadline = time.time() + o.max_wait
    log("补测自动启动器：等 B 退出 + .6/.8/.9/.10/.11 全空闲 → 跑 EP4/EP64 各 18 例")
    write_status("waiting", started=time.strftime("%F %T"),
                 supp_nodes=SUPP_HOSTS,
                 note="EP64=[2,4,5,6]=.6/.8/.9/.10, EP4=[7]=.11; 排除 .4/.5 与 .7; 18例/EP")

    proc = None
    relaunches = 0
    while time.time() < deadline:
        try:
            done = completed_cases()
            if done >= 36:
                write_status("completed", cases=done, relaunches=relaunches,
                             note="EP4+EP64 各 18 例已写入 round3_perf_supp.csv")
                log("DONE: 补测完成 %d/36 例（重启 %d 次）" % (done, relaunches))
                return 0
            idle = supp_idle()
            n_idle = sum(1 for v in idle.values() if v >= 16)
            b = b_running()
            sr = supp_running()
            if proc is None:
                ready = (n_idle == 5 and not b and not sr)
                log("空闲 %d/5，B=%s，补测控制器=%s，已完成 %d/36，ready=%s，节点=%s" %
                    (n_idle, b, sr, done, ready, idle))
                if ready:
                    proc = launch_supp()
                    relaunches = 0
                    write_status("running", supp_pid=proc.pid, cases=done,
                                 relaunches=relaunches, idle_npus=idle)
                else:
                    write_status("waiting", idle_npus=idle, b_running=b,
                                 supp_running=sr, cases=done, ready=ready)
            else:
                rc = proc.poll()
                if rc is not None:
                    relaunches += 1
                    log("补测控制器退出 rc=%s 已完成 %d/36 → 重启第 %d 次"
                        % (rc, done, relaunches))
                    proc = launch_supp()
                    write_status("running", supp_pid=proc.pid, cases=done,
                                 relaunches=relaunches, idle_npus=idle)
                else:
                    write_status("running", supp_pid=proc.pid, cases=done,
                                 relaunches=relaunches)
                    log("补测运行中 已完成 %d/36 relaunches=%d" % (done, relaunches))
        except Exception as exc:
            log("轮询异常（继续）: %s" % str(exc)[:200])
        time.sleep(o.interval)
    write_status("timeout", cases=completed_cases(), relaunches=relaunches,
                 note="超过 max_wait，补测未完成")
    log("TIMEOUT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
