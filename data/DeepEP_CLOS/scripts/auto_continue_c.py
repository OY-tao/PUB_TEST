#!/usr/bin/env python3
"""等 .4/.5 释放后自动接续 R3 阶段 C（EP128 泛化精度），并监控重启。

用户 2026-11-28 指令：「在 .4/.5 释放后自动接续 C(EP128)」+
「C 也带『被kill/出错继续重试』，注意精度复用统一buffer」。

C 的执行体是 `r3_c_ep128/run_retry.py`（持久 8 机/128 rank 单进程组 +
compatibility_key 分组复用 Buffer + checkpoint + 40 次 attempt 自动重试），
不再是逐 case 独立 torchrun 的 run_round3.py precision。

两层重试：
  * 内层（run_retry.py）：每次 attempt 后 collect_and_judge 结算已通过用例，
    只重跑 remaining；worker 被 kill / 超时 / 异常都自动进入下一 attempt。
  * 外层（本脚本）：监控控制器进程本身，若被 kill / 异常退出且 status.json
    未 completed，则重新拉起 `run_retry.py --run`（启动会 recover_completed
    从容器磁盘恢复），直到完成或 max_wait。

触发条件三者全满足才开跑：
  1. B 阶段 perf 控制器已退出（避免与 C 抢 .6~.11 的卡）；
  2. 8 台 NPU 全部 16/16 空闲（C 需要 128 卡整齐，指南 §八/§九）；
  3. 没有别的 run_round3 / run_retry 控制器在跑（防重复触发）。
满足后先重跑 gate 要求 full_pass=True（8 台全空闲），再启动 C。

用法： python3 -u auto_continue_c.py [--interval 60] [--max-wait 7200]
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
STATUS = DATA / "round3_auto_c.json"
GATE = DATA / "round3_gate.json"
C_DIR = HERE / "r3_c_ep128"
C_STATUS = C_DIR / "status.json"
C_LOG = "/tmp/r3_c_ep128.log"
HOSTS = ["30.104.128.%d" % i for i in range(4, 12)]
SSH = ["ssh", "-T", "-p", "56000", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
CONT = "deepep_ep128_full_matrix_20261128_073617"


def log(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def idle_count(host):
    """返回该机空闲 NPU 数；不可达返回 -1。`grep -c` 无匹配时退出码 1 但
    stdout 仍是有效数字 0（占满），只看 stdout 是否为数字。"""
    try:
        p = subprocess.run(
            SSH + ["root@" + host, "npu-smi info 2>/dev/null | grep -c 'No running processes found in NPU'"],
            capture_output=True, text=True, timeout=45)
        out = (p.stdout or "").strip()
        if out.isdigit():
            return int(out)
    except Exception:
        pass
    return -1


def all_idle():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(HOSTS, pool.map(idle_count, HOSTS)))


def controllers():
    """本地正在跑的控制器命令行（排除本脚本自身）。

    只看 argv[0] 的 basename 以 python 开头的进程：这样既能命中
    `/usr/bin/python3 -u run_round3.py ...`（补测控制器是这个前缀，
    旧的 startswith 白名单会漏掉它），又不会把 bash -c 包装器误判成控制器。
    同时纳入 run_perf_reuse.py：复用版性能在跑时不许触发 C。
    """
    out = []
    me = os.getpid()
    scripts = (b"run_round3.py", b"run_retry.py", b"run_perf_reuse.py")
    for p in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            pid = int(p.name)
            if pid == me:
                continue
            s = b" ".join((p / "cmdline").read_bytes().split(bytes([0])))
            if not any(name in s for name in scripts):
                continue
            head = s.split(None, 1)[0].rsplit(b"/", 1)[-1]
            if head.startswith(b"python"):
                out.append(s.decode("utf-8", "replace"))
        except (OSError, ValueError, ProcessLookupError, IndexError):
            pass
    return out


def b_still_running():
    return any("run_round3.py" in c and " perf" in c for c in controllers())


def other_controllers():
    return [c for c in controllers() if "auto_continue_c" not in c]


def run_gate():
    """重跑门禁，要求 full_pass（8 台全空闲）。"""
    log("条件满足，重跑 gate（要求 full_pass=True）…")
    p = subprocess.run([sys.executable, "-u", str(HERE / "run_round3.py"), "gate"],
                       cwd=str(HERE), capture_output=True, text=True, timeout=600)
    log("gate rc=%s" % p.returncode)
    for ln in (p.stdout or "").splitlines()[-5:]:
        log("  gate| " + ln)
    try:
        g = json.loads(GATE.read_text(encoding="utf-8"))
    except Exception as exc:
        log("gate.json 读取失败: %s" % exc)
        return False
    log("gate code_pass=%s full_pass=%s blocked=%s" %
        (g.get("code_pass"), g.get("pass"), g.get("blocked_nodes")))
    return bool(g.get("pass")) and p.returncode == 0


def c_status():
    try:
        return json.loads(C_STATUS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def launch_c():
    """拉起 C 控制器（持久进程组 + Buffer 复用 + checkpoint 重试），返回 Popen。"""
    lf = open(C_LOG, "a")
    cmd = [sys.executable, "-u", str(C_DIR / "run_retry.py"), "--run"]
    p = subprocess.Popen(cmd, cwd=str(C_DIR), stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    log("C 控制器已启动 pid=%s log=%s cmd=%s" % (p.pid, C_LOG, " ".join(cmd)))
    return p


def write_status(state, **kw):
    d = {"state": state, "ts": time.strftime("%F %T")}
    d.update(kw)
    STATUS.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def supervise_c(proc, interval, deadline):
    """外层监控：控制器进程被 kill / 异常退出 → 重启，直到 completed。"""
    relaunches = 0
    while time.time() < deadline:
        time.sleep(interval)
        st = c_status()
        completed = len(st.get("completed", {}))
        total = st.get("total_cases", 400)
        state = st.get("state", "unknown")
        rc = proc.poll()
        if state == "completed":
            write_status("c_completed", c_state=state, completed=completed,
                         total=total, relaunches=relaunches,
                         note="C(EP128) 全量完成；Buffer 分组复用 + checkpoint 重试")
            log("DONE: C completed %d/%d（外层重启 %d 次）" %
                (completed, total, relaunches))
            return 0
        if rc is not None:
            relaunches += 1
            log("C 控制器退出 rc=%s state=%s completed=%d/%d → 外层重启第 %d 次"
                "（run_retry 启动会 recover_completed 恢复）" %
                (rc, state, completed, total, relaunches))
            proc = launch_c()
            write_status("c_running", c_state="relaunched", completed=completed,
                         total=total, relaunches=relaunches, c_pid=proc.pid)
        else:
            write_status("c_running", c_state=state, completed=completed,
                         total=total, relaunches=relaunches, c_pid=proc.pid)
            log("C 运行中 state=%s completed=%d/%d relaunches=%d" %
                (state, completed, total, relaunches))
    write_status("timeout", c_state=c_status().get("state"),
                 relaunches=relaunches, note="超过 max_wait，C 未完成")
    log("TIMEOUT: C 未完成")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60, help="轮询间隔秒")
    ap.add_argument("--max-wait", type=int, default=7200, help="最长等待秒")
    o = ap.parse_args()
    t0 = time.time()
    deadline = t0 + o.max_wait
    log("自动接续 C 启动：等 .4/.5 释放 + B 结束 + 8 台全空闲；之后监控重启直到完成")
    write_status("waiting", started=time.strftime("%F %T"), hosts=HOSTS,
                 interval=o.interval, max_wait=o.max_wait,
                 c_controller=str(C_DIR / "run_retry.py"),
                 retry_layers="inner: 40 attempts+checkpoint; outer: process relaunch",
                 buffer_reuse="one process group, compatibility_key grouped Buffer")

    while time.time() < deadline:
        try:
            idle = all_idle()
            b = b_still_running()
            others = other_controllers()
            n_idle = sum(1 for v in idle.values() if v >= 16)
            blocked = [h for h, v in idle.items() if v < 16]
            log("空闲 %d/8 台，B控制器=%s，其它控制器=%d，未空闲=%s" %
                (n_idle, b, len(others), ",".join(blocked) or "-"))
            ready = (n_idle == 8 and not b and not others)
            if ready:
                log("8 台全空闲、B 已结束、无其它控制器 → 准备触发 C")
                if run_gate():
                    proc = launch_c()
                    write_status("c_running", triggered_at=time.strftime("%F %T"),
                                 gate_full_pass=True, c_pid=proc.pid,
                                 c_log=C_LOG, relaunches=0,
                                 note="C = r3_c_ep128/run_retry.py --run")
                    return supervise_c(proc, o.interval, deadline)
                log("gate 未达 full_pass（可能又有任务占卡），继续等待")
            write_status("waiting", idle_npus=idle, b_running=b,
                         other_controllers=others, ready=ready)
        except Exception as exc:
            log("轮询异常（继续等）: %s" % str(exc)[:200])
        time.sleep(o.interval)

    write_status("timeout", note="超过 max_wait 仍未满足条件，C 未触发")
    log("TIMEOUT: C 未触发")
    return 1


if __name__ == "__main__":
    sys.exit(main())
