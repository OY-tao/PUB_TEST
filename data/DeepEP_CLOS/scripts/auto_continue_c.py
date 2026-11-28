#!/usr/bin/env python3
"""等 .4/.5 释放后自动接续 R3 阶段 C（EP128 泛化精度）。

设计（用户 2026-11-28 指令「在 .4/.5 释放后自动接续 C(EP128)」）：

  * 只做「等待 + 触发」，本身不占 NPU、不改用例、不动脚本；
  * 三个条件全部满足才开跑：
      1. B 阶段 perf 控制器已退出（避免与 C 抢 .6~.11 的卡）；
      2. 8 台 NPU 全部 16/16 空闲（C 需要 128 卡整齐，指南 §八/§九）；
      3. 没有别的 run_round3 控制器在跑（防重复触发）。
  * 满足后：先重跑 `gate` 要求 **full_pass=True**（8 台全空闲），再启动
    `precision --eps 128 --epr-tiers main --api v1,v2`（C 阶段）。
  * 触发后写 round3_auto_c.json 并退出；C 自身的进度由控制器日志/CSV 跟踪。
  * 出错不退出，按间隔继续等（用户既定「不中断、出错继续」口径）。

用法：
    python3 -u auto_continue_c.py [--interval 60] [--max-wait 7200]
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
HOSTS = ["30.104.128.%d" % i for i in range(4, 12)]
SSH = ["ssh", "-T", "-p", "56000", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
CONT = "deepep_ep128_full_matrix_20261128_073617"


def log(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def idle_count(host):
    """返回该机空闲 NPU 数；不可达返回 -1（不误判为全空闲）。

    注意 `grep -c` 无匹配时退出码是 1、但 stdout 仍是有效数字 `0`
    （卡被占满）。所以只看 stdout 是否为数字，不能要求 returncode==0，
    否则「占满」会被误写成「不可达」。
    """
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
    """本地正在跑的 run_round3 控制器命令行列表（排除本脚本自身）。"""
    out = []
    me = os.getpid()
    for p in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            pid = int(p.name)
            if pid == me:
                continue
            s = b" ".join((p / "cmdline").read_bytes().split(bytes([0])))
            if b"run_round3.py" in s and s.startswith((b"python3", b"python", b"/opt/conda")):
                out.append(s.decode("utf-8", "replace"))
        except (OSError, ValueError, ProcessLookupError):
            pass
    return out


def b_still_running():
    return any(" run_round3.py " in c and " perf" in c for c in controllers())


def other_controllers():
    return [c for c in controllers() if "auto_continue_c" not in c]


def run_gate():
    """重跑门禁，要求 full_pass（8 台全空闲）。"""
    log("条件满足，重跑 gate（要求 full_pass=True）…")
    p = subprocess.run([sys.executable, "-u", str(HERE / "run_round3.py"), "gate"],
                       cwd=str(HERE), capture_output=True, text=True, timeout=600)
    log("gate rc=%s" % p.returncode)
    for ln in (p.stdout or "").splitlines()[-6:]:
        log("  gate| " + ln)
    try:
        g = json.loads(GATE.read_text(encoding="utf-8"))
    except Exception as exc:
        log("gate.json 读取失败: %s" % exc)
        return False
    log("gate code_pass=%s full_pass=%s blocked=%s" %
        (g.get("code_pass"), g.get("pass"), g.get("blocked_nodes")))
    return bool(g.get("pass")) and p.returncode == 0


def launch_c():
    """启动 C 阶段（后台），返回 PID。"""
    log_path = "/tmp/r3_c_ep128.log"
    lf = open(log_path, "w")
    cmd = [sys.executable, "-u", str(HERE / "run_round3.py"), "precision",
           "--eps", "128", "--epr-tiers", "main", "--api", "v1,v2"]
    p = subprocess.Popen(cmd, cwd=str(HERE), stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    log("C 已启动: pid=%s  log=%s  cmd=%s" % (p.pid, log_path, " ".join(cmd)))
    return p.pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60, help="轮询间隔秒")
    ap.add_argument("--max-wait", type=int, default=7200, help="最长等待秒")
    o = ap.parse_args()
    t0 = time.time()
    log("自动接续 C 启动：等待 .4/.5 释放 + B 阶段结束 + 8 台全空闲")
    STATUS.write_text(json.dumps({
        "state": "waiting", "started": time.strftime("%F %T"),
        "hosts": HOSTS, "interval": o.interval, "max_wait": o.max_wait,
        "condition": "B控制器退出 + 8台16/16空闲 + 无其它控制器",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    while time.time() - t0 < o.max_wait:
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
                    pid = launch_c()
                    STATUS.write_text(json.dumps({
                        "state": "triggered", "triggered_at": time.strftime("%F %T"),
                        "gate_full_pass": True, "c_pid": pid,
                        "c_log": "/tmp/r3_c_ep128.log",
                        "note": "C(EP128 precision --eps 128 --epr-tiers main --api v1,v2)",
                    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    log("DONE: C 已自动接续")
                    return 0
                log("gate 未达 full_pass（可能又有任务占卡），继续等待")
            STATUS.write_text(json.dumps({
                "state": "waiting", "ts": time.strftime("%F %T"),
                "idle_npus": idle, "b_running": b, "other_controllers": others,
                "ready": ready,
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception as exc:
            log("轮询异常（继续等）: %s" % str(exc)[:200])
        time.sleep(o.interval)

    STATUS.write_text(json.dumps({
        "state": "timeout", "ended": time.strftime("%F %T"),
        "note": "超过 max_wait 仍未满足条件，C 未触发",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log("TIMEOUT: C 未触发")
    return 1


if __name__ == "__main__":
    sys.exit(main())
