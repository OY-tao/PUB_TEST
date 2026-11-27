#!/usr/bin/env python3
"""核对 DeepEP main 到底"新在哪"——不碰设备，纯 GitHub API。

为什么需要这个脚本
------------------
第二轮需求是"更新 main 分支代码后重测"，但 main 会持续前进（用户 2026-11-27 16:26
明确"还不是最新"）。每次换代码，"哪些路径真的变了"都要重新算一遍，靠记忆或聊天记录
不可靠，而且结论直接决定性能预期：

* dispatch / combine kernel 未变 → 预期 delta ≈ 0（变化只会来自环境）
* dispatch / combine kernel 变了 → 预期有变化，报告里要逐条对标 kernel 改动
* benchmarks/clos 变了         → 口径也变了，与上轮数字不可直接比

判定口径
--------
对每条路径问一句：「这条路径最后一次被修改的 commit，是否落在 `BASE...main` 区间内」。
落在区间内 = 本轮变更动的；否则 = 与基线一致。

用法
----
    python check_main_delta.py                 # 打表（默认基线 83252a5，即上轮实测代码）
    python check_main_delta.py --base <sha>    # 换基线
    python check_main_delta.py --json          # 机器可读，便于写进 round2_gate.json

匿名 API 限速 60 次/小时；本脚本固定约 11 个请求，连跑多次会被限流。
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

API = "https://api.github.com/repos/lueelu/DeepEP"
BASE = "83252a54d2363734a706a7dd8368bebc3094af5a"   # 上轮 48 例性能 + 200 例精度实测代码
UA = {"User-Agent": "deepep-round2-check", "Accept": "application/vnd.github+json"}

# 顺序有意义：前三条是"这轮到底测什么"的核心变量，后面是口径与接口
PATHS = [
    "csrc/kernels/dispatch",
    "csrc/kernels/combine",
    "benchmarks/clos",
    "csrc/kernels/common.hpp",
    "csrc/ops",
    "deep_ep/backends",
    "cmake",
    "csrc/kernels",
]
CRITICAL = {"csrc/kernels/dispatch", "csrc/kernels/combine"}
CALIBER = {"benchmarks/clos"}


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode("utf-8"))


def commit_of(obj):
    c = obj["commit"]
    return {
        "sha": obj["sha"],
        "short": obj["sha"][:7],
        "date": c["author"]["date"],
        "msg": (c["message"] or "").splitlines()[0][:72],
    }


def last_commit(path):
    """某路径最后一次被修改的 commit；路径不存在或空目录返回 None。"""
    try:
        arr = get("%s/commits?path=%s&sha=main&per_page=1" % (API, path))
    except urllib.error.HTTPError as exc:
        return {"error": "HTTP %s" % exc.code}
    if not arr:
        return None
    return commit_of(arr[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE, help="基线 sha（默认上轮实测代码）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tip = commit_of(get(API + "/commits/main"))
    try:
        base = commit_of(get("%s/commits/%s" % (API, args.base)))
    except urllib.error.HTTPError as exc:
        sys.exit("基线 %s 取不到: HTTP %s" % (args.base[:7], exc.code))

    # 区间内的 commit 集合。compare 默认最多 250 条，够用；超了按日期兜底。
    cmp_ = get("%s/compare/%s...%s" % (API, base["sha"], tip["sha"]))
    ahead = cmp_.get("ahead_by", 0)
    in_range = {c["sha"] for c in cmp_.get("commits", [])}
    truncated = ahead > len(in_range)

    rows = []
    for p in PATHS:
        last = last_commit(p)
        if last is None:
            rows.append({"path": p, "state": "路径不存在", "short": "-", "date": "-", "msg": "-"})
            continue
        if "error" in last:
            rows.append({"path": p, "state": last["error"], "short": "-", "date": "-", "msg": "-"})
            continue
        if last["sha"] in in_range:
            state = "改动了"
        elif last["sha"] == base["sha"]:
            state = "未变"
        elif last["date"] <= base["date"] and not truncated:
            state = "未变（早于基线）"
        else:
            state = "需人工确认"
        rows.append({"path": p, "state": state, "short": last["short"],
                     "date": last["date"][:10], "msg": last["msg"]})

    changed = [r["path"] for r in rows if r["state"] == "改动了"]
    crit = [p for p in changed if p in CRITICAL]
    cal = [p for p in changed if p in CALIBER]

    if args.json:
        print(json.dumps({"repo": "lueelu/DeepEP", "branch": "main",
                          "tip": tip, "base": base, "ahead_by": ahead,
                          "changed": changed, "critical_changed": crit,
                          "caliber_changed": cal, "paths": rows},
                         indent=2, ensure_ascii=False))
        return 0

    print("仓     : lueelu/DeepEP  (main)")
    print("main   : %s  %s  %s" % (tip["short"], tip["date"][:19], tip["msg"]))
    print("基线   : %s  %s  (上轮实测)" % (base["short"], base["date"][:19]))
    print("区间   : ahead_by=%d%s" % (ahead, "（compare 被截断，按日期兜底）" if truncated else ""))
    print()
    print("| 路径 | 最后修改 | 日期 | 状态 | 说明 |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| `%s` | %s | %s | %s | %s |" % (r["path"], r["short"], r["date"], r["state"], r["msg"]))
    print()
    if crit:
        print("⚠ 关键 kernel 有改动：%s" % ", ".join(crit))
        print("  → 本轮不能套用\"delta≈0\"预期，报告里需逐条对标这些改动。")
    if cal:
        print("⚠ benchmark 有改动：%s" % ", ".join(cal))
        print("  → 口径可能一起变了，与上轮数字不可直接比；先 diff bench 再决定用例。")
    if not crit and not cal:
        print("✓ dispatch/combine kernel 与 benchmark 均未变：")
        print("  预期性能 delta 落在 CV 内（≈0），若出现 >3% 系统性偏差应查环境而非归因新代码。")
    print()
    print("★ 代码仍是新的，重编译重装不能省（用户要求）。下一步：A2 编译安装 → gate。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
