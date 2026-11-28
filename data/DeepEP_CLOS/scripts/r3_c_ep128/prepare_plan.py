#!/usr/bin/env python3
"""R3 阶段 C：EP128 泛化精度 plan 生成（CPU only，不碰 NPU / 不 ssh）。

用例来源：`gen_precision_cases.py --eps 128` 的 50 条规格（指南 §五·C3），
按 §五·C2 的 epr 双档投影：
  - 主档 main  : epr=4  (E=512，与性能/基线同口径)
  - 探针 probe : epr=32 (E=4096，大 E 泛化，沿用 R2 spec)
两档分开报，不混在一张通过率里。EP=128、nnodes=8、16 rank/机。

Buffer 复用（用户既定口径）：单进程组；V1 一个 Buffer；V2 按
(hidden, num_topk, dtype, huge_pages) 分组，组内取 max_tokens 最大值作为
统一容量。workspace 用 EP32/EP64 校准的 CPU 公式以 num_ranks=128 重算，
**必须 ≤ ACLSHMEM_MAX_LOCAL_SIZE = 40 GiB**（R2 400 例 plan 的 48 GiB
V1 单组正是 status=-2 的根因）。本批最重 shape 约 32 GiB。

Run: python3 prepare_plan.py   # 写 plan.json / plan_smoke.json
"""
import copy
import hashlib
import importlib.util
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
SCRIPTS = BASE.parent          # = data/DeepEP_CLOS/scripts (2-16P 规格与生成器所在)
SPEC = SCRIPTS / "precision_cases_2to16p.json"

EP = 128
NNODES = 8
LOCAL = 16
MAX_LOCAL_MEM = 40 * 1024 ** 3     # ACLSHMEM_MAX_LOCAL_SIZE（shmem 硬上限）
GUARD = 38 * 1024 ** 3             # 留 2 GiB 余量，超过即拒绝生成

# 与 EP32/EP64 校准公式一致（prepare_plan.py of precision_ep128_gen_*）
IB, FB, MS, TI, BF, ME = 8, 4, 2, 4, 2, 8
CA, NW, WA, SA, UD = 128, 128, 128, 128, 8
MAXAIV = 64


def align(x, a):
    return ((x + a - 1) // a) * a


def official_capacity():
    """读取 wheel 官方 size-hint API 实算的权威容量表（capacity_official.json）。

    该表由容器内 `Config.get_nvl_buffer_size_hint()` / `ElasticBuffer.get_buffer_size_hint()`
    对全部 buffer 组逐一实算生成（见 R3_C阶段精度方案_20261128.md §8）。
    旧的自写近似公式 v1_hint/v2_hint 已废弃：v1_hint 漏掉
    `recv_tokens * hidden_bytes` 主导项，17 组容量全部低估，导致
    `dispatch requires ... bytes of symmetric workspace` 首例即挂。
    """
    path = BASE / "capacity_official.json"
    if not path.exists():
        raise RuntimeError("缺少 %s：必须先用 wheel 官方 size-hint API 生成容量表" % path.name)
    table = json.loads(path.read_text(encoding="utf-8"))
    if table.get("num_ranks") != EP:
        raise RuntimeError("容量表 num_ranks=%r 与 plan EP=%d 不符" % (table.get("num_ranks"), EP))
    return table["groups"]


def size_of_group(capacity, key, max_tokens, members):
    """按 buffer 组 key 查官方容量；查不到直接报错，绝不回退到近似公式。"""
    lookup = json.dumps(list(key))
    if lookup not in capacity:
        raise RuntimeError("容量表缺少组 %s（请重新生成 capacity_official.json）" % lookup)
    entry = capacity[lookup]
    size = int(entry["workspace_bytes"])
    if size < 2 * 1024 ** 2:
        raise RuntimeError("组 %s 官方容量 %d B 异常偏小" % (lookup, size))
    # 表是按当前 spec 实算的：组内 max_tokens 与 case 数必须一致，防止 spec 改了表没更新
    if int(entry["max_tokens"]) != max_tokens or int(entry["case_count"]) != len(members):
        raise RuntimeError("组 %s 与容量表不一致（表: max_tokens=%s cases=%s；现: %d/%d），"
                           "请重新生成 capacity_official.json" %
                           (lookup, entry.get("max_tokens"), entry.get("case_count"),
                            max_tokens, len(members)))
    return size


def load_spec():
    spec = importlib.util.spec_from_file_location("gpc", str(SCRIPTS / "gen_precision_cases.py"))
    gpc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpc)
    return gpc, json.loads(SPEC.read_text(encoding="utf-8"))


def build_cases(epr_tiers=("main",)):
    gpc, spec = load_spec()
    out = []
    for tier in epr_tiers:
        cap = {"main": 4, "probe": 32}[tier]
        for c in gpc.build(spec, EP):
            p = dict(c.params)                       # kebab-case keys
            epr = min(int(p.get("experts-per-rank", 32)), cap)
            k = min(int(p.get("num-topk", 8)), gpc.topk_cap(EP, epr))
            params = {
                "num_tokens": int(p.get("num-tokens", 1)),
                "hidden": int(p.get("hidden", 512)),
                "num_topk": k,
                "experts_per_rank": epr,
                "aiv": int(p.get("aiv", 32)),
                "warmup": 10, "iterations": 30, "seed": 20260823,
                "route_mode": str(p.get("route-mode", "uniform")),
                "route_group_size": EP // 2,
                "huge_pages": False,
                "rendezvous_timeout": 900,
            }
            for key in ("route-skew", "route-local-fraction", "route-mask-rate"):
                if key in p:
                    params[key.replace("-", "_")] = p[key]
            for api in ("v1", "v2"):
                out.append({
                    "api": api,
                    "dtype": str(p.get("dtypes", "bf16")),
                    "expand": int(p.get("expand", 0)),
                    "variant": "%s_ep%d_%s_%s" % (tier, EP, api, c.tag),
                    "source_id": c.tag,
                    "group": c.group,
                    "epr_tier": tier,
                    "parameters": params,
                })
    return out


def allocate(cases):
    capacity = official_capacity()
    buckets = {}
    for c in cases:
        p = c["parameters"]
        key = (("v1", p["huge_pages"]) if c["api"] == "v1"
               else ("v2", p["hidden"], p["num_topk"], c["dtype"], p["huge_pages"]))
        buckets.setdefault(key, []).append(c)
    groups = []
    for key, members in sorted(buckets.items(), key=lambda kv: repr(kv[0])):
        max_tokens = max(m["parameters"]["num_tokens"] for m in members)
        size = size_of_group(capacity, key, max_tokens, members)
        if size > GUARD:
            raise RuntimeError("group %s workspace %.2f GiB > guard %.2f GiB" %
                               (key, size / 1024 ** 3, GUARD / 1024 ** 3))
        cfg = {"key": list(key), "max_tokens": max_tokens, "workspace_bytes": size}
        for m in members:
            m["buffer_config"] = cfg
        groups.append(dict(cfg, case_count=len(members)))
    return cases, groups


def main():
    cases = build_cases(("main", "probe"))
    cases, groups = allocate(cases)
    plan = {
        "ep_size": EP, "nnodes": NNODES, "local_ranks_per_node": LOCAL,
        "hosts": ["30.104.128.%d" % i for i in range(4, 12)],
        "case_count_total": len(cases),
        "epr_tiers": ["main", "probe"],
        "precision_only": True,
        "buffer_reuse": ("one persistent 8-node/128-rank process group; V1 one Buffer; "
                         "V2 grouped by (hidden,num_topk,dtype,huge_pages); "
                         "same group reuses the initialized Buffer, no per-case rebuild"),
        "max_local_mem_bytes": MAX_LOCAL_MEM,
        "guard_bytes": GUARD,
        "workspace_source": "PR#37 wheel size-hint API (Config.get_nvl_buffer_size_hint / "
                            "ElasticBuffer.get_buffer_size_hint), num_ranks=128; "
                            "see capacity_official.json",
        "reference": "PUB_TEST/data/DeepEP_CLOS/第三轮测试执行指南_20261128.md §五",
        "remaining": cases,
        "buffer_groups": groups,
        "expected_buffer_initializations_per_rank": len(groups),
        "expected_process_group_initializations_per_rank": 1,
        "source_spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
    }
    (BASE / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    # smoke：每档每 API 取 2 例，共 8 例（同 64P smoke 规模）
    smoke_cases = []
    for tier in ("main", "probe"):
        for api in ("v1", "v2"):
            picked = [c for c in cases if c["epr_tier"] == tier and c["api"] == api][:2]
            smoke_cases.extend(copy.deepcopy(picked))
    smoke = dict(plan, remaining=smoke_cases,
                 buffer_groups=[g for g in groups if any(
                     m["buffer_config"]["key"] == g["key"] for m in smoke_cases)],
                 expected_buffer_initializations_per_rank=len({tuple(m["buffer_config"]["key"])
                                                               for m in smoke_cases}),
                 case_count_total=len(smoke_cases), note="R3 C smoke: 2 cases per tier per API")
    (BASE / "plan_smoke.json").write_text(json.dumps(smoke, indent=2, ensure_ascii=False) + "\n")
    print("plan.json      : %d cases, %d buffer groups" % (len(cases), len(groups)))
    print("plan_smoke.json: %d cases" % len(smoke_cases))
    print("max workspace  : %.2f GiB (hard limit %.2f GiB)" %
          (max(g["workspace_bytes"] for g in groups) / 1024 ** 3, MAX_LOCAL_MEM / 1024 ** 3))
    for t in ("main", "probe"):
        print("  tier %-5s: %d cases" % (t, sum(1 for c in cases if c["epr_tier"] == t)))


if __name__ == "__main__":
    main()
