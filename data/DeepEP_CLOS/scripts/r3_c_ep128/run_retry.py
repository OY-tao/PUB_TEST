"""R3 阶段 C: EP128 persistent precision batch, buffer reuse + auto retry.

One persistent 8-node/128-rank process group runs the R3 C plan
(currently 400 cases = 100 spec x 2 epr tiers x 2 API, 19 Buffer groups;
 the exact count is read from plan.json `case_count_total`, not hardcoded).
Same group reuses the initialized Buffer. If a group is
killed, times out, errors, or fails validation, the controller collects whatever
artifacts were written, marks the fully passed variants as completed, and
resumes ONLY the remaining variants in a fresh attempt (new output root +
new rdzv-id). Automatic retry, never abort.

Inherited from the EP64 controller: 8h ssh timeout, checkpoint settlement on
every path (including exceptions), startup recovery from on-disk results, and
NO artifact write-back to the shared filesystem (judge from container disks).

PREPARATION ONLY: default invocation is plan-only -- no ssh, no deploy, no NPU.
Pass --run to actually deploy and execute. See README.md before doing so.
"""
import argparse
import concurrent.futures
import json
import pathlib
import re
import shlex
import subprocess
import time
import threading

BASE = pathlib.Path(__file__).resolve().parent
TAG = BASE.name  # precision_ep128_r3_20261128_C1
EP = 128
NNODES = 8
LOCAL = 16                      # ranks per node
HOSTS = ["30.104.128.%d" % i for i in range(4, 12)]   # .4 ... .11
CONTAINERS = ['deepep_ep128_full_matrix_20261128_073617'] * NNODES
MOUNTS = ['/data2/deepep_ep64_notify_20261127_2012/deepep_ep128_full_matrix_20261128_073617'] * NNODES
REMOTE_DIR = '/work/' + TAG     # /work is the container-side mount of MOUNTS
SSH = ['ssh', '-T', '-p', '56000', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6']
MASTER_PORT = 29910             # bench-internal shmem/UDMA port (must match batch_worker.py)
RDZV_PORT = 29911               # torchrun static rendezvous port
RUN_TIMEOUT = 8 * 3600          # ssh client timeout; enough for the whole batch
MAX_ATTEMPTS = 40
DEPLOY_FILES = ('batch_worker.py', 'checks_v1.py', 'checks_v2.py', 'plan.json',
                 'common.py', 'stage_metrics.py', 'route_workloads.py',
                 'bandwidth_multinode.py', 'profiling.py')
# The EP128 container's /work/source/benchmarks/clos has NO precision bench.
# These two must be copied into REMOTE_DIR before execution (see README.md).
EXTRA_BENCH_FILES = ('bench_v1_precision.py', 'bench_v2_precision.py')
BENCH_PATH = REMOTE_DIR         # PYTHONPATH entry for bench_v*_precision + checks
# bench_v2_precision.py computes REPO = Path(__file__).resolve().parents[2].
# Copied into /work/<TAG>/ that resolves to "/" instead of the source root, so
# golden `common` and `profiling` must be supplied explicitly via PYTHONPATH.
GOLDEN_PATH = REMOTE_DIR  # common.py 同目录部署（旧版配套）
BENCH_EXTRA_PATH = REMOTE_DIR  # stage_metrics/route_workloads/profiling 同目录
# 128P loads /work/env as the deep_ep runtime (the same location the successful
# 8-node full_matrix job imported). After the PR#37 rebuild, the wheel is
# installed here. Per user directive (2026-11-28), DEEPEP_CLOS_QP_PLANS is
# intentionally NOT set: deep_ep uses its automatic QP policy.
DEEPEP_ENV_SITE = '/work/env/lib/python3.13/site-packages'
DEEPEP_ENV_LIBS = DEEPEP_ENV_SITE + '/deep_ep/.libs'
TORCH_LIBS = '/opt/conda/envs/hunyuan4_tenpu/lib/python3.13/site-packages/torch/lib'
TORCH_NPU_LIBS = '/opt/conda/envs/hunyuan4_tenpu/lib/python3.13/site-packages/torch_npu/lib'
RUNTIME_PATH = ''
ENV_PYTHON = '/work/env/bin/python'
STATE = {}
LOCK = threading.Lock()
STATUS_PATH = BASE / 'status.json'


def save():
    p = STATUS_PATH.with_suffix('.json.tmp')
    p.write_text(json.dumps(STATE, indent=2, ensure_ascii=False))
    p.replace(STATUS_PATH)


def remote(rank, cmd, **kwargs):
    return subprocess.run(SSH + ['root@' + HOSTS[rank], cmd], **kwargs)


def read_remote(rank, cmd, timeout=90):
    r = remote(rank, cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(HOSTS[rank] + ': ' + r.stderr[-400:] + r.stdout[-400:])
    return r.stdout


def cleanup(rank):
    py = ("import os,signal,pathlib\n"
          "for p in pathlib.Path('/proc').glob('[0-9]*'):\n"
          " try:\n"
          "  s=b' '.join(p.joinpath('cmdline').read_bytes().split(bytes([0])))\n"
          "  if b'" + TAG + "' in s and (b'batch_worker.py' in s or b'torch.distributed.run' in s):\n"
          "   os.kill(int(p.name),signal.SIGKILL)\n"
          " except (OSError,ValueError,ProcessLookupError):pass\n"
          "print('cleaned')\n")
    try:
        read_remote(rank, 'docker exec ' + CONTAINERS[rank] + ' python -c ' + shlex.quote(py), timeout=60)
    except Exception:
        pass


def cleanup_all():
    with concurrent.futures.ThreadPoolExecutor(max_workers=NNODES) as pool:
        list(pool.map(cleanup, range(NNODES)))


def check_idle(rank):
    text = read_remote(rank, 'npu-smi info', timeout=30)
    idle = {int(n) for n in re.findall(r'No running processes found in NPU\s+(\d+)', text)}
    health = {int(n): h for n, h in re.findall(
        r'^\|\s*(\d+)\s*\|\s*Ascend[^|]*\|\s*(OK|Warning|Alarm)\s*\|', text, re.M)}
    busy = [n for n in range(16) if n not in idle or health.get(n) not in ('OK', 'Warning')]
    if busy:
        raise RuntimeError('%s NPUs busy/unhealthy: %s' % (HOSTS[rank], busy))
    m = re.search(r'npu-smi\s+(\S+)', text)
    return m.group(1) if m else 'unknown'


def preflight():
    with concurrent.futures.ThreadPoolExecutor(max_workers=NNODES) as pool:
        versions = list(pool.map(check_idle, range(NNODES)))
    if len(set(versions)) != 1:
        raise RuntimeError('driver mismatch: ' + str(versions))
    return versions[0]


def deploy_initial():
    missing = [n for n in EXTRA_BENCH_FILES if not (BASE / n).exists()]
    if missing:
        raise RuntimeError('EP128 container has no precision bench; copy these into this dir first: %s (see README.md)' % missing)
    payload = DEPLOY_FILES + EXTRA_BENCH_FILES
    for rank in range(NNODES):
        read_remote(rank, 'mkdir -p ' + shlex.quote(MOUNTS[rank] + '/' + TAG), timeout=30)
        files = [str(BASE / n) for n in payload]
        subprocess.run(['scp', '-q', '-P', '56000', '-o', 'BatchMode=yes'] + files +
                       ['root@' + HOSTS[rank] + ':' + MOUNTS[rank] + '/' + TAG + '/'], check=True)


def write_plan_attempt(attempt, cases, plan_all):
    plan = dict(plan_all)
    plan['remaining'] = cases
    plan['buffer_groups'] = plan_all['buffer_groups']
    plan['expected_buffer_initializations_per_rank'] = len({tuple(c['buffer_config']['key']) for c in cases})
    name = 'plan_attempt_%03d.json' % attempt
    (BASE / name).write_text(json.dumps(plan, indent=2) + '\n')
    for rank in range(NNODES):
        subprocess.run(['scp', '-q', '-P', '56000', '-o', 'BatchMode=yes', str(BASE / name),
                        'root@' + HOSTS[rank] + ':' + MOUNTS[rank] + '/' + TAG + '/' + name], check=True)
    return name


def torchrun_cmd(rank, attempt, plan_name, output, smoke):
    run_id = '%s_a%03d%s' % (TAG, attempt, '_smoke' if smoke else '')
    # `env -u DEEPEP_CLOS_QP_PLANS` 显式证明纯 auto 起跑（PR #37 已删该变量读取）。
    # 不注入 SHMEM_UDMA_CONNECT_TIMEOUT_MS：PR #37 改为按 Buffer 配置、默认 10 min。
    command = ['env', '-u', 'DEEPEP_CLOS_QP_PLANS',
               ENV_PYTHON, '-m', 'torch.distributed.run', '--max-restarts=0',
               '--nnodes=' + str(NNODES), '--node-rank=' + str(rank), '--rdzv-backend=static',
               '--rdzv-id=' + run_id, '--master-addr=30.104.128.4',
               '--master-port=' + str(RDZV_PORT), '--rdzv-conf=timeout=900',
               '--nproc-per-node=16', REMOTE_DIR + '/batch_worker.py',
               '--plan=' + REMOTE_DIR + '/' + plan_name,
               '--output=' + output, '--node-rank=' + str(rank), '--run-id=' + run_id]
    py_paths = ':'.join(x for x in (DEEPEP_ENV_SITE, REMOTE_DIR, BENCH_PATH,
                                    BENCH_EXTRA_PATH, GOLDEN_PATH, RUNTIME_PATH) if x)
    shell = ('source /usr/local/Ascend/ascend-toolkit/set_env.sh && '
             'unset SHMEM_UID_SESSION_ID ASCEND_RT_VISIBLE_DEVICES && '
             'unset DEEPEP_CLOS_QP_PLANS && '
             'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && '
             'export LD_LIBRARY_PATH=' + DEEPEP_ENV_LIBS + ':' + TORCH_LIBS + ':' +
             TORCH_NPU_LIBS + ':$LD_LIBRARY_PATH && '
             'export PYTHONPATH=' + py_paths + ':$PYTHONPATH && '
             + shlex.join(command))
    return ('docker exec ' + CONTAINERS[rank] + ' bash -lc ' + shlex.quote(shell))


def collect(rank, output):
    """Read result/precision/exit json from a remote output dir; retry on ssh hiccup."""
    code = '''import pathlib,json
p=pathlib.Path(ROOT)
wanted={"result.json","precision.json","exit.json","process.json"}
print(json.dumps({str(f.relative_to(p)):f.read_text() for f in p.rglob("*.json")
 if f.name in wanted or f.name.startswith("batch_rank")}))
'''.replace('ROOT', repr(output))
    last = None
    for _ in range(3):
        try:
            r = remote(rank, 'docker exec ' + CONTAINERS[rank] + ' python -c ' + shlex.quote(code),
                       capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                return json.loads(r.stdout)
            last = r.stderr[-200:]
        except Exception as exc:
            last = str(exc)[:200]
        time.sleep(3)
    print('collect node%d incomplete: %s' % (rank, last), flush=True)
    return {}


def judge(cases, per_node_files):
    passed, failed = [], []
    for c in cases:
        v = c['variant']
        ok = True
        for g in range(EP):
            node = g // LOCAL
            prefix = v + '/rank' + str(g) + '/'
            files = per_node_files.get(node, {})
            res = json.loads(files.get(prefix + 'result.json', '{}'))
            prec = json.loads(files.get(prefix + 'precision.json', '{}'))
            ex = json.loads(files.get(prefix + 'exit.json', '{}'))
            checks = [res.get('precision_passed') is True, res.get('precision_only') is True,
                      res.get('operators') == {}, res.get('buffer_reused') is True,
                      prec.get('passed') is True, ex.get('code') == 0]
            if not all(checks):
                ok = False
                break
        (passed if ok else failed).append(v)
    return passed, failed


def collect_and_judge(out_root, cases):
    """Collect all 8 nodes' artifacts for an attempt and return passed variants.

    Called on EVERY path (success, non-zero rc, timeout, exception) so that
    checkpoint progress is never lost.
    """
    outputs = {r: REMOTE_DIR + '/' + out_root + '/node' + str(r) for r in range(NNODES)}
    per_node = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=NNODES) as pool:
        coll = {rank: pool.submit(collect, rank, outputs[rank]) for rank in range(NNODES)}
        for rank, fut in coll.items():
            try:
                per_node[rank] = fut.result()
            except Exception:
                per_node[rank] = {}
    # Artifacts stay on the container disks (and in per_node memory); do NOT
    # re-write ~17k tiny json files onto the shared NFS during recovery — that
    # is redundant for judging and makes startup take tens of minutes.
    passed, _ = judge(cases, per_node)
    return passed


def list_remote_attempt_dirs():
    """Existing full-attempt output roots still on the containers."""
    code = '''import pathlib,json
p=pathlib.Path(%r)
print(json.dumps(sorted(x.name for x in p.glob("results_attempt_*_full") if x.is_dir())))
''' % REMOTE_DIR
    r = remote(0, 'docker exec ' + CONTAINERS[0] + ' python -c ' + shlex.quote(code),
               capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return []
    return json.loads(r.stdout)


def recover_completed(all_cases):
    """Settle completed variants from all existing attempt dirs (startup recovery)."""
    by_variant = {c['variant']: c for c in all_cases}
    completed = {}
    dirs = list_remote_attempt_dirs()
    print('RECOVER scanning %d existing attempt dirs' % len(dirs), flush=True)
    for d in dirs:
        # candidate variants = union of variant dirs across nodes
        code = '''import pathlib,json
p=pathlib.Path(%r)
print(json.dumps(sorted(x.name for x in p.iterdir() if x.is_dir())))
''' % (REMOTE_DIR + '/' + d + '/node0')
        try:
            r = remote(0, 'docker exec ' + CONTAINERS[0] + ' python -c ' + shlex.quote(code),
                       capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                continue
            candidates = [v for v in json.loads(r.stdout) if v in by_variant]
        except Exception:
            continue
        if not candidates:
            continue
        cases = [by_variant[v] for v in candidates]
        try:
            passed = collect_and_judge(d, cases)
        except Exception as exc:
            print('RECOVER %s failed: %s' % (d, str(exc)[:150]), flush=True)
            continue
        for v in passed:
            completed[v] = d
        print('RECOVER %s -> %d passed variants' % (d, len(passed)), flush=True)
    return completed


def run_attempt(attempt, cases, plan_all, smoke=False):
    mode = 'smoke' if smoke else 'full'
    plan_name = write_plan_attempt(attempt, cases, plan_all)
    out_root = 'results_attempt_%03d_%s' % (attempt, mode)
    outputs = {r: REMOTE_DIR + '/' + out_root + '/node' + str(r) for r in range(NNODES)}
    for rank in range(NNODES):
        read_remote(rank, 'docker exec ' + CONTAINERS[rank] + ' rm -rf ' + shlex.quote(outputs[rank]), timeout=30)
    print('ATTEMPT %d %s cases=%d' % (attempt, mode, len(cases)), flush=True)
    rcs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=NNODES) as pool:
        futs = {}
        for rank in range(NNODES):
            log = BASE / ('run_attempt_%03d_node%d.log' % (attempt, rank))
            f = open(str(log), 'w')
            futs[rank] = (pool.submit(remote, rank,
                                      torchrun_cmd(rank, attempt, plan_name, outputs[rank], smoke),
                                      stdout=f, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT), f)
        for rank, (fut, f) in futs.items():
            try:
                rcs[rank] = fut.result().returncode
            except Exception as exc:
                rcs[rank] = 'EXC:%s' % str(exc)[:80]
            finally:
                f.close()
    # ALWAYS settle checkpoint, regardless of rc/exception
    passed = collect_and_judge(out_root, cases)
    return {'attempt': attempt, 'mode': mode, 'rcs': rcs, 'passed': passed,
            'out_root': out_root}


def next_attempt_no():
    """Avoid overwriting existing attempt dirs after a controller restart."""
    mx = 0
    for d in BASE.glob('results_attempt_*'):
        m = re.search(r'results_attempt_(\d+)_', d.name)
        if m:
            mx = max(mx, int(m.group(1)))
    for d in list_remote_attempt_dirs():
        m = re.search(r'results_attempt_(\d+)_', d)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def main():
    ap = argparse.ArgumentParser(description='EP128 eight-node precision batch controller')
    ap.add_argument('--run', action='store_true',
                    help='actually deploy + execute; WITHOUT this flag the controller is plan-only')
    ap.add_argument('--smoke', action='store_true',
                    help='run the 8-case smoke plan (with --run); full 400-case plan is the default')
    o = ap.parse_args()
    global STATUS_PATH
    STATUS_PATH = BASE / ('status_smoke.json' if o.smoke else 'status.json')
    plan_name = 'plan_smoke.json' if o.smoke else 'plan.json'
    plan_all = json.loads((BASE / plan_name).read_text())
    all_cases = plan_all['remaining']
    total = len(all_cases)
    expected = 8 if o.smoke else int(plan_all.get('case_count_total', total))
    assert total == expected, "plan cases %d != expected %d" % (total, expected)
    if not o.run:
        smoke = json.loads((BASE / 'plan_smoke.json').read_text())
        print(json.dumps({
            'PLAN_ONLY': True, 'mode': 'smoke' if o.smoke else 'full',
            'ep_size': EP, 'nnodes': NNODES, 'local_ranks_per_node': LOCAL,
            'hosts': HOSTS, 'container': CONTAINERS[0],
            'remote_dir': REMOTE_DIR, 'mount': MOUNTS[0],
            'total_cases': len(all_cases),
            'epr_tiers': plan_all.get('epr_tiers'),
            'buffer_groups': len(plan_all['buffer_groups']),
            'smoke_cases': len(smoke['remaining']),
            'buffer_reuse': plan_all.get('buffer_reuse'),
            'max_workspace_GiB': round(max(g['workspace_bytes'] for g in plan_all['buffer_groups']) / 1024 ** 3, 3),
            'deploy_files': list(DEPLOY_FILES + EXTRA_BENCH_FILES),
            'extra_bench_files_present': {n: (BASE / n).exists() for n in EXTRA_BENCH_FILES},
            'master_port': MASTER_PORT, 'rdzv_port': RDZV_PORT,
            'note': 'no ssh / deploy / NPU work started; pass --run to execute',
        }, ensure_ascii=False, indent=2))
        return 0
    STATE.update({'state': 'preflight', 'started': time.strftime('%F %T'),
                  'total_cases': total, 'hosts': HOSTS, 'master_port': MASTER_PORT,
                  'rdzv_port': RDZV_PORT, 'max_attempts': MAX_ATTEMPTS,
                  'completed': {}, 'attempts': [], 'smoke_ok': False,
                  'mode': 'retry-until-complete-v2', 'run_timeout': RUN_TIMEOUT})
    save()
    deploy_initial()
    # startup recovery from on-disk checkpoints
    completed = recover_completed(all_cases)
    STATE['completed'] = dict(completed)
    STATE['state'] = 'running'
    save()
    print('RECOVERED %d / %d completed variants' % (len(completed), len(all_cases)), flush=True)

    by_variant = {c['variant']: c for c in all_cases}
    remaining = [c for c in all_cases if c['variant'] not in completed]
    base_attempt = next_attempt_no()
    attempt_no = 0
    while remaining and attempt_no < MAX_ATTEMPTS:
        attempt_no += 1
        attempt = base_attempt + attempt_no - 1
        try:
            cleanup_all()
            preflight()
            res = run_attempt(attempt, remaining, plan_all, smoke=o.smoke)
            for v in res['passed']:
                completed[v] = res['out_root']
            remaining = [c for c in remaining if c['variant'] not in res['passed']]
            STATE['completed'] = dict(completed)
            STATE['attempts'].append({k: res[k] for k in ('attempt', 'mode', 'rcs', 'out_root')} |
                                     {'passed': len(res['passed']), 'remaining': len(remaining)})
            save()
            print('FULL attempt %d passed=%d remaining=%d rcs=%s' %
                  (attempt, len(res['passed']), len(remaining), res['rcs']), flush=True)
        except Exception as exc:
            STATE['attempts'].append({'attempt': attempt, 'mode': 'full', 'error': str(exc)[:250]})
            save()
            print('FULL attempt %d error: %s' % (attempt, str(exc)[:180]), flush=True)
        if remaining:
            time.sleep(8)
    STATE['state'] = 'completed' if not remaining else 'failed'
    STATE['ended'] = time.strftime('%F %T')
    STATE['remaining_cases'] = [c['variant'] for c in remaining]
    save()
    print('EP128_FINISHED state=%s completed=%d remaining=%d' %
          (STATE['state'], len(completed), len(remaining)), flush=True)
    return 0 if not remaining else 1


if __name__ == '__main__':
    raise SystemExit(main())
