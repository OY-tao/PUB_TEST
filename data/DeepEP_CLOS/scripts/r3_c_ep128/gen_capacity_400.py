"""容器内用 wheel 官方 size-hint API 生成 400 例 C 阶段的权威容量表。

读 --emit-groups 输出的分组清单，对每组用官方 API 实算：
  V1: Config.get_nvl_buffer_size_hint(hidden*2, num_ranks, num_tokens)，
      单组一份 Buffer 覆盖组内全部 case → 取组内所有成员的上界 max。
  V2: ElasticBuffer.get_buffer_size_hint(group, max_tokens, hidden, topk, fp8)，
      正是 elastic.py:809 native 校验所用的同一函数。
输出 capacity_official.json（与 prepare_plan.official_capacity() 读取格式一致）。
"""
import json
import time
from deep_ep.backends.direct.config import Config
from deep_ep.backends.direct.elastic import ElasticBuffer


class Group128:
    def rank(self):
        return 0

    def size(self):
        return 128


manifest = json.load(open('/tmp/c_groups_400.json'))
NRANKS = manifest['num_ranks']
assert NRANKS == 128
cfg = Config()
grp = Group128()
out = {}
for lookup, g in sorted(manifest['groups'].items()):
    key = g['key']
    max_tokens = g['max_tokens']
    if key[0] == 'v1':
        size = max(cfg.get_nvl_buffer_size_hint(
            m['hidden'] * 2, NRANKS, m['num_tokens']) for m in g['members'])
        api = 'Config.get_nvl_buffer_size_hint(hidden*2, 128, num_tokens), max over group'
    else:
        p = g['members'][0]
        size = ElasticBuffer.get_buffer_size_hint(
            grp, max_tokens, p['hidden'], p['num_topk'], p['dtype'] == 'fp8')
        api = 'ElasticBuffer.get_buffer_size_hint(group128, max_tokens, hidden, topk, fp8)'
    out[lookup] = {
        'workspace_bytes': size,
        'max_tokens': max_tokens,
        'case_count': g['case_count'],
        'api': api,
    }

table = {
    'generated': time.strftime('%F %T'),
    'source': 'PR#37 wheel 0d1eca2 官方 size-hint API（容器内按 400 例分组清单实算）',
    'num_ranks': NRANKS,
    'spec_count': 100,
    'case_count_total': sum(v['case_count'] for v in out.values()),
    'shmem_hard_limit_bytes': 40 * 1024 ** 3,
    'guard_bytes': 38 * 1024 ** 3,
    'groups': out,
}
print(json.dumps(table, indent=2))
