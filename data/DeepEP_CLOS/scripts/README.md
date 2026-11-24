# DeepEP CLOS 配套脚本

配套文档：[`../泛化用例测试设计V2.md`](../泛化用例测试设计V2.md)

被测对象是 DeepEP 的 CLOS 组网带宽基准
（`benchmarks/clos/bench_v1_bandwidth.py`、`bench_v2_bandwidth.py`）。
本目录是分析过程中自研的 4 个工具 + 4 个上游脚本快照。

## 目录

```
scripts/
├── gen_cases.py              ★ 按规格生成 bench 命令行批次
├── cases_1148.json           ★ 1148 例规格（单一事实来源）
├── verify_ledger.py          跑测前算 6 种字节分母（纯 CPU，不需 NPU）
├── extract_v1_stages.py      跑测后取 layout/expand/aggregate 三阶段
├── lifecycle_test.py         构造/销毁功能验证（需 torchrun + NPU）
└── upstream/                 上游脚本快照（BSD 2-Clause，见其 LICENSE）
    ├── LICENSE
    ├── bench_v1_bandwidth.py
    ├── bench_v2_bandwidth.py
    ├── route_workloads.py
    └── stage_metrics.py
```

## 四个工具的分工

| 工具 | 何时用 | 输入 | 输出 | 覆盖被测项 |
|---|---|---|---|---|
| `gen_cases.py` | 跑测**前** | `cases_1148.json` | 1148 条 bench 命令 + shell 批次 | 全部 1148 例 |
| `verify_ledger.py` | 跑测**前** | 形状参数 | 6 种字节口径 | —（只算分母） |
| `extract_v1_stages.py` | 跑测**后** | bench 输出目录 | 分阶段耗时 | V1-2 / V1-4 / V1-5 |
| `lifecycle_test.py` | 独立 | torchrun | 构造/销毁耗时 + 失败路径 + 泄漏 | V1-1 / V1-7 / V2-1 / V2-4 |

四个工具都不修改上游代码。

## 用法

### gen_cases.py —— 按 md 生成 1148 例

**这是把文档里的 1148 例变成可执行命令的入口。** 规格全在 `cases_1148.json`，
改范围只改 JSON，不用动代码。

```bash
# 只校验配比（不写文件、不碰设备）
python gen_cases.py --check

# 生成 v1/v2 两套批次
python gen_cases.py --out batches/

# 只生成某几组
python gen_cases.py --out batches/ --groups A B C

# 只看命令，不落盘
python gen_cases.py --dry-run --limit 5
```

产物：

```
batches/
├── v2_A.sh ... v2_H.sh     每组一个脚本，可单独跑
├── v1_A.sh ... v1_H.sh
├── manifest_v2.csv         逐条明细（组/编号/层级/标签/参数/命令）
├── summary.md              配比核对表
└── index.sh                串行执行入口
```

**支持断点续跑**（每个 case 独立 torchrun，中断后不用从头）：

```bash
bash batches/v2_A.sh          # 跑全部 252 条
bash batches/v2_A.sh 128      # 从第 128 条开始
```

`--check` 会逐组核对例数与规格里的 `_expected`，对不上直接报错退出：

```
组    名称                 期望       实际     结果
A    形状                252      252     OK
...
合计                    1148     1148     OK
```

参数名的合法性在生成时就校验：`ep-size`/`nnodes`/`aiv`/`dtypes`/`expand`/
`route-mode` 都在 bench 的 `choices` 范围内，且 `hidden` 必须是 128 的倍数。
生成器只放行 `ALLOWED_KEYS` 白名单里的参数名，防止规格文件里的内部字段
（`_id`/`desc`/`count`）被误当成命令行参数——argparse 的前缀匹配会让这种
错误静默通过，是个坑。

### verify_ledger.py

纯 CPU 计算，不需要 NPU、不需要 torch。依赖同目录 `upstream/` 下的
`route_workloads.py` 与 `stage_metrics.py`，**目录结构不能打散**。

```bash
# 默认形状的完整账本（4 dtype × 4 op）
python verify_ledger.py

# 三个关键扫描
python verify_ledger.py --all
python verify_ledger.py --sweep-topk    # topk 1..32，看 algorithm 与 payload 分离
python verify_ledger.py --sweep-mask    # mask 0..0.9，看 algorithm 恒定
python verify_ledger.py --sweep-skew    # skew 1..32，看 slot 饱和

# 换形状估算
python verify_ledger.py --tokens 16384 --hidden 8192 --topk 4
```

### extract_v1_stages.py

只读 bench 的 `result.json`，不碰 NPU。自动识别两种目录形状：

- `<root>/rankN/result.json`（单 case）
- `<root>/<case_name>/rankN/result.json`（bench 的 `--output` 根）

```bash
python extract_v1_stages.py ./run_v1_ep8
python extract_v1_stages.py ./run_v1_ep8 --csv stages.csv --json stages.json
python extract_v1_stages.py ./run_v1_ep8 --op dispatch
```

输出含「V1 通信算子覆盖检查」，应为 5/5。`expand=0` 的 case 会自动跳过
全 0 的 `expand_us` / `aggregate_us`。

### lifecycle_test.py

**必须用 torchrun 启动**，因为构造 Buffer 需要 EP 进程组。

```bash
# 正常路径：构造/销毁耗时、幂等、泄漏
torchrun --nproc-per-node 8 lifecycle_test.py --api both --ep-size 8 --hidden 6144

# 只跑失败路径（L7-C）
torchrun --nproc-per-node 8 lifecycle_test.py --api v1 --ep-size 8 --fail-paths

# 显式指定卡
docker exec -e ASCEND_RT_VISIBLE_DEVICES=6,7 ...
```

关键判据：

- V2 未设 `explicitly_destroy=True` 时 `destroy()` 必须抛 `RuntimeError`
- V1 有活跃 slot 时 `destroy()` 必须抛 `RuntimeError`
- 两者重复调用 `destroy()` 必须幂等
- 20 次构造→销毁后 HBM 应回到基线（±5%）

已知副作用：测「活跃 slot 拒绝销毁」会 `acquire_slot()` 后无法释放
（`Buffer` 没有 `release_slot` 接口），该实例只能等进程退出回收。
建议这一项单独跑一轮。

## 脚本与文档的对应关系

**`gen_cases.py` + `cases_1148.json` 才是文档 1148 例的可执行实现**，
另外三个是分析工具，覆盖的不是这 1148 例，而是 bench 补不了的部分。

| 工具 | 对应文档的哪部分 | 是否一一对应 |
|---|---|---|
| `gen_cases.py` | 1148 例矩阵（A~H 八组） | ✅ 完全对应，`--check` 可验证 |
| `verify_ledger.py` | 第二/三章的字节口径与分母陷阱 | ❌ 扫描范围更宽，用于画趋势曲线 |
| `extract_v1_stages.py` | V1-2/V1-4/V1-5 的覆盖缺口 | 部分（只补这三个） |
| `lifecycle_test.py` | V1-1/V1-7/V2-1/V2-4 的覆盖缺口 | 部分（L7 的 32 例） |

`verify_ledger.py` 的扫描档位**故意宽于文档**，因为它的用途是证明结论而不是当用例清单：

| 参数 | 文档规格 | verify_ledger 实际 | 原因 |
|---|---|---|---|
| `skew` | 1-8（6 档） | 1, 1.5, 2, 4, 6, 8, **16, 32** | 扫到 32 才能证明「skew>8 已饱和」（收行封顶 65536 就是这么测出来的） |
| `topk` | 1-64（7 档） | 1, 2, 4, 8, 16, 32（**缺 64**） | 论证「algorithm 随 topk 线性涨」用不到 64 |
| `mask` | 0.1-0.9（4 档） | 0.0, 0.1, 0.3, 0.5, 0.7, **0.9** | 多 0.0 做基线、0.7 加密采样 |

**如果直接把 `verify_ledger.py` 的扫描结果当用例清单，会多跑 skew=16/32 两档无用 case。**
1148 例的权威来源是 `cases_1148.json`。

### cases_1148.json 的规格要点

规格里记录的参数名全部对照上游核实过（见文件内 `_verified` 字段）：

- bench **没有** `--op` 参数，每个 case 固定跑 4 个 op（dispatch / cached-dispatch / combine ×2）
- bench **没有** `--tokens`，是 `--num-tokens`
- `--route-mode` 只有 3 种（`uniform`/`rank_hotspot`/`local_group`）；
  **mask 不是独立模式**，是 `--route-mask-rate` 叠加参数
- `--ep-size` choices=(2,4,8,16,32,64,128,256)
- `--nnodes` choices=(1,2,4,8,16,32)
- `--aiv` choices=range(2,65,2)

多机 case（`nnodes>1`，D 组共 12 条）生成时只标注，不自动展开——各节点需
分别执行并注入 `--master-addr` / `--node-rank`。

## ⚠️ 跑测前核对：确认走 `global_clos`

**这是开工前第一件事。** bench 用的是
`from deep_ep.backends.global_clos import Buffer / ElasticBuffer`，
而 `global_clos` 是 `direct` 的别名入口（目录下只有 274 B 的 `__init__.py`）。

**为什么必须核对**：同仓库还有 `mesh_clos`，那是**另一套独立实现**
（`elastic.py` 78 KB vs `direct` 的 64 KB）。一旦走错，
性能与精度数字**都不能和 `direct` 的结果对比**，且从输出上看不出来。

```bash
# 一行判据
python -c "import sys; from deep_ep.backends.global_clos import Buffer, ElasticBuffer; \
bad=[c.__name__ for c in (Buffer,ElasticBuffer) if 'mesh_clos' in c.__module__]; \
print('FAIL: 走了 mesh_clos', bad) if bad else print('OK: global_clos'); sys.exit(1 if bad else 0)"
```

也可直接 grep 源码：

```bash
grep -n "from deep_ep.backends" benchmarks/clos/bench_v1_bandwidth.py
grep -n "from deep_ep.backends" benchmarks/clos/bench_v2_bandwidth.py
```

`Buffer` / `ElasticBuffer` 必须来自 `global_clos`；`Config` 来自 `direct`、
`loaded_identity()` 里的 `direct as clos` 都属正常，别误判。

完整调用链与三个易混淆目录的对照见文档
[`../泛化用例测试设计V2.md`](../泛化用例测试设计V2.md) 的「零」章。

## 环境前置

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
unset SHMEM_UID_SESSION_ID           # 否则 bench 报错
unset ASCEND_RT_VISIBLE_DEVICES      # 改用 --first-device 指定物理卡
```

bench 自身的检查：`npu-smi` 的 Health 必须是 `OK`/`Warning`，
且必须出现 `No running processes found in NPU <id>`。

## upstream/ 的来源与许可

4 个文件是 [lueelu/DeepEP](https://github.com/lueelu/DeepEP) 的原样副本
（未做任何修改），取自 `main` 分支，拉取日期 2026-11-24：

| 文件 | 上游路径 |
|---|---|
| `bench_v1_bandwidth.py` | `benchmarks/clos/bench_v1_bandwidth.py` |
| `bench_v2_bandwidth.py` | `benchmarks/clos/bench_v2_bandwidth.py` |
| `route_workloads.py` | `benchmarks/clos/route_workloads.py` |
| `stage_metrics.py` | `benchmarks/clos/stage_metrics.py` |

**为什么保留快照**：文档里引用的行号（如 `stage_metrics.py:191-241`）
对应这一版快照。上游若更新，行号会漂移；用 `diff` 可看出改了什么、
原有结论是否还成立。

上游以 **BSD 2-Clause License** 发布，Copyright (c) 2026, Lu Lu，
全文见 [`upstream/LICENSE`](upstream/LICENSE)。按该许可证要求，
再分发源码时保留版权声明、许可证与免责声明。

自研的 3 个脚本（`verify_ledger.py`、`extract_v1_stages.py`、
`lifecycle_test.py`）为分析工具，不包含上游代码。
