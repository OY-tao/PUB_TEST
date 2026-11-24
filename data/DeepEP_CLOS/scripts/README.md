# DeepEP CLOS 配套脚本

配套文档：[`../泛化用例测试设计V2.md`](../泛化用例测试设计V2.md)

被测对象是 DeepEP 的 CLOS 组网带宽基准
（`benchmarks/clos/bench_v1_bandwidth.py`、`bench_v2_bandwidth.py`）。
本目录是分析过程中自研的 3 个工具 + 4 个上游脚本快照。

## 目录

```
scripts/
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

## 三个工具的分工

| 工具 | 何时用 | 输入 | 输出 | 覆盖被测项 |
|---|---|---|---|---|
| `verify_ledger.py` | 跑测**前** | 形状参数 | 6 种字节口径 | —（只算分母） |
| `extract_v1_stages.py` | 跑测**后** | bench 输出目录 | 分阶段耗时 | V1-2 / V1-4 / V1-5 |
| `lifecycle_test.py` | 独立 | torchrun | 构造/销毁耗时 + 失败路径 + 泄漏 | V1-1 / V1-7 / V2-1 / V2-4 |

三者都不修改上游代码。

## 用法

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
