# FlashAttnBwdSeqused 边界（Boundary）用例分析

> 对象：`cases/all_FlashAttnBwdSeqused_Boundary_950.json` / `_910.json`  
> 分析日期：2026-11-13

## 一、总览

| 指标 | 值 |
|---|---|
| 每个 SoC 用例数 | 55 |
| 950 / 910 两个文件 | 形状与属性完全一致，仅 `backend` 与 id 段不同 |
| id 段 | 950: 90000–90054；910: 90055–90109（无重叠） |
| 唯一边界轴（tag） | 55 |
| `is_boundary` 标记 | 55/55 全为 True |
| 结构性自洽 | 110/110 例通过生成器 `validate()` 重跑，0 违规 |

两个文件是**同一套用例的双份拷贝**：生成器 `gen_boundary_cases.py` 先造一批，
再深拷贝并把 `backend` 改为 `ascend910`、id 顺延，保证每个 SoC 都有各自可跑的完整套件。

## 二、设计意图

生成器头部明确了一个核心原则：

> **boundary 指「合法域的边缘」，不是「非法输入」。非法输入归 -NEG 生成器管。**

每个用例只围绕**一条**边界轴构造，轴编码在 `name` 后缀 `_BND_<TAG>` 里，
便于在结果报告里直接 grep 定位。

## 三、边界轴覆盖（13 组 / 55 例）

| 组 | 用例数 | 覆盖的边界 |
|---|---|---|
| L1 | 3 | region 长度恰为 1（无法承载严格前缀，只能做 mixed batch 或另一侧严格） |
| USED | 9 | used==1 / used==region 单侧 / used==region-1（丢一个尾 token） |
| TILE | 15 | used 落在 16/64/128/256 的 ±1 上（tile 对齐边界） |
| HEADDIM | 4 | headdim 1 / 127 / 128 / 129（129 跨过 ≤128 的结构分支） |
| ALIGN | 4 | headdim×2B = 32/64/96/128B（行字节对齐边界） |
| GQA | 5 | kv_heads=1、heads=1、组数 1/2/4/8/32 |
| BATCH | 3 | batch=1 与 batch=6（生成器上限） |
| MAXLEN | 3 | region 触及 golden 预算上限（8192） |
| CAUSAL | 3 | causal 下 uq==uk（对角）与 uk==uq+1 |
| NONCAUSAL | 2 | 非 causal 的极端 q/k 长度不对称 |
| DT | 4 | fp16/bf16 × causal/non-causal 四象限 |

### 3.1 tile 边界覆盖细节

| tile | 命中 used 值 | 用例 |
|---|---|---|
| 16 | [15, 16, 17] | TILE_16_M1, TILE_16_EQ, TILE_16_P1 |
| 64 | [63, 64, 65] | TILE_64_M1, TILE_64_EQ, TILE_64_P1, TILE_Q_64_M1, ALIGN_32B_H16, ALIGN_64B_H32 … |
| 128 | [127, 128, 129] | L1_Q_ONLY, TILE_128_M1, TILE_128_EQ, TILE_128_P1, TILE_K_128_EQ, TILE_K_128_P1 … |
| 256 | [255, 256, 257] | L1_MIXED_CAUSAL, TILE_256_M1, TILE_256_EQ, TILE_256_P1, TILE_K_128_EQ, TILE_K_128_P1 … |
| 512 | [511, 512] | USED_DIFF1, USED_DIFF1_ASYM, USED_DIFF1_CAUSAL, BATCH_1, MAXLEN_BOTH, NONCAUSAL_ASYM_Q … |
| 1024 | [1023, 1024] | USED_EQ1_Q, USED_EQ1_K, USED_DIFF1_ASYM, CAUSAL_K_LONG |
| 2048 | **未覆盖** |  |

## 四、检出力分析（核心结论）

设计得再「边界」，如果内核忽略 seqused 也看不出差异，用例就是无效的。
这里对每例算一个量：**假设内核完全忽略 seqused、按整个 region 做注意力，
输出会偏多少**（相对误差），再和该 dtype 的量化噪声底比较。

模型：忽略尾部等于多平均 `tail=region-used` 个随机 key

```
|dOut|/|Out| ~ frac/(1+frac) * sqrt(1 + 1/(1+frac)),   frac = tail/used
```

噪声底：fp16 `2^-11 = 4.88e-4`，bf16 `2^-8 = 3.91e-3`。

| 判定 | 判据 | 用例数 |
|---|---|---|
| 检出强 | 误差 >= 20x 噪声底 | **52** |
| 可检出 | 5-20x | 0 |
| 偏弱 | 2-5x | 1 |
| **低于噪声底** | < 2x | **2** |

### 4.1 问题用例：低于噪声底（检不出 seqused 缺陷）

| 用例 | dtype | 相对误差 | 噪声倍数 | 说明 |
|---|---|---|---|---|
| `USED_DIFF1_CAUSAL` | bf16 | 0.00138 | **0.4x** | q 侧 used=512 region=513 |
| `USED_DIFF1` | bf16 | 0.00276 | **0.7x** | q 侧 used=512 region=513 |

另有 1 例刚过线但很勉强：

- `USED_DIFF1_ASYM`（fp16）相对误差 0.00138，仅 **2.8x** 噪声底 —— k 侧 used=1023 region=1024

`USED_DIFF1` 系列（3 例）的共同点：只丢**一个**尾部 token。
尾部占比 `1/512`，均摊到输出上只有约 0.28% 的偏差 ——
**bf16 的量化噪声就有 0.39%，完全把这个信号埋掉了**。

它们作为「used==region-1」这条轴的语义标注是正确的（边界定义本身没问题），
但**不能用来验证精度**。要验证「丢一个尾 token 是否被正确排除」，
必须靠逐元素/位宽级对比（如 golden 侧显式截断检查），不能指望端到端精度阈值。

> 注意 `USED_DIFF1_ASYM` 之所以勉强过线（2.8x），是因为它是 fp16——
> 噪声底比 bf16 低 8 倍。同一形状换成 bf16 就会掉进盲区。

## 五、覆盖盲区（结构化缺口）

### 5.1 属性轴完全没有变化

| 属性 | 55 例中的取值 | 影响 |
|---|---|---|
| `layout` | 全 `TND` | 合理（bwd seqused 只支持 TND，BSND/BNSD 被源码硬拒） |
| `softcap` | 全 `0.0` | **缺口**：softcap≠0 会改分数量化路径，未测 |
| `deterministic` | 全 `False` | 与 seqused **源码级硬互斥**，非用例缺失 |
| `window_size` | 全 `-1`（无限窗） | **缺口**：滑动窗与 seqused 的交互完全未覆盖 |

`layout` 固定是对的。
`deterministic` 与 `seqused` 在 `mha_bwd.cpp` 里是硬互斥（直接 TORCH_CHECK 拒绝），
所以这条不算用例缺失，是能力边界。
但 `softcap` 和 `window_size` 是**真实缺口**：
window 归一化用的是 `max_seqlen` 而非 used 长度（`attention_window.h`），
要和 seqused 组合才能暴露归一化错误；而本套件 window 全为 -1。

### 5.2 数值范围全为 U(-1,1)，数值路径未激活

全部 55 例 `range_values = [-1, 1]`。此时：

```
q,k ~ U(-1,1),  scale = 1/sqrt(d)
var(logit) = d * (1/3) * (1/3) / d = 1/9
std(logit) = 1/3 = 0.333
```

- softmax 熵约 0.99，输出接近**均匀加权**
- fp16 的 exp 溢出阈值是 `logit ≈ 11.09`，当前 0.333 差 **33 倍**
- 于是这三条路径在本套件里是**死代码**：
  1. online-softmax 的 rescale（需要 max 值跨度大）
  2. exp 溢出保护（需要 logit 接近 11）
  3. P 的 fp16 次正规数处理（当前 P 全在正常区）

真实 LLM 的 attention logit `std` 通常在 2-6，等效 `U(-3,3)` 或 `N(0,2)`。
本套件比真实场景**温和 5-20 倍**。

> 注：`RANGE_ND`（N(0,1)）与 `RANGE_PM5`（U(-5,5)）两档数值范围在
> `Accu_Generalization` 系列里已有覆盖，不在 Boundary 里重复 —— 这是合理分层。

### 5.3 headdim=256 缺失

| headdim | 例数 |
|---|---|
| 1 | 1 |
| 16 | 3 |
| 32 | 2 |
| 48 | 1 |
| 64 | 43 |
| 127 | 1 |
| 128 | 3 |
| 129 | 1 |

`headdim` 覆盖了 1/16/32/48/64/127/128/129，但**没测 256**。
256 是 950 上 head_dim 的**运行时上限**（`(0,256]` 全支持），
且 129 已跨过 `qkHeadDim_ <= 128U` 的结构分支（K 常驻 → K 流式），
256 是这条流式路径的最远端，值得补一条。

### 5.4 batch 多样性不足

| batch | 例数 | 占比 |
|---|---|---|
| 1 | 51 | 93% |
| 2 | 2 | 4% |
| 6 | 2 | 4% |

93% 是单 batch。`BATCH_6_MIXED` 是唯一真正检验「部分 region 严格前缀、
部分 region 不严格」的用例 —— 这正是 `validate()` 用 `any` 而非 `all` 的原因。
这类 mixed batch 只有 2 例，偏少。

## 六、Causal 掩码的检出力盲区（重要）

golden 的 causal 掩码实现（`fa_small_op_golden_same_input.py:64-67`）：

```python
offset = seq_k - seq_q
mask = torch.triu(torch.ones(seq_q, seq_k), diagonal=offset + 1)
```

即**右下角对齐**。当 `used_q == used_k` 时 `offset=0`，
第 i 行恰好只能看到 key `0..i` —— **尾部键全部落在被屏蔽的上三角里**。

这意味着：若内核忽略 seqused 按 region 算，尾部键依然被掩码，
**输出完全相同**。

受影响的用例：

| 用例 | used_q | used_k | region_q | region_k | k 侧尾部是否可达 |
|---|---|---|---|---|---|
| `L1_MIXED_CAUSAL` | 1 | 256 | 512 | 512 | **否（被掩码）** |
| `USED_EQ1_CAUSAL` | 1 | 1 | 1024 | 1024 | **否（被掩码）** |
| `USED_DIFF1_CAUSAL` | 512 | 512 | 513 | 513 | **否（被掩码）** |
| `CAUSAL_EQ` | 256 | 256 | 512 | 512 | **否（被掩码）** |
| `CAUSAL_D1` | 256 | 257 | 512 | 512 | **否（被掩码）** |
| `DT_fp16_CAUSAL` | 100 | 128 | 256 | 256 | **否（被掩码）** |
| `DT_bf16_CAUSAL` | 100 | 128 | 256 | 256 | **否（被掩码）** |

共 **7 / 8** 个 causal 用例的 k 侧检出力为零。

**但这不代表用例写错了。** 以 `CAUSAL_D1` 为例：

```
sq=512  sk=512  uq=256  uk=257
```
`uk = uq + 1` 的设计意图是测「对角线上方多一个 key」这条路径。
在 used 域里 `offset = uk - uq = 1 > 0`，第 i 行能看到 `0..i+1`，
那个多出来的 key **确实被访问**，语义是有效的。

盲区的成因是：一旦内核忽略 seqused 退回 region 域，`offset` 变成 `sk-sq = 0`，
尾部键反而被掩码 —— 于是「错的行为」和「对的行为」输出一致。

**结论**：这是 causal 结构本身的固有性质，不是用例缺陷。
想覆盖 causal 下的 k 侧尾部检测，必须构造 `sk > sq` 的用例（如 `CAUSAL_K_LONG`）：
`sq=512 sk=1024` 时 `offset=512`，忽略 seqused 后尾部键会被访问，可检出。
`CAUSAL_K_LONG` 正是唯一不在此盲区里的 causal 用例。

## 七、与 generalize 套件的分工（避免重复评价）

| 维度 | Boundary（55） | Gen-950/910（各 5000） | RANGE_ND / PM5 |
|---|---|---|---|
| 目的 | 每轴取边界点 | 广域随机采样 | 数值范围配对 |
| 形状 | 手写、可解释 | 随机生成 | 随机生成 |
| 数值范围 | 全 U(-1,1) | 全 U(-1,1) | N(0,1) / U(-5,5) |
| 覆盖密度 | 稀疏但精准 | 密集 | 密集 |

所以「Boundary 没有数值范围覆盖」这件事，**不该单独归责于 Boundary**——
它和 Gen/RANGE 是分层设计，数值范围由后者负责。
真正值得补的是：**Boundary 独有的边界点（headdim=256、window 非 -1）** 上没有数值范围组合。

## 八、结论与建议

### 做得好的地方

1. **一轴一例**的设计非常清晰，tag 可 grep，出问题能立刻定位到边界轴
2. `validate()` 在**生成期**就拦住非法组合（causal 下 uq>uk、无严格前缀等），
   不会把坏用例送上 NPU —— 110/110 例重跑 0 违规可证
3. `USED_EQ1` 系列检出力极强（127-2047x 噪声底），单 token decode 场景覆盖好
4. `L1_*` 对「region=1 无法承载严格前缀」这个逻辑陷阱处理得对：
   用 mixed batch 或单侧模式解决，而不是硬造一个无效用例
5. `HEADDIM_129` 精准命中 `<=128` 结构分支的跨越点，这是有源码依据的设计

### 建议补充

| 优先级 | 建议 | 理由 |
|---|---|---|
| P0 | `USED_DIFF1` 系列改用逐元素断言而非精度阈值 | 3 例目前检出力低于噪声底，等于空跑 |
| P1 | 补 `HEADDIM_256` | 950 的 head_dim 上限，流式路径最远端 |
| P1 | 补 `window_size` 非 -1 与 seqused 的组合 | window 归一化用 max_seqlen，易错点 |
| P2 | 补 `softcap` 非 0 与 seqused 的组合 | 改分数量化路径，未覆盖 |
| P2 | 增加 mixed batch 用例（当前仅 2 例） | 检验 per-region 独立性最有效 |
| P3 | causal 下补 `sk >> sq` 且 uk<sk 的组合 | 绕开对角掩码盲区 |

### 一句话总结

这套 Boundary 用例在**形状边界**上覆盖得相当扎实（tile、head_dim、GQA、batch 都有精准打点），
生成期自检也严谨；但**检出力**存在两类系统性盲区：
① 只丢 1 个 token 的用例淹没在量化噪声里；
② causal 且 uq==uk 时尾部键被掩码。
前者需要换判定方式，后者是结构固有限制、需换形状来绕开。
