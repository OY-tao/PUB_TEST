# Copyright (c) Huawei Technologies Co., Ltd. 2012-2026. All rights reserved.
"""
Case generator for FlashAttention v3 TND backward seqused_q/seqused_k.

Supports Ascend950 (csrc/ascend950/flash_attn_npu_3) and Ascend910
(csrc/ascend910/flash_attn_npu_3). The yaml attr ``backend`` selects which
SoC each case targets: ascend950 / ascend910 (or a list to sample).

Each case packs Q/K/V by allocated lengths (sq/sk). seqused lists (uq/uk) are
prefixes of those regions. The kernel must attend only on the prefixes.
Mode is one of:
  both   : pass seqused_q and seqused_k, both may be shorter than allocated
  q_only : pass seqused_q only; K length stays the cu region
  k_only : pass seqused_k only; Q length stays the cu region
"""
import json
import random
from typing import List, Union

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig

Q_INDEX = 0
K_INDEX = 1
V_INDEX = 2
CU_SEQLENS_Q_INDEX = 3
CU_SEQLENS_K_INDEX = 4
MAX_SEQLEN_Q_INDEX = 5
MAX_SEQLEN_K_INDEX = 6
LAYOUT_INDEX = 7
SOFTMAX_SCALE_INDEX = 8
CAUSAL_INDEX = 9
WINDOW_SIZE_INDEX = 10
SOFTCAP_INDEX = 11
DETERMINISTIC_INDEX = 12
BACKEND_INDEX = 13

_TOKEN_LIMIT = 2_000_000
_ATTN_LIMIT = 4_000_000

_BACKEND_ALIASES = {
    "950": "ascend950",
    "a5": "ascend950",
    "a5v3": "ascend950",
    "ascend950": "ascend950",
    "910": "ascend910",
    "910b": "ascend910",
    "910c": "ascend910",
    "ascend910": "ascend910",
}


def _normalize_backend(value):
    key = str(value).strip().lower().replace("-", "").replace("_", "")
    # Keep common spellings after stripping separators.
    for alias, canonical in _BACKEND_ALIASES.items():
        if key == alias.replace("-", "").replace("_", ""):
            return canonical
    raise ValueError(
        f"unsupported backend={value!r}; use ascend950 or ascend910"
    )


def _pick_backend(case_config):
    if len(case_config.inputs) <= BACKEND_INDEX:
        return "ascend950"
    cfg = case_config.inputs[BACKEND_INDEX].range_values
    if isinstance(cfg, str):
        return _normalize_backend(cfg)
    if isinstance(cfg, (list, tuple)) and cfg:
        return _normalize_backend(random.choice(list(cfg)))
    return "ascend950"


def _finalize_softmax_scale(range_values, head_size):
    if range_values is None or range_values == "default":
        return float(1.0 / (head_size ** 0.5))
    if isinstance(range_values, str) and range_values.lower() == "default":
        return float(1.0 / (head_size ** 0.5))
    try:
        scale = float(range_values)
    except (TypeError, ValueError):
        scale = random.uniform(1e-6, 1.0)
    if scale <= 0.0 or scale > 1.0:
        scale = random.uniform(1e-6, 1.0)
    return float(scale)


def _sample_heads_and_dim():
    # Cube FA paths are most stable on headdim multiples of 16.
    head_size = random.randint(1, 128)
    nheads_k = random.randint(1, 8)
    multiple = random.randint(1, 4)
    return head_size, nheads_k * multiple, nheads_k


def _clamp_used(sq, sk, uq, uk, causal):
    uq = [min(max(1, int(u)), int(s)) for u, s in zip(uq, sq)]
    uk = [min(max(1, int(u)), int(s)) for u, s in zip(uk, sk)]
    if causal:
        uq = [min(u, k) for u, k in zip(uq, uk)]
        uk = [max(u, q) if k >= q else k for u, q, k in zip(uk, uq, sk)]
        uk = [min(u, s) for u, s in zip(uk, sk)]
        uq = [min(u, k) for u, k in zip(uq, uk)]
    return uq, uk


def _ensure_strict(sq, sk, uq, uk, mode, causal):
    """Keep at least one passed seqused side strictly inside its region."""
    uq, uk = _clamp_used(sq, sk, uq, uk, causal)

    def _shrink_q():
        candidates = [i for i, (u, s) in enumerate(zip(uq, sq)) if s > 1 and u == s]
        if not candidates:
            candidates = [i for i, s in enumerate(sq) if s > 1]
        if not candidates:
            return
        idx = max(candidates, key=lambda i: sq[i])
        uq[idx] = random.randint(1, sq[idx] - 1)
        if causal:
            uk[idx] = max(uk[idx], uq[idx])
            uk[idx] = min(uk[idx], sk[idx])

    def _shrink_k():
        lo = [(uq[i] if causal else 1) for i in range(len(sk))]
        candidates = [
            i for i, (u, s, floor) in enumerate(zip(uk, sk, lo)) if s > floor and u == s
        ]
        if not candidates:
            candidates = [i for i, (s, floor) in enumerate(zip(sk, lo)) if s > floor]
        if not candidates:
            return
        idx = max(candidates, key=lambda i: sk[i] - lo[i])
        floor = lo[idx]
        uk[idx] = random.randint(floor, sk[idx] - 1)

    if mode in ("both", "q_only") and all(u == s for u, s in zip(uq, sq)):
        _shrink_q()
    if mode in ("both", "k_only") and all(u == s for u, s in zip(uk, sk)):
        _shrink_k()
    return _clamp_used(sq, sk, uq, uk, causal)


def _maybe_tile_boundary(sq, sk, uq, uk, mode, causal):
    if random.random() >= 0.2:
        return uq, uk
    idx = random.randrange(len(sq))
    boundary = random.choice([1, 16, 64, 127, 128, 129, 256])
    if mode != "k_only" and sq[idx] > boundary:
        uq[idx] = boundary
    if mode != "q_only" and sk[idx] > boundary:
        floor = uq[idx] if causal else 1
        if sk[idx] > max(boundary, floor) or boundary >= floor:
            uk[idx] = max(boundary, floor)
    return _clamp_used(sq, sk, uq, uk, causal)


def _shrink_volume(sq, sk, uq, uk, num_heads, head_size, causal):
    for _ in range(20):
        tokens = sum(sq) * num_heads * head_size
        attn = sum(a * b for a, b in zip(sq, sk)) * num_heads
        if tokens <= _TOKEN_LIMIT and attn <= _ATTN_LIMIT:
            break
        sq = [max(1, length // 2) for length in sq]
        sk = [max(1, length // 2) for length in sk]
        if causal:
            sk = [max(kv, q) for q, kv in zip(sq, sk)]
        uq = [min(u, s) for u, s in zip(uq, sq)]
        uk = [min(u, s) for u, s in zip(uk, sk)]
        uq, uk = _clamp_used(sq, sk, uq, uk, causal)
    return sq, sk, uq, uk


def _ensure_feasible(sq, sk, mode, causal):
    """Make sure the mode's passed side can hold a strict prefix.

    Some sampled batches are mathematically degenerate and would end up with
    seqused == the allocated region, which tests nothing:
      - q_only / both with every sq == 1: a length-1 sequence cannot shrink;
      - k_only with causal and sq == sk: causal forces uk >= uq == sq == sk.
    Bump the allocated length so a strict prefix still exists.
    """
    if mode in ("both", "q_only") and all(s <= 1 for s in sq):
        idx = random.randrange(len(sq))
        sq[idx] = random.randint(2, 8)
        if causal:
            sk[idx] = max(sk[idx], sq[idx])
    if mode in ("both", "k_only"):
        floors = [q if causal else 1 for q in sq]
        if all(k <= floor for k, floor in zip(sk, floors)):
            idx = random.randrange(len(sk))
            sk[idx] = floors[idx] + random.randint(1, 8)
    return sq, sk


def _sample_seqlens(causal):
    batch = random.randint(1, 6)
    max_q = random.randint(1, 8192)
    max_k = random.randint(max_q if causal else 1, 8192)
    sq = [random.randint(1, max_q) for _ in range(batch)]
    if causal:
        sk = [random.randint(q, max_k) for q in sq]
    else:
        sk = [random.randint(1, max_k) for _ in range(batch)]

    mode = random.choice(["both", "q_only", "k_only"])
    sq, sk = _ensure_feasible(sq, sk, mode, causal)
    uq = []
    uk = []
    for q_len, k_len in zip(sq, sk):
        if mode == "q_only":
            uq.append(random.randint(1, q_len))
            uk.append(k_len)
        elif mode == "k_only":
            uq.append(q_len)
            floor = min(q_len, k_len) if causal else 1
            uk.append(random.randint(floor, k_len))
        else:
            used_q = random.randint(1, q_len)
            floor = min(used_q, k_len) if causal else 1
            uq.append(used_q)
            uk.append(random.randint(floor, k_len))
    uq, uk = _maybe_tile_boundary(sq, sk, uq, uk, mode, causal)
    uq, uk = _ensure_strict(sq, sk, uq, uk, mode, causal)
    return mode, sq, sk, uq, uk


@GENERATOR_REGISTRY.register("ascend_generate_FlashAttnBwdSeqused")
class FlashAttnBwdSequsedGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        q_shape = case_config.inputs[Q_INDEX].shape
        if len(q_shape) == 0 or 0 in q_shape:
            return case_config

        backend = _pick_backend(case_config)
        head_size, num_heads, kv_heads = _sample_heads_and_dim()
        causal_cfg = case_config.inputs[CAUSAL_INDEX].range_values
        if isinstance(causal_cfg, bool):
            causal = causal_cfg
        else:
            causal = bool(random.choice([True, False]))

        mode, seqlens_q, seqlens_k, used_q, used_k = _sample_seqlens(causal)
        seqlens_q, seqlens_k, used_q, used_k = _shrink_volume(
            seqlens_q, seqlens_k, used_q, used_k, num_heads, head_size, causal
        )
        # _shrink_volume may halve lengths down to 1 and re-create a degenerate
        # batch, so re-check feasibility before forcing a strict prefix.
        seqlens_q, seqlens_k = _ensure_feasible(seqlens_q, seqlens_k, mode, causal)
        used_q, used_k = _ensure_strict(seqlens_q, seqlens_k, used_q, used_k, mode, causal)
        batch = len(seqlens_q)

        total_q = int(sum(seqlens_q))
        total_k = int(sum(seqlens_k))
        case_config.inputs[Q_INDEX].shape = [total_q, num_heads, head_size]
        case_config.inputs[K_INDEX].shape = [total_k, kv_heads, head_size]
        case_config.inputs[V_INDEX].shape = [total_k, kv_heads, head_size]
        case_config.inputs[CU_SEQLENS_Q_INDEX].shape = [batch + 1]
        case_config.inputs[CU_SEQLENS_K_INDEX].shape = [batch + 1]
        case_config.inputs[CU_SEQLENS_Q_INDEX].dtype = "int32"
        case_config.inputs[CU_SEQLENS_K_INDEX].dtype = "int32"
        case_config.inputs[MAX_SEQLEN_Q_INDEX].range_values = int(max(seqlens_q))
        case_config.inputs[MAX_SEQLEN_K_INDEX].range_values = int(max(seqlens_k))

        dtype = case_config.inputs[Q_INDEX].dtype
        case_config.inputs[K_INDEX].dtype = dtype
        case_config.inputs[V_INDEX].dtype = dtype
        case_config.inputs[LAYOUT_INDEX].range_values = "TND"
        # Prefer 1/sqrt(d); random scales with large |x| make FA bwd unstable.
        case_config.inputs[SOFTMAX_SCALE_INDEX].range_values = float(
            1.0 / (head_size ** 0.5)
        )
        case_config.inputs[CAUSAL_INDEX].range_values = causal
        case_config.inputs[WINDOW_SIZE_INDEX][0].range_values = -1
        case_config.inputs[WINDOW_SIZE_INDEX][1].range_values = -1
        case_config.inputs[SOFTCAP_INDEX].range_values = 0.0
        # seqused + deterministic is rejected by both 910/950 bwd hosts.
        case_config.inputs[DETERMINISTIC_INDEX].range_values = False
        if len(case_config.inputs) > BACKEND_INDEX:
            case_config.inputs[BACKEND_INDEX].range_values = backend

        seq_meta = json.dumps(
            {
                "layout": "TND",
                "backend": backend,
                "mode": mode,
                "sq": seqlens_q,
                "sk": seqlens_k,
                "uq": used_q,
                "uk": used_k,
                "pass_q": mode in ("both", "q_only"),
                "pass_k": mode in ("both", "k_only"),
            },
            separators=(",", ":"),
        )
        case_config.aclnn_name = seq_meta
        case_config.save_name = seq_meta
        return case_config
