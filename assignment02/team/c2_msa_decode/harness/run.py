"""C2 harness 入口:Triton 基线独立运行 + SDPA 对拍 + decode 计时。

用法(不需要安装 vllm,只要 torch + triton):
    python run.py check    # 三组形状对拍(含尾块/短序列/投机 decode)
    python run.py bench    # batch 扫描,基线 profile 的起点
"""
import sys

import torch

import vllm_shim
from ref_sdpa import sdpa_ref
from synth import make_case

sa = vllm_shim.load_sparse_attn()


def run_decode(case):
    out = torch.empty_like(case["q"])
    sa.minimax_m3_sparse_attn_decode(
        case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
        case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
        case["decode_query_len"])
    return out


def err_ratio(x, ref):
    x, ref = x.double().flatten(), ref.double().flatten()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def check():
    cases = [
        ("常规 batch=4", dict(num_reqs=4, seq_range=(1024, 8192), seed=0)),
        ("短序列+尾块", dict(num_reqs=3, seq_range=(50, 300), seed=1)),
        ("投机 decode dql=2", dict(num_reqs=2, seq_range=(2048, 4096),
                                   decode_query_len=2, seed=2)),
    ]
    ok = True
    for name, cfg in cases:
        case = make_case(**cfg)
        got = run_decode(case)
        ref = sdpa_ref(case)
        e = err_ratio(got, ref)
        good = e < 2e-2
        ok &= good
        print(f"{name:24s} err_ratio={e:.3e}  {'PASS' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


def bench():
    print(f"{'batch':>6} {'us/call':>9}   (seq~8192, topk=16, kv_heads=4, "
          f"gqa=16, dql=1)")
    for n in (1, 4, 8, 16, 32, 64):
        case = make_case(num_reqs=n, seq_range=(8192, 8192), seed=0)
        for _ in range(10):
            run_decode(case)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(100):
            run_decode(case)
        e.record()
        torch.cuda.synchronize()
        print(f"{n:>6} {s.elapsed_time(e) / 100 * 1e3:>9.1f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    check() if mode == "check" else bench()
