"""SDPA 稠密参考:逐 (token, kv_head) 把选中块的 K/V 聚齐,fp32 标准
softmax attention。慢而直白,只用于对拍。语义逐条对齐 Triton kernel:
逻辑块经 block_table 取物理页;只取前 real_topk 个索引;块内位置
>= kv_len 的行掩掉。
"""
import math

import torch

PAGE = 128


def sdpa_ref(case):
    q = case["q"].float()
    kv = case["kv_cache"].float()
    topk_idx = case["topk_idx"].cpu()
    bt = case["block_table"].cpu()
    seq_lens = case["seq_lens"].cpu()
    G = case["gqa_group_size"]
    d = case["head_dim"]
    dql = case["decode_query_len"]
    scale = case["sm_scale"]
    total_q, num_heads, _ = q.shape
    out = torch.empty_like(q)

    for t in range(total_q):
        r = t // dql
        qpos = int(seq_lens[r]) - dql + t % dql
        kv_len = qpos + 1
        nb = math.ceil(kv_len / PAGE)
        real_topk = min(case["topk"], nb)
        for kh in range(topk_idx.shape[0]):
            Ks, Vs = [], []
            for blk in topk_idx[kh, t, :real_topk].tolist():
                page = int(bt[r, blk])
                lo = blk * PAGE
                n = min(PAGE, kv_len - lo)
                Ks.append(kv[page, kh, :n, :d])
                Vs.append(kv[page, kh, :n, d:])
            K = torch.cat(Ks)  # [L, d]
            V = torch.cat(Vs)
            Q = q[t, kh * G:(kh + 1) * G]           # [G, d]
            p = torch.softmax(Q @ K.T * scale, dim=-1)
            out[t, kh * G:(kh + 1) * G] = p @ V
    return out.to(case["q"].dtype)
