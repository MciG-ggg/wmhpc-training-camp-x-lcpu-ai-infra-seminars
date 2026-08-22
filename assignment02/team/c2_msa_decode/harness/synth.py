"""合成 decode 负载:paged KV cache + top-k 逻辑块索引。

布局与 vllm 完全一致(见 vllm_msa_ref/sparse_attn.py 文件头):
  kv_cache    [num_pages, num_kv_heads, 128, 2*head_dim](前半 K 后半 V)
  block_table [num_reqs, max_blocks]  逻辑块 -> 物理页(物理页故意打乱,
              让两级间接寻址真实)
  topk_idx    [num_kv_heads, total_q, topk]  逻辑块号;kernel 只读前
              min(topk, ceil(kv_len/128)) 个,这些槽必须是互不相同的
              有效块(重复会被算两次),恒包含当前块(与 indexer 行为
              一致),其余槽位填 0 被忽略
  seq_lens 含正在生成的 token:query 位置 = seq_len - decode_query_len
              + 局部序号,它只看得见 <= 自己位置的 KV(kernel 内逐
              token 因果掩码)
"""
import math

import torch

PAGE = 128


def make_case(num_reqs=4, seq_range=(1024, 8192), topk=16, num_kv_heads=4,
              gqa_group_size=16, head_dim=128, decode_query_len=1,
              dtype=torch.bfloat16, seed=0, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    seq_lens = torch.randint(seq_range[0], seq_range[1] + 1, (num_reqs,),
                             generator=g, dtype=torch.int32)
    nblocks = [math.ceil(int(s) / PAGE) for s in seq_lens]
    max_blocks = max(nblocks)
    total_pages = sum(nblocks)

    perm = torch.randperm(total_pages, generator=g)
    block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32)
    off = 0
    for r, nb in enumerate(nblocks):
        block_table[r, :nb] = perm[off:off + nb]
        off += nb

    num_heads = num_kv_heads * gqa_group_size
    total_q = num_reqs * decode_query_len
    kv_cache = (torch.randn(total_pages, num_kv_heads, PAGE, 2 * head_dim,
                            generator=g) * 0.5).to(dtype)
    q = (torch.randn(total_q, num_heads, head_dim, generator=g) * 0.5).to(dtype)

    topk_idx = torch.zeros(num_kv_heads, total_q, topk, dtype=torch.int32)
    for t in range(total_q):
        r = t // decode_query_len
        qpos = int(seq_lens[r]) - decode_query_len + t % decode_query_len
        nb = math.ceil((qpos + 1) / PAGE)
        k = min(topk, nb)
        for kh in range(num_kv_heads):
            sel = torch.randperm(nb, generator=g)[:k]
            if (sel == nb - 1).sum() == 0:
                sel[0] = nb - 1  # 恒包含当前块
            topk_idx[kh, t, :k] = sel.to(torch.int32)

    dev = dict(device=device)
    return dict(
        q=q.to(**dev), kv_cache=kv_cache.to(**dev),
        topk_idx=topk_idx.to(**dev), block_table=block_table.to(**dev),
        seq_lens=seq_lens.to(**dev),
        num_kv_heads=num_kv_heads, gqa_group_size=gqa_group_size,
        head_dim=head_dim, topk=topk, decode_query_len=decode_query_len,
        sm_scale=1.0 / math.sqrt(head_dim),
    )
