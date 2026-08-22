# C2 harness:Triton 基线独立运行

不装 vllm,只要 torch + triton:

    python run.py check    # SDPA 稠密参考对拍(常规 / 短序列尾块 / 投机 decode)
    python run.py bench    # batch 扫描计时,基线 profile 的起点

四个文件:`vllm_shim.py`(vllm import 垫片,PDL 关闭)、`synth.py`
(合成 paged KV + top-k 逻辑块索引,物理页打乱使两级间接寻址真实;
布局约定见其 docstring)、`ref_sdpa.py`(fp32 稠密参考,只用于对拍)、
`run.py`(入口)。

改 kernel 后先过 check 再谈性能;自定验收方案(讨论点 6)可以从
这里的 err_ratio 口径出发挑毛病。FP8 KV cache 路径 harness 未覆盖
(讨论点 4 自行扩展,上游口径见 ../vllm_msa_ref/test_sparse_attn_fp8_scale.py)。
