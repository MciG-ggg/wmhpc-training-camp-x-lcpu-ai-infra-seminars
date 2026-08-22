"""让 vendored 的 sparse_attn.py 脱离 vllm 独立运行的 import 垫片。

sparse_attn.py 只从 vllm 拿三样:triton、triton.language、
current_platform.is_arch_support_pdl()。这里注入同名模块;PDL 统一
关闭(kernel 内所有 PDL 路径都在 USE_PDL constexpr 后面,关闭即整段
剪掉,不影响语义)。import 本模块必须发生在 import sparse_attn 之前。
"""
import sys
import types

import triton
import triton.language as tl


class _Platform:
    @staticmethod
    def is_arch_support_pdl() -> bool:
        return False


def _mod(name: str) -> types.ModuleType:
    m = sys.modules.get(name)
    if m is None:
        m = types.ModuleType(name)
        sys.modules[name] = m
    return m


_mod("vllm")
_mod("vllm.platforms").current_platform = _Platform()
tu = _mod("vllm.triton_utils")
tu.triton = triton
tu.tl = tl


def load_sparse_attn():
    """按文件路径加载 ../vllm_msa_ref/sparse_attn.py,返回模块。"""
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "vllm_msa_ref",
                     "sparse_attn.py")
    spec = importlib.util.spec_from_file_location("sparse_attn", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
