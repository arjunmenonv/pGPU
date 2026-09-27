"""
Vector Addition (A + B = C) Kernel Example for pGPU Simulator.

Implements a 1024x1 vector addition kernel using SPMD WarpKernel assembly abstractions
and executes it on the top-level `pGPU` device object using `dram_alloc` and `dram_copy`.

Location: pgpu/examples/vector_add.py
"""

from typing import List, Optional, Tuple
import numpy as np

from pgpu.arch.gpu_device import CopyDirection, pGPU
from pgpu.arch.isa import Instruction, WarpKernel


def build_vector_add_kernel(
    n: int = 1024,
    base_a: int = 0,
    base_b: int = 1024,
    base_c: int = 2048,
    num_warps: int = 4,
    debug: bool = False,
) -> WarpKernel:
    """
    Build a single-threaded SPMD WarpKernel for 1D vector addition:
        C[i] = A[i] + B[i]  for i in [0, n)
    
    Uses SPMD programming abstractions:
    - Global thread index calculation via special registers:
        `gtid = SR_WARPID * 32 + SR_LANEID`
    - Grid-stride loop computed dynamically from hardware special register `SR_NUMWARPS`:
        `stride = SR_NUMWARPS * 32`
    - Predicated execution using `p_cond = (idx < limit)`
    - Memory loads/stores via `dram.ld` and `dram.st`
    - Conditional backward branch via `jmp loop_start, pred=p_cond`
    """
    kb = WarpKernel(num_warps=num_warps, debug=debug)

    with kb.reg_scope("outer_scope") as outer_scope:
        r_base_a = outer_scope.alloc("base_a")
        r_base_b = outer_scope.alloc("base_b")
        r_base_c = outer_scope.alloc("base_c")
        r_stride = outer_scope.alloc("stride")
        r_limit  = outer_scope.alloc("limit")
        r_idx    = outer_scope.alloc("idx")
        p_cond   = outer_scope.alloc_pred("p_cond")

        # Initialize base pointers, loop bounds, and dynamic grid stride using a scoped temporary
        kb.asm("set.imm", r_base_a, float(base_a))
        kb.asm("set.imm", r_base_b, float(base_b))
        kb.asm("set.imm", r_base_c, float(base_c))
        kb.asm("set.imm", r_limit, float(n))

        with kb.reg_scope("init_scope") as init_scope:
            r_scale = init_scope.alloc("scale")
            r_tid   = init_scope.alloc("tid")
            kb.asm("set.imm", r_scale, 32.0)
            kb.asm("mul", r_stride, kb.sr_numwarps, r_scale)
            # Compute global thread ID: gtid = warpid * 32 + laneid
            kb.asm("fma", r_tid, kb.sr_warpid, r_scale, kb.sr_laneid)
            kb.asm("mov", r_idx, r_tid)

        # Grid-stride loop
        kb.label("loop_start")
        kb.asm("cmp.lt", p_cond, r_idx, r_limit)

        with kb.reg_scope("loop_scope") as loop_scope:
            r_addr_a = loop_scope.alloc("addr_a")
            r_addr_b = loop_scope.alloc("addr_b")
            r_addr_c = loop_scope.alloc("addr_c")
            r_val_a  = loop_scope.alloc("val_a")
            r_val_b  = loop_scope.alloc("val_b")
            r_val_c  = loop_scope.alloc("val_c")

            kb.asm("add", r_addr_a, r_base_a, r_idx, pred=p_cond)
            kb.asm("add", r_addr_b, r_base_b, r_idx, pred=p_cond)
            kb.asm("add", r_addr_c, r_base_c, r_idx, pred=p_cond)

            kb.asm("dram.ld", r_val_a, r_addr_a, pred=p_cond)
            kb.asm("dram.ld", r_val_b, r_addr_b, pred=p_cond)
            kb.asm("add", r_val_c, r_val_a, r_val_b, pred=p_cond)
            kb.asm("dram.st", r_addr_c, r_val_c, pred=p_cond)

        # Advance index and branch if any active lane continues
        kb.asm("add", r_idx, r_idx, r_stride)
        kb.asm("cmp.lt", p_cond, r_idx, r_limit)
        kb.asm("jmp", "loop_start", pred=p_cond)

    return kb


def run_vector_add(
    a: np.ndarray,
    b: np.ndarray,
    num_warps: int = 4,
    device: Optional[pGPU] = None,
) -> Tuple[np.ndarray, int]:
    """
    Execute 1D vector add on the top-level pGPU device simulator.
    
    Args:
        a: 1D NumPy float array (length N)
        b: 1D NumPy float array (length N)
        num_warps: Number of warps per SMSP (default 4)
        device: Optional pre-configured `pGPU` device instance
        
    Returns:
        (c, total_cycles): Result 1D array of length N, and total elapsed simulation cycles.
    """
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    n = len(a_arr)
    if len(b_arr) != n:
        raise ValueError(f"Vectors A and B must have identical size, got {n} and {len(b_arr)}.")

    dev = device if device is not None else pGPU(num_warps_per_smsp=num_warps)

    # 1. Allocate DRAM buffers via dram_alloc (allocates without copying)
    alloc_a = dev.dram_alloc("vec_a", n)
    alloc_b = dev.dram_alloc("vec_b", n)
    alloc_c = dev.dram_alloc("vec_c", n)

    # 2. Copy input arrays H2D into DRAM via dram_copy
    dev.dram_copy(alloc_a, a_arr, direction=CopyDirection.H2D)
    dev.dram_copy(alloc_b, b_arr, direction=CopyDirection.H2D)

    # 3. Build SPMD WarpKernel and launch on pGPU
    kernel = build_vector_add_kernel(
        n=n,
        base_a=alloc_a.base_addr,
        base_b=alloc_b.base_addr,
        base_c=alloc_c.base_addr,
        num_warps=dev.total_warps,
    )
    total_cycles = dev.launch(kernel)

    # 4. Copy output buffer D2H back to host NumPy array via dram_copy
    c = dev.dram_copy(alloc_c, direction=CopyDirection.D2H)

    return c, total_cycles


if __name__ == "__main__":
    np.random.seed(42)
    build_vector_add_kernel(debug=False)
    # experimental feature to print reg live ranges at entry and exit points of reg_scope
    # will eventually be replaced with objdump tool with per-asm live range tracking
    # build_vector_add_kernel(debug=True) 
    N = 1024
    vec_a = np.random.randn(N).astype(np.float32)
    vec_b = np.random.randn(N).astype(np.float32)
    expected = vec_a + vec_b

    nw = 4
    dev = pGPU(num_warps_per_smsp=nw)
    out_c, total_cycles = run_vector_add(vec_a, vec_b, device=dev)
    max_err = float(np.max(np.abs(out_c - expected)))
    print(
        f"[vector_add 1024x1 | num_warps={nw}] "
        f"total_cycles={total_cycles:,} "
        f"max_abs_err={max_err:.2e})"
    )
