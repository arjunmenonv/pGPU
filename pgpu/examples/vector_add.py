"""
Vector Addition (A + B = C) Kernel Example for pGPU Simulator.

Implements a 1024x1 vector addition kernel using SPMD WarpKernel assembly abstractions
and executes it on the top-level `pGPU` device object using `dram_alloc` and `dram_copy`.

Location: pgpu/examples/vector_add.py
"""

from typing import Any, List, Optional, Tuple
import numpy as np

from pgpu.arch.gpu_device import CopyDirection, pGPU
from pgpu.arch.isa import Instruction
from pgpu.sw.kernel import WarpKernel, build_kernel


def vector_add_kernel(
    kb: WarpKernel,
    r_base_a: Any,
    r_base_b: Any,
    r_base_c: Any,
    r_limit: Any,
) -> None:
    """
    User-written SPMD Vector Addition Kernel:
        C[i] = A[i] + B[i]  for i in [0, n)

    Receives symbolic VirtualReg handles for array base addresses and limits.
    Completely decoupled from physical memory addresses, allowing it to be compiled
    once and launched repeatedly across different buffer allocations.
    """
    with kb.reg_scope("outer_scope") as outer_scope:
        r_stride = outer_scope.alloc("stride")
        r_idx    = outer_scope.alloc("idx")
        p_cond   = outer_scope.alloc_pred("p_cond")

        with kb.reg_scope("init_scope") as init_scope:
            r_scale = init_scope.alloc("scale")
            r_tid   = init_scope.alloc("tid")
            kb.asm("set.imm", r_scale, 32.0)
            kb.asm("mul", r_stride, kb.sr_numwarps, r_scale)
            # Compute global thread ID: gtid = warpid * 32 + laneid
            kb.asm("mul", r_tid, kb.sr_warpid, r_scale)
            kb.asm("add", r_tid, r_tid, kb.sr_laneid)
            kb.asm("mov", r_idx, r_tid)

        # Grid-stride loop
        kb.trace_start("VectorAddLoop")
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
        kb.trace_end("VectorAddLoop")



def build_vector_add_kernel(
    n: int = 1024,
    base_a: int = 0,
    base_b: int = 1024,
    base_c: int = 2048,
    num_warps: int = 4,
    debug: bool = False,
) -> Any:
    """
    Build a compiled SPMD kernel for 1D vector addition using the factory API.
    """
    return build_kernel(vector_add_kernel, num_warps=num_warps, debug=debug)


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

    # 3. Build SPMD Kernel and launch with dynamic arguments on pGPU
    compiled = build_kernel(vector_add_kernel, num_warps=dev.total_warps)

    # --- TEMPORARY TRACER INJECTION ---
    from pgpu.devtools.kernel_tracer import KernelTracer
    tracer = KernelTracer("vector_add_trace")
    tracer.attach(dev, compiled)
    # ----------------------------------

    total_cycles = dev.launch(compiled, alloc_a, alloc_b, alloc_c, n)

    # --- TEMPORARY TRACER INJECTION ---
    tracer.export()
    # ----------------------------------

    # 4. Copy output buffer D2H back to host NumPy array via dram_copy
    c = dev.dram_copy(alloc_c, direction=CopyDirection.D2H)

    return c, total_cycles


def batched_vector_add_sram_kernel(
    kb: WarpKernel,
    r_base_a: Any,
    r_base_b: Any,
    r_base_c: Any,
    r_num_rows: Any,
) -> None:
    """
    Batched Vector Addition with SRAM Caching:
        C[n, w] = A[n, w] + B[w]  for n in [0, N), w in [0, 32)

    A has shape (N, 32) in DRAM.
    B has shape (32,) in DRAM.
    C has shape (N, 32) in DRAM.

    Demonstrates the dynamic SRAM interface:
    1. Allocates shared memory for B using `kb.sram.alloc("b_shared", 32)`.
       The returned `r_sram_b` is a symbolic VirtualReg populated dynamically at launch.
    2. Warp 0 loads B from DRAM and stores it into SRAM.
    3. All warps synchronize on a barrier (`kb.asm("sync")`).
    4. Each warp streams rows of A from DRAM, reads B from SRAM, and writes C to DRAM.
    """
    W = 32
    # 1. Allocate SRAM buffer for vector B (32 words) - returns a VirtualReg directly!
    r_sram_b = kb.sram.alloc("b_shared", size=W)

    with kb.reg_scope("preload_scope") as p_scope:
        p_is_w0 = p_scope.alloc_pred("p_is_w0")
        r_b_dram_addr = p_scope.alloc("b_dram_addr")
        r_b_sram_addr = p_scope.alloc("b_sram_addr")
        r_b_val = p_scope.alloc("b_val")

        # Predicate: only warp 0 loads B from DRAM into SRAM
        kb.asm("cmp.eq", p_is_w0, kb.sr_warpid, kb.rz)
        kb.asm("add", r_b_dram_addr, r_base_b, kb.sr_laneid, pred=p_is_w0)
        kb.asm("dram.ld", r_b_val, r_b_dram_addr, pred=p_is_w0)
        kb.asm("add", r_b_sram_addr, r_sram_b, kb.sr_laneid, pred=p_is_w0)
        kb.asm("sram.st", r_b_sram_addr, r_b_val, pred=p_is_w0)

    # Barrier: ensure B is fully written into SRAM before other warps read it
    kb.asm("sync")

    # 2. Stream rows of A from DRAM and add cached B from SRAM
    with kb.reg_scope("stream_scope") as s_scope:
        r_row = s_scope.alloc("row")
        r_stride = s_scope.alloc("stride")
        r_w_const = s_scope.alloc("w_const")
        r_b_sram_addr = s_scope.alloc("b_sram_addr")
        r_cached_b = s_scope.alloc("cached_b")
        p_row_valid = s_scope.alloc_pred("p_row_valid")

        kb.asm("set.imm", r_w_const, float(W))
        kb.asm("mov", r_stride, kb.sr_numwarps)
        kb.asm("mov", r_row, kb.sr_warpid)

        # Pre-read B[w] from SRAM into a local register (lane w reads B[w])
        kb.asm("add", r_b_sram_addr, r_sram_b, kb.sr_laneid)
        kb.asm("sram.ld", r_cached_b, r_b_sram_addr)

        kb.label("stream_loop")
        kb.asm("cmp.lt", p_row_valid, r_row, r_num_rows)

        with kb.reg_scope("row_compute_scope") as rc_scope:
            r_row_offset = rc_scope.alloc("row_offset")
            r_addr_a = rc_scope.alloc("addr_a")
            r_addr_c = rc_scope.alloc("addr_c")
            r_val_a = rc_scope.alloc("val_a")
            r_val_c = rc_scope.alloc("val_c")

            # row_offset = row * 32 + laneid
            kb.asm("mul", r_row_offset, r_row, r_w_const)
            kb.asm("add", r_row_offset, r_row_offset, kb.sr_laneid, pred=p_row_valid)
            kb.asm("add", r_addr_a, r_base_a, r_row_offset, pred=p_row_valid)
            kb.asm("add", r_addr_c, r_base_c, r_row_offset, pred=p_row_valid)

            kb.asm("dram.ld", r_val_a, r_addr_a, pred=p_row_valid)
            kb.asm("add", r_val_c, r_val_a, r_cached_b, pred=p_row_valid)
            kb.asm("dram.st", r_addr_c, r_val_c, pred=p_row_valid)

        kb.asm("add", r_row, r_row, r_stride)
        kb.asm("cmp.lt", p_row_valid, r_row, r_num_rows)
        kb.asm("jmp", "stream_loop", pred=p_row_valid)


def build_batched_vector_add_sram_kernel(
    num_warps: int = 4,
    debug: bool = False,
) -> Any:
    """Compile the batched vector add SRAM kernel using build_kernel."""
    return build_kernel(batched_vector_add_sram_kernel, num_warps=num_warps, debug=debug)


def run_batched_vector_add_sram(
    a: np.ndarray,
    b: np.ndarray,
    num_warps: int = 4,
    device: Optional[pGPU] = None,
) -> Tuple[np.ndarray, int]:
    """
    Execute batched vector add A (N x 32) + B (32) on pGPU using SRAM caching.
    """
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    if a_arr.ndim != 2 or a_arr.shape[1] != 32:
        raise ValueError(f"Matrix A must have shape (N, 32), got {a_arr.shape}.")
    if b_arr.ndim != 1 or len(b_arr) != 32:
        raise ValueError(f"Vector B must have length 32, got {len(b_arr)}.")

    num_rows = a_arr.shape[0]
    dev = device if device is not None else pGPU(num_warps_per_smsp=num_warps)

    alloc_a = dev.dram_alloc("batch_a", a_arr.size)
    alloc_b = dev.dram_alloc("batch_b", b_arr.size)
    alloc_c = dev.dram_alloc("batch_c", a_arr.size)

    dev.dram_copy(alloc_a, a_arr.ravel(), direction=CopyDirection.H2D)
    dev.dram_copy(alloc_b, b_arr, direction=CopyDirection.H2D)

    compiled = build_batched_vector_add_sram_kernel(num_warps=dev.total_warps)
    total_cycles = dev.launch(compiled, alloc_a, alloc_b, alloc_c, float(num_rows))

    c = dev.dram_copy(alloc_c, direction=CopyDirection.D2H).reshape(num_rows, 32)
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
        f"max_abs_err={max_err:.2e}"
    )

    # Demonstrate SRAM batched vector add
    num_rows = 64
    w_cols = 32
    mat_a = np.random.randn(num_rows, w_cols).astype(np.float32)
    vec_b_sram = np.random.randn(w_cols).astype(np.float32)
    expected_sram = mat_a + vec_b_sram[None, :]

    dev2 = pGPU(num_warps_per_smsp=nw)
    out_c_sram, sram_cycles = run_batched_vector_add_sram(mat_a, vec_b_sram, device=dev2)
    sram_err = float(np.max(np.abs(out_c_sram - expected_sram)))
    print(
        f"[batched_vector_add_sram {num_rows}x{w_cols} | num_warps={nw}] "
        f"total_cycles={sram_cycles:,} "
        f"max_abs_err={sram_err:.2e}"
    )
