"""
Device-Level Intrinsics & Helper Functions for pGPU Kernels.

Provides reusable in-kernel device functions (such as `memset`) that emit
instruction sequences into a `WarpKernel` builder using scoped temporary registers.

Author: Arjun Vadakkeveedu (arjunmenonv@alumni.iitm.ac.in)
September 2026
"""

from typing import List, Optional, Union, TYPE_CHECKING, Any
from pgpu.arch.isa import DEFAULT_TAYLOR_N

if TYPE_CHECKING:
    from pgpu.arch.isa import Instruction
    from pgpu.arch.memory import MemoryAllocation
    from pgpu.sw.kernel import WarpKernel


def memset(
    kb: "WarpKernel",
    target: Union["MemoryAllocation", int],
    value: float = 0.0,
    size: Optional[int] = None,
    mem_space: Optional[str] = None,
) -> List["Instruction"]:
    """
    Device function callable from inside a WarpKernel that performs a parallel memset
    on a DRAMResource or SRAMResource allocation by emitting a stream of stores.

    Addressing pattern:
    - Uses standard linear addressing across threads within a warp (`0..31` contiguous offsets):
        `tid = SR_WARPID * 32 + SR_LANEID`
    - Grid-stride increment per loop iteration:
        `stride = SR_NUMWARPS * 32`
    - Guarded by `p_cond = (idx < limit)` so non-multiple sizes are masked cleanly.
    """
    from pgpu.arch.memory import MemoryAllocation
    from pgpu.sw.kernel import WarpKernel

    if hasattr(target, "base_addr") and hasattr(target, "size"):
        base_addr = target.base_addr
        num_words = size if size is not None else target.size
        space = (mem_space or getattr(target, "mem_space", "dram")).lower()
        if hasattr(target, "allocation") and getattr(target.allocation, "mem_space", None) == "sram":
            space = "sram"
    elif isinstance(target, int):
        if size is None:
            raise ValueError("`size` must be specified when `target` is an integer base address.")
        base_addr = target
        num_words = size
        space = (mem_space or "dram").lower()
    else:
        raise TypeError(f"Expected MemoryAllocation or int base_addr, got {type(target).__name__}.")

    if space not in ("dram", "sram"):
        raise ValueError(f"Invalid mem_space '{space}'; expected 'dram' or 'sram'.")

    store_mnemonic = "sram.st" if space == "sram" else "dram.st"
    loop_label = f"_memset_loop_{kb._next_internal_id()}"

    start_idx = len(kb.instructions)
    with kb.reg_scope() as scope:
        r_base   = scope.alloc("memset_base")
        r_val    = scope.alloc("memset_val")
        r_scale  = scope.alloc("memset_scale")
        r_stride = scope.alloc("memset_stride")
        r_limit  = scope.alloc("memset_limit")
        r_idx    = scope.alloc("memset_idx")
        r_addr   = scope.alloc("memset_addr")
        p_cond   = scope.alloc_pred("memset_cond")

        kb.asm("set.imm", r_base, float(base_addr))
        kb.asm("set.imm", r_val, float(value))
        kb.asm("set.imm", r_scale, 32.0)
        kb.asm("set.imm", r_limit, float(num_words))

        # stride = SR_NUMWARPS * 32
        kb.asm("mul", r_stride, kb.sr_numwarps, r_scale)
        # linear thread index: idx = SR_WARPID * 32 + SR_LANEID
        kb.asm("fma", r_idx, kb.sr_warpid, r_scale, kb.sr_laneid)

        kb.label(loop_label)
        kb.asm("cmp.lt", p_cond, r_idx, r_limit)
        kb.asm("add", r_addr, r_base, r_idx, pred=p_cond)
        kb.asm(store_mnemonic, r_addr, r_val, pred=p_cond)
        kb.asm("add", r_idx, r_idx, r_stride)
        kb.asm("cmp.lt", p_cond, r_idx, r_limit)
        kb.asm("jmp", loop_label, pred=p_cond)

    return kb.instructions[start_idx:]


# Alias
emit_memset = memset

def emit_fast_trans(
    kb: "WarpKernel",
    func_name: str,
    dest: Any,
    src_x: Any,
    n: int = DEFAULT_TAYLOR_N,
    pred: Optional[Union["VirtualPred", int]] = None,
) -> List["Instruction"]:
    """
    Intrinsic wrapper for `<func>.fast.<n>`: emits `set.imm r_n, n` followed by
    `<func>.fast dest, src_x, r_n` using a scoped temporary register.
    """
    with kb.reg_scope() as scope:
        r_n = scope.alloc(name=f"{func_name}_n")
        # Hardware validation requires all 32 lanes of R_n to hold the identical Taylor count,
        # so this set.imm MUST NOT be predicated.
        i_imm = kb.asm("set.imm", r_n, int(n), pred=None)
        i_op = kb.asm(f"avx.{func_name.lower()}.fast", dest, src_x, r_n, pred=pred)
        return [i_imm, i_op]

from typing import Tuple

def emit_mma_8x16_8x8(
    kb: "WarpKernel",
    a_sram: Any,
    b_sram: Any,
    c_sram: Optional[Any],
    d_regs: Tuple[Any, Any],
) -> List["Instruction"]:
    """
    Intrinsic for invoking the MMA instruction on 8x16 tiles A and B sitting in scratch memory.
    Executes D = A * B + C, where D and C are 8x8 tiles.

    Args:
        kb: WarpKernel builder.
        a_sram: VirtualReg containing the base address of tile A (8x16) in SRAM.
        b_sram: VirtualReg containing the base address of tile B (8x16) in SRAM.
        c_sram: VirtualReg containing the base address of tile C (8x8) in SRAM. 
                If None, RZ is passed as the C operand.
        d_regs: Tuple of 2 VirtualRegs to hold the 8x8 result tile D.
                Passed explicitly by caller to avoid dangling allocations upon scope exit.
    """
    start_idx = len(kb.instructions)

    with kb.reg_scope("mma_8x16_8x8") as scope:
        # Compute thread's quad coordinates
        r_a_base = scope.alloc("a_base")
        if hasattr(a_sram, 'base_addr'):
            kb.asm("set.imm", r_a_base, float(a_sram.base_addr))
        else:
            kb.asm("mov", r_a_base, a_sram)

        r_b_base = scope.alloc("b_base")
        if hasattr(b_sram, 'base_addr'):
            kb.asm("set.imm", r_b_base, float(b_sram.base_addr))
        else:
            kb.asm("mov", r_b_base, b_sram)

        r_row = scope.alloc("row")
        r_col = scope.alloc("col")
        r_thresh = scope.alloc("thresh")
        r_one = scope.alloc("one")

        kb.asm("set.imm", r_row, 0.0)
        kb.asm("set.imm", r_one, 1.0)

        # row = tid // 8 (since tid is float, we use predicated addition)
        for thresh in [8.0, 16.0, 24.0]:
            p_ge = scope.alloc_pred(f"p_ge_{int(thresh)}")
            kb.asm("set.imm", r_thresh, thresh)
            kb.asm("cmp.lt", p_ge, kb.sr_laneid, r_thresh)
            kb.asm("add", r_row, r_row, r_one, pred=p_ge, pred_inv=True)

        # col = tid - row * 8
        r_eight = scope.alloc("eight")
        r_row_x_8 = scope.alloc("row_x_8")
        kb.asm("set.imm", r_eight, 8.0)
        kb.asm("mul", r_row_x_8, r_row, r_eight)
        kb.asm("sub", r_col, kb.sr_laneid, r_row_x_8)

        # Base offsets for the thread within the tiles
        # A & B (8x16 tiles): base = row * 16 + col
        r_16 = scope.alloc("sixteen")
        r_ab_base = scope.alloc("ab_base")
        kb.asm("set.imm", r_16, 16.0)
        kb.asm("fma", r_ab_base, r_row, r_16, r_col)

        # Constants for quad offsets
        r_64 = scope.alloc("sixtyfour")
        r_8 = scope.alloc("eight_off")
        r_72 = scope.alloc("seventytwo")
        r_tmp_addr = scope.alloc("tmp_addr")
        kb.asm("set.imm", r_64, 64.0)
        kb.asm("set.imm", r_8, 8.0)
        kb.asm("set.imm", r_72, 72.0)

        a_regs = (scope.alloc("a_0"), scope.alloc("a_1"), scope.alloc("a_2"), scope.alloc("a_3"))
        b_regs = (scope.alloc("b_0"), scope.alloc("b_1"), scope.alloc("b_2"), scope.alloc("b_3"))

        # Load A (Q00, Q10, Q01, Q11)
        r_a_base_thread = scope.alloc("a_base_thread")
        kb.asm("add", r_a_base_thread, r_a_base, r_ab_base)
        kb.asm("sram.ld", a_regs[0], r_a_base_thread)
        kb.asm("add", r_tmp_addr, r_a_base_thread, r_64)
        kb.asm("sram.ld", a_regs[1], r_tmp_addr)
        kb.asm("add", r_tmp_addr, r_a_base_thread, r_8)
        kb.asm("sram.ld", a_regs[2], r_tmp_addr)
        kb.asm("add", r_tmp_addr, r_a_base_thread, r_72)
        kb.asm("sram.ld", a_regs[3], r_tmp_addr)

        # Load B (Q00, Q10, Q01, Q11)
        r_b_base_thread = scope.alloc("b_base_thread")
        kb.asm("add", r_b_base_thread, r_b_base, r_ab_base)
        kb.asm("sram.ld", b_regs[0], r_b_base_thread)
        kb.asm("add", r_tmp_addr, r_b_base_thread, r_64)
        kb.asm("sram.ld", b_regs[1], r_tmp_addr)
        kb.asm("add", r_tmp_addr, r_b_base_thread, r_8)
        kb.asm("sram.ld", b_regs[2], r_tmp_addr)
        kb.asm("add", r_tmp_addr, r_b_base_thread, r_72)
        kb.asm("sram.ld", b_regs[3], r_tmp_addr)

        # Load C (Q00, Q10)
        if c_sram is not None:
            c_regs = (scope.alloc("c_0"), scope.alloc("c_1"))
            r_c_base = scope.alloc("c_base")
            # C (8x8 tile): base = row * 8 + col
            kb.asm("fma", r_c_base, r_row, r_eight, r_col)

            r_c_base_thread = scope.alloc("c_base_thread")
            r_32 = scope.alloc("thirtytwo")
            kb.asm("set.imm", r_32, 32.0)

            r_c_base_reg = scope.alloc("c_base_reg")
            if hasattr(c_sram, 'base_addr'):
                kb.asm("set.imm", r_c_base_reg, float(c_sram.base_addr))
            else:
                kb.asm("mov", r_c_base_reg, c_sram)

            kb.asm("add", r_c_base_thread, r_c_base_reg, r_c_base)

            kb.asm("sram.ld", c_regs[0], r_c_base_thread)
            kb.asm("add", r_tmp_addr, r_c_base_thread, r_32)
            kb.asm("sram.ld", c_regs[1], r_tmp_addr)
        else:
            c_regs = (kb.rz, kb.rz)

        # Execute MMA
        kb.asm("avx.mma", d_regs, a_regs, b_regs, c_regs)

    return kb.instructions[start_idx:]

mma_8x16_8x8 = emit_mma_8x16_8x8

def emit_warp_reduce(
    kb: "WarpKernel",
    sram_buf: Any,
    op: str = "add"
) -> List["Instruction"]:
    """
    Intrinsic for a warp-wide parallel tree reduction in shared memory.
    
    Args:
        kb: WarpKernel builder.
        sram_buf: VirtualReg or MemoryAllocation pointing to a 32-element SRAM buffer.
        op: The reduction operation to perform ("add", "max", "min"). Defaults to "add".
            The final reduced data sits in element 0 of the SRAM buffer.
    """
    start_idx = len(kb.instructions)

    with kb.reg_scope("warp_reduce") as scope:
        r_base = scope.alloc("base")
        if hasattr(sram_buf, 'base_addr'):
            kb.asm("set.imm", r_base, float(sram_buf.base_addr))
        else:
            kb.asm("mov", r_base, sram_buf)

        r_tid = kb.sr_laneid

        # For each stride in [16, 8, 4, 2, 1]
        for stride in [16.0, 8.0, 4.0, 2.0, 1.0]:
            r_stride = scope.alloc(f"stride_{int(stride)}")
            kb.asm("set.imm", r_stride, stride)

            # Active threads: tid < stride
            p_active = scope.alloc_pred(f"p_active_{int(stride)}")
            kb.asm("cmp.lt", p_active, r_tid, r_stride)

            # addr1 = base + tid
            r_addr1 = scope.alloc(f"addr1_{int(stride)}")
            kb.asm("add", r_addr1, r_base, r_tid, pred=p_active)

            # addr2 = base + tid + stride
            r_addr2 = scope.alloc(f"addr2_{int(stride)}")
            r_tid_plus_stride = scope.alloc(f"tid_plus_stride_{int(stride)}")
            kb.asm("add", r_tid_plus_stride, r_tid, r_stride, pred=p_active)
            kb.asm("add", r_addr2, r_base, r_tid_plus_stride, pred=p_active)

            # Load val1, val2
            r_val1 = scope.alloc(f"val1_{int(stride)}")
            r_val2 = scope.alloc(f"val2_{int(stride)}")
            kb.asm("sram.ld", r_val1, r_addr1, pred=p_active)
            kb.asm("sram.ld", r_val2, r_addr2, pred=p_active)

            # Compute op
            r_res = scope.alloc(f"res_{int(stride)}")
            kb.asm(op, r_res, r_val1, r_val2, pred=p_active)

            # Store result
            kb.asm("sram.st", r_addr1, r_res, pred=p_active)

            # Sync warp before next step since write/read threads overlap in next step
            kb.asm("sync.warp")

    return kb.instructions[start_idx:]

warp_reduce = emit_warp_reduce
