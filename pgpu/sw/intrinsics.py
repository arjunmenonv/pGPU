"""
Device-Level Intrinsics & Helper Functions for pGPU Kernels.

Provides reusable in-kernel device functions (such as `memset`) that emit
instruction sequences into a `WarpKernel` builder using scoped temporary registers.

Location: pgpu/sw/intrinsics.py
"""

from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pgpu.arch.isa import Instruction, WarpKernel
    from pgpu.arch.memory import MemoryAllocation


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
    - Uses standard linear addressing across threads within a warp (`0..31` contiguous offsets,
      which is uncoalesced for DRAM in this architecture):
        `tid = SR_WARPID * 32 + SR_LANEID`
    - Grid-stride increment per loop iteration:
        `stride = SR_NUMWARPS * 32`
    - Guarded by `p_cond = (idx < limit)` so non-multiple sizes are masked cleanly.
    """
    from pgpu.arch.memory import MemoryAllocation

    if isinstance(target, MemoryAllocation):
        base_addr = target.base_addr
        num_words = size if size is not None else target.size
        space = (mem_space or getattr(target, "mem_space", "dram")).lower()
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
