'''
Firmware-level Driver Helpers for pGPU Simulator.

Contains hardware microarchitectural functions:
- Index modifier (swizzle layout) functions for Scratchpad SRAM allocations
- Warp context switch save and restore routines

Author: Arjun Vadakkeveedu (arjunmenonv@alumni.iitm.ac.in)
September 2026
'''

from typing import Dict, List, Optional, Set, Tuple
import numpy as np


def swizzle_indices(offset: int) -> int:
    """
    Default firmware index modifier mapping a logical intra-tile word offset
    to a physical intra-tile word offset.
    """
    row = offset >> 5
    col = offset & 0x1F
    swz_col = col ^ (row & 0x3)
    return (row << 5) | swz_col


# ============================================================================
# System Context-Save Scratchpad Layout (Dedicated high SRAM region)
# ============================================================================
# Base address placed above user scratchpad capacity (24K words = 96 KB)
CONTEXT_SAVE_BASE_ADDR: int = 24 * 1024
VECTOR_REGS_COUNT: int = 64
PRED_REGS_COUNT: int = 8
WARP_CONTEXT_ENTRIES: int = VECTOR_REGS_COUNT + PRED_REGS_COUNT  # 72 vectors
WARP_CONTEXT_WORDS: int = WARP_CONTEXT_ENTRIES * 32              # 2304 words per warp


def get_warp_scratch_base(warp_id: int) -> int:
    """Return the base physical SRAM word address for `warp_id`'s context-save slot."""
    return CONTEXT_SAVE_BASE_ADDR + warp_id * WARP_CONTEXT_WORDS


def get_vector_scratch_addr(warp_id: int, reg_id: int) -> np.ndarray:
    """Return the 32-lane physical SRAM word address vector for vector register `reg_id`."""
    base = get_warp_scratch_base(warp_id) + reg_id * 32
    return base + np.arange(32, dtype=np.int64)


def get_pred_scratch_addr(warp_id: int, pred_idx: int) -> np.ndarray:
    """Return the 32-lane physical SRAM word address vector for predicate register `pred_idx`."""
    base = get_warp_scratch_base(warp_id) + (VECTOR_REGS_COUNT + pred_idx) * 32
    return base + np.arange(32, dtype=np.int64)


def get_pending_long_lat_dests(warp) -> Set[int]:
    """
    Return the set of register IDs (vector 0..63, predicates 68..75) for which there is
    an active in-flight pending write in `warp.scoreboard` (e.g., DRAM_LD, SRAM_LD, or multi-cycle op).
    """
    return set(warp.scoreboard.keys())


def emit_context_save_instructions(scheduler, warp_id: int):
    """
    Emit context-save operations to save all 64 vector registers and 8 predicate
    registers for `warp_id` to its dedicated scratch buffer slot in SRAM,
    skipping registers with an active in-flight long-latency load instruction.
    
    Returns a list of (Instruction, address_vector, value_vector).
    """
    from pgpu.arch.isa import Instruction, OpClass, OpCode
    from pgpu.arch.smsp import RegisterFile

    warp = scheduler.get_warp(warp_id) if hasattr(scheduler, 'get_warp') else scheduler.warps[warp_id]
    pending_long_lat = get_pending_long_lat_dests(warp)
    warp.saved_skipped_regs = set(pending_long_lat)
    save_ops = []

    # 1. Vector registers R0..R63
    for r in range(RegisterFile.NUM_VECTOR_REGS):
        if r in pending_long_lat:
            continue
        addr = get_vector_scratch_addr(warp_id, r)
        val = warp.regs[r].copy()
        instr = Instruction(
            opcode=OpCode.SRAM_ST,
            op_class=OpClass.SRAM,
            dest=[r],
            srcs=[],
        )
        save_ops.append((instr, addr, val))

    # 2. Predicate registers P0..P7
    for p in range(RegisterFile.NUM_PRED_REGS):
        pred_reg_id = RegisterFile.PRED_BASE_ID + p
        if pred_reg_id in pending_long_lat:
            continue
        addr = get_pred_scratch_addr(warp_id, p)
        val = warp.preds[p].astype(np.float32)
        instr = Instruction(
            opcode=OpCode.SRAM_ST,
            op_class=OpClass.SRAM,
            dest=[pred_reg_id],
            srcs=[],
        )
        save_ops.append((instr, addr, val))

    return save_ops


def emit_context_restore_instructions(scheduler, warp_id: int):
    """
    Emit context-restore operations to load all 64 vector registers and 8 predicate
    registers for `warp_id` from its dedicated scratch buffer slot in SRAM,
    skipping registers with an active in-flight long-latency load instruction.
    
    Returns a list of (Instruction, address_vector).
    """
    from pgpu.arch.isa import Instruction, OpClass, OpCode
    from pgpu.arch.smsp import RegisterFile

    warp = scheduler.get_warp(warp_id) if hasattr(scheduler, 'get_warp') else scheduler.warps[warp_id]
    pending_long_lat = get_pending_long_lat_dests(warp) | getattr(warp, "saved_skipped_regs", set())
    restore_ops = []

    # 1. Vector registers R0..R63
    for r in range(RegisterFile.NUM_VECTOR_REGS):
        if r in pending_long_lat:
            continue
        addr = get_vector_scratch_addr(warp_id, r)
        instr = Instruction(
            opcode=OpCode.SRAM_LD,
            op_class=OpClass.SRAM,
            dest=[r],
            srcs=[],
        )
        restore_ops.append((instr, addr))

    # 2. Predicate registers P0..P7
    for p in range(RegisterFile.NUM_PRED_REGS):
        pred_reg_id = RegisterFile.PRED_BASE_ID + p
        if pred_reg_id in pending_long_lat:
            continue
        addr = get_pred_scratch_addr(warp_id, p)
        instr = Instruction(
            opcode=OpCode.SRAM_LD,
            op_class=OpClass.SRAM,
            dest=[pred_reg_id],
            srcs=[],
        )
        restore_ops.append((instr, addr))

    return restore_ops


def apply_context_switch(scheduler, old_warp_id: int, new_warp_id: int) -> int:
    """
    Perform warp context switch by saving all vector and predicate registers
    of `old_warp_id` to the dedicated scratch buffer and restoring registers of
    `new_warp_id` from the scratch buffer, skipping registers that have an
    active in-flight long-latency load instruction.
    
    Returns the total cycles elapsed across the SRAM save and restore operations.
    """
    if old_warp_id == new_warp_id:
        return 0

    from pgpu.arch.smsp import RegisterFile
    from pgpu.arch.memory import MemoryAllocation

    if scheduler.sram is not None and "__context_save__" not in scheduler.sram.allocations:
        scheduler.sram.allocations["__context_save__"] = MemoryAllocation(
            name="__context_save__",
            base_addr=CONTEXT_SAVE_BASE_ADDR,
            size=16 * WARP_CONTEXT_WORDS,
            mem_space="sram",
        )

    start_cycle = scheduler.current_cycle
    curr_time = scheduler.current_cycle

    # 1. Context Save: store registers of old_warp_id to SRAM
    save_ops = emit_context_save_instructions(scheduler, old_warp_id)
    for instr, addrs, vals in save_ops:
        curr_time = max(curr_time, scheduler.sram.next_available_store_cycle)
        state, comp, tokens = scheduler.sram.store(
            addrs, vals, current_cycle=curr_time, warp_id=old_warp_id
        )
        if tokens:
            scheduler.sram.InstrRetireRoutine(old_warp_id, tokens)
        curr_time = comp

    # 2. Context Restore: load registers of new_warp_id from SRAM
    warp_in = scheduler.get_warp(new_warp_id) if hasattr(scheduler, 'get_warp') else scheduler.warps[new_warp_id]
    restore_ops = emit_context_restore_instructions(scheduler, new_warp_id)
    for instr, addrs in restore_ops:
        curr_time = max(curr_time, scheduler.sram.next_available_load_cycle)
        state, comp, tokens, results = scheduler.sram.load(
            addrs, current_cycle=curr_time, warp_id=new_warp_id
        )
        if tokens:
            scheduler.sram.InstrRetireRoutine(new_warp_id, tokens)
        if results is not None:
            reg_id = instr.dest[0]
            if reg_id < RegisterFile.NUM_VECTOR_REGS:
                warp_in.regs[reg_id] = results
            elif RegisterFile.PRED_BASE_ID <= reg_id < RegisterFile.PRED_BASE_ID + RegisterFile.NUM_PRED_REGS:
                pred_idx = reg_id - RegisterFile.PRED_BASE_ID
                warp_in.preds[pred_idx] = (results != 0.0)
        curr_time = comp

    if hasattr(warp_in, "saved_skipped_regs"):
        warp_in.saved_skipped_regs.clear()

    total_cost = max(curr_time - start_cycle, 0)
    return total_cost


# Backwards compatibility alias
apply_context_switch_cost = apply_context_switch
