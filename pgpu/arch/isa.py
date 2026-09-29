"""
Instruction Set Architecture (ISA) Specification, Tokens, Register Allocator, and Assembler.

Defines:
- OpClass(IntEnum): Hardware execution class tokens (DRAM, SRAM, VALU, GEMMCORE, TRANS, SYNC)
- OpCode(IntEnum): Instruction opcode tokens across all 6 classes
- InstrState(IntEnum): Instruction lifecycle tokens (PENDING, IN_FLIGHT, COMPLETE)
- VirtualReg, VirtualRegTuple, VirtualPred, RegisterAllocator, RegScope
- Instruction: Dataclass representing an instruction with Latch-at-Issue, Commit-at-Retire semantics
- assemble_instruction / WarpKernel: Assembly mnemonic parser (match-case) and kernel builder
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pgpu.arch.gpu_device import AbstractResource
    from pgpu.arch.memory import InFlightMemOp
    from pgpu.arch.smsp import RegScoreboardEntry


# Import register structures and namespace from SMSP (their architectural owner)
from pgpu.arch.smsp import (
    RegisterFile,
    RegisterSpillError,
    VirtualReg,
    VirtualRegTuple,
    VirtualPred,
    RegisterAllocator,
    RegScope,
    NUM_VECTOR_REGS,
    RZ_ID,
    SR_LANEID_ID,
    SR_WARPID_ID,
    SR_NUMWARPS_ID,
    PRED_BASE_ID,
    NUM_PRED_REGS,
    SIMD_WIDTH,
)

DEFAULT_TAYLOR_N: int = 64


# ============================================================================
# IntEnum Tokens for Opcode Classes, Opcodes, and Instruction States
# ============================================================================
class OpClass(Enum):
    """Hardware opcode classes mapping instructions to execution resources."""
    DRAM = 0
    SRAM = 1
    VALU = 2
    GEMMCORE = 3
    TRANS = 4
    SYNC = 5


class InstrState(Enum):
    """Lifecycle state tokens for an Instruction."""
    PENDING = 0
    IN_FLIGHT = 1
    COMPLETE = 2


class OpCode(Enum):
    """Unique integer tokens for all instructions in the pGPU ISA."""
    # --- DRAM Class ---
    DRAM_LD = 0
    DRAM_ST = 1

    # --- SRAM Class ---
    SRAM_LD = 10
    SRAM_ST = 11

    # --- VALU Class ---
    ADD = 20
    SUB = 21
    MUL = 22
    DIV = 23
    FMA = 24
    MIN = 25
    MAX = 26
    SHL = 27
    SHR = 28
    AND = 29
    OR = 30
    XOR = 31
    NOT = 32
    CMP_LT = 33
    CMP_EQ = 34
    SELECT = 35
    MOV = 36
    BCAST = 37
    SET_IMM = 38
    JMP = 40

    # --- GEMMCORE Class ---
    AVX_MMA = 50
    AVX_MMRED_MAX = 51
    AVX_MMRED_MIN = 52

    # --- TRANS Class ---
    EXP_TRANS = 60
    AVX_EXP_FAST = 61
    SIN_TRANS = 62
    AVX_SIN_FAST = 63
    COS_TRANS = 64
    AVX_COS_FAST = 65
    TANH_TRANS = 66
    AVX_TANH_FAST = 67
    LOG_TRANS = 68
    AVX_LOG_FAST = 69

    # --- SYNC Class ---
    SYNC = 80
    SYNC_WARP = 81
    YIELD = 82


OPCODE_TO_CLASS: Dict[OpCode, OpClass] = {
    OpCode.DRAM_LD: OpClass.DRAM,
    OpCode.DRAM_ST: OpClass.DRAM,
    OpCode.SRAM_LD: OpClass.SRAM,
    OpCode.SRAM_ST: OpClass.SRAM,
    OpCode.ADD: OpClass.VALU,
    OpCode.SUB: OpClass.VALU,
    OpCode.MUL: OpClass.VALU,
    OpCode.DIV: OpClass.VALU,
    OpCode.FMA: OpClass.VALU,
    OpCode.MIN: OpClass.VALU,
    OpCode.MAX: OpClass.VALU,
    OpCode.SHL: OpClass.VALU,
    OpCode.SHR: OpClass.VALU,
    OpCode.AND: OpClass.VALU,
    OpCode.OR: OpClass.VALU,
    OpCode.XOR: OpClass.VALU,
    OpCode.NOT: OpClass.VALU,
    OpCode.CMP_LT: OpClass.VALU,
    OpCode.CMP_EQ: OpClass.VALU,
    OpCode.SELECT: OpClass.VALU,
    OpCode.MOV: OpClass.VALU,
    OpCode.BCAST: OpClass.VALU,
    OpCode.SET_IMM: OpClass.VALU,
    OpCode.JMP: OpClass.VALU,
# AVX GEMMCORE Instructions
    OpCode.AVX_MMA: OpClass.GEMMCORE,           
    OpCode.AVX_MMRED_MAX: OpClass.GEMMCORE,
    OpCode.AVX_MMRED_MIN: OpClass.GEMMCORE,
    OpCode.EXP_TRANS: OpClass.TRANS,
    OpCode.SIN_TRANS: OpClass.TRANS,
    OpCode.COS_TRANS: OpClass.TRANS,
    OpCode.TANH_TRANS: OpClass.TRANS,
    OpCode.LOG_TRANS: OpClass.TRANS,
# AVX FAST MATH Instructions
    OpCode.AVX_EXP_FAST: OpClass.TRANS,
    OpCode.AVX_SIN_FAST: OpClass.TRANS,
    OpCode.AVX_COS_FAST: OpClass.TRANS,
    OpCode.AVX_TANH_FAST: OpClass.TRANS,
    OpCode.AVX_LOG_FAST: OpClass.TRANS,
    OpCode.SYNC: OpClass.SYNC,
    # SYNC_WARP is used for intra-warp sync, when there is memory dependency across threads in a warp. 
    # this instr waits until all threads complete their pending instructions, but does not necessarily wait for other warps to reach the same point.
    OpCode.SYNC_WARP: OpClass.SYNC,     
    OpCode.YIELD: OpClass.SYNC,
}


# ============================================================================
# Instruction Dataclass (Static Decode Fields + Dynamic Issue/Retire State)
# ============================================================================
@dataclass
class Instruction:
    """
    Represents a pGPU instruction.
    
    Static attributes (`opcode`, `op_class`, `dest`, `srcs`, `pred`, `pred_inv`, `imm`)
    are populated at assembly/parse time.
    Dynamic attributes (`state`, `issue_time`, `completion_time`, `warp_id`, `resources`,
    `scoreboard_entries`, `mem_tokens`, `pending_writes`, `branch_taken`, `branch_target`)
    are populated by the SMSP state machine at issue time and committed at retire time.
    """
    opcode: OpCode
    op_class: OpClass
    dest: List[int] = field(default_factory=list)
    srcs: List[int] = field(default_factory=list)
    pred: Optional[int] = None
    pred_inv: bool = False
    imm: Optional[Union[int, float, str]] = None

    # --- Dynamic Runtime State (Filled by SMSP at Issue, Committed at Retire) ---
    state: InstrState = InstrState.PENDING
    issue_time: Optional[int] = None
    completion_time: Optional[int] = None
    warp_id: Optional[int] = None
    resources: List["AbstractResource"] = field(default_factory=list)
    scoreboard_entries: List["RegScoreboardEntry"] = field(default_factory=list)
    mem_tokens: List[Optional["InFlightMemOp"]] = field(default_factory=list)
    # List of (reg_id, 32-element result array, 32-element boolean active lane mask)
    pending_writes: List[Tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    branch_taken: bool = False
    branch_target: Optional[int] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate operand arity and register ID bounds for the instruction opcode."""
        if not isinstance(self.opcode, OpCode):
            raise TypeError(f"opcode must be an OpCode IntEnum, got {type(self.opcode)}.")
        if not isinstance(self.op_class, OpClass):
            raise TypeError(f"op_class must be an OpClass IntEnum, got {type(self.op_class)}.")
        if OPCODE_TO_CLASS[self.opcode] != self.op_class:
            raise ValueError(
                f"OpCode {self.opcode.name} belongs to {OPCODE_TO_CLASS[self.opcode].name}, "
                f"not {self.op_class.name}."
            )
        if self.pred is not None and not (0 <= self.pred < NUM_PRED_REGS):
            raise ValueError(f"Invalid guard predicate index {self.pred}; expected 0..{NUM_PRED_REGS - 1}.")

        for d in self.dest:
            if not (0 <= d < PRED_BASE_ID + NUM_PRED_REGS):
                raise ValueError(f"Invalid destination register ID {d} in {self.opcode.name}.")
        for s in self.srcs:
            if not (0 <= s <= SR_NUMWARPS_ID):
                raise ValueError(f"Invalid source register ID {s} in {self.opcode.name}.")

        if self.op_class == OpClass.GEMMCORE:
            if len(self.dest) != 2 or len(self.srcs) != 10:
                raise ValueError(
                    f"{self.opcode.name} requires 2 dest regs (D) and 10 src regs (4 A + 4 B + 2 C), "
                    f"got dest={len(self.dest)}, srcs={len(self.srcs)}."
                )

    def clone_for_issue(self, warp_id: int, issue_time: int) -> "Instruction":
        """
        Return a fresh in-flight copy of this static instruction template for `warp_id`
        at `issue_time`, ensuring loops (`JMP`) and multi-warp execution never collide.
        """
        inst = copy.copy(self)
        inst.dest = list(self.dest)
        inst.srcs = list(self.srcs)
        inst.state = InstrState.IN_FLIGHT
        inst.warp_id = warp_id
        inst.issue_time = issue_time
        inst.completion_time = None
        inst.resources = []
        inst.scoreboard_entries = []
        inst.mem_tokens = []
        inst.pending_writes = []
        inst.branch_taken = False
        inst.branch_target = None
        return inst

    def active_lane_mask(self, warp_preds: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the 32-lane boolean active mask from the warp predicate file (`shape=(8, 32)`).
        """
        if self.pred is None or warp_preds is None:
            return np.ones(SIMD_WIDTH, dtype=bool)
        base_mask = np.asarray(warp_preds[self.pred], dtype=bool)
        return (~base_mask) if self.pred_inv else base_mask.copy()

    def retire(
        self,
        current_cycle: int,
        warp_regs: Optional[np.ndarray] = None,
        warp_preds: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Commit latched pending writes to `warp_regs` / `warp_preds`, clear scoreboard entries,
        and invoke `resource.InstrRetireRoutine` once `current_cycle >= self.completion_time`.
        """
        if self.state == InstrState.COMPLETE:
            return True
        if self.completion_time is None or current_cycle < self.completion_time:
            return False

        if warp_regs is not None:
            for reg_id, values, mask in self.pending_writes:
                if 0 <= reg_id < NUM_VECTOR_REGS:
                    warp_regs[reg_id, mask] = np.asarray(values, dtype=np.float32)[mask]
                elif reg_id == RZ_ID:
                    pass  # Writes to RZ are discarded
                elif PRED_BASE_ID <= reg_id < PRED_BASE_ID + NUM_PRED_REGS and warp_preds is not None:
                    pred_idx = reg_id - PRED_BASE_ID
                    warp_preds[pred_idx, mask] = np.asarray(values)[mask].astype(bool)

        self.pending_writes.clear()
        self.state = InstrState.COMPLETE
        self.scoreboard_entries.clear()
        for resource in self.resources:
            resource.InstrRetireRoutine(self.warp_id, self.mem_tokens)
        return True

    def disassemble(self) -> str:
        """Format instruction into canonical human-readable assembly syntax."""
        pred_prefix = ""
        if self.pred is not None:
            pred_prefix = f"@!P{self.pred} " if self.pred_inv else f"@P{self.pred} "

        def _fmt_reg(r: int) -> str:
            if r == RZ_ID:
                return "RZ"
            if r == SR_LANEID_ID:
                return "SR_LANEID"
            if r == SR_WARPID_ID:
                return "SR_WARPID"
            if r == SR_NUMWARPS_ID:
                return "SR_NUMWARPS"
            if r >= PRED_BASE_ID:
                return f"P{r - PRED_BASE_ID}"
            return f"R{r}"

        mnemonic = self.opcode.name.lower().replace("_", ".")
        if self.op_class == OpClass.GEMMCORE:
            d_str = "[" + ", ".join(_fmt_reg(r) for r in self.dest) + "]"
            a_str = "[" + ", ".join(_fmt_reg(r) for r in self.srcs[0:4]) + "]"
            b_str = "[" + ", ".join(_fmt_reg(r) for r in self.srcs[4:8]) + "]"
            c_str = "[" + ", ".join(_fmt_reg(r) for r in self.srcs[8:10]) + "]"
            return f"{pred_prefix}{mnemonic} {d_str}, {a_str}, {b_str}, {c_str}"

        parts: List[str] = []
        for d in self.dest:
            parts.append(_fmt_reg(d))
        for s in self.srcs:
            parts.append(_fmt_reg(s))
        if self.imm is not None:
            parts.append(str(self.imm))

        operands_str = ", ".join(parts)
        return f"{pred_prefix}{mnemonic}" + (f" {operands_str}" if operands_str else "")

    def __repr__(self) -> str:
        return f"Instruction('{self.disassemble()}', state={self.state.name})"


