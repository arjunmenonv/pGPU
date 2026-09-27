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
    MMA = 50
    MMRED_MAX = 51
    MMRED_MIN = 52

    # --- TRANS Class ---
    EXP_TRANS = 60
    EXP_FAST = 61
    SIN_TRANS = 62
    SIN_FAST = 63
    COS_TRANS = 64
    COS_FAST = 65
    TANH_TRANS = 66
    TANH_FAST = 67
    LOG_TRANS = 68
    LOG_FAST = 69

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
    OpCode.MMA: OpClass.GEMMCORE,
    OpCode.MMRED_MAX: OpClass.GEMMCORE,
    OpCode.MMRED_MIN: OpClass.GEMMCORE,
    OpCode.EXP_TRANS: OpClass.TRANS,
    OpCode.EXP_FAST: OpClass.TRANS,
    OpCode.SIN_TRANS: OpClass.TRANS,
    OpCode.SIN_FAST: OpClass.TRANS,
    OpCode.COS_TRANS: OpClass.TRANS,
    OpCode.COS_FAST: OpClass.TRANS,
    OpCode.TANH_TRANS: OpClass.TRANS,
    OpCode.TANH_FAST: OpClass.TRANS,
    OpCode.LOG_TRANS: OpClass.TRANS,
    OpCode.LOG_FAST: OpClass.TRANS,
    OpCode.SYNC: OpClass.SYNC,
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


# ============================================================================
# Assembly Mnemonic Parser (match-case) & WarpKernel Builder
# ============================================================================
def _unwrap_reg_id(operand: Union[VirtualReg, VirtualPred, int, str], allow_pred: bool = False) -> int:
    """Extract the unified integer register ID from a VirtualReg, VirtualPred, string, or int."""
    if isinstance(operand, VirtualReg):
        return operand.id
    if isinstance(operand, VirtualPred):
        if not allow_pred:
            raise TypeError(f"Predicate register {operand} is not valid in a vector source operand slot.")
        return operand.id
    if isinstance(operand, int):
        return operand
    if isinstance(operand, str):
        tok = operand.strip().upper()
        if tok == "RZ":
            return RZ_ID
        if tok in ("SR_LANEID", "LANE_ID", "LANEID", "SR_LANE_ID"):
            return SR_LANEID_ID
        if tok in ("SR_WARPID", "WARP_ID", "WARPID", "SR_WARP_ID"):
            return SR_WARPID_ID
        if tok in ("SR_NUMWARPS", "NUM_WARPS", "NUMWARPS", "SR_NUM_WARPS"):
            return SR_NUMWARPS_ID
        if tok.startswith("R") and tok[1:].isdigit():
            return int(tok[1:])
        if allow_pred and tok.startswith("P") and tok[1:].isdigit():
            return PRED_BASE_ID + int(tok[1:])
    raise TypeError(f"Unsupported register operand: {operand!r}")


def _unwrap_reg_list(
    operand: Union[VirtualRegTuple, Sequence[Union[VirtualReg, int, str]]],
    expected_len: int,
    role: str,
) -> List[int]:
    """Extract a list of register IDs from a VirtualRegTuple or sequence of registers."""
    if isinstance(operand, VirtualRegTuple):
        ids = operand.ids
    elif isinstance(operand, (list, tuple)):
        ids = [_unwrap_reg_id(r) for r in operand]
    else:
        raise TypeError(f"Operand {role} must be a VirtualRegTuple({expected_len}) or list of registers, got {operand!r}")
    if len(ids) != expected_len:
        raise ValueError(f"Operand {role} requires {expected_len} registers, got {len(ids)}.")
    return ids


def assemble_instruction(
    mnemonic: str,
    *operands: Any,
    pred: Optional[Union[VirtualPred, int]] = None,
    pred_inv: bool = False,
) -> Instruction:
    """
    Parse an assembly mnemonic string and operands into a static `Instruction` object
    using structural pattern matching (`match-case`).
    
    Supports optional inline guard predicates (e.g., `"@P0 add"`, `"@!P1 dram.st"`)
    or the `pred=` keyword argument (`pred=p0` or `pred=~p0`).
    """
    raw = mnemonic.strip()

    # 1. Parse optional inline predicate prefix: "@P0 ..." or "@!P0 ..."
    if raw.startswith("@"):
        parts = raw.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid predicated instruction string: {mnemonic!r}")
        pred_tok, raw = parts[0], parts[1].strip()
        if pred_tok.upper().startswith("@!P"):
            pred_idx = int(pred_tok[3:])
            pred_inv = True
        elif pred_tok.upper().startswith("@P"):
            pred_idx = int(pred_tok[2:])
            pred_inv = False
        else:
            raise ValueError(f"Invalid guard predicate prefix {pred_tok!r}; expected '@P<k>' or '@!P<k>'.")
    elif pred is not None:
        if isinstance(pred, VirtualPred):
            pred_idx = pred.pred_idx
            pred_inv = pred.inverted ^ pred_inv
        elif isinstance(pred, int):
            pred_idx = pred
        else:
            raise TypeError(f"Invalid pred argument: {pred!r}")
    else:
        pred_idx = None

    # Normalize mnemonic token: lowercase, convert underscores to dots for canonical matching
    op_tok = raw.lower().replace("_", ".")

    # 2. Match-case over canonical assembly mnemonics
    match op_tok:
        # --- DRAM Class ---
        case "dram.ld":
            if len(operands) != 2:
                raise ValueError(f"dram.ld expects (dest, addr), got {len(operands)} operands.")
            return Instruction(
                opcode=OpCode.DRAM_LD,
                op_class=OpClass.DRAM,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[_unwrap_reg_id(operands[1])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )
        case "dram.st":
            if len(operands) != 2:
                raise ValueError(f"dram.st expects (addr, data), got {len(operands)} operands.")
            return Instruction(
                opcode=OpCode.DRAM_ST,
                op_class=OpClass.DRAM,
                dest=[],
                srcs=[_unwrap_reg_id(operands[0]), _unwrap_reg_id(operands[1])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- SRAM Class ---
        case "sram.ld":
            if len(operands) != 2:
                raise ValueError(f"sram.ld expects (dest, addr), got {len(operands)} operands.")
            return Instruction(
                opcode=OpCode.SRAM_LD,
                op_class=OpClass.SRAM,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[_unwrap_reg_id(operands[1])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )
        case "sram.st":
            if len(operands) != 2:
                raise ValueError(f"sram.st expects (addr, data), got {len(operands)} operands.")
            return Instruction(
                opcode=OpCode.SRAM_ST,
                op_class=OpClass.SRAM,
                dest=[],
                srcs=[_unwrap_reg_id(operands[0]), _unwrap_reg_id(operands[1])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- VALU Binary Arithmetic / Logic / Shifts / Min / Max ---
        case "add" | "sub" | "mul" | "div" | "min" | "max" | "shl" | "shr" | "and" | "or" | "xor":
            if len(operands) != 3:
                raise ValueError(f"{op_tok} expects (dest, src_a, src_b), got {len(operands)} operands.")
            bin_map = {
                "add": OpCode.ADD,
                "sub": OpCode.SUB,
                "mul": OpCode.MUL,
                "div": OpCode.DIV,
                "min": OpCode.MIN,
                "max": OpCode.MAX,
                "shl": OpCode.SHL,
                "shr": OpCode.SHR,
                "and": OpCode.AND,
                "or": OpCode.OR,
                "xor": OpCode.XOR,
            }
            return Instruction(
                opcode=bin_map[op_tok],
                op_class=OpClass.VALU,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[_unwrap_reg_id(operands[1]), _unwrap_reg_id(operands[2])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- VALU Ternary (FMA, SELECT) ---
        case "fma" | "select":
            if len(operands) != 4:
                raise ValueError(f"{op_tok} expects 4 operands, got {len(operands)}.")
            opc = OpCode.FMA if op_tok == "fma" else OpCode.SELECT
            return Instruction(
                opcode=opc,
                op_class=OpClass.VALU,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[
                    _unwrap_reg_id(operands[1]),
                    _unwrap_reg_id(operands[2]),
                    _unwrap_reg_id(operands[3]),
                ],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- VALU Unary (NOT, MOV, BCAST) ---
        case "not" | "mov" | "bcast":
            if len(operands) != 2:
                raise ValueError(f"{op_tok} expects (dest, src), got {len(operands)} operands.")
            un_map = {"not": OpCode.NOT, "mov": OpCode.MOV, "bcast": OpCode.BCAST}
            return Instruction(
                opcode=un_map[op_tok],
                op_class=OpClass.VALU,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[_unwrap_reg_id(operands[1])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- VALU Comparisons (Destination can be R_d or P_d!) ---
        case "cmp.lt" | "cmp.eq":
            if len(operands) != 3:
                raise ValueError(f"{op_tok} expects (dest_reg_or_pred, src_a, src_b), got {len(operands)}.")
            opc = OpCode.CMP_LT if op_tok == "cmp.lt" else OpCode.CMP_EQ
            return Instruction(
                opcode=opc,
                op_class=OpClass.VALU,
                dest=[_unwrap_reg_id(operands[0], allow_pred=True)],
                srcs=[_unwrap_reg_id(operands[1]), _unwrap_reg_id(operands[2])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- VALU Immediate / Lane ID / Control Flow ---
        case "set.imm":
            if len(operands) != 2:
                raise ValueError(f"set.imm expects (dest, imm_value), got {len(operands)}.")
            return Instruction(
                opcode=OpCode.SET_IMM,
                op_class=OpClass.VALU,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[],
                imm=operands[1],
                pred=pred_idx,
                pred_inv=pred_inv,
            )
        case "jmp":
            if len(operands) != 1:
                raise ValueError(f"jmp expects (target_pc_or_label,), got {len(operands)}.")
            return Instruction(
                opcode=OpCode.JMP,
                op_class=OpClass.VALU,
                dest=[],
                srcs=[],
                imm=operands[0],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- GEMMCORE Class (MMA, MMRED_MAX, MMRED_MIN) ---
        case "mma" | "mmred.max" | "mmred.min":
            if len(operands) != 4:
                raise ValueError(
                    f"{op_tok} expects 4 VirtualRegTuple operands (d_tuple, a_tuple, b_tuple, c_tuple), "
                    f"got {len(operands)}."
                )
            gemm_map = {
                "mma": OpCode.MMA,
                "mmred.max": OpCode.MMRED_MAX,
                "mmred.min": OpCode.MMRED_MIN,
            }
            d_ids = _unwrap_reg_list(operands[0], 2, "D")
            a_ids = _unwrap_reg_list(operands[1], 4, "A")
            b_ids = _unwrap_reg_list(operands[2], 4, "B")
            c_ids = _unwrap_reg_list(operands[3], 2, "C")
            return Instruction(
                opcode=gemm_map[op_tok],
                op_class=OpClass.GEMMCORE,
                dest=d_ids,
                srcs=a_ids + b_ids + c_ids,
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- TRANS Class: Transcendental Mode (.trans) ---
        case "exp.trans" | "sin.trans" | "cos.trans" | "tanh.trans" | "log.trans":
            if len(operands) != 2:
                raise ValueError(f"{op_tok} expects (dest, src_x), got {len(operands)} operands.")
            trans_map = {
                "exp.trans": OpCode.EXP_TRANS,
                "sin.trans": OpCode.SIN_TRANS,
                "cos.trans": OpCode.COS_TRANS,
                "tanh.trans": OpCode.TANH_TRANS,
                "log.trans": OpCode.LOG_TRANS,
            }
            return Instruction(
                opcode=trans_map[op_tok],
                op_class=OpClass.TRANS,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[_unwrap_reg_id(operands[1])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- TRANS Class: Fast Taylor-Series Mode (.fast) ---
        case "exp.fast" | "sin.fast" | "cos.fast" | "tanh.fast" | "log.fast":
            if len(operands) != 3:
                raise ValueError(
                    f"{op_tok} expects (dest, src_x, src_n_reg), where src_n_reg holds n via SET_IMM; "
                    f"got {len(operands)} operands."
                )
            fast_map = {
                "exp.fast": OpCode.EXP_FAST,
                "sin.fast": OpCode.SIN_FAST,
                "cos.fast": OpCode.COS_FAST,
                "tanh.fast": OpCode.TANH_FAST,
                "log.fast": OpCode.LOG_FAST,
            }
            return Instruction(
                opcode=fast_map[op_tok],
                op_class=OpClass.TRANS,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[_unwrap_reg_id(operands[1]), _unwrap_reg_id(operands[2])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- SYNC Class ---
        case "sync" | "sync.warp" | "yield":
            if len(operands) != 0:
                raise ValueError(f"{op_tok} takes no register operands, got {len(operands)}.")
            sync_map = {
                "sync": OpCode.SYNC,
                "sync.warp": OpCode.SYNC_WARP,
                "yield": OpCode.YIELD,
            }
            return Instruction(
                opcode=sync_map[op_tok],
                op_class=OpClass.SYNC,
                dest=[],
                srcs=[],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        case _:
            raise ValueError(f"Unknown assembly instruction mnemonic: {mnemonic!r}")


class KernelProgram(list):
    """
    A list of assembled `Instruction` objects that also carries the static
    shared memory (`sram.alloc` / `sram.free`) descriptors declared inside `WarpKernel`.
    """
    def __init__(self, instructions: Sequence[Instruction], sram_interface: Optional["_KernelSRAMInterface"] = None):
        super().__init__(instructions)
        self.sram_interface = sram_interface


class _KernelSRAMInterface:
    """
    In-kernel static shared memory manager (`kb.sram.alloc` / `kb.sram.free`).
    Enforces static shared memory allocation from inside a WarpKernel (no host-side sram.copy).
    """
    USER_SRAM_CAPACITY_WORDS: int = 24 * 1024  # 24K words (96 KB) user region below context-save area

    def __init__(self, kernel: "WarpKernel"):
        self._kernel = kernel
        self.allocations: Dict[str, Any] = {}
        self.all_allocations: List[Any] = []
        self.allocated_words: int = 0
        self._next_alloc_base: int = 0

    def alloc(
        self,
        name: str,
        size: int,
        index_modifier: Optional[Any] = None,
    ) -> Any:
        """Allocate `size` 32-bit words of static shared memory in SRAM from inside the kernel."""
        from pgpu.arch.memory import MemoryAllocation, SRAMResource
        if name in self.allocations:
            raise ValueError(f"Static SRAM allocation '{name}' already exists in kernel.")
        if self.allocated_words + size > self.USER_SRAM_CAPACITY_WORDS:
            raise MemoryError(
                f"Static SRAM capacity exceeded. Requested {size} words, "
                f"available {self.USER_SRAM_CAPACITY_WORDS - self.allocated_words} words."
            )
        if index_modifier is not None and size > 0:
            SRAMResource.validate_bijection(name, size, index_modifier)

        aligned_base = (self._next_alloc_base + 31) & ~31
        base_addr = aligned_base % (1 << 32)
        self._next_alloc_base = (base_addr + size) % (1 << 32)
        allocation = MemoryAllocation(
            name=name,
            base_addr=base_addr,
            size=size,
            index_modifier=index_modifier,
            mem_space="sram",
        )
        self.allocations[name] = allocation
        self.all_allocations.append(allocation)
        self.allocated_words += size
        return allocation

    def free(self, name: str) -> None:
        """Free a static shared memory allocation from inside the kernel."""
        if name not in self.allocations:
            raise KeyError(f"Static SRAM allocation '{name}' not found in kernel.")
        allocation = self.allocations.pop(name)
        self.allocated_words -= allocation.size
        if not self.allocations:
            self._next_alloc_base = 0

    def apply_to_sram(self, sram: Any) -> None:
        """Register static shared memory allocations on an SM's SRAMResource prior to execution."""
        if sram is None:
            return
        for alloc in self.all_allocations:
            sram.allocations[alloc.name] = alloc
        sram.allocated_words = self.allocated_words
        sram._next_alloc_base = self._next_alloc_base


class WarpKernel:
    """
    SPMD Kernel Program Builder.
    
    The programmer writes a single straight-line assembly program representing the
    instruction stream executed across threads in a warp and across warps in the SM.
    Warp differentiation (address offsets, matrix partitioning) is achieved using
    hardware special registers (`kb.sr_warpid`, `kb.sr_laneid`, `kb.sr_numwarps`).
    """
    def __init__(self, num_warps: int = 4, debug: bool = False):
        self.num_warps = num_warps
        # TEMPORARY INTERNAL FEATURE: `debug` / `_print_live_regs` is for internal developer
        # verification only and must not be exposed as the user-facing API. The user-facing
        # interface will be `pgpu.devtools.objdump.objdump(kernel, reg_live_range=True)`
        # (nvdisasm-style per-line register live range table).
        self.debug = debug
        self.instructions: List[Instruction] = []
        self.labels: Dict[str, int] = {}
        self.allocator: RegisterAllocator = RegisterAllocator()
        self.sram: _KernelSRAMInterface = _KernelSRAMInterface(self)
        self._internal_id_counter: int = 0
        self._scope_counter: int = 0
        if self.debug:
            print("=== WarpKernel Register Liveness Trace (debug=True) ===")
            self._print_live_regs("Kernel start")

    def _print_live_regs(self, stage: str = "") -> None:
        """
        [Temporary Internal Helper] Print currently live physical vector and predicate
        registers when `self.debug` is True. To be replaced by `objdump(..., reg_live_range=True)`.
        """
        if not self.debug:
            return
        alloc = self.allocator
        live_vregs = [f"R{i}" for i, is_used in enumerate(alloc.used) if is_used]
        live_preds = [f"P{i}" for i, is_used in enumerate(alloc.pred_used) if is_used]
        print(
            f"  [{stage:<28}] "
            f"alive_vregs={alloc.num_alive:2d}/64 {live_vregs} | "
            f"alive_preds={alloc.num_preds_alive}/8 {live_preds} | "
            f"next_reg_idx=R{alloc.next_reg_idx}"
        )

    print_live_regs = _print_live_regs

    def _next_internal_id(self) -> int:
        val = self._internal_id_counter
        self._internal_id_counter += 1
        return val

    def sram_alloc(self, name: str, size: int, index_modifier: Optional[Any] = None) -> Any:
        """Allocate static shared memory in SRAM from inside the kernel."""
        return self.sram.alloc(name=name, size=size, index_modifier=index_modifier)

    def sram_free(self, name: str) -> None:
        """Free static shared memory in SRAM from inside the kernel."""
        self.sram.free(name=name)

    def memset(
        self,
        target: Any,
        value: float = 0.0,
        size: Optional[int] = None,
        mem_space: Optional[str] = None,
    ) -> List[Instruction]:
        """Emit an in-kernel parallel memset stream of stores into a DRAM or SRAM allocation."""
        from pgpu.sw.intrinsics import memset as _emit_memset
        return _emit_memset(self, target=target, value=value, size=size, mem_space=mem_space)

    @property
    def rz(self) -> VirtualReg:
        """Return the dedicated zero register `RZ`."""
        return self.allocator.rz

    @property
    def sr_laneid(self) -> VirtualReg:
        """Return the dedicated special register `SR_LANEID`."""
        return self.allocator.sr_laneid

    @property
    def sr_warpid(self) -> VirtualReg:
        """Return the dedicated special register `SR_WARPID`."""
        return self.allocator.sr_warpid

    @property
    def sr_numwarps(self) -> VirtualReg:
        """Return the dedicated special register `SR_NUMWARPS`."""
        return self.allocator.sr_numwarps

    def alloc_reg(self, *args: Any, **kwargs: Any) -> VirtualReg:
        """Allocate a virtual vector register. Accepts optional name."""
        name = kwargs.get("name", None)
        if args:
            if isinstance(args[0], int) and len(args) > 1:
                name = args[1]
            elif isinstance(args[0], str):
                name = args[0]
        return self.allocator.alloc(name=name)

    def alloc_tuple(self, *args: Any, **kwargs: Any) -> VirtualRegTuple:
        """Allocate a VirtualRegTuple. Accepts count and optional name."""
        name = kwargs.get("name", None)
        count = kwargs.get("count", None)
        if args:
            if isinstance(args[0], int) and len(args) >= 2 and isinstance(args[1], int):
                count = args[1]
                if len(args) > 2:
                    name = args[2]
            else:
                count = args[0]
                if len(args) > 1:
                    name = args[1]
        return self.allocator.alloc_tuple(count=count, name=name)

    def alloc_pred(self, *args: Any, **kwargs: Any) -> VirtualPred:
        """Allocate a virtual predicate register. Accepts optional name."""
        name = kwargs.get("name", None)
        if args:
            if isinstance(args[0], int) and len(args) > 1:
                name = args[1]
            elif isinstance(args[0], str):
                name = args[0]
        return self.allocator.alloc_pred(name=name)

    def reg_scope(self, name: Optional[str] = None) -> RegScope:
        """Return a RegScope context manager for scoped register allocation."""
        if name is None or isinstance(name, int):
            scope_name = f"scope_{self._scope_counter}"
            self._scope_counter += 1
        else:
            scope_name = str(name)
        return RegScope(
            self.allocator,
            name=scope_name,
            debug_hook=self._print_live_regs if self.debug else None,
        )

    def label(self, *args: Any) -> int:
        """Bind a symbolic jump label to the current instruction index. Accepts label_name."""
        label_name = args[1] if len(args) > 1 and isinstance(args[0], int) else args[0]
        pc = len(self.instructions)
        self.labels[label_name] = pc
        return pc

    def asm(
        self,
        *args: Any,
        pred: Optional[Union[VirtualPred, int]] = None,
        pred_inv: bool = False,
    ) -> Instruction:
        """Assemble a mnemonic + VirtualReg/VirtualRegTuple operands and append to program."""
        if args and isinstance(args[0], int):
            mnemonic = args[1]
            operands = args[2:]
        else:
            mnemonic = args[0]
            operands = args[1:]
        instr = assemble_instruction(mnemonic, *operands, pred=pred, pred_inv=pred_inv)
        self.instructions.append(instr)
        return instr

    def emit_fast_trans(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[Instruction]:
        """
        Intrinsic wrapper for `<func>.fast.<n>`: emits `set.imm r_n, n` followed by
        `<func>.fast dest, src_x, r_n` using a scoped temporary register.
        """
        if args and isinstance(args[0], int):
            func_name = args[1]
            dest = args[2]
            src_x = args[3]
            n = args[4] if len(args) > 4 else kwargs.get("n", DEFAULT_TAYLOR_N)
        else:
            func_name = args[0]
            dest = args[1]
            src_x = args[2]
            n = args[3] if len(args) > 3 else kwargs.get("n", DEFAULT_TAYLOR_N)
        pred = kwargs.get("pred", None)

        with self.reg_scope() as scope:
            r_n = scope.alloc(name=f"{func_name}_n")
            i_imm = self.asm("set.imm", r_n, int(n), pred=pred)
            i_op = self.asm(f"{func_name.lower()}.fast", dest, src_x, r_n, pred=pred)
            return [i_imm, i_op]

    def get_program(self) -> KernelProgram:
        """Resolve any symbolic jump labels to integer PCs and return the unified program stream."""
        for instr in self.instructions:
            if instr.opcode == OpCode.JMP and isinstance(instr.imm, str):
                if instr.imm not in self.labels:
                    raise KeyError(f"Unresolved jump label {instr.imm!r}.")
                instr.imm = self.labels[instr.imm]
        return KernelProgram(self.instructions, sram_interface=self.sram)
