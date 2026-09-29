"""
Kernel Programming Interface, Assembly Parser, and Kernel Builders for pGPU.

Provides software-level kernel authoring, assembling, and building abstractions:
- assemble_instruction: Structural match-case mnemonic parser
- KernelProgram / CompiledKernel: Container for instruction streams and memory interfaces
- SRAMVirtualReg: Symbolic register handle for in-kernel SRAM allocations
- _KernelSRAMInterface: Static shared memory manager for kernels
- WarpKernel: High-level SPMD kernel builder
- build_kernel: Factory compiler API for dynamic kernel arguments

Author: Arjun Vadakkeveedu (arjunmenonv@alumni.iitm.ac.in)
September 2026
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np

from pgpu.arch.isa import Instruction, OpClass, OpCode
from pgpu.arch.smsp import (
    RegisterFile,
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

if TYPE_CHECKING:
    from pgpu.arch.memory import MemoryAllocation


# ============================================================================
# Architectural Parameter Buffer in DRAM
# ============================================================================
PARAM_BUFFER_BASE: int = 0


# ============================================================================
# Operand Unwrapping Helpers
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


# ============================================================================
# Assembly Mnemonic Parser (match-case)
# ============================================================================
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

        # --- VALU Binary Arithmetic & Logic ---
        case "add" | "sub" | "mul" | "div" | "min" | "max" | "shl" | "shr" | "and" | "or" | "xor":
            if len(operands) != 3:
                raise ValueError(f"{op_tok} expects (dest, src1, src2), got {len(operands)} operands.")
            bin_map = {
                "add": OpCode.ADD, "sub": OpCode.SUB, "mul": OpCode.MUL, "div": OpCode.DIV,
                "min": OpCode.MIN, "max": OpCode.MAX, "shl": OpCode.SHL, "shr": OpCode.SHR,
                "and": OpCode.AND, "or": OpCode.OR, "xor": OpCode.XOR,
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
        case "fma":
            if len(operands) != 4:
                raise ValueError(f"fma expects (dest, a, b, c), got {len(operands)} operands.")
            return Instruction(
                opcode=OpCode.FMA,
                op_class=OpClass.VALU,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[_unwrap_reg_id(operands[1]), _unwrap_reg_id(operands[2]), _unwrap_reg_id(operands[3])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )
        case "select":
            if len(operands) != 4:
                raise ValueError(f"select expects (dest, cond, true_val, false_val), got {len(operands)} operands.")
            return Instruction(
                opcode=OpCode.SELECT,
                op_class=OpClass.VALU,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[_unwrap_reg_id(operands[1]), _unwrap_reg_id(operands[2]), _unwrap_reg_id(operands[3])],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- VALU Predicate Comparison (writes P0..P7) ---
        case "cmp.lt" | "cmp.eq":
            if len(operands) != 3:
                raise ValueError(f"{op_tok} expects (dest_pred, src1, src2), got {len(operands)} operands.")
            p_dest = operands[0]
            if isinstance(p_dest, VirtualPred):
                pred_dest_id = p_dest.id
            elif isinstance(p_dest, int):
                pred_dest_id = PRED_BASE_ID + p_dest if p_dest < PRED_BASE_ID else p_dest
            else:
                pred_dest_id = _unwrap_reg_id(p_dest, allow_pred=True)
            cmp_op = OpCode.CMP_LT if op_tok == "cmp.lt" else OpCode.CMP_EQ
            return Instruction(
                opcode=cmp_op,
                op_class=OpClass.VALU,
                dest=[pred_dest_id],
                srcs=[_unwrap_reg_id(operands[1]), _unwrap_reg_id(operands[2])],
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

        # --- VALU Immediate (SET_IMM) ---
        case "set.imm":
            if len(operands) != 2:
                raise ValueError(f"set.imm expects (dest, imm_value), got {len(operands)} operands.")
            return Instruction(
                opcode=OpCode.SET_IMM,
                op_class=OpClass.VALU,
                dest=[_unwrap_reg_id(operands[0])],
                srcs=[],
                imm=float(operands[1]),
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- Control Flow (JMP) ---
        case "jmp":
            if len(operands) != 1:
                raise ValueError(f"jmp expects (target_label_or_pc,), got {len(operands)} operands.")
            target = operands[0]
            imm_val = int(target) if isinstance(target, (int, np.integer)) else str(target)
            return Instruction(
                opcode=OpCode.JMP,
                op_class=OpClass.VALU,
                dest=[],
                srcs=[],
                imm=imm_val,
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        # --- GEMMCORE Class: MMA & MMRED ---
        case "avx.mma" | "avx.mmred.max" | "avx.mmred.min":
            if len(operands) != 4:
                raise ValueError(
                    f"{op_tok} expects 4 VirtualRegTuple operands (d_tuple, a_tuple, b_tuple, c_tuple), "
                    f"got {len(operands)}."
                )
            gemm_map = {
                "avx.mma": OpCode.AVX_MMA,
                "avx.mmred.max": OpCode.AVX_MMRED_MAX,
                "avx.mmred.min": OpCode.AVX_MMRED_MIN,
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

        # --- TRANS Transcendental Functions ---
        case "exp.trans" | "sin.trans" | "cos.trans" | "tanh.trans" | "log.trans":
            if len(operands) != 2:
                raise ValueError(f"{op_tok} expects (dest, src), got {len(operands)} operands.")
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

        # --- TRANS Fast Taylor Approximations ---
        case "avx.exp.fast" | "avx.sin.fast" | "avx.cos.fast" | "avx.tanh.fast" | "avx.log.fast":
            if len(operands) != 3:
                raise ValueError(f"{op_tok} expects (dest, src_x, src_n), got {len(operands)} operands.")
            fast_map = {
                "avx.exp.fast": OpCode.AVX_EXP_FAST,
                "avx.sin.fast": OpCode.AVX_SIN_FAST,
                "avx.cos.fast": OpCode.AVX_COS_FAST,
                "avx.tanh.fast": OpCode.AVX_TANH_FAST,
                "avx.log.fast": OpCode.AVX_LOG_FAST,
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
        case "sync":
            return Instruction(
                opcode=OpCode.SYNC,
                op_class=OpClass.SYNC,
                dest=[],
                srcs=[],
                pred=pred_idx,
                pred_inv=pred_inv,
            )
        case "sync.warp":
            return Instruction(
                opcode=OpCode.SYNC_WARP,
                op_class=OpClass.SYNC,
                dest=[],
                srcs=[],
                pred=pred_idx,
                pred_inv=pred_inv,
            )
        case "yield":
            return Instruction(
                opcode=OpCode.YIELD,
                op_class=OpClass.SYNC,
                dest=[],
                srcs=[],
                pred=pred_idx,
                pred_inv=pred_inv,
            )

        case _:
            raise ValueError(f"Unknown assembly instruction mnemonic: {mnemonic!r}")


# ============================================================================
# Kernel Program Containers & SRAM Interface
# ============================================================================
class KernelProgram(list):
    """
    A list of assembled `Instruction` objects that also carries the static
    shared memory (`sram.alloc` / `sram.free`) descriptors declared inside `WarpKernel`.
    """
    def __init__(self, instructions: Sequence[Instruction], sram_interface: Optional["_KernelSRAMInterface"] = None, trace_regions: Optional[Dict[str, Tuple[int, int]]] = None):
        super().__init__(instructions)
        self.sram_interface = sram_interface
        self.trace_regions = trace_regions or {}

class CompiledKernel(KernelProgram):
    """
    Executable compiled kernel program with formal argument and shared memory metadata.
    """
    def __init__(
        self,
        instructions: Sequence[Instruction],
        sram_interface: Optional["_KernelSRAMInterface"] = None,
        arg_names: Optional[List[str]] = None,
        sram_bindings: Optional[List[Tuple[str, "VirtualReg"]]] = None,
        trace_regions: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        super().__init__(instructions, sram_interface=sram_interface, trace_regions=trace_regions)
        self.arg_names: List[str] = list(arg_names) if arg_names is not None else []
        self.sram_bindings: List[Tuple[str, "VirtualReg"]] = list(sram_bindings) if sram_bindings is not None else []



class SRAMVirtualReg(VirtualReg):
    """
    VirtualReg representing an in-kernel SRAM allocation.
    Acts as a standard VirtualReg handle for instructions while retaining .allocation, .base_addr, and .size.
    """
    def __init__(self, phys_id: int, allocator: "RegisterAllocator", allocation: Any):
        super().__init__(phys_id, allocator, name=f"sram_{allocation.name}")
        self.allocation = allocation
        self.base_addr = allocation.base_addr
        self.size = allocation.size


class _KernelSRAMInterface:
    """
    In-kernel static shared memory manager (`kb.sram.alloc` / `kb.sram.free`).
    Delegates allocation and deallocation directly to `SRAMResource.alloc` / `SRAMResource.free`.
    """

    def __init__(self, kernel: "WarpKernel"):
        from pgpu.arch.memory import SRAMResource
        from pgpu.sw.driver import WARP_CONTEXT_WORDS
        self._kernel = kernel
        self._dummy_sram = SRAMResource(name="StaticSRAM")
        self.all_allocations: List[Any] = []
        self.sram_bindings: List[Tuple[str, Any]] = []
        
        # Driver init: reserve space for context save region before any user allocation.
        # This ensures it uses the standard allocation path and raises OOM if space runs out.
        ctx_alloc = self._dummy_sram.alloc(
            name="__context_save__", 
            size=self._kernel.num_warps * WARP_CONTEXT_WORDS
        )
        self.all_allocations.append(ctx_alloc)

    @property
    def allocations(self) -> Dict[str, Any]:
        return self._dummy_sram.allocations

    @property
    def allocated_words(self) -> int:
        return self._dummy_sram.allocated_words

    @property
    def _next_alloc_base(self) -> int:
        return self._dummy_sram._next_alloc_base

    def alloc(
        self,
        name: str,
        size: int,
        index_modifier: Optional[Any] = None,
        reg: Optional[VirtualReg] = None,
    ) -> Any:
        """Allocate `size` 32-bit words of static shared memory via `SRAMResource.alloc`."""
        allocation = self._dummy_sram.alloc(name=name, size=size, index_modifier=index_modifier)
        self.all_allocations.append(allocation)

        if getattr(self._kernel, "_is_building", False):
            vreg = reg if reg is not None else self._kernel.alloc_reg(name=f"sram_{name}")
            sram_reg = SRAMVirtualReg(vreg.id, self._kernel.allocator, allocation)
            self.sram_bindings.append((name, sram_reg))

            # Emit dynamic parameter load from PARAM_BUFFER
            slot_idx = self._kernel._next_param_slot()
            if slot_idx == 0:
                self._kernel.asm("dram.ld", sram_reg, self._kernel.rz)
            else:
                with self._kernel.reg_scope() as scope:
                    r_addr = scope.alloc(name=f"sram_param_{name}_addr")
                    self._kernel.asm("set.imm", r_addr, float(slot_idx))
                    self._kernel.asm("dram.ld", sram_reg, r_addr)
            return sram_reg

        return allocation

    def free(self, name: str) -> None:
        """Free a static shared memory allocation via `SRAMResource.free`."""
        self._dummy_sram.free(name=name)

    def apply_to_sram(self, sram: Any) -> None:
        """Register static shared memory allocations on an SM's SRAMResource prior to execution."""
        if sram is None:
            return
        for alloc in self.all_allocations:
            if alloc.name not in sram.allocations:
                sram.alloc(
                    name=alloc.name, 
                    size=alloc.size, 
                    index_modifier=alloc.index_modifier
                )
        sram.allocated_words = self._dummy_sram.allocated_words
        sram._next_alloc_base = self._dummy_sram._next_alloc_base


# ============================================================================
# SPMD WarpKernel Program Builder
# ============================================================================
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
        self.debug = debug
        self.instructions: List[Instruction] = []
        self.labels: Dict[str, int] = {}
        self.trace_regions: Dict[str, Tuple[int, int]] = {}
        self._open_trace_regions: Dict[str, int] = {}
        self.allocator: RegisterAllocator = RegisterAllocator()
        self.sram: _KernelSRAMInterface = _KernelSRAMInterface(self)
        self._internal_id_counter: int = 0
        self._scope_counter: int = 0
        self._is_building: bool = False
        self._param_slot_counter: int = 0
        if self.debug:
            print("=== WarpKernel Register Liveness Trace (debug=True) ===")
            self._print_live_regs("Kernel start")

    def _next_param_slot(self) -> int:
        slot = self._param_slot_counter
        self._param_slot_counter += 1
        return slot

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

    def sram_alloc(
        self,
        name: str,
        size: int,
        index_modifier: Optional[Any] = None,
        reg: Optional[VirtualReg] = None,
    ) -> Any:
        """Allocate static shared memory in SRAM from inside the kernel."""
        return self.sram.alloc(name=name, size=size, index_modifier=index_modifier, reg=reg)

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

    def alloc_reg(self, name: Optional[str] = None) -> VirtualReg:
        """Allocate a single general-purpose vector register (`R0..R63`)."""
        return self.allocator.alloc(name=name)

    def alloc_regs(self, count: int, name: Optional[str] = None) -> List[VirtualReg]:
        """Allocate a list of `count` independent general-purpose registers."""
        return [self.allocator.alloc(name=f"{name}_{i}" if name else None) for i in range(count)]

    def alloc_tuple(self, count: int, name: Optional[str] = None) -> VirtualRegTuple:
        """Allocate an ordered tuple of `count` registers."""
        return self.allocator.alloc_tuple(count=count, name=name)

    def alloc_pred(self, name: Optional[str] = None) -> VirtualPred:
        """Allocate a predicate register (`P0..P7`)."""
        return self.allocator.alloc_pred(name=name)

    def free_reg(self, reg: VirtualReg) -> None:
        """Manually free a `VirtualReg`."""
        reg.free()

    def free_tuple(self, reg_tuple: VirtualRegTuple) -> None:
        """Manually free a `VirtualRegTuple`."""
        reg_tuple.free()

    def free_pred(self, pred: VirtualPred) -> None:
        """Manually free a `VirtualPred`."""
        pred.free()

    def reg_scope(self, name: Optional[str] = None) -> RegScope:
        """
        Create a RAII context manager scope for automatic register reclamation.
        All registers/predicates allocated within this block are automatically freed on exit.
        """
        self._scope_counter += 1
        scope_name = name or f"scope_{self._scope_counter}"
        hook = self._print_live_regs if self.debug else None
        return RegScope(self.allocator, name=scope_name, debug_hook=hook)

    def label(self, label_name: str) -> int:
        """Define a jump label at the current instruction PC."""
        pc = len(self.instructions)
        if label_name in self.labels:
            raise ValueError(f"Duplicate jump label {label_name!r} at PC {pc}.")
        self.labels[label_name] = pc
        return pc

    def trace_start(self, region_name: str) -> int:
        """Define the entry point of a trace region at the current instruction PC."""
        pc = len(self.instructions)
        self._open_trace_regions[region_name] = pc
        return pc

    def trace_end(self, region_name: str) -> int:
        """Define the exit point of a trace region (inclusive of the last emitted instruction)."""
        if region_name not in self._open_trace_regions:
            raise ValueError(f"Trace region {region_name!r} ended without being started.")
        start_pc = self._open_trace_regions.pop(region_name)
        end_pc = max(start_pc, len(self.instructions) - 1)
        self.trace_regions[region_name] = (start_pc, end_pc)
        return end_pc

    def asm(
        self,
        mnemonic: str,
        *operands: Any,
        pred: Optional[Union[VirtualPred, int]] = None,
        pred_inv: bool = False,
    ) -> Instruction:
        """Assemble a mnemonic + VirtualReg/VirtualRegTuple operands and append to program."""
        instr = assemble_instruction(mnemonic, *operands, pred=pred, pred_inv=pred_inv)
        self.instructions.append(instr)
        return instr


    def get_program(self) -> KernelProgram:
        """Resolve any symbolic jump labels to integer PCs and return the unified program stream."""
        for instr in self.instructions:
            if instr.opcode == OpCode.JMP and isinstance(instr.imm, str):
                if instr.imm not in self.labels:
                    raise KeyError(f"Unresolved jump label {instr.imm!r}.")
                instr.imm = self.labels[instr.imm]
        return KernelProgram(self.instructions, sram_interface=self.sram)


# ============================================================================
# Kernel Factory & Compiler API
# ============================================================================
def build_kernel(
    kernel_fn: Callable[..., None],
    num_warps: int = 4,
    debug: bool = False,
) -> CompiledKernel:
    """
    Compile a high-level user kernel function into an executable CompiledKernel.
    
    1. Inspects `kernel_fn` signature to discover kernel argument names.
    2. Allocates symbolic VirtualReg inputs for each argument.
    3. Emits prologue instructions loading each argument from the DRAM parameter buffer.
    4. Executes `kernel_fn(kb, *arg_regs)` to populate the body.
    5. Emits epilogue instructions (barrier synchronization).
    6. Resolves jump labels and returns an executable `CompiledKernel`.
    """
    sig = inspect.signature(kernel_fn)
    params = list(sig.parameters.values())
    if len(params) < 1:
        raise ValueError("Kernel function must take at least one argument (the WarpKernel builder instance).")

    arg_names = [p.name for p in params[1:]]
    kb = WarpKernel(num_warps=num_warps, debug=debug)
    kb._is_building = True
    kb._param_slot_counter = 0

    # Allocate VirtualReg for each kernel argument
    arg_regs = [kb.alloc_reg(name=name) for name in arg_names]

    # Emit Prologue: load arguments from PARAM_BUFFER in DRAM
    for i, reg in enumerate(arg_regs):
        slot_addr = kb._next_param_slot()
        if slot_addr == 0:
            kb.asm("dram.ld", reg, kb.rz)
        else:
            with kb.reg_scope() as scope:
                r_addr = scope.alloc(name=f"param_addr_{i}")
                kb.asm("set.imm", r_addr, float(slot_addr))
                kb.asm("dram.ld", reg, r_addr)

    # Execute user kernel body
    kernel_fn(kb, *arg_regs)

    # Emit Epilogue: thread-block barrier
    kb.asm("sync")

    # Finalize jump labels
    for instr in kb.instructions:
        if instr.opcode == OpCode.JMP and isinstance(instr.imm, str):
            if instr.imm not in kb.labels:
                raise KeyError(f"Unresolved jump label {instr.imm!r}.")
            instr.imm = kb.labels[instr.imm]

    compiled = CompiledKernel(
        instructions=kb.instructions,
        sram_interface=kb.sram,
        arg_names=arg_names,
        sram_bindings=list(kb.sram.sram_bindings),
        trace_regions=kb.trace_regions,
    )
    return compiled
