"""
Defines SM Sub-Partition (SMSP) hardware state tokens, register file structures, and scoreboard entries:
- WarpState(Enum): Hardware warp state signals routed by resources and the scoreboard decoder.
- RegScoreboardEntry: Register scoreboard token tracking completion_time and producer_resource.
- RegisterFile: Hardware register file namespace, capacity limits, and special registers.
- VirtualReg / VirtualRegTuple / VirtualPred: Opaque handles to physical vector/predicate registers.
- RegisterAllocator: Per-warp circular next-fit register allocator tracking live registers.
- RegScope: Automatic lifetime context manager for scoped register allocation.

Author: Arjun Vadakkeveedu (arjunmenonv@alumni.iitm.ac.in)
September 2026
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pgpu.arch.gpu_device import AbstractResource


class WarpState(Enum):
    """
    Hardware warp state signals emitted by the scoreboard decoder and memory/execution resources.
    """
    WARP_READY = 0
    WARP_LONG_LAT_STALL = 1
    WARP_SRAM_STALL = 2
    WARP_COMPUTE_STALL = 3
    WARP_BARRIER_STALL = 4
    WARP_YIELD = 5
    WARP_COMPLETED = 6
    WARP_SYNC_STALL = 7


@dataclass
class RegScoreboardEntry:
    """
    Tracks an in-flight write to a destination register `(warp_id, reg_id)`.

    Deviation from a real GPU: no read scoreboards maintained by the SMSP. why?
    
    Holding a reference to `producer_resource` allows the SMSP scoreboard decoder
    to emit the exact stall signal (`producer_resource.stall_signal`:
      - `WarpState.WARP_LONG_LAT_STALL` when waiting on a `DRAMResource` load,
      - `WarpState.WARP_SRAM_STALL` when waiting on an `SRAMResource` load,
      - `WarpState.WARP_COMPUTE_STALL` when waiting on an `ALU`/`MMA`/`SFU` op)
    at the point when a dependent instruction attempts to issue.
    """
    warp_id: int
    reg_id: int
    completion_time: int
    producer_resource: "AbstractResource"


# ============================================================================
# RegisterFile Architectural Namespace & Layout
# ============================================================================
class RegisterFile:
    """
    Defines the architectural register file namespace, capacity limits, and special registers.
    Now also owns the physical register file arrays for an SMSP.
    """
    NUM_VECTOR_REGS: int = 64        # R0 .. R63 -> IDs 0 .. 63
    RZ_ID: int = 64                  # Dedicated hardwired zero register RZ -> ID 64
    SR_LANEID_ID: int = 65           # Special read-only register: lane ID [0..31]
    SR_WARPID_ID: int = 66           # Special read-only register: warp ID [wid..wid]
    SR_NUMWARPS_ID: int = 67         # Special read-only register: total warps [num_warps..num_warps]
    PRED_BASE_ID: int = 68           # P0 .. P7  -> IDs 68 .. 75
    NUM_PRED_REGS: int = 8           # 8 predicate registers per warp
    SIMD_WIDTH: int = 32             # 32 lanes per warp

    def __init__(self, num_warps: int = 4):
        self.num_warps = num_warps
        self.regs = np.zeros((num_warps, self.NUM_VECTOR_REGS, self.SIMD_WIDTH), dtype=np.float32)
        self.preds = np.zeros((num_warps, self.NUM_PRED_REGS, self.SIMD_WIDTH), dtype=bool)


# Module-level aliases for backwards compatibility
NUM_VECTOR_REGS = RegisterFile.NUM_VECTOR_REGS
RZ_ID = RegisterFile.RZ_ID
SR_LANEID_ID = RegisterFile.SR_LANEID_ID
SR_WARPID_ID = RegisterFile.SR_WARPID_ID
SR_NUMWARPS_ID = RegisterFile.SR_NUMWARPS_ID
PRED_BASE_ID = RegisterFile.PRED_BASE_ID
NUM_PRED_REGS = RegisterFile.NUM_PRED_REGS
SIMD_WIDTH = RegisterFile.SIMD_WIDTH


# ============================================================================
# Virtual Register Handles & Circular Next-Fit Register Allocator
# ============================================================================
class RegisterSpillError(RuntimeError):
    """Raised when a warp attempts to allocate more than 64 live vector registers (or 8 predicates)."""
    pass


class VirtualReg:
    """
    Opaque handle to a physical vector register (R0..R63, RZ=64, SR_LANEID=65, SR_WARPID=66, SR_NUMWARPS=67).
    """
    def __init__(
        self,
        phys_id: int,
        allocator: Optional["RegisterAllocator"] = None,
        name: Optional[str] = None,
    ):
        if not (0 <= phys_id <= RegisterFile.SR_NUMWARPS_ID):
            raise ValueError(
                f"Invalid physical vector/special register ID {phys_id}; expected 0..{RegisterFile.SR_NUMWARPS_ID}."
            )
        self.id: int = int(phys_id)
        self._allocator = allocator
        self.name = name
        self._freed: bool = False

    def free(self) -> None:
        """Return this register to the allocator in place."""
        if self.id in (
            RegisterFile.RZ_ID,
            RegisterFile.SR_LANEID_ID,
            RegisterFile.SR_WARPID_ID,
            RegisterFile.SR_NUMWARPS_ID,
        ):
            return
        if not self._freed and self._allocator is not None:
            self._allocator.free(self)
            self._freed = True

    def __repr__(self) -> str:
        if self.id == RegisterFile.RZ_ID:
            return "RZ"
        if self.id == RegisterFile.SR_LANEID_ID:
            return "SR_LANEID"
        if self.id == RegisterFile.SR_WARPID_ID:
            return "SR_WARPID"
        if self.id == RegisterFile.SR_NUMWARPS_ID:
            return "SR_NUMWARPS"
        return f"R{self.id}" if not self.name else f"R{self.id}({self.name})"


class VirtualRegTuple:
    """
    Ordered tuple of `VirtualReg` handles (e.g. 4 registers for an 8x16 GEMMCORE operand
    or 2 registers for an 8x8 accumulator).
    
    The underlying physical registers in `self.regs` do NOT need to be contiguous in R0..R63.
    """
    def __init__(
        self,
        regs: Sequence[VirtualReg],
        allocator: Optional["RegisterAllocator"] = None,
        name: Optional[str] = None,
    ):
        self.regs: List[VirtualReg] = list(regs)
        self.count: int = len(self.regs)
        self._allocator = allocator
        self.name = name
        self._freed: bool = False

    def __getitem__(self, idx: int) -> VirtualReg:
        return self.regs[idx]

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(self.regs)

    @property
    def ids(self) -> List[int]:
        return [r.id for r in self.regs]

    def free(self) -> None:
        """Free all constituent VirtualRegs in place."""
        if not self._freed:
            if self._allocator is not None:
                self._allocator.free_tuple(self)
            else:
                for r in self.regs:
                    r.free()
            self._freed = True

    def __repr__(self) -> str:
        inner = ", ".join(f"R{r.id}" for r in self.regs)
        return f"VirtualRegTuple([{inner}])"


class VirtualPred:
    """
    Opaque handle to a physical predicate register (P0..P7).
    Supports bitwise negation `~p` to express inverted guard predication (`@!P`).
    """
    def __init__(
        self,
        pred_idx: int,
        allocator: Optional["RegisterAllocator"] = None,
        inverted: bool = False,
        name: Optional[str] = None,
    ):
        if not (0 <= pred_idx < RegisterFile.NUM_PRED_REGS):
            raise ValueError(
                f"Invalid predicate index {pred_idx}; expected 0..{RegisterFile.NUM_PRED_REGS - 1}."
            )
        self.pred_idx: int = int(pred_idx)
        self.inverted: bool = bool(inverted)
        self._allocator = allocator
        self.name = name
        self._freed: bool = False

    @property
    def id(self) -> int:
        """Unified register ID (68..75) for scoreboard and destination tracking."""
        return RegisterFile.PRED_BASE_ID + self.pred_idx

    def __invert__(self) -> "VirtualPred":
        """Return an inverted view (`@!P`) of this predicate register."""
        return VirtualPred(
            pred_idx=self.pred_idx,
            allocator=self._allocator,
            inverted=not self.inverted,
            name=self.name,
        )

    def free(self) -> None:
        if not self._freed and self._allocator is not None:
            self._allocator.free_pred(self)
            self._freed = True

    def __repr__(self) -> str:
        prefix = "@!P" if self.inverted else "@P"
        return f"{prefix}{self.pred_idx}"


class RegisterAllocator:
    """
    Per-warp circular next-fit register allocator.
    
    Allocates physical register numbers sequentially (`0 -> 1 -> ... -> 63`),
    freeing registers in place (`self.used[id] = False`, `self.num_alive -= 1`)
    when `.free()` is called. Once the pointer reaches `R63`, it wraps around to `R0`
    and scans forward for the next available free register hole. Raises `RegisterSpillError`
    if `self.num_alive == 64` (no register available).
    """
    TOTAL_REGISTERS = RegisterFile.NUM_VECTOR_REGS
    TOTAL_PREDS = RegisterFile.NUM_PRED_REGS

    def __init__(self):
        self.used: List[bool] = [False] * self.TOTAL_REGISTERS
        self.num_alive: int = 0
        self.next_reg_idx: int = 0

        self.pred_used: List[bool] = [False] * self.TOTAL_PREDS
        self.num_preds_alive: int = 0
        self.next_pred_idx: int = 0

        self._rz = VirtualReg(RegisterFile.RZ_ID, self, name="RZ")
        self._sr_laneid = VirtualReg(RegisterFile.SR_LANEID_ID, self, name="SR_LANEID")
        self._sr_warpid = VirtualReg(RegisterFile.SR_WARPID_ID, self, name="SR_WARPID")
        self._sr_numwarps = VirtualReg(RegisterFile.SR_NUMWARPS_ID, self, name="SR_NUMWARPS")

    @property
    def rz(self) -> VirtualReg:
        """Dedicated hardwired zero register (`RZ`)."""
        return self._rz

    @property
    def sr_laneid(self) -> VirtualReg:
        """Dedicated special register holding hardware lane ID [0..31]."""
        return self._sr_laneid

    @property
    def sr_warpid(self) -> VirtualReg:
        """Dedicated special register holding hardware warp ID [wid..wid]."""
        return self._sr_warpid

    @property
    def sr_numwarps(self) -> VirtualReg:
        """Dedicated special register holding total warps [num_warps..num_warps]."""
        return self._sr_numwarps

    def alloc(self, name: Optional[str] = None) -> VirtualReg:
        """Allocate the next available register starting from `self.next_reg_idx` (wrapping R63 -> R0)."""
        if self.num_alive >= self.TOTAL_REGISTERS:
            raise RegisterSpillError(
                "Out of registers! All 64 registers (R0..R63) are currently alive."
            )
        for offset in range(self.TOTAL_REGISTERS):
            candidate = (self.next_reg_idx + offset) % self.TOTAL_REGISTERS
            if not self.used[candidate]:
                self.used[candidate] = True
                self.num_alive += 1
                self.next_reg_idx = (candidate + 1) % self.TOTAL_REGISTERS
                return VirtualReg(candidate, self, name=name)
        raise RegisterSpillError("Out of registers! Limit of 64 registers exceeded.")

    def alloc_tuple(self, count: int, name: Optional[str] = None) -> VirtualRegTuple:
        """
        Allocate `count` registers into a `VirtualRegTuple`.
        Constituent registers do NOT need to be contiguous in R0..R63.
        """
        if count <= 0:
            raise ValueError(f"Tuple count must be positive, got {count}.")
        if self.num_alive + count > self.TOTAL_REGISTERS:
            raise RegisterSpillError(
                f"Cannot allocate VirtualRegTuple({count}): only "
                f"{self.TOTAL_REGISTERS - self.num_alive} free registers remaining."
            )
        regs = [
            self.alloc(name=f"{name}_{i}" if name else None)
            for i in range(count)
        ]
        return VirtualRegTuple(regs, self, name=name)

    def free(self, reg: VirtualReg) -> None:
        """Free a `VirtualReg` in place, creating a reusable hole without moving `next_reg_idx`."""
        if reg.id in (
            RegisterFile.RZ_ID,
            RegisterFile.SR_LANEID_ID,
            RegisterFile.SR_WARPID_ID,
            RegisterFile.SR_NUMWARPS_ID,
        ):
            return
        if not self.used[reg.id]:
            raise ValueError(f"Double-free or invalid free on R{reg.id}!")
        self.used[reg.id] = False
        self.num_alive -= 1
        reg._freed = True

    def free_tuple(self, reg_tuple: VirtualRegTuple) -> None:
        """Free all constituent `VirtualReg`s of a `VirtualRegTuple` in place."""
        for reg in reg_tuple.regs:
            if not reg._freed:
                self.free(reg)
        reg_tuple._freed = True

    def alloc_pred(self, name: Optional[str] = None) -> VirtualPred:
        """Allocate a predicate register (P0..P7) using circular next-fit."""
        if self.num_preds_alive >= self.TOTAL_PREDS:
            raise RegisterSpillError("Out of predicate registers! All 8 predicates (P0..P7) are alive.")
        for offset in range(self.TOTAL_PREDS):
            candidate = (self.next_pred_idx + offset) % self.TOTAL_PREDS
            if not self.pred_used[candidate]:
                self.pred_used[candidate] = True
                self.num_preds_alive += 1
                self.next_pred_idx = (candidate + 1) % self.TOTAL_PREDS
                return VirtualPred(candidate, self, name=name)
        raise RegisterSpillError("Out of predicate registers!")

    def free_pred(self, pred: VirtualPred) -> None:
        if not self.pred_used[pred.pred_idx]:
            raise ValueError(f"Double-free on P{pred.pred_idx}!")
        self.pred_used[pred.pred_idx] = False
        self.num_preds_alive -= 1
        pred._freed = True


class RegScope:
    """Context manager that automatically frees any registers/tuples/predicates allocated within its block."""
    def __init__(self, allocator: RegisterAllocator, name: Optional[str] = None, debug_hook: Optional[Any] = None):
        self.allocator = allocator
        self.name = name or "scope"
        self._debug_hook = debug_hook
        self.allocated: List[Union[VirtualReg, VirtualRegTuple, VirtualPred]] = []

    def __enter__(self) -> "RegScope":
        if self._debug_hook is not None:
            self._debug_hook(f"Enter {self.name}")
        return self

    def alloc(self, name: Optional[str] = None) -> VirtualReg:
        r = self.allocator.alloc(name=name)
        self.allocated.append(r)
        return r

    alloc_reg = alloc

    def alloc_tuple(self, count: int, name: Optional[str] = None) -> VirtualRegTuple:
        t = self.allocator.alloc_tuple(count, name=name)
        self.allocated.append(t)
        return t

    def alloc_pred(self, name: Optional[str] = None) -> VirtualPred:
        p = self.allocator.alloc_pred(name=name)
        self.allocated.append(p)
        return p

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._debug_hook is not None:
            self._debug_hook(f"Inside {self.name} (pre-free)")
        for item in self.allocated:
            item.free()
        if self._debug_hook is not None:
            self._debug_hook(f"Exit {self.name} (post-free)")


# ============================================================================
# SMSP Hardware Model & Round-Robin Warp Scheduler
# ============================================================================
from pgpu.sw.driver import apply_context_switch


@dataclass
class WarpContext:
    """
    Hardware execution context for a single warp on an SMSP.
    """
    warp_id: int
    pc: int = 0
    state: WarpState = WarpState.WARP_READY
    at_barrier: bool = False
    completed: bool = False
    # Local register file view: R0..R63
    regs: np.ndarray = field(init=False)
    # Local predicate file view: P0..P7
    preds: np.ndarray = field(init=False)
    # Active register scoreboard mapping reg_id -> RegScoreboardEntry
    scoreboard: Dict[int, RegScoreboardEntry] = field(default_factory=dict)
    # Registers skipped during context save due to in-flight long-latency operations
    saved_skipped_regs: Set[int] = field(default_factory=set)


class SMSP:
    """
    SM Sub-Partition (SMSP) Hardware Architecture & Warp Scheduler.
    
    Houses:
    - Up to 4 active warp contexts (`WarpContext`).
    - Dedicated execution units: VALUResource, GEMMCoreResource, TransResource.
    - Interfaces to SRAMResource (scratchpad) and DRAMResource (global memory).
    - Round-robin warp scheduler with analytical timestamp model, hazard scoreboarding,
      fast-forwarding on stalls, and context-switch penalty accounting.
    """
    MAX_WARPS: int = 4

    def __init__(
        self,
        num_warps: int = 4,
        smsp_id: int = 0,
        sm_id: int = 0,
        warp_id_offset: int = 0,
        total_warps: Optional[int] = None,
        dram: Optional["DRAMResource"] = None,
        sram: Optional["SRAMResource"] = None,
        valu: Optional["VALUResource"] = None,
        gemm: Optional["GEMMCoreResource"] = None,
        trans: Optional["TransResource"] = None,
    ):
        from pgpu.arch.valu import VALUResource, GEMMCoreResource, TransResource
        if not (1 <= num_warps <= self.MAX_WARPS):
            raise ValueError(f"num_warps must be between 1 and {self.MAX_WARPS}, got {num_warps}.")
        self.smsp_id = smsp_id
        self.sm_id = sm_id
        self.num_warps = num_warps
        self.warp_id_offset = warp_id_offset
        self.total_warps = total_warps if total_warps is not None else num_warps
        self.register_file = RegisterFile(num_warps=num_warps)

        self.warps: List[WarpContext] = []
        for w in range(num_warps):
            warp = WarpContext(warp_id=self.warp_id_offset + w)
            warp.regs = self.register_file.regs[w]
            warp.preds = self.register_file.preds[w]
            self.warps.append(warp)
            
        self._warp_map: Dict[int, WarpContext] = {w.warp_id: w for w in self.warps}

        # Memory resource pointers (decoupled from instantiation; wired via bind_memory or constructor)
        self.dram: Optional["DRAMResource"] = dram
        self.sram: Optional["SRAMResource"] = sram

        # Owned private compute execution pipelines
        self.valu = valu if valu is not None else VALUResource()
        self.gemm = gemm if gemm is not None else GEMMCoreResource()
        self.trans = trans if trans is not None else TransResource()

        # Timing and scheduler state
        self.current_cycle: int = 0
        self.last_active_warp: Optional[int] = None
        self.total_context_switch_cycles: int = 0
        self.context_switch_count: int = 0

        # In-flight instructions tracking: list of (warp_id, clone)
        self.in_flight_instructions: List[Tuple[int, "Instruction"]] = []

        # Optional instruction trace hook: called with (warp_id, clone_instr) after every issue
        # Set to a callable to record trace data; None by default (no overhead).
        self.trace_hook: Optional[Any] = None

    def bind_memory(
        self,
        dram: Optional["DRAMResource"] = None,
        sram: Optional["SRAMResource"] = None,
    ) -> None:
        """Bind external DRAMResource and SRAMResource pointers to this SMSP."""
        if dram is not None:
            self.dram = dram
        if sram is not None:
            self.sram = sram

    def set_warp_offset(self, warp_id_offset: int, total_warps: int) -> None:
        """Update global warp_id offset and total_warps count when attached to an SM/pGPU."""
        self.warp_id_offset = warp_id_offset
        self.total_warps = total_warps
        for idx, w in enumerate(self.warps):
            w.warp_id = warp_id_offset + idx
        self._warp_map = {w.warp_id: w for w in self.warps}

    def get_warp(self, warp_id: int) -> WarpContext:
        """Retrieve WarpContext by its global or local warp_id."""
        if warp_id in self._warp_map:
            return self._warp_map[warp_id]
        return self.warps[warp_id - self.warp_id_offset]

    def reset(self):
        """Reset SMSP compute state, register files, and execution pipelines (does not wipe DRAM/SRAM)."""
        self.current_cycle = 0
        self.last_active_warp = None
        self.total_context_switch_cycles = 0
        self.context_switch_count = 0
        self.in_flight_instructions.clear()
        self.register_file.regs.fill(0)
        self.register_file.preds.fill(0)
        self.warps = []
        for w in range(self.num_warps):
            warp = WarpContext(warp_id=self.warp_id_offset + w)
            warp.regs = self.register_file.regs[w]
            warp.preds = self.register_file.preds[w]
            self.warps.append(warp)
        self._warp_map = {w.warp_id: w for w in self.warps}
        self.valu.reset()
        self.gemm.reset()
        self.trans.reset()

    def retire_completed_instructions(self) -> None:
        """
        Scan all in-flight instructions across resources. Any instruction whose
        `completion_time <= current_cycle` is retired: committing writes to `warp.regs`/`warp.preds`,
        popping scoreboard entries, and triggering resource retire routines (LSQ token cleanup).
        """
        still_in_flight: List[Tuple[int, "Instruction"]] = []
        for wid, clone in self.in_flight_instructions:
            if clone.completion_time is not None and clone.completion_time <= self.current_cycle:
                warp = self.get_warp(wid)
                clone.retire(self.current_cycle, warp_regs=warp.regs, warp_preds=warp.preds)
                # Remove dest entries from scoreboard
                for d in clone.dest:
                    if d in warp.scoreboard and warp.scoreboard[d].completion_time <= self.current_cycle:
                        del warp.scoreboard[d]
            else:
                still_in_flight.append((wid, clone))
        self.in_flight_instructions = still_in_flight

    def decode_warp_state(self, warp: WarpContext, program: List["Instruction"]) -> Tuple[WarpState, Optional[int]]:
        from pgpu.arch.isa import OpClass, OpCode
        """
        Query scoreboards, resources, and barrier state for the given warp,
        emitting (WarpState, earliest_unblock_cycle).
        """
        if warp.completed or warp.pc >= len(program):
            return WarpState.WARP_COMPLETED, None

        if warp.at_barrier:
            return WarpState.WARP_BARRIER_STALL, None

        instr = program[warp.pc]

        if instr.opcode == OpCode.YIELD:
            return WarpState.WARP_YIELD, self.current_cycle

        if instr.opcode == OpCode.SYNC:
            return WarpState.WARP_BARRIER_STALL, self.current_cycle

        if instr.opcode == OpCode.SYNC_WARP:
            # Memory fence: Wait for all pending loads and stores for this warp to complete
            pending_mem = [
                clone for wid, clone in self.in_flight_instructions
                if wid == warp.warp_id and clone.op_class in (OpClass.DRAM, OpClass.SRAM)
            ]
            if pending_mem:
                unblock = max(
                    (c.completion_time for c in pending_mem if c.completion_time is not None),
                    default=self.current_cycle
                )
                if unblock > self.current_cycle:
                    # Treat memory fence stall identically to a long-latency memory load stall
                    return WarpState.WARP_LONG_LAT_STALL, unblock
            
            # Fence resolved: all memory ops complete. It proceeds as a 1-cycle no-op.
            return WarpState.WARP_READY, self.current_cycle

        # 1. Scoreboard RAW Dependency Check (srcs and guard pred)
        stalling_entries: List[RegScoreboardEntry] = []
        for s in instr.srcs:
            if s in warp.scoreboard:
                stalling_entries.append(warp.scoreboard[s])
        if instr.pred is not None:
            pred_reg_id = RegisterFile.PRED_BASE_ID + instr.pred
            if pred_reg_id in warp.scoreboard:
                stalling_entries.append(warp.scoreboard[pred_reg_id])

        # 2. Scoreboard WAW Dependency Check (dest)
        for d in instr.dest:
            if d in warp.scoreboard:
                stalling_entries.append(warp.scoreboard[d])

        if stalling_entries:
            unblock_time = max(e.completion_time for e in stalling_entries)
            # If ANY blocking entry comes from DRAM or has WARP_LONG_LAT_STALL, route that signal
            if any(e.producer_resource.stall_signal == WarpState.WARP_LONG_LAT_STALL for e in stalling_entries):
                return WarpState.WARP_LONG_LAT_STALL, unblock_time
            if any(e.producer_resource.stall_signal == WarpState.WARP_SRAM_STALL for e in stalling_entries):
                return WarpState.WARP_SRAM_STALL, unblock_time
            return WarpState.WARP_COMPUTE_STALL, unblock_time

        # 3. Structural Resource Availability Check
        res = self._get_resource_for_instruction(instr)
        if res is not None:
            from pgpu.arch.memory import MemoryResource, MemOpType
            from pgpu.arch.valu import _read_reg
            if isinstance(res, MemoryResource):
                mask = instr.active_lane_mask(warp.preds)
                addr_vec = _read_reg(warp.regs, instr.srcs[0], warp_id=warp.warp_id, num_warps=self.total_warps).astype(np.int64)
                op_t = MemOpType.LOAD if instr.opcode in (OpCode.DRAM_LD, OpCode.SRAM_LD) else MemOpType.STORE
                mem_state, mem_unblock = res.check_ready(
                    op_type=op_t,
                    addresses=addr_vec,
                    current_cycle=self.current_cycle,
                    warp_id=warp.warp_id,
                    mask=mask,
                )
                if mem_state != WarpState.WARP_READY:
                    return mem_state, mem_unblock
            elif not res.is_available(self.current_cycle):
                return res.stall_signal, res.next_available_cycle

        return WarpState.WARP_READY, self.current_cycle

    def _get_resource_for_instruction(self, instr: "Instruction") -> Optional["AbstractResource"]:
        from pgpu.arch.isa import OpClass, OpCode
        """Return the target hardware execution or memory resource for `instr`."""
        match instr.op_class:
            case OpClass.VALU:
                return self.valu
            case OpClass.GEMMCORE:
                return self.gemm
            case OpClass.TRANS:
                return self.trans
            case OpClass.SRAM:
                if self.sram is None:
                    raise RuntimeError("SRAMResource is not bound to this SMSP.")
                return self.sram
            case OpClass.DRAM:
                if self.dram is None:
                    raise RuntimeError("DRAMResource is not bound to this SMSP.")
                return self.dram
            case _:
                return None

    def issue(self, warp: WarpContext, instr: "Instruction") -> int:
        from pgpu.arch.isa import OpClass, OpCode
        from pgpu.arch.memory import MemoryResource
        """
        Issue an instruction for `warp` at `self.current_cycle`.
        Returns completion_cycle.
        """
        clone = instr.clone_for_issue(issue_time=self.current_cycle, warp_id=warp.warp_id)
        completion_cycle = self.current_cycle + 1

        match instr.op_class:
            case OpClass.VALU:
                _, comp, _ = self.valu.issue(
                    clone, warp.regs, warp.preds, current_cycle=self.current_cycle, warp_id=warp.warp_id, num_warps=self.total_warps
                )
                completion_cycle = comp
            case OpClass.GEMMCORE:
                _, comp, _ = self.gemm.issue(
                    clone, warp.regs, warp.preds, current_cycle=self.current_cycle, warp_id=warp.warp_id, num_warps=self.total_warps
                )
                completion_cycle = comp
            case OpClass.TRANS:
                _, comp, _ = self.trans.issue(
                    clone, warp.regs, warp.preds, current_cycle=self.current_cycle, warp_id=warp.warp_id, num_warps=self.total_warps
                )
                completion_cycle = comp
            case OpClass.DRAM | OpClass.SRAM:
                res: MemoryResource = self.dram if instr.op_class == OpClass.DRAM else self.sram
                mask = clone.active_lane_mask(warp.preds)
                # Address vector read from src 0 (for LD) or src 1 (for ST)
                from pgpu.arch.valu import _read_reg
                if instr.opcode in (OpCode.DRAM_LD, OpCode.SRAM_LD):
                    addr_vec = _read_reg(warp.regs, instr.srcs[0], warp_id=warp.warp_id, num_warps=self.total_warps).astype(np.int64)
                    _, comp, tokens, results = res.load(
                        addr_vec, current_cycle=self.current_cycle, warp_id=warp.warp_id, mask=mask
                    )
                    clone.mem_tokens = tokens if tokens is not None else []
                    clone.completion_time = comp
                    if results is not None and clone.dest:
                        clone.pending_writes = [(clone.dest[0], results, mask)]
                    completion_cycle = comp
                else:
                    # STORE: dest=[], srcs=[src_addr, src_val]
                    addr_vec = _read_reg(warp.regs, instr.srcs[0], warp_id=warp.warp_id, num_warps=self.total_warps).astype(np.int64)
                    val_vec = _read_reg(warp.regs, instr.srcs[1], warp_id=warp.warp_id, num_warps=self.total_warps)
                    _, comp, tokens = res.store(
                        addr_vec, val_vec, current_cycle=self.current_cycle, warp_id=warp.warp_id, mask=mask
                    )
                    clone.mem_tokens = tokens if tokens is not None else []
                    clone.completion_time = comp
                    completion_cycle = comp
                if res not in clone.resources:
                    clone.resources.append(res)

        # Populate scoreboard entries for destination registers
        producer_res = self._get_resource_for_instruction(instr)
        if producer_res is not None:
            for d in clone.dest:
                entry = RegScoreboardEntry(
                    warp_id=warp.warp_id,
                    reg_id=d,
                    completion_time=completion_cycle,
                    producer_resource=producer_res,
                )
                warp.scoreboard[d] = entry

        if clone.completion_time is None:
            clone.completion_time = completion_cycle
        self.in_flight_instructions.append((warp.warp_id, clone))

        # Fire the optional instruction trace hook
        if self.trace_hook is not None:
            self.trace_hook(warp.warp_id, warp.pc, clone, self.current_cycle)

        # Advance PC or branch
        if instr.opcode == OpCode.JMP and clone.branch_taken:
            warp.pc = int(clone.branch_target)
        else:
            warp.pc += 1

        return completion_cycle

    def handle_sync(self, warp: WarpContext, active_warps: List[WarpContext]) -> None:
        """
        Thread-block barrier: marks warp as arrived. When all active warps arrive,
        releases the barrier by advancing clock to the maximum in-flight completion time.
        """
        warp.at_barrier = True
        if all(w.at_barrier or w.completed for w in active_warps):
            # All warps reached barrier -> release
            max_comp = self.current_cycle
            for wid, clone in self.in_flight_instructions:
                if clone.completion_time is not None and clone.completion_time > max_comp:
                    max_comp = clone.completion_time
            self.current_cycle = max_comp
            self.retire_completed_instructions()
            for w in active_warps:
                w.at_barrier = False

    def run(self, program: Union[List["Instruction"], Any]) -> int:
        """
        Execute the SPMD `program` (or `WarpKernel`) across all warps using round-robin scheduling.
        Returns total simulation cycles (`self.current_cycle`).
        """
        if hasattr(program, "get_program"):
            program = program.get_program()
        if hasattr(program, "sram_interface") and program.sram_interface is not None and self.sram is not None:
            program.sram_interface.apply_to_sram(self.sram)

        active_warps = list(self.warps)
        warp_ptr = 0

        while active_warps:
            self.retire_completed_instructions()
            warp = active_warps[warp_ptr]

            state, unblock_time = self.decode_warp_state(warp, program)

            if state == WarpState.WARP_COMPLETED:
                warp.completed = True
                active_warps.remove(warp)
                if not active_warps:
                    break
                warp_ptr %= len(active_warps)
                continue

            if state == WarpState.WARP_BARRIER_STALL:
                if not warp.at_barrier:
                    warp.pc += 1
                self.handle_sync(warp, active_warps)
                warp_ptr = (warp_ptr + 1) % len(active_warps)
                next_wid = active_warps[warp_ptr].warp_id
                if self.last_active_warp is not None and self.last_active_warp != next_wid:
                    cost = apply_context_switch(self, self.last_active_warp, next_wid)
                    self.current_cycle += cost
                    self.total_context_switch_cycles += cost
                    self.context_switch_count += 1
                self.last_active_warp = next_wid
                continue

            if state == WarpState.WARP_YIELD:
                warp.pc += 1
                # Trigger warp switch
                warp_ptr = (warp_ptr + 1) % len(active_warps)
                next_wid = active_warps[warp_ptr].warp_id
                if self.last_active_warp is not None and self.last_active_warp != next_wid:
                    cost = apply_context_switch(self, self.last_active_warp, next_wid)
                    self.current_cycle += cost
                    self.total_context_switch_cycles += cost
                    self.context_switch_count += 1
                self.last_active_warp = next_wid
                continue

            if state == WarpState.WARP_LONG_LAT_STALL:
                # Trigger warp switch
                warp_ptr = (warp_ptr + 1) % len(active_warps)
                next_wid = active_warps[warp_ptr].warp_id
                if self.last_active_warp is not None and self.last_active_warp != next_wid:
                    cost = apply_context_switch(self, self.last_active_warp, next_wid)
                    self.current_cycle += cost
                    self.total_context_switch_cycles += cost
                    self.context_switch_count += 1
                self.last_active_warp = next_wid

                # Check if all schedulable warps are stalled
                schedulable = [w for w in active_warps if not w.at_barrier]
                all_stalled = True
                earliest_unblock = float("inf")
                for w in schedulable:
                    st, ub = self.decode_warp_state(w, program)
                    if st == WarpState.WARP_READY:
                        all_stalled = False
                        break
                    if ub is not None and ub > self.current_cycle:
                        earliest_unblock = min(earliest_unblock, ub)

                if all_stalled and earliest_unblock != float("inf"):
                    self.current_cycle = int(earliest_unblock)
                    self.retire_completed_instructions()
                continue

            if state in (WarpState.WARP_SRAM_STALL, WarpState.WARP_COMPUTE_STALL):
                # Spin-wait in place to avoid SRAM deadlock and unnecessary context overhead
                if unblock_time is not None and unblock_time > self.current_cycle:
                    self.current_cycle = unblock_time
                    self.retire_completed_instructions()
                continue

            if state == WarpState.WARP_READY:
                # Context switch penalty if previous warp completed (no context switch performed)
                if self.last_active_warp is not None and self.last_active_warp != warp.warp_id:
                    cost = apply_context_switch(self, self.last_active_warp, warp.warp_id)
                    self.current_cycle += cost
                    self.total_context_switch_cycles += cost
                    self.context_switch_count += 1
                self.last_active_warp = warp.warp_id

                # Issue instruction
                instr = program[warp.pc]
                if self.trace_hook is not None:
                    # In trace mode: catch DRAM/SRAM bounds errors so we still record
                    # the instruction stream even when running with dummy allocations.
                    from pgpu.arch.memory import MemoryAccessError
                    try:
                        self.issue(warp, instr)
                    except MemoryAccessError:
                        warp.pc += 1
                else:
                    self.issue(warp, instr)

                self.current_cycle += 1
                continue

        # Final drain of any remaining in-flight instructions
        if self.in_flight_instructions:
            max_comp = max(c.completion_time for _, c in self.in_flight_instructions if c.completion_time is not None)
            if max_comp > self.current_cycle:
                self.current_cycle = max_comp
            self.retire_completed_instructions()

        return self.current_cycle
