"""
Compute Execution Resources for the pGPU Simulator:
- VALUResource: Vector ALU & Control unit (1 cycle, DIV = 4 cycles; non-pipelined)
- GEMMCoreResource: 8x16x8 Tensor Core unit for MMA, MMRED_MAX, MMRED_MIN (8 cycles; non-pipelined)
- TransResource: Transcendental (24 cycles) & Taylor-series Fast Math unit (non-pipelined)

All compute resources enforce:
1. Non-pipelined execution within each unit (`next_available_cycle` port reservation;
   co-schedulable across independent units `VALU`, `GEMMCORE`, `TRANS` in SMSP).
2. Latch Inputs at Issue, Commit Outputs at Retire (`instr.pending_writes` committed
   inside `instr.retire(current_cycle, warp_regs, warp_preds)`).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from pgpu.arch.gpu_device import AbstractResource
from pgpu.arch.isa import (
    Instruction,
    OpClass,
    OpCode,
    RegisterFile,
    NUM_VECTOR_REGS,
    RZ_ID,
    SR_LANEID_ID,
    SR_WARPID_ID,
    SR_NUMWARPS_ID,
    SIMD_WIDTH,
)
from pgpu.arch.smsp import WarpState


def _read_reg(warp_regs: np.ndarray, reg_id: int, warp_id: int = 0, num_warps: int = 4) -> np.ndarray:
    """
    Latch a 32-lane vector register (`R0..R63`) or dedicated special registers
    (`RZ`=64, `SR_LANEID`=65, `SR_WARPID`=66, `SR_NUMWARPS`=67).
    """
    if reg_id == RZ_ID:
        return np.zeros(SIMD_WIDTH, dtype=np.float32)
    if reg_id == SR_LANEID_ID:
        return np.arange(SIMD_WIDTH, dtype=np.float32)
    if reg_id == SR_WARPID_ID:
        return np.full(SIMD_WIDTH, float(warp_id), dtype=np.float32)
    if reg_id == SR_NUMWARPS_ID:
        return np.full(SIMD_WIDTH, float(num_warps), dtype=np.float32)
    if 0 <= reg_id < NUM_VECTOR_REGS:
        return np.asarray(warp_regs[reg_id], dtype=np.float32).copy()
    raise ValueError(f"Invalid vector/special source register ID {reg_id}.")


# ============================================================================
# Taylor-Series Fast Math Implementations (Overflow-Safe float64 Recurrences)
# ============================================================================
VALID_TAYLOR_TERMS = (4, 8, 16, 32, 64, 128)


def exp_taylor(x: np.ndarray, n: int = 64) -> np.ndarray:
    """n-term Taylor series for exp(x) = sum_{k=0}^{n-1} x^k / k!"""
    x64 = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x64, dtype=np.float64)
    term = np.ones_like(x64, dtype=np.float64)
    for k in range(1, n):
        term = term * (x64 / k)
        out = out + term
    return out.astype(np.float32)


def sin_taylor(x: np.ndarray, n: int = 64) -> np.ndarray:
    """n-term Taylor series for sin(x) = sum_{k=0}^{n-1} (-1)^k x^{2k+1} / (2k+1)!"""
    x64 = np.asarray(x, dtype=np.float64)
    x_sq = x64 * x64
    out = x64.copy()
    term = x64.copy()
    for k in range(1, n):
        term = -term * x_sq / ((2 * k) * (2 * k + 1))
        out = out + term
    return out.astype(np.float32)


def cos_taylor(x: np.ndarray, n: int = 64) -> np.ndarray:
    """n-term Taylor series for cos(x) = sum_{k=0}^{n-1} (-1)^k x^{2k} / (2k)!"""
    x64 = np.asarray(x, dtype=np.float64)
    x_sq = x64 * x64
    out = np.ones_like(x64, dtype=np.float64)
    term = np.ones_like(x64, dtype=np.float64)
    for k in range(1, n):
        term = -term * x_sq / ((2 * k - 1) * (2 * k))
        out = out + term
    return out.astype(np.float32)


def tanh_taylor(x: np.ndarray, n: int = 64) -> np.ndarray:
    """
    n-term Taylor series ratio for tanh(x) = sinh_taylor_n(x) / cosh_taylor_n(x),
    where sinh(x) = sum_{k=0}^{n-1} x^{2k+1}/(2k+1)! and cosh(x) = sum_{k=0}^{n-1} x^{2k}/(2k)!.
    """
    x64 = np.asarray(x, dtype=np.float64)
    x_sq = x64 * x64

    sinh_out = x64.copy()
    sinh_term = x64.copy()

    cosh_out = np.ones_like(x64, dtype=np.float64)
    cosh_term = np.ones_like(x64, dtype=np.float64)

    for k in range(1, n):
        sinh_term = sinh_term * x_sq / ((2 * k) * (2 * k + 1))
        sinh_out = sinh_out + sinh_term

        cosh_term = cosh_term * x_sq / ((2 * k - 1) * (2 * k))
        cosh_out = cosh_out + cosh_term

    return (sinh_out / cosh_out).astype(np.float32)


def log_taylor(x: np.ndarray, n: int = 64) -> np.ndarray:
    """
    n-term hyperbolic-artanh Taylor expansion for log(x) (convergent for all x > 0):
    log(x) = 2 * sum_{k=0}^{n-1} (1 / (2k + 1)) * ((x - 1) / (x + 1))^{2k + 1}
    """
    x64 = np.asarray(x, dtype=np.float64)
    z = (x64 - 1.0) / (x64 + 1.0)
    z_sq = z * z
    power = z.copy()
    out = power.copy()
    for k in range(1, n):
        power = power * z_sq
        out = out + (power / (2 * k + 1))
    return (2.0 * out).astype(np.float32)


# ============================================================================
# Base Single-Port Non-Pipelined Compute Resource
# ============================================================================
class ComputeResource(AbstractResource):
    """
    Base class for single-port compute execution units (VALU, GEMMCore, Trans)
    that track availability via `next_available_cycle`.
    """
    def __init__(self, name: str, base_latency: int = 1, stall_signal: WarpState = WarpState.WARP_COMPUTE_STALL):
        super().__init__(name=name, base_latency=base_latency, stall_signal=stall_signal)
        self.next_available_cycle: int = 0

    def is_available(self, current_cycle: int) -> bool:
        """Check if the compute unit port is available at current_cycle."""
        return current_cycle >= self.next_available_cycle

    def get_actual_start(self, issue_cycle: int) -> int:
        """Return the earliest cycle when an instruction can issue on this compute unit."""
        return max(issue_cycle, self.next_available_cycle)

    def reserve(self, issue_cycle: int, duration: int) -> int:
        """Reserve the compute unit port for an operation starting at issue_cycle."""
        actual_start = self.get_actual_start(issue_cycle)
        self.next_available_cycle = actual_start + duration
        return actual_start

    def reset(self):
        """Reset compute unit port state for a new simulation run."""
        super().reset()
        self.next_available_cycle = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', base_lat={self.base_latency}, next_avail={self.next_available_cycle})"


# ============================================================================
# 1. VALUResource (Non-Pipelined; 1 cycle for all ops except DIV = 4 cycles)
# ============================================================================
class VALUResource(ComputeResource):
    """
    Vector Arithmetic & Logic Unit (`OpClass.VALU`).
    
    - Latency: 1 cycle for all instructions except `OpCode.DIV` (4 cycles).
    - Non-pipelined: cannot accept a new VALU instruction while occupied (`current_cycle < next_available_cycle`).
    - Co-schedulable with `GEMMCoreResource` and `TransResource`.
    """
    DIV_LATENCY: int = 4

    def __init__(self, name: str = "VALU", base_latency: int = 1, div_latency: int = DIV_LATENCY):
        super().__init__(
            name=name,
            base_latency=base_latency,
            stall_signal=WarpState.WARP_COMPUTE_STALL,
        )
        self.div_latency = div_latency

    def compute_latency(self, instr: Instruction) -> int:
        if instr.opcode == OpCode.DIV:
            return self.div_latency
        return self.base_latency

    def issue(
        self,
        instr: Instruction,
        warp_regs: np.ndarray,
        warp_preds: Optional[np.ndarray],
        current_cycle: int,
        warp_id: int = 0,
        num_warps: int = 4,
    ) -> Tuple[WarpState, int, List[Tuple[int, np.ndarray, np.ndarray]]]:
        """
        Attempt to issue `instr` on the non-pipelined VALU resource at `current_cycle`.
        If the unit is occupied (`current_cycle < self.next_available_cycle`), returns
        `(WarpState.WARP_COMPUTE_STALL, self.next_available_cycle, [])`.
        Otherwise latches source registers, computes results into `instr.pending_writes`,
        and returns `(WarpState.WARP_READY, completion_cycle, instr.pending_writes)`.
        """
        if instr.op_class != OpClass.VALU:
            raise ValueError(f"VALUResource cannot execute {instr.opcode.name} ({instr.op_class.name}).")

        if not self.is_available(current_cycle):
            return self.stall_signal, self.next_available_cycle, []

        latency = self.compute_latency(instr)
        self.reserve(current_cycle, latency)
        completion_cycle = current_cycle + latency

        mask = instr.active_lane_mask(warp_preds)
        wid = getattr(instr, "warp_id", None) if (warp_id == 0 and getattr(instr, "warp_id", None) is not None) else warp_id
        src_vals = [_read_reg(warp_regs, r, warp_id=wid, num_warps=num_warps) for r in instr.srcs]

        result: Optional[np.ndarray] = None
        match instr.opcode:
            case OpCode.ADD:
                result = (src_vals[0] + src_vals[1]).astype(np.float32)
            case OpCode.SUB:
                result = (src_vals[0] - src_vals[1]).astype(np.float32)
            case OpCode.MUL:
                result = (src_vals[0] * src_vals[1]).astype(np.float32)
            case OpCode.DIV:
                result = (src_vals[0] / src_vals[1]).astype(np.float32)
            case OpCode.FMA:
                result = (src_vals[0] * src_vals[1] + src_vals[2]).astype(np.float32)
            case OpCode.MIN:
                result = np.minimum(src_vals[0], src_vals[1]).astype(np.float32)
            case OpCode.MAX:
                result = np.maximum(src_vals[0], src_vals[1]).astype(np.float32)
            case OpCode.SHL:
                a_i = src_vals[0].view(np.uint32).astype(np.uint64)
                b_i = src_vals[1].view(np.uint32).astype(np.uint64)
                result = ((a_i << b_i) & 0xFFFFFFFF).astype(np.uint32).view(np.float32)
            case OpCode.SHR:
                a_u = src_vals[0].view(np.uint32).astype(np.uint64)
                b_u = src_vals[1].view(np.uint32).astype(np.uint64)
                result = (a_u >> b_u).astype(np.uint32).view(np.float32)
            case OpCode.AND:
                result = (src_vals[0].view(np.int32) & src_vals[1].view(np.int32)).view(np.float32)
            case OpCode.OR:
                result = (src_vals[0].view(np.int32) | src_vals[1].view(np.int32)).view(np.float32)
            case OpCode.XOR:
                result = (src_vals[0].view(np.int32) ^ src_vals[1].view(np.int32)).view(np.float32)
            case OpCode.NOT:
                result = (~src_vals[0].view(np.int32)).view(np.float32)
            case OpCode.CMP_LT:
                result = (src_vals[0] < src_vals[1]).astype(np.float32)
            case OpCode.CMP_EQ:
                result = (src_vals[0] == src_vals[1]).astype(np.float32)
            case OpCode.SELECT:
                cond = src_vals[0] != 0.0
                result = np.where(cond, src_vals[1], src_vals[2]).astype(np.float32)
            case OpCode.MOV:
                result = src_vals[0].copy()
            case OpCode.BCAST:
                # Cooperative across all 32 lanes: lane 0 value broadcast to all 32 lanes;
                # active_lane_mask is applied at writeback during Instruction.retire().
                result = np.full(SIMD_WIDTH, src_vals[0][0], dtype=np.float32)
            case OpCode.SET_IMM:
                if instr.imm is None or isinstance(instr.imm, str):
                    raise ValueError(f"SET_IMM requires a numeric immediate, got {instr.imm!r}.")
                result = np.full(SIMD_WIDTH, float(instr.imm), dtype=np.float32)
            case OpCode.JMP:
                if not isinstance(instr.imm, int):
                    raise ValueError(f"JMP requires resolved integer target PC, got {instr.imm!r}.")
                instr.branch_taken = bool(np.any(mask))
                instr.branch_target = int(instr.imm) if instr.branch_taken else None

        instr.completion_time = completion_cycle
        if self not in instr.resources:
            instr.resources.append(self)

        if result is not None and len(instr.dest) == 1:
            instr.pending_writes = [(instr.dest[0], result, mask)]

        return WarpState.WARP_READY, completion_cycle, instr.pending_writes


# ============================================================================
# 2. GEMMCoreResource (Non-Pipelined; Latency = 8 cycles)
# ============================================================================
class GEMMCoreResource(ComputeResource):
    """
    8x16x8 Tensor Core Matrix Unit (`OpClass.GEMMCORE`) executing `MMA`, `MMRED_MAX`, `MMRED_MIN`.
    
    - Latency: 8 cycles, non-pipelined.
    - Register Ownership Layout (Column-Major 8x8 Sub-Blocks of 4x8 Tiles):
      * `a_tuple[0]` / `b_tuple[0]`: Rows 0..3, Cols 0..7   (4x8)
      * `a_tuple[1]` / `b_tuple[1]`: Rows 4..7, Cols 0..7   (4x8)
      * `a_tuple[2]` / `b_tuple[2]`: Rows 0..3, Cols 8..15  (4x8)
      * `a_tuple[3]` / `b_tuple[3]`: Rows 4..7, Cols 8..15  (4x8)
      * `c_tuple[0]` / `d_tuple[0]`: Rows 0..3, Cols 0..7   (4x8)
      * `c_tuple[1]` / `d_tuple[1]`: Rows 4..7, Cols 0..7   (4x8)
    - Cooperative instruction: all 32 lanes participate in computing D(8x8), and `active_lane_mask`
      is applied at writeback when `Instruction.retire()` commits `pending_writes`.
    """
    DEFAULT_LATENCY: int = 8

    def __init__(self, name: str = "GEMMCORE", base_latency: int = DEFAULT_LATENCY):
        super().__init__(
            name=name,
            base_latency=base_latency,
            stall_signal=WarpState.WARP_COMPUTE_STALL,
        )

    @staticmethod
    def unpack_8x16_operand(reg_arrays: Sequence[np.ndarray]) -> np.ndarray:
        """
        Reconstruct an 8x16 matrix from 4 32-lane vector register arrays:
        - reg_arrays[0]: Rows 0..3, Cols 0..7
        - reg_arrays[1]: Rows 4..7, Cols 0..7
        - reg_arrays[2]: Rows 0..3, Cols 8..15
        - reg_arrays[3]: Rows 4..7, Cols 8..15
        """
        mat = np.zeros((8, 16), dtype=np.float32)
        mat[0:4, 0:8] = np.asarray(reg_arrays[0], dtype=np.float32).reshape(4, 8)
        mat[4:8, 0:8] = np.asarray(reg_arrays[1], dtype=np.float32).reshape(4, 8)
        mat[0:4, 8:16] = np.asarray(reg_arrays[2], dtype=np.float32).reshape(4, 8)
        mat[4:8, 8:16] = np.asarray(reg_arrays[3], dtype=np.float32).reshape(4, 8)
        return mat

    @staticmethod
    def unpack_8x8_operand(reg_arrays: Sequence[np.ndarray]) -> np.ndarray:
        """
        Reconstruct an 8x8 matrix from 2 32-lane vector register arrays:
        - reg_arrays[0]: Rows 0..3, Cols 0..7
        - reg_arrays[1]: Rows 4..7, Cols 0..7
        """
        mat = np.zeros((8, 8), dtype=np.float32)
        mat[0:4, 0:8] = np.asarray(reg_arrays[0], dtype=np.float32).reshape(4, 8)
        mat[4:8, 0:8] = np.asarray(reg_arrays[1], dtype=np.float32).reshape(4, 8)
        return mat

    @staticmethod
    def pack_8x8_result(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pack an 8x8 matrix into two 32-lane vector register arrays (top 4x8 and bottom 4x8)."""
        d0 = np.asarray(mat[0:4, 0:8], dtype=np.float32).reshape(SIMD_WIDTH).copy()
        d1 = np.asarray(mat[4:8, 0:8], dtype=np.float32).reshape(SIMD_WIDTH).copy()
        return d0, d1

    def issue(
        self,
        instr: Instruction,
        warp_regs: np.ndarray,
        warp_preds: Optional[np.ndarray],
        current_cycle: int,
        warp_id: int = 0,
        num_warps: int = 4,
    ) -> Tuple[WarpState, int, List[Tuple[int, np.ndarray, np.ndarray]]]:
        if instr.op_class != OpClass.GEMMCORE:
            raise ValueError(f"GEMMCoreResource cannot execute {instr.opcode.name} ({instr.op_class.name}).")

        if not self.is_available(current_cycle):
            return self.stall_signal, self.next_available_cycle, []

        latency = self.base_latency
        self.reserve(current_cycle, latency)
        completion_cycle = current_cycle + latency

        # Latch all 10 source registers (4 A + 4 B + 2 C) and guard predicate mask at issue_time
        mask = instr.active_lane_mask(warp_preds)
        wid = getattr(instr, "warp_id", None) if (warp_id == 0 and getattr(instr, "warp_id", None) is not None) else warp_id
        a_regs = [_read_reg(warp_regs, r, warp_id=wid, num_warps=num_warps) for r in instr.srcs[0:4]]
        b_regs = [_read_reg(warp_regs, r, warp_id=wid, num_warps=num_warps) for r in instr.srcs[4:8]]
        c_regs = [_read_reg(warp_regs, r, warp_id=wid, num_warps=num_warps) for r in instr.srcs[8:10]]

        A = self.unpack_8x16_operand(a_regs)  # (8, 16)
        B = self.unpack_8x16_operand(b_regs)  # (8, 16)
        C = self.unpack_8x8_operand(c_regs)   # (8, 8)

        match instr.opcode:
            case OpCode.MMA:
                D = (A @ B.T + C).astype(np.float32)
            case OpCode.MMRED_MAX:
                pairwise_prod = A[:, None, :] * B[None, :, :]  # (8, 8, 16)
                D = np.maximum(C, np.max(pairwise_prod, axis=2)).astype(np.float32)
            case OpCode.MMRED_MIN:
                pairwise_prod = A[:, None, :] * B[None, :, :]  # (8, 8, 16)
                D = np.minimum(C, np.min(pairwise_prod, axis=2)).astype(np.float32)
            case _:
                raise ValueError(f"Unsupported GEMMCORE opcode: {instr.opcode}")

        d0_arr, d1_arr = self.pack_8x8_result(D)

        instr.completion_time = completion_cycle
        if self not in instr.resources:
            instr.resources.append(self)

        instr.pending_writes = [
            (instr.dest[0], d0_arr, mask),
            (instr.dest[1], d1_arr, mask.copy()),
        ]
        return WarpState.WARP_READY, completion_cycle, instr.pending_writes


# ============================================================================
# 3. TransResource (Non-Pipelined; Trans Mode = 24 cyc, Fast Mode = 4+ceil(1.5 log2 n) / 6+3 log2 n)
# ============================================================================
class TransResource(ComputeResource):
    """
    Transcendental & Fast Taylor-Series Approximation Unit (`OpClass.TRANS`).
    
    - Transcendental Mode (`*.trans`): Latency = 24 cycles.
    - Fast Mode (`*.fast.<n>`):
      * `n` is read from `instr.srcs[1]` (`R_n`, populated via `SET_IMM`), where `n in {4, 8, 16, 32, 64, 128}`.
      * Latency for `EXP_FAST`, `SIN_FAST`, `COS_FAST`, `LOG_FAST`: `4 + ceil(1.5 * log2(n))` cycles.
      * Latency for `TANH_FAST`: `6 + 3 * log2(n)` cycles.
    - Non-pipelined: cannot accept another `TRANS` instruction until `current_cycle >= next_available_cycle`.
    """
    TRANS_MODE_LATENCY: int = 24

    def __init__(self, name: str = "TRANS", trans_latency: int = TRANS_MODE_LATENCY):
        super().__init__(
            name=name,
            base_latency=trans_latency,
            stall_signal=WarpState.WARP_COMPUTE_STALL,
        )
        self.trans_latency = trans_latency

    @staticmethod
    def extract_and_validate_n(n_vec: np.ndarray) -> int:
        """Extract Taylor term count `n` from `R_n` and verify `n in {4, 8, 16, 32, 64, 128}`."""
        n_val = int(round(float(n_vec[0])))
        if not np.all(np.round(n_vec).astype(np.int64) == n_val):
            raise ValueError("All 32 lanes of R_n must hold the same Taylor term count n.")
        if n_val not in VALID_TAYLOR_TERMS:
            raise ValueError(
                f"Invalid Taylor series term count n={n_val}; allowed values are {VALID_TAYLOR_TERMS}."
            )
        return n_val

    def compute_latency(self, instr: Instruction, warp_regs: np.ndarray) -> int:
        """Compute instruction latency in cycles based on `.trans` vs `.fast` mode and `n`."""
        if instr.opcode in (
            OpCode.EXP_TRANS,
            OpCode.SIN_TRANS,
            OpCode.COS_TRANS,
            OpCode.TANH_TRANS,
            OpCode.LOG_TRANS,
        ):
            return self.trans_latency

        n_vec = _read_reg(warp_regs, instr.srcs[1])
        n = self.extract_and_validate_n(n_vec)
        log2_n = int(round(math.log2(n)))

        if instr.opcode == OpCode.TANH_FAST:
            return 6 + 3 * log2_n
        return 4 + int(math.ceil(1.5 * log2_n))

    def issue(
        self,
        instr: Instruction,
        warp_regs: np.ndarray,
        warp_preds: Optional[np.ndarray],
        current_cycle: int,
        warp_id: int = 0,
        num_warps: int = 4,
    ) -> Tuple[WarpState, int, List[Tuple[int, np.ndarray, np.ndarray]]]:
        if instr.op_class != OpClass.TRANS:
            raise ValueError(f"TransResource cannot execute {instr.opcode.name} ({instr.op_class.name}).")

        if not self.is_available(current_cycle):
            return self.stall_signal, self.next_available_cycle, []

        latency = self.compute_latency(instr, warp_regs)
        self.reserve(current_cycle, latency)
        completion_cycle = current_cycle + latency

        mask = instr.active_lane_mask(warp_preds)
        wid = getattr(instr, "warp_id", None) if (warp_id == 0 and getattr(instr, "warp_id", None) is not None) else warp_id
        x = _read_reg(warp_regs, instr.srcs[0], warp_id=wid, num_warps=num_warps)

        match instr.opcode:
            case OpCode.EXP_TRANS:
                result = np.exp(x).astype(np.float32)
            case OpCode.SIN_TRANS:
                result = np.sin(x).astype(np.float32)
            case OpCode.COS_TRANS:
                result = np.cos(x).astype(np.float32)
            case OpCode.TANH_TRANS:
                result = np.tanh(x).astype(np.float32)
            case OpCode.LOG_TRANS:
                result = np.log(x).astype(np.float32)
            case OpCode.EXP_FAST:
                n = self.extract_and_validate_n(_read_reg(warp_regs, instr.srcs[1], warp_id=wid))
                result = exp_taylor(x, n=n)
            case OpCode.SIN_FAST:
                n = self.extract_and_validate_n(_read_reg(warp_regs, instr.srcs[1], warp_id=wid))
                result = sin_taylor(x, n=n)
            case OpCode.COS_FAST:
                n = self.extract_and_validate_n(_read_reg(warp_regs, instr.srcs[1], warp_id=wid))
                result = cos_taylor(x, n=n)
            case OpCode.TANH_FAST:
                n = self.extract_and_validate_n(_read_reg(warp_regs, instr.srcs[1], warp_id=wid))
                result = tanh_taylor(x, n=n)
            case OpCode.LOG_FAST:
                n = self.extract_and_validate_n(_read_reg(warp_regs, instr.srcs[1], warp_id=wid))
                result = log_taylor(x, n=n)
            case _:
                raise ValueError(f"Unsupported TRANS opcode: {instr.opcode}")

        instr.completion_time = completion_cycle
        if self not in instr.resources:
            instr.resources.append(self)

        instr.pending_writes = [(instr.dest[0], result, mask)]
        return WarpState.WARP_READY, completion_cycle, instr.pending_writes
