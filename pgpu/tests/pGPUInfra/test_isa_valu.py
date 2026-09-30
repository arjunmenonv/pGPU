'''
Unit tests for pGPU ISA, Register Allocator, Assembler, and Compute Resources (VALU, GEMMCORE, TRANS).

Location: pgpu/tests/pGPUInfra/test_isa_valu.py
'''

import unittest
import numpy as np

from pgpu.arch.isa import (
    OpClass,
    OpCode,
    InstrState,
    RegisterSpillError,
    VirtualReg,
    VirtualRegTuple,
    VirtualPred,
    RegisterAllocator,
    RegScope,
    Instruction,
    RZ_ID,
    PRED_BASE_ID,
    SIMD_WIDTH,
)
from pgpu.sw.kernel import (
    assemble_instruction,
    WarpKernel,
)

from pgpu.arch.valu import (
    VALUResource,
    GEMMCoreResource,
    TransResource,
    exp_taylor,
    sin_taylor,
    cos_taylor,
    tanh_taylor,
    log_taylor,
)
from pgpu.arch.smsp import WarpState


class TestEnumTypeSafety(unittest.TestCase):
    def test_cross_enum_comparison_returns_false(self):
        """Verify that OpClass, OpCode, InstrState, WarpState, and MemOpType are type-safe Enums."""
        from pgpu.arch.memory import MemOpType
        # With standard Enum, comparing different enums with the same integer value must return False
        self.assertFalse(OpClass.DRAM == InstrState.PENDING)
        self.assertFalse(OpClass.DRAM == OpCode.DRAM_LD)
        self.assertFalse(OpClass.DRAM == WarpState.WARP_READY)
        self.assertFalse(OpClass.DRAM == MemOpType.LOAD)
        self.assertFalse(OpClass.DRAM == 0)
        self.assertFalse(InstrState.PENDING == 0)
        self.assertFalse(OpCode.DRAM_LD == 0)

        # But integer value can still be accessed via .value
        self.assertEqual(OpClass.DRAM.value, 0)
        self.assertEqual(InstrState.PENDING.value, 0)
        self.assertEqual(OpCode.DRAM_LD.value, 0)


class TestRegisterAllocatorAndVirtualRegs(unittest.TestCase):
    def setUp(self):
        self.allocator = RegisterAllocator()

    def test_circular_next_fit_and_spill(self):
        """Verify sequential allocation, spill error at 64, and wrapping around to fill freed holes."""
        regs = []
        for i in range(64):
            r = self.allocator.alloc()
            self.assertEqual(r.id, i)
            regs.append(r)

        self.assertEqual(self.allocator.num_alive, 64)
        with self.assertRaises(RegisterSpillError):
            self.allocator.alloc()

        # Free registers 5 and 10 in place
        regs[5].free()
        regs[10].free()
        self.assertEqual(self.allocator.num_alive, 62)

        # Next allocations should wrap around R63 -> R0 and find holes 5 and 10
        r_new1 = self.allocator.alloc()
        self.assertEqual(r_new1.id, 5)
        r_new2 = self.allocator.alloc()
        self.assertEqual(r_new2.id, 10)
        self.assertEqual(self.allocator.num_alive, 64)

    def test_virtual_reg_tuple_non_contiguous(self):
        """Verify VirtualRegTuple can wrap non-contiguous registers."""
        # Allocate some, free some to create non-contiguous sequence
        r0 = self.allocator.alloc()
        r1 = self.allocator.alloc()
        r2 = self.allocator.alloc()
        r1.free()  # hole at 1

        # alloc_tuple(2) will allocate 3, then wrap or take next free
        # Let's allocate 61 more to fill up to 63
        for _ in range(61):
            self.allocator.alloc()
        # Next alloc will wrap to hole 1, then next hole
        r0.free()
        t = self.allocator.alloc_tuple(2)
        self.assertEqual(len(t), 2)
        self.assertEqual(t.ids, [0, 1])

        t.free()
        self.assertTrue(t._freed)

    def test_virtual_pred_and_inversion(self):
        """Verify VirtualPred allocation, negation (~P), and unified register IDs."""
        p0 = self.allocator.alloc_pred("flag")
        self.assertEqual(p0.pred_idx, 0)
        self.assertEqual(p0.id, PRED_BASE_ID)
        self.assertFalse(p0.inverted)

        not_p0 = ~p0
        self.assertEqual(not_p0.pred_idx, 0)
        self.assertEqual(not_p0.id, PRED_BASE_ID)
        self.assertTrue(not_p0.inverted)

        p0.free()
        self.assertEqual(self.allocator.num_preds_alive, 0)


class TestInstructionAndAssembler(unittest.TestCase):
    def test_assemble_and_disassemble(self):
        """Verify assembly parsing via match-case and disassembly formatting."""
        alloc = RegisterAllocator()
        r0 = alloc.alloc()
        r1 = alloc.alloc()
        r2 = alloc.alloc()
        p0 = alloc.alloc_pred()

        # 1. VALU op
        i_add = assemble_instruction("add", r0, r1, r2)
        self.assertEqual(i_add.opcode, OpCode.ADD)
        self.assertEqual(i_add.op_class, OpClass.VALU)
        self.assertEqual(i_add.dest, [0])
        self.assertEqual(i_add.srcs, [1, 2])
        self.assertEqual(i_add.disassemble(), "add R0, R1, R2")

        # 2. Predicated inline op
        i_pred = assemble_instruction("@P0 mul", r0, r1, r2)
        self.assertEqual(i_pred.pred, 0)
        self.assertFalse(i_pred.pred_inv)
        self.assertEqual(i_pred.disassemble(), "@P0 mul R0, R1, R2")

        # 3. Inverted predicate via ~p0
        i_inv = assemble_instruction("fma", r0, r1, r2, alloc.rz, pred=~p0)
        self.assertEqual(i_inv.pred, 0)
        self.assertTrue(i_inv.pred_inv)
        self.assertEqual(i_inv.srcs[2], RZ_ID)
        self.assertEqual(i_inv.disassemble(), "@!P0 fma R0, R1, R2, RZ")

    def test_pipeline_latch_at_issue_commit_at_retire(self):
        """Verify that pending_writes are not committed until Instruction.retire()."""
        warp_regs = np.zeros((64, 32), dtype=np.float32)
        warp_preds = np.zeros((8, 32), dtype=bool)

        instr = assemble_instruction("add", 0, 1, 2)
        instr.pending_writes = [(0, np.full(32, 42.0, dtype=np.float32), np.ones(32, dtype=bool))]
        instr.completion_time = 10

        # Before completion time, retire returns False and does not commit
        self.assertFalse(instr.retire(current_cycle=5, warp_regs=warp_regs, warp_preds=warp_preds))
        self.assertEqual(warp_regs[0, 0], 0.0)

        # At completion time, retire returns True and commits
        self.assertTrue(instr.retire(current_cycle=10, warp_regs=warp_regs, warp_preds=warp_preds))
        self.assertEqual(warp_regs[0, 0], 42.0)
        self.assertEqual(instr.state, InstrState.COMPLETE)


class TestVALUResource(unittest.TestCase):
    def setUp(self):
        self.valu = VALUResource()
        self.regs = np.zeros((64, 32), dtype=np.float32)
        self.preds = np.zeros((8, 32), dtype=bool)

    def test_non_pipelined_structural_stall(self):
        """Verify that VALU is non-pipelined and rejects back-to-back issues while busy."""
        self.regs[1] = 10.0
        self.regs[2] = 2.0
        i_div = assemble_instruction("div", 0, 1, 2)

        # Issue DIV at cycle 0 (latency 4 -> busy until cycle 4)
        state, comp, _ = self.valu.issue(i_div, self.regs, self.preds, current_cycle=0)
        self.assertEqual(state, WarpState.WARP_READY)
        self.assertEqual(comp, 4)

        # Attempt to issue ADD at cycle 2 -> structural stall
        i_add = assemble_instruction("add", 3, 1, 2)
        stall_state, unblock, _ = self.valu.issue(i_add, self.regs, self.preds, current_cycle=2)
        self.assertEqual(stall_state, WarpState.WARP_COMPUTE_STALL)
        self.assertEqual(unblock, 4)

        # At cycle 4, VALU is free again
        state2, comp2, _ = self.valu.issue(i_add, self.regs, self.preds, current_cycle=4)
        self.assertEqual(state2, WarpState.WARP_READY)
        self.assertEqual(comp2, 5)

    def test_special_registers_sr_laneid_warpid_numwarps(self):
        """Verify that SR_LANEID, SR_WARPID, and SR_NUMWARPS can be read directly by any ALU op."""
        # 1. MOV R0, SR_LANEID
        i_lane = assemble_instruction("mov", 0, "SR_LANEID")
        _, comp_lane, _ = self.valu.issue(i_lane, self.regs, self.preds, current_cycle=0)
        i_lane.retire(comp_lane, self.regs, self.preds)
        np.testing.assert_array_equal(self.regs[0], np.arange(32, dtype=np.float32))

        # 2. Global Thread ID via FMA: R2 = SR_WARPID * 32 + SR_LANEID (for warp 3)
        self.valu.reset()
        self.regs[1] = 32.0
        i_fma = assemble_instruction("fma", 2, "SR_WARPID", 1, "SR_LANEID")
        i_fma.warp_id = 3
        _, comp_fma, _ = self.valu.issue(i_fma, self.regs, self.preds, current_cycle=1)
        i_fma.retire(comp_fma, self.regs, self.preds)
        expected_gtid = 3 * 32 + np.arange(32, dtype=np.float32)
        np.testing.assert_array_equal(self.regs[2], expected_gtid)

    def test_functional_valu_ops(self):
        """Verify functional arithmetic, comparison, and bcast operations."""
        self.regs[1] = np.arange(32, dtype=np.float32)
        self.regs[2] = 5.0

        # CMP_LT into predicate register P0 (ID = PRED_BASE_ID)
        i_cmp = assemble_instruction("cmp.lt", "P0", 1, 2)
        _, _, pw = self.valu.issue(i_cmp, self.regs, self.preds, current_cycle=0)
        i_cmp.retire(1, self.regs, self.preds)
        # Lanes 0..4 should be True, 5..31 False
        self.assertTrue(np.all(self.preds[0, :5]))
        self.assertFalse(np.any(self.preds[0, 5:]))

        # BCAST lane 0 of R1 (which is 0.0)
        self.regs[1, 0] = 99.0
        i_bcast = assemble_instruction("bcast", 3, 1)
        self.valu.reset()
        _, _, _ = self.valu.issue(i_bcast, self.regs, self.preds, current_cycle=2)
        i_bcast.retire(3, self.regs, self.preds)
        self.assertTrue(np.all(self.regs[3] == 99.0))


class TestGEMMCoreResource(unittest.TestCase):
    def setUp(self):
        self.gemm = GEMMCoreResource()
        self.regs = np.zeros((64, 32), dtype=np.float32)
        self.preds = np.zeros((8, 32), dtype=bool)

    def test_mma_non_contiguous_layout_and_math(self):
        """
        Verify that GEMMCore correctly gathers A(8x16) and B(8x16) from 4 non-contiguous
        registers, computes D = A @ B^T + C, and scatters D(8x8) into 2 registers.
        """
        # A: 8x16 matrix of all 1.0s
        # B: 8x16 matrix of all 2.0s
        # Inner product sum_{k=0..15} (1.0 * 2.0) = 32.0
        # C: 8x8 matrix of 3.0s -> D should be 35.0 everywhere
        a_regs = [4, 17, 9, 44]  # non-contiguous
        b_regs = [1, 2, 3, 5]
        c_regs = [10, 11]
        d_regs = [20, 21]

        for r in a_regs:
            self.regs[r] = 1.0
        for r in b_regs:
            self.regs[r] = 2.0
        for r in c_regs:
            self.regs[r] = 3.0

        i_mma = assemble_instruction("avx.mma", d_regs, a_regs, b_regs, c_regs)
        self.assertEqual(i_mma.op_class, OpClass.GEMMCORE)
        self.assertEqual(len(i_mma.dest), 2)
        self.assertEqual(len(i_mma.srcs), 10)

        # Issue MMA at cycle 0 (latency 8 cycles, non-pipelined)
        state, comp, _ = self.gemm.issue(i_mma, self.regs, self.preds, current_cycle=0)
        self.assertEqual(state, WarpState.WARP_READY)
        self.assertEqual(comp, 8)

        # Next MMA at cycle 2 stalls
        stall, unblock, _ = self.gemm.issue(i_mma, self.regs, self.preds, current_cycle=2)
        self.assertEqual(stall, WarpState.WARP_COMPUTE_STALL)
        self.assertEqual(unblock, 8)

        # Retire at cycle 8
        i_mma.retire(8, self.regs, self.preds)
        self.assertTrue(np.all(self.regs[20] == 35.0))
        self.assertTrue(np.all(self.regs[21] == 35.0))

    def test_mmred_max(self):
        """Verify MMRED_MAX computes D[r, c] = max(C[r, c], max_k (A[r, k] * B[c, k]))."""
        a_regs = [0, 1, 2, 3]
        b_regs = [4, 5, 6, 7]
        c_regs = [8, 9]
        d_regs = [10, 11]

        for r in a_regs:
            self.regs[r] = 4.0
        for r in b_regs:
            self.regs[r] = 5.0
        # A[r, k] * B[c, k] = 20.0
        # C = 15.0 -> max(15.0, 20.0) = 20.0
        for r in c_regs:
            self.regs[r] = 15.0

        i_max = assemble_instruction("avx.mmred.max", d_regs, a_regs, b_regs, c_regs)
        self.gemm.issue(i_max, self.regs, self.preds, current_cycle=0)
        i_max.retire(8, self.regs, self.preds)
        self.assertTrue(np.all(self.regs[10] == 20.0))
        self.assertTrue(np.all(self.regs[11] == 20.0))


class TestTransResource(unittest.TestCase):
    def setUp(self):
        self.trans = TransResource()
        self.regs = np.zeros((64, 32), dtype=np.float32)
        self.preds = np.zeros((8, 32), dtype=bool)

    def test_transcendental_mode_latency_and_accuracy(self):
        """Verify .trans mode has 24-cycle latency and exact NumPy math."""
        self.regs[1] = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
        i_exp = assemble_instruction("exp.trans", 0, 1)

        state, comp, _ = self.trans.issue(i_exp, self.regs, self.preds, current_cycle=0)
        self.assertEqual(state, WarpState.WARP_READY)
        self.assertEqual(comp, 24)

        i_exp.retire(24, self.regs, self.preds)
        np.testing.assert_allclose(self.regs[0], np.exp(self.regs[1]), rtol=1e-5)

    def test_fast_mode_latencies_and_taylor(self):
        """
        Verify fast mode latencies:
        - EXP/SIN/COS/LOG: 4 + ceil(1.5 * log2(n)) -> for n=64, 4 + 9 = 13 cycles
        - TANH: 6 + 3 * log2(n) -> for n=64, 6 + 18 = 24 cycles
        """
        self.regs[1] = 0.5
        self.regs[2] = 64.0  # R_n = 64

        # 1. exp.fast
        i_exp_fast = assemble_instruction("avx.exp.fast", 0, 1, 2)
        state, comp, _ = self.trans.issue(i_exp_fast, self.regs, self.preds, current_cycle=0)
        self.assertEqual(state, WarpState.WARP_READY)
        self.assertEqual(comp, 13)
        i_exp_fast.retire(13, self.regs, self.preds)
        np.testing.assert_allclose(self.regs[0], np.exp(0.5), rtol=1e-4)

        # 2. tanh.fast
        self.trans.reset()
        i_tanh_fast = assemble_instruction("avx.tanh.fast", 3, 1, 2)
        state_t, comp_t, _ = self.trans.issue(i_tanh_fast, self.regs, self.preds, current_cycle=13)
        self.assertEqual(state_t, WarpState.WARP_READY)
        self.assertEqual(comp_t, 13 + 24)
        i_tanh_fast.retire(comp_t, self.regs, self.preds)
        np.testing.assert_allclose(self.regs[3], np.tanh(0.5), rtol=1e-4)

    def test_fast_mode_rejects_invalid_n(self):
        """Verify that fast mode raises ValueError if n is not in (4, 8, 16, 32, 64, 128)."""
        self.regs[1] = 0.5
        self.regs[2] = 25.0  # Invalid n
        i_bad = assemble_instruction("avx.exp.fast", 0, 1, 2)
        with self.assertRaises(ValueError):
            self.trans.issue(i_bad, self.regs, self.preds, current_cycle=0)


class TestCoScheduling(unittest.TestCase):
    def test_cross_unit_co_scheduling(self):
        """
        Verify that VALU, GEMMCORE, and TRANS can issue on consecutive cycles
        when operating on independent registers without structural interference.
        """
        valu = VALUResource()
        gemm = GEMMCoreResource()
        trans = TransResource()

        regs = np.zeros((64, 32), dtype=np.float32)
        preds = np.zeros((8, 32), dtype=bool)

        # Cycle 0: Issue GEMMCORE (latency 8 -> finishes at 8)
        i_gemm = assemble_instruction("avx.mma", [10, 11], [0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
        s0, c0, _ = gemm.issue(i_gemm, regs, preds, current_cycle=0)
        self.assertEqual(s0, WarpState.WARP_READY)
        self.assertEqual(c0, 8)

        # Cycle 1: GEMMCORE is busy, but TRANS is free -> issue exp.trans at cycle 1
        i_trans = assemble_instruction("exp.trans", 12, 13)
        s1, c1, _ = trans.issue(i_trans, regs, preds, current_cycle=1)
        self.assertEqual(s1, WarpState.WARP_READY)
        self.assertEqual(c1, 1 + 24)

        # Cycle 2: VALU is free -> issue fma at cycle 2
        i_valu = assemble_instruction("fma", 14, 15, 16, 17)
        s2, c2, _ = valu.issue(i_valu, regs, preds, current_cycle=2)
        self.assertEqual(s2, WarpState.WARP_READY)
        self.assertEqual(c2, 2 + 1)


if __name__ == "__main__":
    unittest.main()
