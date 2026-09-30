'''
Unit tests for ISA tokens, RegisterAllocator, Assembler (match-case), Instruction lifecycle,
and all compute resources (VALUResource, GEMMCoreResource, TransResource).

Location: pgpu/tests/pGPUInfra/test_isa.py
'''

import math
import unittest
import numpy as np

from pgpu.arch.isa import (
    OpClass,
    OpCode,
    InstrState,
    OPCODE_TO_CLASS,
    VirtualReg,
    VirtualRegTuple,
    VirtualPred,
    RegisterAllocator,
    RegScope,
    RegisterSpillError,
    Instruction,
    NUM_VECTOR_REGS,
    RZ_ID,
    PRED_BASE_ID,
    NUM_PRED_REGS,
    SIMD_WIDTH,
)
from pgpu.sw.kernel import (
    assemble_instruction,
    WarpKernel,
)

from pgpu.arch.valu import (
    VALUResource, GEMMCoreResource, TransResource,
    exp_taylor, sin_taylor, cos_taylor, tanh_taylor, log_taylor,
    VALID_TAYLOR_TERMS,
)
from pgpu.arch.smsp import WarpState


def make_warp_state():
    """Helper: create fresh warp register file (64x32 float32) and pred file (8x32 bool)."""
    regs = np.zeros((NUM_VECTOR_REGS, SIMD_WIDTH), dtype=np.float32)
    preds = np.zeros((NUM_PRED_REGS, SIMD_WIDTH), dtype=bool)
    return regs, preds


class TestOpCodeTokens(unittest.TestCase):
    """Verify IntEnum tokens, OPCODE_TO_CLASS completeness, and uniqueness."""

    def test_all_opcodes_have_class_mapping(self):
        for opc in OpCode:
            self.assertIn(opc, OPCODE_TO_CLASS, f"OpCode.{opc.name} missing from OPCODE_TO_CLASS.")

    def test_opcode_class_consistency(self):
        self.assertEqual(OPCODE_TO_CLASS[OpCode.ADD], OpClass.VALU)
        self.assertEqual(OPCODE_TO_CLASS[OpCode.DRAM_LD], OpClass.DRAM)
        self.assertEqual(OPCODE_TO_CLASS[OpCode.SRAM_ST], OpClass.SRAM)
        self.assertEqual(OPCODE_TO_CLASS[OpCode.AVX_MMA], OpClass.GEMMCORE)
        self.assertEqual(OPCODE_TO_CLASS[OpCode.EXP_TRANS], OpClass.TRANS)
        self.assertEqual(OPCODE_TO_CLASS[OpCode.SYNC_WARP], OpClass.SYNC)

    def test_opcode_values_unique(self):
        vals = [opc.value for opc in OpCode]
        self.assertEqual(len(vals), len(set(vals)), "Duplicate OpCode integer values detected.")


class TestRegisterAllocator(unittest.TestCase):
    """Verify circular next-fit allocation, wrap-around, free, and spill detection."""

    def setUp(self):
        self.alloc = RegisterAllocator()

    def test_sequential_allocation_starts_at_r0(self):
        r0 = self.alloc.alloc("first")
        self.assertEqual(r0.id, 0)
        r1 = self.alloc.alloc()
        self.assertEqual(r1.id, 1)

    def test_circular_wrap_around_after_r63(self):
        # Allocate all 64 registers
        regs = [self.alloc.alloc() for _ in range(64)]
        self.assertEqual(self.alloc.num_alive, 64)
        # Free R0 and R32
        regs[0].free()
        regs[32].free()
        self.assertEqual(self.alloc.num_alive, 62)
        # next_reg_idx should be at 0 (wrapped), so the next alloc should find R0 first
        r_new = self.alloc.alloc()
        self.assertEqual(r_new.id, 0)
        r_new2 = self.alloc.alloc()
        self.assertEqual(r_new2.id, 32)

    def test_spill_error_on_exhaustion(self):
        for _ in range(64):
            self.alloc.alloc()
        with self.assertRaises(RegisterSpillError):
            self.alloc.alloc()

    def test_double_free_raises(self):
        r = self.alloc.alloc()
        r.free()
        with self.assertRaises(ValueError):
            self.alloc.free(r)

    def test_rz_free_is_noop(self):
        rz = self.alloc.rz
        rz.free()
        rz.free()  # Should not raise

    def test_alloc_tuple_non_contiguous(self):
        # Allocate 60 regs, free every other one, then allocate a tuple of 4
        regs = [self.alloc.alloc() for _ in range(60)]
        for i in range(0, 60, 2):
            regs[i].free()
        self.assertEqual(self.alloc.num_alive, 30)
        tup = self.alloc.alloc_tuple(4, "test")
        self.assertEqual(len(tup), 4)
        self.assertEqual(self.alloc.num_alive, 34)
        # Registers in the tuple need not be contiguous
        tup.free()
        self.assertEqual(self.alloc.num_alive, 30)

    def test_tuple_spill_error(self):
        for _ in range(62):
            self.alloc.alloc()
        with self.assertRaises(RegisterSpillError):
            self.alloc.alloc_tuple(3)

    def test_pred_allocation_and_free(self):
        p0 = self.alloc.alloc_pred("mask0")
        self.assertEqual(p0.pred_idx, 0)
        self.assertEqual(p0.id, PRED_BASE_ID)
        p1 = self.alloc.alloc_pred()
        self.assertEqual(p1.pred_idx, 1)
        p0.free()
        p2 = self.alloc.alloc_pred()
        # Should wrap and find P0 free
        self.assertEqual(p2.pred_idx, 2)  # next_pred_idx was at 2
        self.assertEqual(self.alloc.num_preds_alive, 2)

    def test_pred_spill_error(self):
        for _ in range(8):
            self.alloc.alloc_pred()
        with self.assertRaises(RegisterSpillError):
            self.alloc.alloc_pred()

    def test_virtual_pred_inversion(self):
        p = self.alloc.alloc_pred()
        self.assertFalse(p.inverted)
        p_inv = ~p
        self.assertTrue(p_inv.inverted)
        self.assertEqual(p.pred_idx, p_inv.pred_idx)
        p_inv_inv = ~p_inv
        self.assertFalse(p_inv_inv.inverted)

    def test_reg_scope_auto_free(self):
        with self.alloc_scope() as scope:
            r0 = scope.alloc("tmp")
            t = scope.alloc_tuple(2, "pair")
            p = scope.alloc_pred("cond")
        self.assertEqual(self.alloc.num_alive, 0)
        self.assertEqual(self.alloc.num_preds_alive, 0)
        self.assertTrue(r0._freed)
        self.assertTrue(t._freed)
        self.assertTrue(p._freed)

    def alloc_scope(self):
        return RegScope(self.alloc)


class TestAssembler(unittest.TestCase):
    """Verify match-case assembly mnemonic parsing for all OpClasses."""

    def test_valu_binops(self):
        for mnemonic, expected_opc in [
            ("add", OpCode.ADD), ("sub", OpCode.SUB), ("mul", OpCode.MUL),
            ("div", OpCode.DIV), ("min", OpCode.MIN), ("max", OpCode.MAX),
            ("shl", OpCode.SHL), ("shr", OpCode.SHR),
            ("and", OpCode.AND), ("or", OpCode.OR), ("xor", OpCode.XOR),
        ]:
            instr = assemble_instruction(mnemonic, 0, 1, 2)
            self.assertEqual(instr.opcode, expected_opc)
            self.assertEqual(instr.op_class, OpClass.VALU)
            self.assertEqual(instr.dest, [0])
            self.assertEqual(instr.srcs, [1, 2])

    def test_valu_unary(self):
        for mn, opc in [("not", OpCode.NOT), ("mov", OpCode.MOV), ("bcast", OpCode.BCAST)]:
            instr = assemble_instruction(mn, 3, 5)
            self.assertEqual(instr.opcode, opc)
            self.assertEqual(instr.dest, [3])
            self.assertEqual(instr.srcs, [5])

    def test_valu_ternary(self):
        instr = assemble_instruction("fma", 0, 1, 2, 3)
        self.assertEqual(instr.opcode, OpCode.FMA)
        self.assertEqual(instr.srcs, [1, 2, 3])

        instr = assemble_instruction("select", 4, 5, 6, 7)
        self.assertEqual(instr.opcode, OpCode.SELECT)

    def test_cmp_to_pred_register(self):
        # CMP_LT writing to P0 (ID = 65)
        instr = assemble_instruction("cmp.lt", PRED_BASE_ID, 1, 2)
        self.assertEqual(instr.dest, [PRED_BASE_ID])

    def test_set_imm(self):
        instr = assemble_instruction("set.imm", 0, 42.0)
        self.assertEqual(instr.opcode, OpCode.SET_IMM)
        self.assertEqual(instr.imm, 42.0)

    def test_special_registers_in_instructions(self):
        instr = assemble_instruction("mov", 5, "SR_LANEID")
        self.assertEqual(instr.opcode, OpCode.MOV)
        self.assertEqual(instr.dest, [5])
        self.assertEqual(instr.srcs, [65])

    def test_jmp(self):
        instr = assemble_instruction("jmp", 10)
        self.assertEqual(instr.opcode, OpCode.JMP)
        self.assertEqual(instr.imm, 10)

    def test_dram_ld_st(self):
        ld = assemble_instruction("dram.ld", 5, 0)
        self.assertEqual(ld.opcode, OpCode.DRAM_LD)
        self.assertEqual(ld.dest, [5])
        self.assertEqual(ld.srcs, [0])

        st = assemble_instruction("dram.st", 0, 5)
        self.assertEqual(st.opcode, OpCode.DRAM_ST)
        self.assertEqual(st.dest, [])
        self.assertEqual(st.srcs, [0, 5])

    def test_sram_ld_st(self):
        ld = assemble_instruction("sram.ld", 3, 1)
        self.assertEqual(ld.opcode, OpCode.SRAM_LD)
        st = assemble_instruction("sram.st", 1, 3)
        self.assertEqual(st.opcode, OpCode.SRAM_ST)

    def test_gemmcore_mma(self):
        alloc = RegisterAllocator()
        d = alloc.alloc_tuple(2, "D")
        a = alloc.alloc_tuple(4, "A")
        b = alloc.alloc_tuple(4, "B")
        c = alloc.alloc_tuple(2, "C")
        instr = assemble_instruction("avx.mma", d, a, b, c)
        self.assertEqual(instr.opcode, OpCode.AVX_MMA)
        self.assertEqual(instr.op_class, OpClass.GEMMCORE)
        self.assertEqual(len(instr.dest), 2)
        self.assertEqual(len(instr.srcs), 10)

    def test_gemmcore_mmred(self):
        alloc = RegisterAllocator()
        d = alloc.alloc_tuple(2)
        a = alloc.alloc_tuple(4)
        b = alloc.alloc_tuple(4)
        c = alloc.alloc_tuple(2)
        for mn, opc in [("avx.mmred.max", OpCode.AVX_MMRED_MAX), ("avx.mmred.min", OpCode.AVX_MMRED_MIN)]:
            instr = assemble_instruction(mn, d, a, b, c)
            self.assertEqual(instr.opcode, opc)

    def test_trans_transcendental(self):
        for mn, opc in [
            ("exp.trans", OpCode.EXP_TRANS), ("sin.trans", OpCode.SIN_TRANS),
            ("cos.trans", OpCode.COS_TRANS), ("tanh.trans", OpCode.TANH_TRANS),
            ("log.trans", OpCode.LOG_TRANS),
        ]:
            instr = assemble_instruction(mn, 5, 3)
            self.assertEqual(instr.opcode, opc)
            self.assertEqual(instr.dest, [5])
            self.assertEqual(instr.srcs, [3])

    def test_trans_fast(self):
        for mn, opc in [
            ("avx.exp.fast", OpCode.AVX_EXP_FAST), ("avx.sin.fast", OpCode.AVX_SIN_FAST),
            ("avx.cos.fast", OpCode.AVX_COS_FAST), ("avx.tanh.fast", OpCode.AVX_TANH_FAST),
            ("avx.log.fast", OpCode.AVX_LOG_FAST),
        ]:
            instr = assemble_instruction(mn, 5, 3, 7)
            self.assertEqual(instr.opcode, opc)
            self.assertEqual(instr.srcs, [3, 7])

    def test_sync_instructions(self):
        for mn, opc in [("sync", OpCode.SYNC), ("sync.warp", OpCode.SYNC_WARP), ("yield", OpCode.YIELD)]:
            instr = assemble_instruction(mn)
            self.assertEqual(instr.opcode, opc)
            self.assertEqual(instr.dest, [])
            self.assertEqual(instr.srcs, [])

    def test_inline_predicate_prefix(self):
        instr = assemble_instruction("@P0 add", 0, 1, 2)
        self.assertEqual(instr.pred, 0)
        self.assertFalse(instr.pred_inv)

        instr = assemble_instruction("@!P3 sub", 0, 1, 2)
        self.assertEqual(instr.pred, 3)
        self.assertTrue(instr.pred_inv)

    def test_keyword_predicate(self):
        p = VirtualPred(2)
        instr = assemble_instruction("add", 0, 1, 2, pred=p)
        self.assertEqual(instr.pred, 2)
        self.assertFalse(instr.pred_inv)

        instr = assemble_instruction("add", 0, 1, 2, pred=~p)
        self.assertEqual(instr.pred, 2)
        self.assertTrue(instr.pred_inv)

    def test_virtualreg_operands(self):
        alloc = RegisterAllocator()
        rd = alloc.alloc("d")
        ra = alloc.alloc("a")
        rb = alloc.alloc("b")
        instr = assemble_instruction("add", rd, ra, rb)
        self.assertEqual(instr.dest, [rd.id])
        self.assertEqual(instr.srcs, [ra.id, rb.id])

    def test_unknown_mnemonic_raises(self):
        with self.assertRaises(ValueError):
            assemble_instruction("nonexistent_op", 0, 1, 2)

    def test_wrong_arity_raises(self):
        with self.assertRaises(ValueError):
            assemble_instruction("add", 0, 1)  # Missing operand
        with self.assertRaises(ValueError):
            assemble_instruction("set.imm", 0)  # Missing immediate


class TestInstructionLifecycle(unittest.TestCase):
    """Verify Instruction clone_for_issue, active_lane_mask, retire with pending_writes."""

    def test_clone_for_issue_is_independent(self):
        static = assemble_instruction("add", 0, 1, 2)
        self.assertEqual(static.state, InstrState.PENDING)
        clone = static.clone_for_issue(warp_id=0, issue_time=10)
        self.assertEqual(clone.state, InstrState.IN_FLIGHT)
        self.assertEqual(clone.warp_id, 0)
        self.assertEqual(clone.issue_time, 10)
        # Static template is unmodified
        self.assertEqual(static.state, InstrState.PENDING)
        self.assertIsNone(static.warp_id)

    def test_active_lane_mask_no_pred(self):
        instr = assemble_instruction("add", 0, 1, 2)
        mask = instr.active_lane_mask(None)
        self.assertTrue(np.all(mask))

    def test_active_lane_mask_with_pred(self):
        _, preds = make_warp_state()
        preds[2, 0:16] = True
        preds[2, 16:32] = False
        instr = assemble_instruction("@P2 add", 0, 1, 2)
        mask = instr.active_lane_mask(preds)
        self.assertTrue(np.all(mask[:16]))
        self.assertFalse(np.any(mask[16:]))

    def test_active_lane_mask_inverted(self):
        _, preds = make_warp_state()
        preds[1, :] = True
        preds[1, 0] = False
        instr = assemble_instruction("@!P1 add", 0, 1, 2)
        mask = instr.active_lane_mask(preds)
        self.assertTrue(mask[0])
        self.assertFalse(np.any(mask[1:]))

    def test_retire_commits_pending_writes(self):
        regs, preds = make_warp_state()
        regs[1, :] = np.arange(32, dtype=np.float32)
        regs[2, :] = np.ones(32, dtype=np.float32)

        instr = assemble_instruction("add", 0, 1, 2)
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        clone.completion_time = 1
        mask = np.ones(32, dtype=bool)
        result = (regs[1] + regs[2]).astype(np.float32)
        clone.pending_writes = [(0, result, mask)]

        self.assertFalse(clone.retire(0, regs, preds))
        self.assertTrue(clone.retire(1, regs, preds))
        self.assertEqual(clone.state, InstrState.COMPLETE)
        np.testing.assert_array_equal(regs[0], np.arange(32, dtype=np.float32) + 1.0)

    def test_retire_to_rz_is_discarded(self):
        regs, preds = make_warp_state()
        instr = assemble_instruction("add", RZ_ID, 1, 2)
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        clone.completion_time = 1
        mask = np.ones(32, dtype=bool)
        clone.pending_writes = [(RZ_ID, np.ones(32, dtype=np.float32), mask)]
        self.assertTrue(clone.retire(1, regs, preds))

    def test_retire_to_pred_register(self):
        regs, preds = make_warp_state()
        instr = assemble_instruction("cmp.lt", PRED_BASE_ID + 3, 0, 1)
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        clone.completion_time = 1
        mask = np.ones(32, dtype=bool)
        result = np.zeros(32, dtype=np.float32)
        result[:16] = 1.0
        clone.pending_writes = [(PRED_BASE_ID + 3, result, mask)]
        clone.retire(1, regs, preds)
        self.assertTrue(np.all(preds[3, :16]))
        self.assertFalse(np.any(preds[3, 16:]))

    def test_retire_with_predicated_mask(self):
        regs, preds = make_warp_state()
        regs[0, :] = np.full(32, -1.0, dtype=np.float32)
        instr = assemble_instruction("add", 0, 1, 2)
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        clone.completion_time = 1
        mask = np.zeros(32, dtype=bool)
        mask[:8] = True
        result = np.full(32, 99.0, dtype=np.float32)
        clone.pending_writes = [(0, result, mask)]
        clone.retire(1, regs, preds)
        np.testing.assert_array_equal(regs[0, :8], np.full(8, 99.0, dtype=np.float32))
        np.testing.assert_array_equal(regs[0, 8:], np.full(24, -1.0, dtype=np.float32))

    def test_disassemble_formatting(self):
        instr = assemble_instruction("add", 0, 1, 2)
        self.assertIn("add", instr.disassemble())
        self.assertIn("R0", instr.disassemble())

        instr = assemble_instruction("@!P1 sub", 3, 4, 5)
        self.assertIn("@!P1", instr.disassemble())

        alloc = RegisterAllocator()
        d = alloc.alloc_tuple(2)
        a = alloc.alloc_tuple(4)
        b = alloc.alloc_tuple(4)
        c = alloc.alloc_tuple(2)
        instr = assemble_instruction("avx.mma", d, a, b, c)
        dis = instr.disassemble()
        self.assertIn("avx.mma", dis)


class TestVALUResource(unittest.TestCase):
    """Verify VALUResource issue, non-pipelined stalling, all opcodes, and BCAST cooperative semantics."""

    def setUp(self):
        self.valu = VALUResource()
        self.regs, self.preds = make_warp_state()

    def _issue(self, mnemonic, *operands, pred=None, **kwargs):
        instr = assemble_instruction(mnemonic, *operands, pred=pred)
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        return self.valu.issue(clone, self.regs, self.preds, kwargs.get("cycle", 0)), clone

    def test_add_basic(self):
        self.regs[1, :] = np.arange(32, dtype=np.float32)
        self.regs[2, :] = np.ones(32, dtype=np.float32)
        (state, comp, pw), clone = self._issue("add", 0, 1, 2)
        self.assertEqual(state, WarpState.WARP_READY)
        self.assertEqual(comp, 1)
        clone.retire(1, self.regs, self.preds)
        np.testing.assert_array_almost_equal(self.regs[0], np.arange(32) + 1.0)

    def test_div_latency_4(self):
        self.regs[1, :] = 10.0
        self.regs[2, :] = 2.0
        (state, comp, _), clone = self._issue("div", 0, 1, 2)
        self.assertEqual(state, WarpState.WARP_READY)
        self.assertEqual(comp, 4)
        clone.retire(4, self.regs, self.preds)
        np.testing.assert_array_almost_equal(self.regs[0], np.full(32, 5.0))

    def test_non_pipelined_stall(self):
        self.regs[1, :] = 1.0
        self.regs[2, :] = 2.0
        # Issue DIV at cycle 0 (occupies VALU until cycle 4)
        (s1, c1, _), _ = self._issue("div", 0, 1, 2)
        self.assertEqual(s1, WarpState.WARP_READY)
        self.assertEqual(c1, 4)

        # Attempt ADD at cycle 1 -> stall
        instr2 = assemble_instruction("add", 3, 1, 2)
        clone2 = instr2.clone_for_issue(warp_id=0, issue_time=1)
        (s2, unblock, _) = self.valu.issue(clone2, self.regs, self.preds, 1)
        self.assertEqual(s2, WarpState.WARP_COMPUTE_STALL)
        self.assertEqual(unblock, 4)

        # Issue ADD at cycle 4 -> succeeds
        (s3, c3, _) = self.valu.issue(clone2, self.regs, self.preds, 4)
        self.assertEqual(s3, WarpState.WARP_READY)
        self.assertEqual(c3, 5)

    def test_fma(self):
        self.regs[1, :] = 3.0
        self.regs[2, :] = 4.0
        self.regs[3, :] = 5.0
        (state, comp, _), clone = self._issue("fma", 0, 1, 2, 3)
        clone.retire(comp, self.regs, self.preds)
        np.testing.assert_array_almost_equal(self.regs[0], np.full(32, 17.0))

    def test_min_max(self):
        self.regs[1, :] = np.arange(32, dtype=np.float32)
        self.regs[2, :] = 15.0
        (_, _, _), clone_min = self._issue("min", 0, 1, 2)
        clone_min.retire(1, self.regs, self.preds)
        np.testing.assert_array_almost_equal(self.regs[0], np.minimum(np.arange(32), 15.0))

        self.valu.reset()
        (_, _, _), clone_max = self._issue("max", 3, 1, 2)
        clone_max.retire(1, self.regs, self.preds)
        np.testing.assert_array_almost_equal(self.regs[3], np.maximum(np.arange(32), 15.0))

    def test_cmp_lt_to_pred(self):
        self.regs[0, :] = np.arange(32, dtype=np.float32)
        self.regs[1, :] = 16.0
        (_, comp, _), clone = self._issue("cmp.lt", PRED_BASE_ID + 0, 0, 1)
        clone.retire(comp, self.regs, self.preds)
        self.assertTrue(np.all(self.preds[0, :16]))
        self.assertFalse(np.any(self.preds[0, 16:]))

    def test_set_imm(self):
        (_, comp, _), clone = self._issue("set.imm", 5, 42.0)
        clone.retire(comp, self.regs, self.preds)
        np.testing.assert_array_almost_equal(self.regs[5], np.full(32, 42.0))

    def test_special_registers_evaluation(self):
        # Read SR_LANEID into R0
        (_, comp, _), clone = self._issue("mov", 0, "SR_LANEID")
        clone.retire(comp, self.regs, self.preds)
        np.testing.assert_array_equal(self.regs[0], np.arange(32, dtype=np.float32))

        # Compute global TID: R2 = SR_WARPID * 32 + SR_LANEID (wid = 2)
        self.regs[1] = 32.0
        clone_fma = assemble_instruction("fma", 2, "SR_WARPID", 1, "SR_LANEID")
        clone_fma.warp_id = 2
        self.valu.reset()
        _, comp_fma, _ = self.valu.issue(clone_fma, self.regs, self.preds, current_cycle=comp)
        clone_fma.retire(comp_fma, self.regs, self.preds)
        expected_gtid = 2 * 32 + np.arange(32, dtype=np.float32)
        np.testing.assert_array_equal(self.regs[2], expected_gtid)

    def test_bcast_cooperative(self):
        """BCAST is cooperative: all 32 lanes get lane 0 value; predicate applied at writeback."""
        self.regs[1, 0] = 77.0
        self.regs[1, 1:] = -1.0
        self.regs[0, :] = 0.0
        self.preds[0, :16] = True
        self.preds[0, 16:] = False
        (_, comp, _), clone = self._issue("bcast", 0, 1, pred=VirtualPred(0))
        clone.retire(comp, self.regs, self.preds)
        np.testing.assert_array_almost_equal(self.regs[0, :16], np.full(16, 77.0))
        np.testing.assert_array_almost_equal(self.regs[0, 16:], np.full(16, 0.0))

    def test_select(self):
        self.regs[1, :16] = 1.0
        self.regs[1, 16:] = 0.0
        self.regs[2, :] = 10.0
        self.regs[3, :] = 20.0
        (_, comp, _), clone = self._issue("select", 0, 1, 2, 3)
        clone.retire(comp, self.regs, self.preds)
        np.testing.assert_array_almost_equal(self.regs[0, :16], np.full(16, 10.0))
        np.testing.assert_array_almost_equal(self.regs[0, 16:], np.full(16, 20.0))

    def test_bitwise_not(self):
        val = np.array([0x0000FFFF] * 32, dtype=np.uint32)
        self.regs[1, :] = val.view(np.float32)
        (_, comp, _), clone = self._issue("not", 0, 1)
        clone.retire(comp, self.regs, self.preds)
        result_int = self.regs[0].view(np.int32)
        expected_int = (~val.view(np.int32))
        np.testing.assert_array_equal(result_int, expected_int)

    def test_jmp_sets_branch_fields(self):
        (_, comp, _), clone = self._issue("jmp", 42)
        self.assertTrue(clone.branch_taken)
        self.assertEqual(clone.branch_target, 42)


class TestGEMMCoreResource(unittest.TestCase):
    """Verify GEMMCoreResource MMA, MMRED_MAX, MMRED_MIN, non-pipelined stall, and register ownership."""

    def setUp(self):
        self.gemm = GEMMCoreResource()
        self.regs, self.preds = make_warp_state()

    def test_mma_identity(self):
        """MMA with A = identity(8, 16), B = identity(8, 16), C = zeros -> D = I(8x8)."""
        # A is 8x16 identity-like: A[i, i] = 1.0 for i in [0..7], rest 0
        A = np.zeros((8, 16), dtype=np.float32)
        for i in range(8):
            A[i, i] = 1.0
        B = A.copy()

        # Pack A into 4 registers (column-major 8x8 sub-blocks of 4x8 tiles)
        a_ids = [0, 1, 2, 3]
        self.regs[0, :] = A[0:4, 0:8].reshape(32)
        self.regs[1, :] = A[4:8, 0:8].reshape(32)
        self.regs[2, :] = A[0:4, 8:16].reshape(32)
        self.regs[3, :] = A[4:8, 8:16].reshape(32)

        b_ids = [4, 5, 6, 7]
        self.regs[4, :] = B[0:4, 0:8].reshape(32)
        self.regs[5, :] = B[4:8, 0:8].reshape(32)
        self.regs[6, :] = B[0:4, 8:16].reshape(32)
        self.regs[7, :] = B[4:8, 8:16].reshape(32)

        c_ids = [8, 9]
        self.regs[8, :] = 0.0
        self.regs[9, :] = 0.0

        d_ids = [10, 11]
        instr = Instruction(
            opcode=OpCode.AVX_MMA, op_class=OpClass.GEMMCORE,
            dest=d_ids,
            srcs=a_ids + b_ids + c_ids,
        )
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        state, comp, _ = self.gemm.issue(clone, self.regs, self.preds, 0)
        self.assertEqual(state, WarpState.WARP_READY)
        self.assertEqual(comp, 8)

        clone.retire(8, self.regs, self.preds)

        D = np.zeros((8, 8), dtype=np.float32)
        D[0:4, 0:8] = self.regs[10].reshape(4, 8)
        D[4:8, 0:8] = self.regs[11].reshape(4, 8)
        expected = A @ B.T
        np.testing.assert_array_almost_equal(D, expected)

    def test_mma_accumulate_into_c(self):
        """MMA in-place: D = A @ B^T + C, where D aliases C."""
        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(8, 16).astype(np.float32)
        C = np.random.randn(8, 8).astype(np.float32)

        self.regs[0, :] = A[0:4, 0:8].reshape(32)
        self.regs[1, :] = A[4:8, 0:8].reshape(32)
        self.regs[2, :] = A[0:4, 8:16].reshape(32)
        self.regs[3, :] = A[4:8, 8:16].reshape(32)

        self.regs[4, :] = B[0:4, 0:8].reshape(32)
        self.regs[5, :] = B[4:8, 0:8].reshape(32)
        self.regs[6, :] = B[0:4, 8:16].reshape(32)
        self.regs[7, :] = B[4:8, 8:16].reshape(32)

        # C in regs 8,9 — D also targets regs 8,9 (in-place accumulation)
        self.regs[8, :] = C[0:4, 0:8].reshape(32)
        self.regs[9, :] = C[4:8, 0:8].reshape(32)

        instr = Instruction(
            opcode=OpCode.AVX_MMA, op_class=OpClass.GEMMCORE,
            dest=[8, 9], srcs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        self.gemm.issue(clone, self.regs, self.preds, 0)
        clone.retire(8, self.regs, self.preds)

        D = np.zeros((8, 8), dtype=np.float32)
        D[0:4, 0:8] = self.regs[8].reshape(4, 8)
        D[4:8, 0:8] = self.regs[9].reshape(4, 8)
        expected = (A @ B.T + C).astype(np.float32)
        np.testing.assert_array_almost_equal(D, expected, decimal=1)

    def test_mmred_max(self):
        """MMRED_MAX: D[r,c] = max(C[r,c], max_k(A[r,k]*B[c,k]))."""
        A = np.ones((8, 16), dtype=np.float32) * 2.0
        B = np.ones((8, 16), dtype=np.float32) * 3.0
        C = np.full((8, 8), -999.0, dtype=np.float32)

        for i, reg_id in enumerate([0, 1, 2, 3]):
            row_slice = slice(0, 4) if i % 2 == 0 else slice(4, 8)
            col_slice = slice(0, 8) if i < 2 else slice(8, 16)
            self.regs[reg_id, :] = A[row_slice, col_slice].reshape(32)
        for i, reg_id in enumerate([4, 5, 6, 7]):
            row_slice = slice(0, 4) if i % 2 == 0 else slice(4, 8)
            col_slice = slice(0, 8) if i < 2 else slice(8, 16)
            self.regs[reg_id, :] = B[row_slice, col_slice].reshape(32)
        self.regs[8, :] = C[0:4, 0:8].reshape(32)
        self.regs[9, :] = C[4:8, 0:8].reshape(32)

        instr = Instruction(
            opcode=OpCode.AVX_MMRED_MAX, op_class=OpClass.GEMMCORE,
            dest=[10, 11], srcs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        self.gemm.issue(clone, self.regs, self.preds, 0)
        clone.retire(8, self.regs, self.preds)

        D = np.zeros((8, 8), dtype=np.float32)
        D[0:4, 0:8] = self.regs[10].reshape(4, 8)
        D[4:8, 0:8] = self.regs[11].reshape(4, 8)
        # A[r,k]*B[c,k] = 2*3 = 6 for all k; max over 16 k-values = 6; max(C=-999, 6) = 6
        np.testing.assert_array_almost_equal(D, np.full((8, 8), 6.0))

    def test_non_pipelined_stall(self):
        """Two consecutive MMA instructions: second must stall for 8 cycles."""
        instr1 = Instruction(
            opcode=OpCode.AVX_MMA, op_class=OpClass.GEMMCORE,
            dest=[10, 11], srcs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        clone1 = instr1.clone_for_issue(warp_id=0, issue_time=0)
        s1, c1, _ = self.gemm.issue(clone1, self.regs, self.preds, 0)
        self.assertEqual(s1, WarpState.WARP_READY)
        self.assertEqual(c1, 8)

        instr2 = Instruction(
            opcode=OpCode.AVX_MMA, op_class=OpClass.GEMMCORE,
            dest=[12, 13], srcs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        clone2 = instr2.clone_for_issue(warp_id=0, issue_time=2)
        s2, unblock, _ = self.gemm.issue(clone2, self.regs, self.preds, 2)
        self.assertEqual(s2, WarpState.WARP_COMPUTE_STALL)
        self.assertEqual(unblock, 8)

    def test_cooperative_predication(self):
        """GEMMCORE computes full 8x8 result; guard predicate only masks writeback lanes."""
        A = np.eye(8, 16, dtype=np.float32)
        B = np.eye(8, 16, dtype=np.float32)

        self.regs[0, :] = A[0:4, 0:8].reshape(32)
        self.regs[1, :] = A[4:8, 0:8].reshape(32)
        self.regs[2, :] = A[0:4, 8:16].reshape(32)
        self.regs[3, :] = A[4:8, 8:16].reshape(32)
        self.regs[4, :] = B[0:4, 0:8].reshape(32)
        self.regs[5, :] = B[4:8, 0:8].reshape(32)
        self.regs[6, :] = B[0:4, 8:16].reshape(32)
        self.regs[7, :] = B[4:8, 8:16].reshape(32)
        self.regs[8, :] = 0.0
        self.regs[9, :] = 0.0
        self.regs[10, :] = -99.0
        self.regs[11, :] = -99.0

        # Predicate: only lanes 0..15 active
        self.preds[0, :16] = True
        self.preds[0, 16:] = False

        instr = Instruction(
            opcode=OpCode.AVX_MMA, op_class=OpClass.GEMMCORE,
            dest=[10, 11], srcs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            pred=0, pred_inv=False,
        )
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        self.gemm.issue(clone, self.regs, self.preds, 0)
        clone.retire(8, self.regs, self.preds)

        # Lanes 0..15 should have the computed D values
        expected_D = (A @ B.T).astype(np.float32)
        d0_top16 = self.regs[10, :16]
        expected_top16 = expected_D[0:4, 0:8].reshape(32)[:16]
        np.testing.assert_array_almost_equal(d0_top16, expected_top16)
        # Lanes 16..31 should retain -99.0
        np.testing.assert_array_almost_equal(self.regs[10, 16:], np.full(16, -99.0))


class TestTransResource(unittest.TestCase):
    """Verify TransResource issue, latencies, Taylor series implementations, and non-pipelined stall."""

    def setUp(self):
        self.trans = TransResource()
        self.regs, self.preds = make_warp_state()

    def test_trans_mode_latency_24(self):
        self.regs[1, :] = 1.0
        instr = assemble_instruction("exp.trans", 0, 1)
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        s, comp, _ = self.trans.issue(clone, self.regs, self.preds, 0)
        self.assertEqual(s, WarpState.WARP_READY)
        self.assertEqual(comp, 24)
        clone.retire(24, self.regs, self.preds)
        np.testing.assert_array_almost_equal(self.regs[0], np.full(32, np.e, dtype=np.float32), decimal=5)

    def test_fast_mode_latency_formula(self):
        """Verify 4 + ceil(1.5 * log2(n)) for exp/sin/cos/log and 6 + 3*log2(n) for tanh."""
        expected_latencies = {4: 7, 8: 9, 16: 10, 32: 12, 64: 13, 128: 15}
        expected_tanh_latencies = {4: 12, 8: 15, 16: 18, 32: 21, 64: 24, 128: 27}

        for n in VALID_TAYLOR_TERMS:
            self.regs[1, :] = 1.0
            self.regs[2, :] = float(n)
            self.trans.reset()

            instr = assemble_instruction("avx.exp.fast", 0, 1, 2)
            clone = instr.clone_for_issue(warp_id=0, issue_time=0)
            _, comp, _ = self.trans.issue(clone, self.regs, self.preds, 0)
            self.assertEqual(comp, expected_latencies[n], f"exp.fast n={n}")

            self.trans.reset()
            instr2 = assemble_instruction("avx.tanh.fast", 0, 1, 2)
            clone2 = instr2.clone_for_issue(warp_id=0, issue_time=0)
            _, comp2, _ = self.trans.issue(clone2, self.regs, self.preds, 0)
            self.assertEqual(comp2, expected_tanh_latencies[n], f"tanh.fast n={n}")

    def test_non_pipelined_stall(self):
        self.regs[1, :] = 1.0
        instr1 = assemble_instruction("exp.trans", 0, 1)
        clone1 = instr1.clone_for_issue(warp_id=0, issue_time=0)
        s1, c1, _ = self.trans.issue(clone1, self.regs, self.preds, 0)
        self.assertEqual(s1, WarpState.WARP_READY)
        self.assertEqual(c1, 24)

        instr2 = assemble_instruction("sin.trans", 3, 1)
        clone2 = instr2.clone_for_issue(warp_id=0, issue_time=5)
        s2, unblock, _ = self.trans.issue(clone2, self.regs, self.preds, 5)
        self.assertEqual(s2, WarpState.WARP_COMPUTE_STALL)
        self.assertEqual(unblock, 24)

    def test_exp_taylor_no_overflow_n128(self):
        """exp_taylor with n=128 should not produce NaN (multiplicative recurrence avoids factorial overflow)."""
        x = np.array([1.0, 2.0, -1.0, 0.5] * 8, dtype=np.float32)
        result = exp_taylor(x, n=128)
        self.assertFalse(np.any(np.isnan(result)), "exp_taylor n=128 produced NaN!")
        np.testing.assert_array_almost_equal(result, np.exp(x).astype(np.float32), decimal=3)

    def test_sin_cos_taylor_convergence(self):
        x = np.linspace(-3.0, 3.0, 32, dtype=np.float32)
        sin_approx = sin_taylor(x, n=64)
        cos_approx = cos_taylor(x, n=64)
        np.testing.assert_array_almost_equal(sin_approx, np.sin(x).astype(np.float32), decimal=4)
        np.testing.assert_array_almost_equal(cos_approx, np.cos(x).astype(np.float32), decimal=4)

    def test_tanh_taylor_convergence(self):
        x = np.linspace(-2.0, 2.0, 32, dtype=np.float32)
        tanh_approx = tanh_taylor(x, n=64)
        np.testing.assert_array_almost_equal(tanh_approx, np.tanh(x).astype(np.float32), decimal=3)

    def test_log_taylor_convergence(self):
        x = np.linspace(0.5, 3.0, 32, dtype=np.float32)
        log_approx = log_taylor(x, n=64)
        np.testing.assert_array_almost_equal(log_approx, np.log(x).astype(np.float32), decimal=2)

    def test_invalid_n_raises(self):
        self.regs[1, :] = 1.0
        self.regs[2, :] = 5.0  # Not in {4, 8, 16, 32, 64, 128}
        instr = assemble_instruction("avx.exp.fast", 0, 1, 2)
        clone = instr.clone_for_issue(warp_id=0, issue_time=0)
        with self.assertRaises(ValueError):
            self.trans.issue(clone, self.regs, self.preds, 0)

    def test_all_5_trans_mode_functions(self):
        """Verify all 5 transcendental mode functions return correct np.* results."""
        self.regs[1, :] = 1.0
        for mn, np_fn in [
            ("exp.trans", np.exp), ("sin.trans", np.sin), ("cos.trans", np.cos),
            ("tanh.trans", np.tanh), ("log.trans", np.log),
        ]:
            self.trans.reset()
            instr = assemble_instruction(mn, 0, 1)
            clone = instr.clone_for_issue(warp_id=0, issue_time=0)
            self.trans.issue(clone, self.regs, self.preds, 0)
            clone.retire(24, self.regs, self.preds)
            expected = np_fn(np.full(32, 1.0)).astype(np.float32)
            np.testing.assert_array_almost_equal(self.regs[0], expected, decimal=5,
                                                  err_msg=f"{mn} failed")


class TestWarpKernel(unittest.TestCase):
    """Verify single-threaded SPMD WarpKernel builder: alloc, asm, label/jmp resolution, and emit_fast_trans."""

    def test_basic_kernel_emission(self):
        kb = WarpKernel()
        r0 = kb.alloc_reg("lane")
        r1 = kb.alloc_reg("val")
        kb.asm("mov", r0, kb.sr_laneid)
        kb.asm("set.imm", r1, 42.0)
        prog = kb.get_program()
        self.assertEqual(len(prog), 2)
        self.assertEqual(prog[0].opcode, OpCode.MOV)
        self.assertEqual(prog[1].opcode, OpCode.SET_IMM)

    def test_label_and_jmp_resolution(self):
        kb = WarpKernel()
        r0 = kb.alloc_reg()
        kb.label("loop_start")
        kb.asm("add", r0, r0, r0)
        kb.asm("jmp", "loop_start")
        prog = kb.get_program()
        jmp_instr = prog[-1]
        self.assertEqual(jmp_instr.opcode, OpCode.JMP)
        self.assertEqual(jmp_instr.imm, 0)  # PC 0 is where the label was bound

    def test_unresolved_label_raises(self):
        kb = WarpKernel()
        kb.asm("jmp", "nonexistent")
        with self.assertRaises(KeyError):
            kb.get_program()

    def test_predicated_asm(self):
        kb = WarpKernel()
        r0 = kb.alloc_reg()
        p0 = kb.alloc_pred("mask")
        kb.asm("add", r0, r0, r0, pred=p0)
        instr = kb.instructions[0]
        self.assertEqual(instr.pred, 0)
        self.assertFalse(instr.pred_inv)

        kb.asm("sub", r0, r0, r0, pred=~p0)
        instr2 = kb.instructions[1]
        self.assertEqual(instr2.pred, 0)
        self.assertTrue(instr2.pred_inv)

    def test_gemmcore_via_warp_kernel(self):
        kb = WarpKernel()
        d = kb.alloc_tuple(2, "D")
        a = kb.alloc_tuple(4, "A")
        b = kb.alloc_tuple(4, "B")
        c = kb.alloc_tuple(2, "C")
        kb.asm("avx.mma", d, a, b, c)
        instr = kb.instructions[0]
        self.assertEqual(instr.opcode, OpCode.AVX_MMA)
        self.assertEqual(len(instr.dest), 2)
        self.assertEqual(len(instr.srcs), 10)

    def test_emit_fast_trans_intrinsic(self):
        kb = WarpKernel()
        r_dest = kb.alloc_reg("result")
        r_x = kb.alloc_reg("input")
        from pgpu.sw.intrinsics import emit_fast_trans
        instrs = emit_fast_trans(kb, "exp", r_dest, r_x, n=64)
        self.assertEqual(len(instrs), 2)
        self.assertEqual(instrs[0].opcode, OpCode.SET_IMM)
        self.assertEqual(instrs[0].imm, 64)
        self.assertEqual(instrs[1].opcode, OpCode.AVX_EXP_FAST)


if __name__ == "__main__":
    unittest.main()
