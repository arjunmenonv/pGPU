'''
Unit tests for the pGPU SMSP hardware model and Round-Robin Warp Scheduler.

Location: pgpu/tests/pGPUInfra/test_warp_scheduler.py
'''

import unittest
import numpy as np

from pgpu.sw.kernel import WarpKernel
from pgpu.arch.isa import (
    OpCode,
)

from pgpu.arch.smsp import (
    SMSP,
    WarpState,
)
from pgpu.arch.memory import DRAMResource, SRAMResource


class TestWarpScheduler(unittest.TestCase):
    def setUp(self):
        self.dram = DRAMResource()
        self.sram = SRAMResource()
        self.smsp = SMSP(num_warps=4, dram=self.dram, sram=self.sram)

    def test_single_warp_straight_line(self):
        """Verify basic straight-line kernel execution and cycle progression on 1 warp."""
        smsp = SMSP(num_warps=1, dram=self.dram, sram=self.sram)
        kb = WarpKernel()
        r0 = kb.alloc_reg("r0")
        r1 = kb.alloc_reg("r1")
        r2 = kb.alloc_reg("r2")

        kb.asm("set.imm", r0, 10.0)
        kb.asm("set.imm", r1, 20.0)
        kb.asm("add", r2, r0, r1)
        program = kb.get_program()

        cycles = smsp.run(program)
        self.assertGreater(cycles, 0)
        self.assertTrue(np.all(smsp.warps[0].regs[r2.id] == 30.0))
        self.assertEqual(smsp.context_switch_count, 0)

    def test_multi_warp_spmd_differentiation(self):
        """
        Verify that 4 warps executing the exact same SPMD kernel compute unique global thread IDs
        using `SR_WARPID` and `SR_LANEID` without needing per-warp source code.
        """
        kb = WarpKernel()
        r_scale = kb.alloc_reg("scale")
        r_gtid = kb.alloc_reg("gtid")

        kb.asm("set.imm", r_scale, 32.0)
        kb.asm("fma", r_gtid, kb.sr_warpid, r_scale, kb.sr_laneid)
        program = kb.get_program()

        cycles = self.smsp.run(program)
        self.assertGreater(cycles, 0)

        for w in range(4):
            expected = w * 32 + np.arange(32, dtype=np.float32)
            np.testing.assert_array_equal(self.smsp.warps[w].regs[r_gtid.id], expected)

    def test_warp_switch_on_yield(self):
        """Verify that YIELD causes a warp switch and incurs the 80-cycle context switch penalty."""
        kb = WarpKernel()
        r0 = kb.alloc_reg("r0")
        kb.asm("set.imm", r0, 5.0)
        kb.asm("yield")
        kb.asm("add", r0, r0, r0)
        program = kb.get_program()

        cycles = self.smsp.run(program)
        # Switching across 4 warps paying context switch costs
        self.assertGreater(self.smsp.context_switch_count, 0)
        self.assertGreater(self.smsp.total_context_switch_cycles, 0)
        for w in range(4):
            self.assertTrue(np.all(self.smsp.warps[w].regs[r0.id] == 10.0))

    def test_sync_barrier_coordination(self):
        """Verify that SYNC holds all warps until all 4 have arrived."""
        kb = WarpKernel()
        r0 = kb.alloc_reg("r0")
        kb.asm("set.imm", r0, 1.0)
        kb.asm("sync")
        kb.asm("add", r0, r0, r0)
        program = kb.get_program()

        cycles = self.smsp.run(program)
        self.assertGreater(cycles, 0)
        for w in range(4):
            self.assertTrue(np.all(self.smsp.warps[w].regs[r0.id] == 2.0))

    def test_compute_stall_spin_waits_in_place(self):
        """
        Verify that compute dependency (DIV latency 4 -> ADD) spins in place
        without triggering context switches.
        """
        smsp = SMSP(num_warps=1, dram=self.dram, sram=self.sram)
        kb = WarpKernel()
        r0 = kb.alloc_reg("r0")
        r1 = kb.alloc_reg("r1")
        r2 = kb.alloc_reg("r2")

        kb.asm("set.imm", r0, 100.0)
        kb.asm("set.imm", r1, 4.0)
        kb.asm("div", r2, r0, r1)  # 4 cycle latency
        kb.asm("add", r0, r2, r1)  # RAW on r2 -> compute stall
        program = kb.get_program()

        cycles = smsp.run(program)
        self.assertEqual(smsp.context_switch_count, 0)
        self.assertTrue(np.all(smsp.warps[0].regs[r0.id] == 29.0))


    def test_context_switch_skips_in_flight_long_lat_load(self):
        """
        Verify that when a warp issues a long-latency DRAM load and yields,
        the destination register is skipped during context save and restore,
        ensuring that incoming DRAM data is preserved and not overwritten by scratchpad.
        """
        import pgpu.arch.smsp as smsp_mod
        from pgpu.sw.driver import get_vector_scratch_addr

        # Allocate DRAM data
        self.dram.alloc("weights", 1024)
        for i in range(32):
            self.dram.memory[i] = 42.0
            self.dram.memory[32 + i] = 99.0

        kb = WarpKernel()
        r_scale = kb.alloc_reg("scale")
        r_addr = kb.alloc_reg("addr")
        r_data = kb.alloc_reg("data")
        r_out = kb.alloc_reg("out")

        kb.asm("set.imm", r_scale, 32.0)
        kb.asm("mul", r_addr, kb.sr_warpid, r_scale)
        kb.asm("dram.ld", r_data, r_addr)
        kb.asm("yield")
        kb.asm("add", r_out, r_data, r_data)
        program = kb.get_program()

        smsp = SMSP(num_warps=2, dram=self.dram, sram=self.sram)

        # Hook to verify that r_data is skipped at the first context switch
        orig_switch = smsp_mod.apply_context_switch
        sram_at_yield_switch = {}

        def hook_switch(sched, old_w, new_w):
            cost = orig_switch(sched, old_w, new_w)
            if "yield_switch_w0_rdata" not in sram_at_yield_switch:
                addr_0 = get_vector_scratch_addr(sched, 0, r_data.id)
                sram_at_yield_switch["yield_switch_w0_rdata"] = [
                    sched.sram.memory[int(a)] for a in addr_0
                ]
            return cost

        try:
            smsp_mod.apply_context_switch = hook_switch
            cycles = smsp.run(program)
        finally:
            smsp_mod.apply_context_switch = orig_switch

        self.assertGreater(cycles, 0)

        # Warp 0 loaded 42.0 -> 42.0 + 42.0 = 84.0
        np.testing.assert_array_equal(smsp.warps[0].regs[r_out.id], np.full(32, 84.0, dtype=np.float32))
        # Warp 1 loaded 99.0 -> 99.0 + 99.0 = 198.0
        np.testing.assert_array_equal(smsp.warps[1].regs[r_out.id], np.full(32, 198.0, dtype=np.float32))

        # At the yield switch, r_data was in-flight and thus skipped; scratchpad slot remained untouched (0.0)
        self.assertTrue(all(val == 0.0 for val in sram_at_yield_switch["yield_switch_w0_rdata"]))

        # Context switches occurred
        self.assertGreater(smsp.context_switch_count, 0)

    def test_context_save_clashes_with_large_kernel_sram_allocation(self):
        """
        Verify that a kernel allocating most of the 32K-word SRAM causes an
        allocation-time MemoryError due to the driver reserving space for
        the context-save mechanism upfront.

        For 2 warps, context save needs 2 * 2304 = 4608 words.
        If a user then tries to allocate 28K words (28672 words), the total
        is 33280 words, which exceeds the 32768-word capacity.
        """
        from pgpu.arch.gpu_device import pGPU

        device = pGPU(num_warps_per_smsp=2)
        sram = device.sm.sram

        kb = WarpKernel(num_warps=2)
        
        # 4608 words are already reserved for context save.
        # Allocating 28K (28672 words) will now properly raise a MemoryError
        # instead of silently colliding during execution.
        with self.assertRaises(MemoryError):
            kb.sram.alloc("large_tile", 28 * 1024)


if __name__ == "__main__":
    unittest.main()
