"""
Unit tests for 1024x1 Vector Addition Kernel, 3-Tier pGPU Hierarchy (pGPU -> SM -> SMSP),
Static Shared Memory (`sram.alloc`/`sram.free`), In-Kernel `memset`, and Allocation Bounds.

Location: pgpu/tests/test_vecadd.py
"""

import unittest
import numpy as np

from pgpu.arch.gpu_device import CopyDirection, SM, pGPU
from pgpu.arch.isa import WarpKernel
from pgpu.arch.memory import DRAMResource, MemoryAccessError, SRAMResource
from pgpu.arch.smsp import SMSP
from pgpu.examples.vector_add import build_vector_add_kernel, run_vector_add
from pgpu.sw.intrinsics import memset


class TestVectorAddAndDeviceHierarchy(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_vector_add_1024x1_matches_numpy(self):
        """
        Verify that 1024x1 vector addition executed on pGPU via dram_alloc and dram_copy
        matches reference NumPy implementation bit-for-bit.
        """
        n = 1024
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)

        c, cycles = run_vector_add(a, b, num_warps=4)

        expected = a + b
        self.assertGreater(cycles, 0)
        np.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)

    def test_vector_add_non_multiple_of_grid_stride(self):
        """
        Verify that boundary predication handles problem sizes
        that are not exact multiples of grid stride (128).
        """
        n = 250
        a = np.random.uniform(-10.0, 10.0, n).astype(np.float32)
        b = np.random.uniform(-10.0, 10.0, n).astype(np.float32)

        c, cycles = run_vector_add(a, b, num_warps=4)

        expected = a + b
        self.assertGreater(cycles, 0)
        np.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)

    def test_vector_add_zeros_and_negatives(self):
        """Verify vector addition on zeros and negative numbers."""
        n = 128
        a = np.full(n, -5.5, dtype=np.float32)
        b = np.full(n, 5.5, dtype=np.float32)

        c, _ = run_vector_add(a, b, num_warps=4)

        expected = np.zeros(n, dtype=np.float32)
        np.testing.assert_allclose(c, expected, atol=1e-6)

    def test_vector_add_single_warp(self):
        """Verify vector addition running on a single warp."""
        n = 64
        a = np.arange(n, dtype=np.float32)
        b = np.ones(n, dtype=np.float32) * 2.0

        c, _ = run_vector_add(a, b, num_warps=1)

        expected = a + b
        np.testing.assert_allclose(c, expected, atol=1e-6)

    def test_single_sm_single_smsp_hierarchy_and_enforcement(self):
        """
        Verify 3-tier hierarchy (1 pGPU -> 1 SM -> 1 SMSP) with decoupled memory pointers,
        and verify that >1 SM per pGPU or >1 SMSP per SM raises ValueError.
        """
        shared_dram = DRAMResource(name="SharedDRAM")
        sm0_sram = SRAMResource(name="SRAM_0")
        smsp0 = SMSP(num_warps=4)
        sm0 = SM(sm_id=0, sram=sm0_sram, smsps=[smsp0])
        device = pGPU(name="SingleSM_pGPU", dram=shared_dram, sms=[sm0])

        # Verify pointer topology
        self.assertIs(sm0.dram, shared_dram)
        self.assertIs(smsp0.dram, shared_dram)
        self.assertIs(smsp0.sram, sm0_sram)

        # Verify architectural constraint: only 1 SM per pGPU and 1 SMSP per SM supported
        with self.assertRaises(ValueError):
            SM(sm_id=0, num_smsps=2)
        with self.assertRaises(ValueError):
            pGPU(num_sms=2)
        with self.assertRaises(ValueError):
            pGPU(num_smsps_per_sm=2)

        # Execute 1024x1 vector add on the constructed device
        n = 1024
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c, cycles = run_vector_add(a, b, device=device)

        self.assertGreater(cycles, 0)
        np.testing.assert_allclose(c, a + b, rtol=1e-5, atol=1e-5)

    def test_in_kernel_memset_dram_and_static_sram(self):
        """
        Verify that:
        1. `memset` works from inside a WarpKernel on both DRAM and static SRAM allocations.
        2. Static shared memory (`kb.sram.alloc` / `kb.sram.free`) is allocated inside the kernel.
        """
        device = pGPU(num_sms=1, num_smsps_per_sm=1, num_warps_per_smsp=4)
        self.assertFalse(hasattr(device, "sram_copy"))

        n = 256
        dram_buf = device.dram_alloc("out_buf", n)

        kb = WarpKernel(num_warps=4)
        # 1. Static SRAM allocation inside kernel
        sram_tile = kb.sram.alloc("tile", n)

        # 2. Call device function `memset` on DRAM buffer (to 7.5) and on SRAM buffer (to 3.25)
        memset(kb, dram_buf, value=7.5)
        kb.memset(sram_tile, value=3.25)

        # 3. Read one row from SRAM tile, add to DRAM buffer element for warp 0, and store back
        r_addr_s = kb.alloc_reg("addr_s")
        r_addr_d = kb.alloc_reg("addr_d")
        r_val_s  = kb.alloc_reg("val_s")
        r_val_d  = kb.alloc_reg("val_d")
        r_sum    = kb.alloc_reg("sum")
        p_w0     = kb.alloc_pred("p_w0")

        kb.asm("cmp.eq", p_w0, kb.sr_warpid, kb.rz)
        kb.asm("set.imm", r_addr_s, float(sram_tile.base_addr))
        kb.asm("add", r_addr_s, r_addr_s, kb.sr_laneid)
        kb.asm("sram.ld", r_val_s, r_addr_s, pred=p_w0)

        kb.asm("set.imm", r_addr_d, float(dram_buf.base_addr))
        kb.asm("add", r_addr_d, r_addr_d, kb.sr_laneid)
        kb.asm("dram.ld", r_val_d, r_addr_d, pred=p_w0)
        kb.asm("add", r_sum, r_val_d, r_val_s, pred=p_w0)
        kb.asm("dram.st", r_addr_d, r_sum, pred=p_w0)

        kb.sram.free("tile")

        cycles = device.launch(kb)
        self.assertGreater(cycles, 0)

        result = device.dram_copy(dram_buf, direction=CopyDirection.D2H)
        # First 32 elements had 3.25 added to 7.5 -> 10.75; remaining 224 elements are 7.5
        np.testing.assert_allclose(result[:32], np.full(32, 10.75, dtype=np.float32))
        np.testing.assert_allclose(result[32:], np.full(224, 7.5, dtype=np.float32))

    def test_unallocated_kernel_memory_access_raises(self):
        """Verify that a kernel accessing an address beyond allocated DRAM raises MemoryAccessError."""
        device = pGPU(num_warps_per_smsp=1)
        buf = device.dram_alloc("small_buf", 32)

        kb = WarpKernel(num_warps=1)
        r_addr = kb.alloc_reg("oob_addr")
        r_val  = kb.alloc_reg("val")
        # Offset 32..63 is outside [0, 32)
        kb.asm("set.imm", r_addr, float(buf.base_addr + 32))
        kb.asm("add", r_addr, r_addr, kb.sr_laneid)
        kb.asm("dram.ld", r_val, r_addr)

        with self.assertRaises(MemoryAccessError):
            device.launch(kb)


if __name__ == "__main__":
    unittest.main()
