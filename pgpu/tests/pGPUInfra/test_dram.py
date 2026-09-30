'''
Unit tests for DRAMResource hardware model and address mapping.

Location: pgpu/tests/pGPUInfra/test_dram.py
'''

import unittest
import numpy as np
from pgpu.arch.memory import DRAMResource, SRAMResource, InFlightMemOp, MemOpType
from pgpu.arch.smsp import WarpState


class TestDRAMResource(unittest.TestCase):
    def setUp(self):
        self.dram = DRAMResource()

    def test_allocation_and_free(self):
        """Test memory allocation and capacity tracking against 16GB limit."""
        size = 1024 * 1024  # 1M words (4MB)
        allocated = self.dram.alloc("tensor_a", size)
        self.assertEqual(allocated, size)
        self.assertEqual(self.dram.allocated_words, size)

        with self.assertRaises(ValueError):
            self.dram.alloc("tensor_a", 512)

        with self.assertRaises(MemoryError):
            self.dram.alloc("overflow_tensor", self.dram.CAPACITY_WORDS + 1)

        # Unaligned allocation size (10 words) should align next base_addr to 32-word boundary
        u1 = self.dram.alloc("unaligned_1", 10)
        self.assertEqual(u1.base_addr, size)
        u2 = self.dram.alloc("unaligned_2", 16)
        self.assertEqual(u2.base_addr, size + 32)
        self.dram.free("unaligned_1")
        self.dram.free("unaligned_2")

        self.dram.free("tensor_a")
        self.assertEqual(self.dram.allocated_words, 0)

    def test_latency_computation_patterns(self):
        """Test DRAM coalescing latency for contiguous vs tiled load patterns."""
        # (a) Contiguous 32 addresses -> 4 row activations in channel 0
        contig_addrs = np.arange(32, dtype=np.uint64)
        lat_contig = self.dram.compute_latency(contig_addrs)
        self.assertEqual(lat_contig, 240)  # 40 + 4 * 50 = 240

        # (b) 4x8 2D tile, stride 32 -> 1 row activation across 4 channels
        tiled_32 = np.array([(i // 8) * 32 + (i % 8) for i in range(32)], dtype=np.uint64)
        lat_tiled_32 = self.dram.compute_latency(tiled_32)
        self.assertEqual(lat_tiled_32, 90)  # 40 + 1 * 50 = 90

        # (c) 4x8 2D tile, stride 64 -> 1 row activation across 4 channels
        tiled_64 = np.array([(i // 8) * 64 + (i % 8) for i in range(32)], dtype=np.uint64)
        lat_tiled_64 = self.dram.compute_latency(tiled_64)
        self.assertEqual(lat_tiled_64, 90)  # 40 + 1 * 50 = 90

        # (d) 8x4 2D tile, stride 32 -> 1 row activation across 8 channels
        tiled_8x4_32 = np.array([(i // 4) * 32 + (i % 4) for i in range(32)], dtype=np.uint64)
        lat_8x4_32 = self.dram.compute_latency(tiled_8x4_32)
        self.assertEqual(lat_8x4_32, 90)  # 40 + 1 * 50 = 90

        # (e) 8x4 2D tile, stride 64 -> 2 DRAM_REQs (rank 0 & rank 1)
        tiled_8x4_64 = np.array([(i // 4) * 64 + (i % 4) for i in range(32)], dtype=np.uint64)
        lat_8x4_64 = self.dram.compute_latency(tiled_8x4_64)
        self.assertEqual(lat_8x4_64, 140)  # 40 + 2 * 50 = 140

        # (f) Different (rank, bg, bank, row, col >> 3) across DIFFERENT channels coalesce into 1 DRAM_REQ
        diff_rows_across_channels = np.array([0, 32 + 8], dtype=np.uint64)  # ch 0 row 0, ch 1 row 1
        self.assertEqual(self.dram.compute_latency(diff_rows_across_channels), 90)  # 1 DRAM_REQ

        # (g) Two different 8-col bursts (col_hi = 0 and col_hi = 1) within the SAME channel -> 2 DRAM_REQs
        two_col_bursts = np.array([0, 1 << 25], dtype=np.uint64)
        self.assertEqual(self.dram.compute_latency(two_col_bursts), 140)  # 2 DRAM_REQs

    def test_functional_store_load_and_retire(self):
        """Test functional DRAM store, load, WarpState routing, and token-pointer InstrRetireRoutine."""
        self.dram.alloc("buf", 2048)
        addresses = np.arange(100, 132, dtype=np.int64)
        values = np.array([float(i * 1.5) for i in range(32)], dtype=np.float32)

        state_1, end_st, tokens_1 = self.dram.store(addresses, values, current_cycle=0, warp_id=0)
        self.assertEqual(state_1, WarpState.WARP_READY)
        self.assertEqual(end_st, 240)
        self.assertEqual(tokens_1[0].op_type, MemOpType.STORE)
        self.assertEqual(len(self.dram.store_queues[0][0]), 1)

        # Attempting to load the SAME addresses at cycle 10 while store is in-flight in LSQ
        # must route WarpState.WARP_LONG_LAT_STALL with unblock_cycle = 220!
        stall_state, unblock_cyc, stall_tokens, stall_res = self.dram.load(
            addresses, current_cycle=10, warp_id=0
        )
        self.assertEqual(stall_state, WarpState.WARP_LONG_LAT_STALL)
        self.assertEqual(unblock_cyc, 240)
        self.assertIsNone(stall_tokens)
        self.assertIsNone(stall_res)

        # Issue a second fire-and-forget store to distinct addresses at cycle 10:
        # Since LSQ has only 1 entry (< 16), it succeeds immediately with WARP_READY!
        state_2, end_st2, tokens_2 = self.dram.store(addresses + 32, values, current_cycle=10, warp_id=0)
        self.assertEqual(state_2, WarpState.WARP_READY)
        self.assertEqual(end_st2, 440)
        self.assertEqual(len(self.dram.store_queues[0][0]), 2)

        # Retire first store at cycle 220 by passing (warp_id=0, mem_tokens=tokens_1)
        self.dram.InstrRetireRoutine(warp_id=0, mem_tokens=tokens_1)
        self.assertEqual(len(self.dram.store_queues[0][0]), 1)
        self.assertIs(self.dram.store_queues[0][0][0], tokens_2[0])

        # Now loading `addresses` at cycle 220 succeeds immediately with WARP_READY
        # even while store #2 is still draining on the Store Pipe until cycle 440!
        ld_state, end_ld, ld_tokens, read_vals = self.dram.load(addresses, current_cycle=240, warp_id=0)
        self.assertEqual(ld_state, WarpState.WARP_READY)
        self.assertEqual(end_ld, 480)
        self.assertEqual(ld_tokens[0].op_type, MemOpType.LOAD)
        np.testing.assert_array_almost_equal(read_vals, values)

        self.dram.InstrRetireRoutine(warp_id=0, mem_tokens=ld_tokens)
        self.dram.InstrRetireRoutine(warp_id=0, mem_tokens=tokens_2)
        self.assertEqual(len(self.dram.store_queues[0][0]), 0)

    def test_waw_fifo_queueing_and_war_hazard_stall(self):
        """
        Verify:
        1. WAW (ST -> ST to same address): Second store is fire-and-forget (WARP_READY at cycle 10),
           while its completion_cycle is serialized via next_available_store_cycle (440).
        2. WAR (LD -> ST to same address): Store is fire-and-forget (WARP_READY at cycle 10),
           while its Store Pipe drain waits for the conflicting in-flight load to finish at 220.
        """
        self.dram.alloc("buf", 2048)
        addrs = np.arange(100, 132, dtype=np.int64)
        vals1 = np.ones(32, dtype=np.float32)
        vals2 = np.full(32, 2.0, dtype=np.float32)

        state_1, comp_1, _ = self.dram.store(addrs, vals1, current_cycle=0, warp_id=0)
        self.assertEqual(state_1, WarpState.WARP_READY)
        self.assertEqual(comp_1, 240)

        state_2, comp_2, _ = self.dram.store(addrs, vals2, current_cycle=10, warp_id=0)
        self.assertEqual(state_2, WarpState.WARP_READY)
        self.assertEqual(comp_2, 440)

        self.dram.reset()
        self.dram.alloc("buf", 2048)
        ld_state, end_ld, _, _ = self.dram.load(addrs, current_cycle=0, warp_id=0)
        self.assertEqual(ld_state, WarpState.WARP_READY)
        self.assertEqual(end_ld, 240)

        st_state, comp_st, _ = self.dram.store(addrs, vals2, current_cycle=10, warp_id=0)
        self.assertEqual(st_state, WarpState.WARP_READY)
        self.assertEqual(comp_st, 480)

    def test_lsq_capacity_full_routes_warp_long_lat_stall(self):
        self.dram.alloc("buf", 2048)
        """
        Verify that 16 fire-and-forget stores succeed with WARP_READY, and the 17th store
        at cycle 0 routes WARP_LONG_LAT_STALL with unblock_cycle = 220.
        """
        addrs = np.arange(100, 132, dtype=np.int64)
        vals = np.ones(32, dtype=np.float32)

        completions = []
        for i in range(16):
            state, comp_c, _ = self.dram.store(addrs + i * 32, vals, current_cycle=0, warp_id=0)
            self.assertEqual(state, WarpState.WARP_READY)
            completions.append(comp_c)

        # 17th store at cycle 0 hits full LSQ (16 entries) -> routes WARP_LONG_LAT_STALL until 220
        state_17, unblock_17, tokens_17 = self.dram.store(addrs + 16 * 32, vals, current_cycle=0, warp_id=0)
        self.assertEqual(state_17, WarpState.WARP_LONG_LAT_STALL)
        self.assertEqual(unblock_17, completions[0])
        self.assertIsNone(tokens_17)


    def test_upper_bitfields_bankgroup_bank_rowhi(self):
        """Verify that distinct bank_group [13:11], bank [16:14], and row_hi [24:17] in the same channel add DRAM_REQs."""
        # 4 addresses in Channel 0: base, +bank_group(1<<11), +bank(1<<14), +row_hi(1<<17)
        addrs = np.array([0, 1 << 11, 1 << 14, 1 << 17], dtype=np.uint64)
        self.assertEqual(self.dram.compute_latency(addrs), 40 + 4 * 50)

    def test_predicated_mask_on_load_and_store(self):
        self.dram.alloc("buf", 2048)
        """Verify predicated execution (mask) suppresses inactive lane writes, LSQ tokens, and latency penalties."""
        # Lanes 0..7 active (all in channel 0, row 0 -> 1 DRAM_REQ = 70 cycles);
        # Lanes 8..31 masked off (even though their addresses point to rows 1, 2, 3!)
        addrs = np.arange(32, dtype=np.int64)
        vals = np.arange(1, 33, dtype=np.float32)
        mask = np.array([i < 8 for i in range(32)], dtype=bool)

        st_state, comp_st, st_tokens = self.dram.store(addrs, vals, current_cycle=0, warp_id=0, mask=mask)
        self.assertEqual(st_state, WarpState.WARP_READY)
        self.assertEqual(comp_st, 90)  # Only active lanes 0..7 counted -> 1 DRAM_REQ (70), not 4 (220)!
        self.assertIsNotNone(st_tokens[0])
        self.assertIsNone(st_tokens[8])  # Masked-out lane has no LSQ token
        self.assertEqual(len(self.dram.store_queues[0][8]), 0)

        self.dram.InstrRetireRoutine(warp_id=0, mem_tokens=st_tokens)

        ld_state, comp_ld, ld_tokens, read_vals = self.dram.load(addrs, current_cycle=90, warp_id=0, mask=mask)
        self.assertEqual(ld_state, WarpState.WARP_READY)
        self.assertEqual(comp_ld, 180)
        np.testing.assert_array_almost_equal(read_vals[:8], vals[:8])
        np.testing.assert_array_equal(read_vals[8:], np.zeros(24, dtype=np.float32))

    def test_load_pipe_busy_and_multi_warp_isolation(self):
        self.dram.alloc("buf", 2048)
        """
        Verify:
        1. Back-to-back independent LD -> LD stalls on Load Pipe Busy (WARP_LONG_LAT_STALL).
        2. Per-warp LSQ is isolated across warps (Warp 0 in-flight store does not RAW-stall Warp 1),
           while physical Load/Store pipes are shared across warps.
        """
        addrs_w0 = np.arange(100, 132, dtype=np.int64)
        addrs_w1 = np.arange(500, 532, dtype=np.int64)
        vals = np.ones(32, dtype=np.float32)

        # Warp 0 issues store to addrs_w0 at cycle 0 (completes at 220)
        self.dram.store(addrs_w0, vals, current_cycle=0, warp_id=0)

        # Warp 1 loading addrs_w0 at cycle 10 does NOT suffer an intra-thread RAW stall from Warp 0's LSQ
        # (cross-warp memory dependencies require explicit SYNC), so Warp 1 issues on the free Load Pipe!
        ld1_state, ld1_end, _, _ = self.dram.load(addrs_w0, current_cycle=10, warp_id=1)
        self.assertEqual(ld1_state, WarpState.WARP_READY)
        self.assertEqual(ld1_end, 250)  # 10 + 40 (issue) + 200 (perform_request) = 250

        # Load 1 is in Stage 1 (issue) during [10, 50]: trying to issue at cycle 30 stalls until 50 (when ld1.is_issued=True)!
        ld0_state, ld0_unblock, _, _ = self.dram.load(addrs_w1, current_cycle=30, warp_id=0)
        self.assertEqual(ld0_state, WarpState.WARP_LONG_LAT_STALL)
        self.assertEqual(ld0_unblock, 50)

        # At cycle 50 (as soon as ld1.is_issued is True), Warp 0 issues ld2! Its 40-cycle issue [50, 90] overlaps with ld1's perform_request [50, 250], completing at 250 + 200 = 450!
        ld2_state, ld2_end, ld2_tokens, _ = self.dram.load(addrs_w1, current_cycle=50, warp_id=0)
        self.assertEqual(ld2_state, WarpState.WARP_READY)
        self.assertEqual(ld2_end, 400)  # 250 + 3*50 = 400 (40-cycle issue hidden)
        self.assertTrue(ld2_tokens[0].is_issued)
        self.assertTrue(ld2_tokens[0].is_retired)

    def test_alloc_32bit_modulo_wraparound(self):
        """Verify that base_addr and _next_alloc_base wrap modulo (1 << 32)."""
        big_chunk = (1 << 31)
        a1 = self.dram.alloc("half_1", big_chunk)
        self.assertEqual(a1.base_addr, 0)
        a2 = self.dram.alloc("half_2", big_chunk - 64)
        self.assertEqual(a2.base_addr, big_chunk)

        # Free half_1 so capacity is available, then allocate 128 words -> wraps around (1 << 32) to 0!
        self.dram.free("half_1")
        a3 = self.dram.alloc("wrapped", 64)
        self.assertEqual(a3.base_addr, (1 << 32) - 64)
        self.assertEqual(self.dram._next_alloc_base, 0)


    def test_unallocated_address_access_raises_error(self):
        """Verify that accessing unallocated addresses or exceeding allocation bounds raises MemoryError."""
        # 1. No allocation exists -> raises error
        addrs = np.arange(32, dtype=np.int64)
        vals = np.ones(32, dtype=np.float32)
        with self.assertRaises(MemoryError):
            self.dram.load(addrs, current_cycle=0)
        with self.assertRaises(MemoryError):
            self.dram.store(addrs, vals, current_cycle=0)

        # 2. Allocate 32 words [0, 32): intra-allocation succeeds, [16, 48) fails
        self.dram.alloc("buf", 32)
        state, _, _, _ = self.dram.load(addrs, current_cycle=0)
        self.assertEqual(state, WarpState.WARP_READY)

        oob_addrs = np.arange(16, 48, dtype=np.int64)
        with self.assertRaises(MemoryError):
            self.dram.load(oob_addrs, current_cycle=300)
        with self.assertRaises(MemoryError):
            self.dram.store(oob_addrs, vals, current_cycle=300)


if __name__ == "__main__":
    unittest.main()
