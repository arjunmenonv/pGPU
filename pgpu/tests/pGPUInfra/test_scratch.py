'''
Unit tests for SRAMResource hardware model and bijective per-allocation index modifiers.

Location: pgpu/tests/pGPUInfra/test_scratch.py
'''

import unittest
import numpy as np
from pgpu.arch.memory import SRAMResource
from pgpu.arch.smsp import WarpState
from pgpu.sw.driver import swizzle_indices


class TestSRAMResource(unittest.TestCase):
    def setUp(self):
        self.sram = SRAMResource()

    def test_per_allocation_index_modifier_bank_conflicts(self):
        """
        Verify that different tiles allocated in the same SRAMResource can attach
        distinct bijective index_modifier functions (or none at all).
        """
        tile_a = self.sram.alloc("tile_a", 1024, index_modifier=swizzle_indices)
        self.assertEqual(tile_a.base_addr, 0)

        # Buffer B: Allocated with index_modifier=None (unswizzled linear buffer)
        buf_b = self.sram.alloc("buf_b", 1024, index_modifier=None)
        self.assertEqual(buf_b.base_addr, 1024)

        # 1. 4x8 tiled read on Tile A (4-way conflict -> 2 + 4*12 = 50 cycles)
        tile_4x8_a = np.array([tile_a.base_addr + (lane // 8) * 32 + (lane % 8) for lane in range(32)], dtype=np.int64)
        self.assertEqual(self.sram.compute_latency(tile_4x8_a), 38)  # 6 + 4*8 = 38

        # 2. Linear 32-word row read on Buffer B (1-way conflict-free -> 2 + 1*12 = 14 cycles)
        row_addrs_b = np.arange(buf_b.base_addr, buf_b.base_addr + 32, dtype=np.int64)
        self.assertEqual(self.sram.compute_latency(row_addrs_b), 14)

    def test_worst_case_32_way_bank_conflict_and_row_vs_col_swizzles(self):
        """
        Verify worst-case 32-way bank conflict (2 + 32*12 = 386 cycles) on unswizzled column reads,
        and verify that row reads (stride-1) remain conflict-free (14 cycles).
        """
        unswizzled = self.sram.alloc("unswizzled_tile", 1024, index_modifier=None)
        tile_swz = self.sram.alloc("tile_swz", 1024, index_modifier=swizzle_indices)

        # Unswizzled column 0 read hits bank 0 across all 32 lanes -> 32-way conflict (2 + 32*12 = 386 cycles)
        col_unswizzled = np.array([unswizzled.base_addr + r * 32 for r in range(32)], dtype=np.int64)
        self.assertEqual(self.sram.compute_latency(col_unswizzled), 262)  # 6 + 32*8 = 262

        # Row 5 read (stride-1) on swizzle_indices is conflict-free (14 cycles)
        row_swz = np.arange(tile_swz.base_addr + 5 * 32, tile_swz.base_addr + 6 * 32, dtype=np.int64)
        self.assertEqual(self.sram.compute_latency(row_swz), 14)

    def test_bijectivity_enforcement_rejects_padding_and_collisions(self):
        """
        Verify that SRAMResource.alloc() strictly enforces that index_modifier is a bijection
        on [0, size), rejecting padded layouts, non-injective collisions, wrong shapes, and negative indices.
        """
        # 1. Padded layout (e.g., adding +1 pad per 32-word row expands range beyond 1024)
        padded_modifier = lambda idx: idx + (idx // 32)
        with self.assertRaises(ValueError):
            self.sram.alloc("padded_tile", 1024, index_modifier=padded_modifier)

        # 2. Colliding layout (non-injective function folding 1024 indices into 512 slots)
        colliding_modifier = lambda idx: idx % 512
        with self.assertRaises(ValueError):
            self.sram.alloc("colliding_tile", 1024, index_modifier=colliding_modifier)

        # 3. Wrong output shape (2D instead of 1D)
        shape_2d_modifier = lambda idx: idx.reshape(32, 32)
        with self.assertRaises(ValueError):
            self.sram.alloc("shape_2d_tile", 1024, index_modifier=shape_2d_modifier)

        # 4. Negative output indices
        negative_modifier = lambda idx: idx - 1
        with self.assertRaises(ValueError):
            self.sram.alloc("negative_tile", 1024, index_modifier=negative_modifier)

        # Failed allocations must not leak SRAM address space or register names
        self.assertEqual(len(self.sram.allocations), 0)

        # 5. Arbitrary CuTe-style 32x32 transpose layout (valid bijection on [0, 1024))
        transpose_layout = lambda idx: (idx % 32) * 32 + (idx // 32)
        tile_t = self.sram.alloc("transposed_tile", 1024, index_modifier=transpose_layout)
        self.assertEqual(tile_t.base_addr, 0)
        self.assertEqual(tile_t.size, 1024)

    def test_sram_capacity_alignment_duplicate_and_free(self):
        """Verify 32-word alignment, duplicate name rejection, free(), and 8192-word capacity overflow."""
        # Non-multiple of 32 size rounds next base_addr up to a 32-word boundary
        a1 = self.sram.alloc("buf_unaligned", 50)
        self.assertEqual(a1.base_addr, 0)
        a2 = self.sram.alloc("buf_aligned_next", 32)
        self.assertEqual(a2.base_addr, 64)

        # Duplicate allocation name raises ValueError until freed
        with self.assertRaises(ValueError):
            self.sram.alloc("buf_unaligned", 32)
        self.sram.free("buf_unaligned")
        a1_reused = self.sram.alloc("buf_unaligned", 32)
        self.assertEqual(a1_reused.base_addr, 96)

        # Default capacity is 32768 words (128 KB); exceeding remaining space raises MemoryError
        self.assertEqual(self.sram.capacity_words, 32768)
        with self.assertRaises(MemoryError):
            self.sram.alloc("overflow_tile", 32768)

    def test_functional_store_and_load_with_index_modifier(self):
        """Test functional SRAM store and load operations across SIMD lanes with an index modifier."""
        # Non-involutive cyclic shift bijection to verify single-pass physical address translation
        cyclic_modifier = lambda idx: (idx + 7) % 1024
        tile = self.sram.alloc("tile", 1024, index_modifier=cyclic_modifier)
        addresses = np.arange(tile.base_addr, tile.base_addr + 32, dtype=np.int64)
        values = np.array([float(i * 2.0) for i in range(32)], dtype=np.float32)

        st_state, end_st, store_tokens = self.sram.store(addresses, values, current_cycle=0, warp_id=0)
        self.assertEqual(st_state, WarpState.WARP_READY)
        self.assertEqual(end_st, 14)

        # Attempting to load same addresses at cycle 5 before store completes routes WARP_SRAM_STALL
        stall_state, unblock_cyc, _, _ = self.sram.load(addresses, current_cycle=5, warp_id=0)
        self.assertEqual(stall_state, WarpState.WARP_SRAM_STALL)
        self.assertEqual(unblock_cyc, 14)

        self.sram.InstrRetireRoutine(warp_id=0, mem_tokens=store_tokens)

        ld_state, end_ld, ld_tokens, read_vals = self.sram.load(addresses, current_cycle=end_st, warp_id=0)
        self.assertEqual(ld_state, WarpState.WARP_READY)
        self.assertEqual(end_ld, 28)
        self.sram.InstrRetireRoutine(warp_id=0, mem_tokens=ld_tokens)
        np.testing.assert_array_almost_equal(read_vals, values)

    def test_predicated_mask_suppresses_bank_conflicts_and_writes(self):
        """
        Verify that masked-out lanes in SRAMResource.store() and load() do not cause false
        bank conflicts, do not overwrite memory, and emit None in mem_tokens for inactive lanes.
        """
        tile = self.sram.alloc("unswizzled", 1024, index_modifier=None)
        # All 32 lanes point to column 0 (bank 0 -> 32-way conflict if all active = 386 cycles)
        col_addrs = np.array([tile.base_addr + r * 32 for r in range(32)], dtype=np.int64)
        vals = np.arange(100, 132, dtype=np.float32)

        # Activate only 2 lanes (lanes 0 and 1) -> 2-way bank conflict: 2 + 2 * 12 = 26 cycles
        mask = np.zeros(32, dtype=bool)
        mask[0] = True
        mask[1] = True

        st_state, end_st, st_tokens = self.sram.store(col_addrs, vals, current_cycle=0, warp_id=0, mask=mask)
        self.assertEqual(st_state, WarpState.WARP_READY)
        self.assertEqual(end_st, 22)  # 6 + 2*8 = 22
        self.assertIsNotNone(st_tokens[0])
        self.assertIsNotNone(st_tokens[1])
        for lane in range(2, 32):
            self.assertIsNone(st_tokens[lane])
        self.sram.InstrRetireRoutine(warp_id=0, mem_tokens=st_tokens)

        # Full-mask load of row 0 and row 1 verifies only lanes 0 and 1 were written
        ld_state, end_ld, ld_tokens, loaded = self.sram.load(
            col_addrs, current_cycle=26, warp_id=0, mask=mask
        )
        self.assertEqual(ld_state, WarpState.WARP_READY)
        self.assertEqual(end_ld, 26 + 22)
        self.assertEqual(loaded[0], 100.0)
        self.assertEqual(loaded[1], 101.0)
        self.assertEqual(loaded[2], 0.0)
        self.sram.InstrRetireRoutine(warp_id=0, mem_tokens=ld_tokens)

    def test_warp_sram_stall_on_pipe_busy_lsq_full_and_war_hazard(self):
        """
        Verify WARP_SRAM_STALL routing on Load Pipe Busy, 17th LSQ Full store,
        and WAR hazard store pipe serialization on SRAMResource.
        """
        tile = self.sram.alloc("tile", 2048, index_modifier=swizzle_indices)
        row0 = np.arange(tile.base_addr, tile.base_addr + 32, dtype=np.int64)
        row1 = np.arange(tile.base_addr + 32, tile.base_addr + 64, dtype=np.int64)

        # 1. Load Pipe Busy stall + WAR hazard: LD at cycle 0 finishes at cycle 14
        ld1_state, end_ld1, ld1_tokens, _ = self.sram.load(row0, current_cycle=0, warp_id=0)
        self.assertEqual(ld1_state, WarpState.WARP_READY)
        self.assertEqual(end_ld1, 14)

        # Second LD at cycle 4 while load pipe is busy stalls with WARP_SRAM_STALL until cycle 14
        ld2_state, unblock_cyc, _, _ = self.sram.load(row1, current_cycle=4, warp_id=1)
        self.assertEqual(ld2_state, WarpState.WARP_SRAM_STALL)
        self.assertEqual(unblock_cyc, 6)  # Stage 1 (issue) of ld1 finishes at cycle 6 (ld1.is_issued=True)

        # Store to row0 at cycle 4 (WAR hazard with in-flight LD1) is fire-and-forget (WARP_READY)
        # but delayed on the store pipe until LD1 finishes at cycle 14 -> completes at 14 + 14 = 28
        vals = np.ones(32, dtype=np.float32)
        st_war_state, end_st_war, st_war_tokens = self.sram.store(row0, vals, current_cycle=4, warp_id=0)
        self.assertEqual(st_war_state, WarpState.WARP_READY)
        self.assertEqual(end_st_war, 28)

        self.sram.InstrRetireRoutine(warp_id=0, mem_tokens=ld1_tokens)
        self.sram.InstrRetireRoutine(warp_id=0, mem_tokens=st_war_tokens)

        # 2. LSQ Capacity (16 entries): 16 stores succeed, 17th stalls with WARP_SRAM_STALL
        active_tokens = []
        for i in range(16):
            addrs = np.arange(tile.base_addr + i * 32, tile.base_addr + (i + 1) * 32, dtype=np.int64)
            st_s, _, toks = self.sram.store(addrs, vals, current_cycle=28 + i, warp_id=2)
            self.assertEqual(st_s, WarpState.WARP_READY)
            active_tokens.append(toks)

        # Store #0 started at cycle 28 and completes at 28 + 14 = 42; if Store #0 starts at cycle 30,
        # at cycle 40 all 16 stores in LSQ are still in-flight (oldest completes at 42 > 40).
        addrs_17 = np.arange(tile.base_addr + 16 * 32, tile.base_addr + 17 * 32, dtype=np.int64)
        st17_state, unblock_17, st17_toks = self.sram.store(addrs_17, vals, current_cycle=40, warp_id=2)
        self.assertEqual(st17_state, WarpState.WARP_SRAM_STALL)
        self.assertIsNone(st17_toks)
        self.assertEqual(unblock_17, 42)

        # Retiring oldest store frees 1 slot in LSQ so the 17th store can issue
        self.sram.InstrRetireRoutine(warp_id=2, mem_tokens=active_tokens[0])
        st17_retry_state, _, _ = self.sram.store(addrs_17, vals, current_cycle=42, warp_id=2)
        self.assertEqual(st17_retry_state, WarpState.WARP_READY)

    def test_multi_allocation_span_address_translation(self):
        """
        Verify that translate_addresses() accurately applies per-allocation index modifiers
        even when a single 32-lane vector access spans two different allocations.
        """
        shift_a = lambda idx: (idx + 3) % 32
        shift_b = lambda idx: (idx + 11) % 32
        alloc_a = self.sram.alloc("alloc_a", 32, index_modifier=shift_a)
        alloc_b = self.sram.alloc("alloc_b", 32, index_modifier=shift_b)

        # First 16 lanes access alloc_a [0..15], next 16 lanes access alloc_b [32..47]
        spanning_addrs = np.concatenate([
            np.arange(alloc_a.base_addr, alloc_a.base_addr + 16, dtype=np.int64),
            np.arange(alloc_b.base_addr, alloc_b.base_addr + 16, dtype=np.int64),
        ])
        phys = self.sram.translate_addresses(spanning_addrs)
        expected = np.concatenate([
            alloc_a.base_addr + shift_a(np.arange(16, dtype=np.int64)),
            alloc_b.base_addr + shift_b(np.arange(16, dtype=np.int64)),
        ])
        np.testing.assert_array_equal(phys, expected)


if __name__ == "__main__":
    unittest.main()
