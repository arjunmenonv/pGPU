'''
Unified Memory Hardware Resource Models for pGPU Simulator.

Defines:
- MemoryAllocation: Metadata for an allocated buffer/tile (base_addr, size, optional index_modifier)
- InFlightStore: Per-thread store buffer entry referencing a MemoryResource object
- MemoryResource: Base class for addressable memory resources
- DRAMResource: Off-chip DRAM hierarchy and channel row-activation latency model
- SRAMResource: On-chip 32-bank Scratchpad SRAM and bank-conflict latency model

Author: Arjun Vadakkeveedu (arjunmenonv@alumni.iitm.ac.in)
September 2026
'''

from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from pgpu.arch.gpu_device import AbstractResource
from pgpu.arch.smsp import WarpState


@dataclass
class MemoryAllocation:
    """
    Represents a named memory allocation (buffer or tile) within a MemoryResource.
    
    Attributes:
        name (str): Identifier for the allocation.
        base_addr (int): Starting 32-bit word address in the memory space.
        size (int): Size of the allocation in 32-bit words.
        index_modifier (Optional[Callable[[np.ndarray], np.ndarray]]): Optional CuTe-style layout/index modifier
            mapping intra-allocation logical index [0, size) -> physical index [0, size).
    """
    name: str
    base_addr: int
    size: int
    index_modifier: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def contains(self, word_addr):
        """Return True (or boolean mask array) if word_addr falls within [base_addr, base_addr + size)."""
        return (self.base_addr <= word_addr) & (word_addr < (self.base_addr + self.size))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.size == other
        if isinstance(other, MemoryAllocation):
            return (
                self.name == other.name
                and self.base_addr == other.base_addr
                and self.size == other.size
                and self.index_modifier == other.index_modifier
            )
        return False


class MemOpType(Enum):
    """Integer enum token identifying memory operation type in the Load-Store Queue."""
    LOAD = 0
    STORE = 1


@dataclass
class InFlightMemOp:
    """
    Tracks an in-flight memory operation (MemOpType.LOAD or MemOpType.STORE) in the per-thread LSQ.
    Pointers to these tokens (one per lane) are held in `instr.mem_tokens` so that
    `MemoryResource.InstrRetireRoutine(warp_id, mem_tokens)` can remove them directly.
    
    Attributes:
        resource (MemoryResource): Target MemoryResource instance.
        op_type (MemOpType): Operation type token (`MemOpType.LOAD` or `MemOpType.STORE`).
        address (int): Physical 32-bit word address accessed by this thread.
        completion_time (int): Simulation cycle when the memory operation completes.
    """
    resource: "MemoryResource"
    op_type: MemOpType
    address: int
    completion_time: int

    def __post_init__(self):
        if not isinstance(self.resource, MemoryResource):
            raise TypeError(
                f"InFlightMemOp requires a MemoryResource instance, got {type(self.resource).__name__}"
            )
        if not isinstance(self.op_type, MemOpType):
            raise TypeError(f"op_type must be a MemOpType Enum token, got {self.op_type!r}")


# Alias for backward compatibility
InFlightStore = InFlightMemOp


class MemoryResource(AbstractResource):
    """
    Abstract base class for addressable memory resources (DRAMResource, SRAMResource).
    
    Provides:
      - Word-addressed data storage (`self.memory: Dict[int, float]`).
      - Allocation tracking (`alloc`, `free`, `allocations`) up to `capacity_words`.
      - Independent bidirectional Read (Load) and Write (Store) pipe timestamps
        (`next_available_load_cycle` and `next_available_store_cycle`).
      - Unified 16-entry per-thread Load-Store Queue (`self.lsq[warp_id][lane_id]`)
        tracking both in-flight LOADs and STOREs using `MemOpType` enum tokens.
      - Hardware `WarpState` signal routing (`WARP_READY` vs `self.stall_signal`).
    """
    MAX_QUEUE_CAPACITY = 16

    def __init__(
        self,
        name: str,
        base_latency: int,
        capacity_words: int,
        stall_signal: WarpState = WarpState.WARP_LONG_LAT_STALL,
    ):
        super().__init__(name=name, base_latency=base_latency, stall_signal=stall_signal)
        self.capacity_words = capacity_words

        # Word-addressed storage mapping 32-bit word address -> 32-bit float value
        self.memory: Dict[int, float] = defaultdict(float)

        # Allocation tracking: name -> MemoryAllocation
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.allocated_words: int = 0
        self._next_alloc_base: int = 0

        # Independent full-duplex (bidirectional) read and write pipe timestamps
        self.next_available_load_cycle: int = 0
        self.next_available_store_cycle: int = 0

        # Unified 16-entry per-thread Load-Store Queue:
        # self.lsq[warp_id][lane_id] = list[InFlightMemOp]
        self.lsq: Dict[int, List[List[InFlightMemOp]]] = defaultdict(lambda: [[] for _ in range(32)])

    @property
    def store_queues(self) -> Dict[int, List[List[InFlightMemOp]]]:
        """Alias to `self.lsq` for inspecting per-thread queue state."""
        return self.lsq

    def alloc(self, name: str, size: int) -> MemoryAllocation:
        """Allocate `size` 32-bit words in this memory resource (32-word aligned)."""
        if name in self.allocations:
            raise ValueError(f"Allocation '{name}' already exists in {self.name}.")
        if self.allocated_words + size > self.capacity_words:
            raise MemoryError(
                f"{self.name} capacity exceeded. Requested {size} words, "
                f"available {self.capacity_words - self.allocated_words} words."
            )
        aligned_base = (self._next_alloc_base + 31) & ~31
        base_addr = aligned_base % (1 << 32)
        self._next_alloc_base = (base_addr + size) % (1 << 32)
        allocation = MemoryAllocation(name=name, base_addr=base_addr, size=size)
        self.allocations[name] = allocation
        self.allocated_words += size
        return allocation

    def free(self, name: str):
        """Free an existing allocation by name."""
        if name not in self.allocations:
            raise KeyError(f"Allocation '{name}' not found in {self.name}.")
        allocation = self.allocations.pop(name)
        self.allocated_words -= allocation.size
        if not self.allocations:
            self._next_alloc_base = 0

    def get_allocation_for_address(self, word_addr: int) -> Optional[MemoryAllocation]:
        """Find the MemoryAllocation containing `word_addr`, or None if unallocated."""
        for allocation in self.allocations.values():
            if allocation.contains(word_addr):
                return allocation
        return None

    def translate_addresses(self, addresses: np.ndarray) -> np.ndarray:
        """
        Translate logical word addresses into physical word addresses.
        Identity mapping by default; overridden by SRAMResource to apply per-allocation index modifiers.
        """
        return np.asarray(addresses, dtype=np.int64)

    def compute_latency(self, addresses: np.ndarray) -> int:
        """Compute access latency for a vector of active lane addresses. Implemented by subclasses."""
        raise NotImplementedError("Subclasses of MemoryResource must implement compute_latency()")

    def _evaluate_lsq_hazards(
        self,
        phys_addrs: np.ndarray,
        current_cycle: int,
        warp_id: int,
        active_indices: np.ndarray,
        conflicting_op_type: MemOpType,
    ) -> Tuple[int, int]:
        """
        Scan the unified per-thread LSQ (`self.lsq[warp_id][lane_id]`) across all active lanes to find:
          1. `max_queue_full_stall`: Earliest cycle when a slot opens in any full (16-entry) lane queue.
          2. `max_addr_hazard_ready`: Latest `completion_time` of any in-flight op in the same lane
             with `op_type == conflicting_op_type` (`MemOpType.STORE` for RAW on LD;
             `MemOpType.LOAD` for WAR on ST) targeting the same physical word address.
        """
        max_queue_full_stall = current_cycle
        max_addr_hazard_ready = current_cycle

        for lane_id in active_indices:
            l_id = int(lane_id)
            p_addr = int(phys_addrs[l_id])
            queue = self.lsq[warp_id][l_id]

            if len(queue) >= self.MAX_QUEUE_CAPACITY:
                sorted_completions = sorted(entry.completion_time for entry in queue)
                earliest_slot_free = sorted_completions[len(queue) - self.MAX_QUEUE_CAPACITY]
                if earliest_slot_free > max_queue_full_stall:
                    max_queue_full_stall = earliest_slot_free

            for entry in queue:
                if (
                    entry.resource is self
                    and entry.op_type == conflicting_op_type
                    and entry.address == p_addr
                ):
                    if entry.completion_time > max_addr_hazard_ready:
                        max_addr_hazard_ready = entry.completion_time

        return max_queue_full_stall, max_addr_hazard_ready

    def _enqueue_lsq_tokens(
        self,
        op_type: MemOpType,
        phys_addrs: np.ndarray,
        completion_cycle: int,
        warp_id: int,
        active_indices: np.ndarray,
    ) -> List[Optional[InFlightMemOp]]:
        """
        Create `InFlightMemOp` tokens for all active lanes, append them to `self.lsq[warp_id][lane_id]`,
        and return the 32-element token pointer list (`InFlightMemOp` or `None` per lane).
        """
        mem_tokens: List[Optional[InFlightMemOp]] = [None] * len(phys_addrs)
        for lane_id in active_indices:
            l_id = int(lane_id)
            token = InFlightMemOp(
                resource=self,
                op_type=op_type,
                address=int(phys_addrs[l_id]),
                completion_time=completion_cycle,
            )
            self.lsq[warp_id][l_id].append(token)
            mem_tokens[l_id] = token
        return mem_tokens

    def InstrRetireRoutine(
        self,
        warp_id: int,
        mem_tokens: List[Optional[InFlightMemOp]],
    ):
        """
        Resource-specific retirement routine invoked by `Instruction.retire()`.
        Takes `warp_id` and `mem_tokens` (a 32-element list, one `InFlightMemOp` token
        or `None` per lane) and removes the retired token from `self.lsq[warp_id][lane_id]`.
        """
        for lane_id, token in enumerate(mem_tokens):
            if token is not None and token.resource is self:
                queue = self.lsq[warp_id][lane_id]
                if token in queue:
                    queue.remove(token)

    def load(
        self,
        addresses: np.ndarray,
        current_cycle: int,
        warp_id: int = 0,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[WarpState, int, Optional[List[Optional[InFlightMemOp]]], Optional[np.ndarray]]:
        """
        Vector memory load operation across 32 SIMD lanes with `WarpState` stall signal routing.
        
        1. Evaluates LSQ capacity full stall, intra-thread RAW (`STORE -> LOAD`) hazard,
           and Load Pipe availability (`next_available_load_cycle`).
        2. If `unblock_cycle > current_cycle`:
           Does NOT mutate state; routes `(self.stall_signal, unblock_cycle, None, None)`.
        3. If `unblock_cycle <= current_cycle`:
           Reserves the Load Pipe at `current_cycle`, enqueues `MemOpType.LOAD` tokens into `self.lsq`,
           reads memory, and routes `(WarpState.WARP_READY, completion_cycle, mem_tokens, results)`.
        """
        addrs = np.asarray(addresses, dtype=np.int64)
        mask_arr = np.ones(len(addrs), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

        phys_addrs = self.translate_addresses(addrs)
        active_indices = np.where(mask_arr)[0]

        max_queue_full_stall, max_raw_ready = self._evaluate_lsq_hazards(
            phys_addrs=phys_addrs,
            current_cycle=current_cycle,
            warp_id=warp_id,
            active_indices=active_indices,
            conflicting_op_type=MemOpType.STORE,
        )

        unblock_cycle = max(current_cycle, max_queue_full_stall, max_raw_ready, self.next_available_load_cycle)
        if unblock_cycle > current_cycle:
            return self.stall_signal, unblock_cycle, None, None

        active_addrs = addrs[mask_arr]
        latency = self.compute_latency(active_addrs)
        actual_load_start = self.reserve_load(current_cycle, duration=latency)
        completion_cycle = actual_load_start + latency

        mem_tokens = self._enqueue_lsq_tokens(
            op_type=MemOpType.LOAD,
            phys_addrs=phys_addrs,
            completion_cycle=completion_cycle,
            warp_id=warp_id,
            active_indices=active_indices,
        )

        get_mem = np.vectorize(lambda a: self.memory[int(a)], otypes=[np.float32])
        results = np.where(mask_arr, get_mem(phys_addrs), np.float32(0.0))

        return WarpState.WARP_READY, completion_cycle, mem_tokens, results

    def store(
        self,
        addresses: np.ndarray,
        values: np.ndarray,
        current_cycle: int,
        warp_id: int = 0,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[WarpState, int, Optional[List[Optional[InFlightMemOp]]]]:
        """
        Vector memory store operation across 32 SIMD lanes with `WarpState` stall signal routing.
        
        Stores are fire-and-forget into the 16-entry LSQ:
        1. Only stalls the warp at issue if the per-thread LSQ is full (`max_queue_full_stall > current_cycle`),
           routing `(self.stall_signal, max_queue_full_stall, None)`.
        2. Otherwise, immediately accepts the store into `self.lsq` with `WarpState.WARP_READY`,
           while scheduling the Store Pipe writeback after `next_available_store_cycle` and any
           in-flight `LOAD` to the same address (`WAR` hazard).
        Returns `(WarpState.WARP_READY, completion_cycle, mem_tokens)`.
        """
        addrs = np.asarray(addresses, dtype=np.int64)
        vals = np.asarray(values, dtype=np.float32)
        mask_arr = np.ones(len(addrs), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

        phys_addrs = self.translate_addresses(addrs)
        active_indices = np.where(mask_arr)[0]

        max_queue_full_stall, max_war_ready = self._evaluate_lsq_hazards(
            phys_addrs=phys_addrs,
            current_cycle=current_cycle,
            warp_id=warp_id,
            active_indices=active_indices,
            conflicting_op_type=MemOpType.LOAD,
        )

        if max_queue_full_stall > current_cycle:
            return self.stall_signal, max_queue_full_stall, None

        active_addrs = addrs[mask_arr]
        latency = self.compute_latency(active_addrs)

        store_pipe_ready = max(current_cycle, max_war_ready)
        actual_store_start = self.reserve_store(store_pipe_ready, duration=latency)
        completion_cycle = actual_store_start + latency

        for lane_id in active_indices:
            l_id = int(lane_id)
            self.memory[int(phys_addrs[l_id])] = float(vals[l_id])

        mem_tokens = self._enqueue_lsq_tokens(
            op_type=MemOpType.STORE,
            phys_addrs=phys_addrs,
            completion_cycle=completion_cycle,
            warp_id=warp_id,
            active_indices=active_indices,
        )

        return WarpState.WARP_READY, completion_cycle, mem_tokens

    def clear_store_queues(self, warp_id: Optional[int] = None):
        """Clear the per-thread LSQ upon SYNC barrier execution."""
        if warp_id is None:
            self.lsq.clear()
        else:
            self.lsq.pop(warp_id, None)

    def is_load_available(self, current_cycle: int) -> bool:
        """Check if the memory read (load) channel is available at current_cycle."""
        return current_cycle >= self.next_available_load_cycle

    def is_store_available(self, current_cycle: int) -> bool:
        """Check if the memory write (store) channel is available at current_cycle."""
        return current_cycle >= self.next_available_store_cycle

    def reserve_load(self, issue_cycle: int, duration: int) -> int:
        """Reserve the memory read (load) port starting at issue_cycle."""
        actual_start = max(issue_cycle, self.next_available_load_cycle)
        self.next_available_load_cycle = actual_start + duration
        return actual_start

    def reserve_store(self, issue_cycle: int, duration: int) -> int:
        """Reserve the memory write (store) port starting at issue_cycle."""
        actual_start = max(issue_cycle, self.next_available_store_cycle)
        self.next_available_store_cycle = actual_start + duration
        return actual_start

    def reset(self):
        """Reset memory resource state, bidirectional port timestamps, data, allocations, and LSQ."""
        super().reset()
        self.next_available_load_cycle = 0
        self.next_available_store_cycle = 0
        self.memory.clear()
        self.allocations.clear()
        self.allocated_words = 0
        self._next_alloc_base = 0
        self.lsq.clear()


class DRAMResource(MemoryResource):
    """
    DRAM Hardware Resource Model.
    
    Inherits from MemoryResource with name="DRAM".
    Models an 8-channel, 8-rank, 8-bank-group, 8-bank hierarchy with 1Kx1K banks (16 GB capacity).
    """
    CAPACITY_BYTES = 16 * 1024 * 1024 * 1024  # 16 GB
    BYTES_PER_WORD = 4
    CAPACITY_WORDS = CAPACITY_BYTES // BYTES_PER_WORD  # 4,294,967,296 words

    def __init__(self, name: str = "DRAM", base_latency: int = 20, request_latency: int = 50):
        super().__init__(
            name=name,
            base_latency=base_latency,
            capacity_words=self.CAPACITY_WORDS,
            stall_signal=WarpState.WARP_LONG_LAT_STALL,
        )
        self.request_latency = request_latency

    def decompose_address(
        self, addrs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose flat 32-bit word addresses into hardware DRAM hierarchy components using NumPy.
        
        Bitfield Extraction:
        - col_lo: bits [2:0]
        - row_lo: bits [4:3]
        - channel: bits [7:5]
        - rank: bits [10:8]
        - bank_group: bits [13:11]
        - bank: bits [16:14]
        - row_hi: bits [24:17]
        - col_hi: bits [31:25]
        
        Returns:
            Tuple of NumPy arrays: (col, channel, row, rank, bank_group, bank)
        """
        addrs_arr = np.asarray(addrs, dtype=np.uint64)
        col_lo     =  addrs_arr        & 0x7
        row_lo     = (addrs_arr >> 3)  & 0x3
        channel    = (addrs_arr >> 5)  & 0x7
        rank       = (addrs_arr >> 8)  & 0x7
        bank_group = (addrs_arr >> 11) & 0x7
        bank       = (addrs_arr >> 14) & 0x7
        row_hi     = (addrs_arr >> 17) & 0xFF
        col_hi     = (addrs_arr >> 25) & 0x7F

        row = (row_hi << 2) | row_lo
        col = (col_hi << 3) | col_lo

        return col, channel, row, rank, bank_group, bank

    def compute_latency(self, addresses: np.ndarray) -> int:
        """
        Compute total access latency for a vector access by counting the number of `DRAM_REQ`s.
        
        A single `DRAM_REQ` can read up to 32 4B words across up to 8 channels, provided all words in the request target:
          1. A single 8-column burst group (`col >> 3`, i.e., 8 cols of 4B each from a bank row)
          2. A single `row` from a DRAM Bank
          3. A single `bank` from a DRAM BankGroup
          4. A single `bank_group` from a DRAM Rank
          5. A single `rank` from the DRAM DIMM
        
        Total Latency = base_latency + num_dram_reqs * request_latency
        """
        addrs = np.asarray(addresses, dtype=np.uint64)
        if addrs.size == 0:
            return self.base_latency

        col, channel, row, rank, bank_group, bank = self.decompose_address(addrs)

        # Within each channel, each unique (rank, bank_group, bank, row, col >> 3) stack
        # requires a separate DRAM_REQ; different channels operate in parallel in the same DRAM_REQ.
        unique_ch_stacks = np.unique(
            np.column_stack((channel, rank, bank_group, bank, row, col >> 3)),
            axis=0,
        )
        reqs_per_channel = np.bincount(unique_ch_stacks[:, 0].astype(np.int64), minlength=8)
        num_dram_reqs = int(np.max(reqs_per_channel))

        return self.base_latency + num_dram_reqs * self.request_latency


class SRAMResource(MemoryResource):
    """
    Scratchpad Memory (SRAM) Hardware Resource Model.
    
    Inherits from MemoryResource with name="SRAM".
    Models a 32-bank on-chip SRAM where bank conflicts serialize requests.
    Base latency = 2 cycles. Request latency = 12 cycles per SRAM request wave
    (best-case conflict-free latency = 2 + 1 * 12 = 14 cycles).
    
    Each allocation in SRAM can specify an optional vectorized `index_modifier` callable
    (analogous to `std::transform` over `[0, size)`) that maps logical intra-allocation
    offsets to physical offsets. The modifier is verified in O(N) via NumPy to be a
    size-preserving bijection.
    """
    NUM_BANKS = 32

    DEFAULT_SIZE_WORDS = 32 * 1024  # 32K words (128 KB total capacity)

    def __init__(
        self,
        name: str = "SRAM",
        base_latency: int = 2,
        request_latency: int = 12,
        size_words: int = DEFAULT_SIZE_WORDS,
    ):
        super().__init__(
            name=name,
            base_latency=base_latency,
            capacity_words=size_words,
            stall_signal=WarpState.WARP_SRAM_STALL,
        )
        self.request_latency = request_latency
        self.conflict_penalty = request_latency

    @staticmethod
    def validate_bijection(
        name: str, size: int, index_modifier: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        """
        Apply `index_modifier` over `[0, size)` (like `std::transform`) and verify in O(size)
        using vectorized NumPy properties that it is a size-preserving bijection:
        
        1. Shape & Type Preservation: `mapped.shape == (size,)` with integer elements.
        2. Codomain Closure (No Padding): `0 <= mapped.min()` and `mapped.max() < size`.
        3. Exact 1-to-1 & Onto Coverage (Bijectivity): `np.all(np.bincount(mapped, minlength=size) == 1)`.
        """
        domain = np.arange(size, dtype=np.int64)
        mapped = np.asarray(index_modifier(domain), dtype=np.int64)

        if mapped.shape != (size,):
            raise ValueError(
                f"Index modifier for '{name}' must return a 1D array of shape ({size},), got {mapped.shape}."
            )

        if mapped.min() < 0 or mapped.max() >= size:
            raise ValueError(
                f"Index modifier for '{name}' mapped outside [0, {size}): "
                f"range=[{int(mapped.min())}, {int(mapped.max())}]. Padding is not permitted."
            )

        counts = np.bincount(mapped, minlength=size)
        if not np.all(counts == 1):
            unique_count = int(np.count_nonzero(counts))
            raise ValueError(
                f"Index modifier for '{name}' is not bijective on [0, {size}): "
                f"mapped {size} indices onto {unique_count} unique slots (collision/hole detected)."
            )

    def alloc(
        self,
        name: str,
        size: int,
        index_modifier: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> MemoryAllocation:
        """
        Allocate `size` 32-bit words in SRAM with an optional vectorized bijective `index_modifier`.
        """
        if index_modifier is not None and size > 0:
            self.validate_bijection(name, size, index_modifier)
        allocation = super().alloc(name=name, size=size)
        if index_modifier is not None:
            allocation.index_modifier = index_modifier
        return allocation

    def translate_addresses(self, addresses: np.ndarray) -> np.ndarray:
        """
        Translate logical word addresses into physical SRAM word addresses by looking up
        the target allocation via `get_allocation_for_address` and applying its `index_modifier`.
        """
        addrs = np.asarray(addresses, dtype=np.int64)
        if addrs.size == 0:
            return addrs

        # Fast path: all lanes in a vector instruction target the same allocation
        allocation = self.get_allocation_for_address(int(addrs[0]))
        if allocation is not None and np.all(allocation.contains(addrs)):
            if allocation.index_modifier is None:
                return addrs
            offsets = addrs - allocation.base_addr
            return allocation.base_addr + np.asarray(
                allocation.index_modifier(offsets), dtype=np.int64
            )

        # General fallback if lanes span multiple allocations
        phys_addrs = addrs.copy()
        for alloc in self.allocations.values():
            if alloc.index_modifier is not None:
                in_bounds = alloc.contains(addrs)
                if np.any(in_bounds):
                    offsets = addrs[in_bounds] - alloc.base_addr
                    phys_addrs[in_bounds] = alloc.base_addr + np.asarray(
                        alloc.index_modifier(offsets), dtype=np.int64
                    )
        return phys_addrs

    def compute_latency(self, addresses: np.ndarray) -> int:
        """
        Compute total access latency for a 32-lane vector access to SRAM using vectorized NumPy operations.
        
        Translates logical addresses to physical SRAM addresses via each allocation's index_modifier,
        maps physical addresses to banks (`phys_addrs % 32`), and counts bank conflicts via `np.bincount`.
        Latency = base_latency (2) + max_bank_conflict * request_latency (12).
        Best-case (1 request wave, conflict-free): 2 + 1 * 12 = 14 cycles.
        """
        addrs = np.asarray(addresses, dtype=np.int64)
        if addrs.size == 0:
            return self.base_latency

        phys_addrs = self.translate_addresses(addrs)
        banks = phys_addrs % self.NUM_BANKS
        bank_counts = np.bincount(banks, minlength=self.NUM_BANKS)
        max_conflict = int(np.max(bank_counts))

        return self.base_latency + max_conflict * self.request_latency
