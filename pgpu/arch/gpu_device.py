"""
Defines the 3-tier pGPU Hardware Device Hierarchy:
- AbstractResource: Base class for all GPU hardware resources (Memory, Execution Units)
- CopyDirection: Enum flag ("H2D" / "D2H") for host-device DRAM memory transfers
- SM: Streaming Multiprocessor housing 1 shared SRAMResource and multiple SMSPs
- pGPU: Top-level GPU device housing 1 shared DRAMResource and multiple SMs

Author: Arjun Vadakkeveedu (arjunmenonv@alumni.iitm.ac.in)
September 2026
"""

from enum import Enum
from typing import Any, List, Optional, Sequence, Union, TYPE_CHECKING
import numpy as np

from pgpu.arch.smsp import WarpState

if TYPE_CHECKING:
    from pgpu.arch.memory import DRAMResource, SRAMResource, MemoryAllocation
    from pgpu.arch.smsp import SMSP


class CopyDirection(Enum):
    """Direction flag for host <-> device DRAM transfers (`pGPU.dram_copy`)."""
    H2D = "H2D"  # Host (NumPy) to Device (DRAMResource)
    D2H = "D2H"  # Device (DRAMResource) to Host (NumPy)


# ============================================================================
# Architectural Parameter Buffer in DRAM
# ============================================================================
PARAM_BUFFER_BASE: int = 0
PARAM_BUFFER_SIZE: int = 1024


class AbstractResource:
    """
    Abstract base class for a hardware resource in the pGPU simulator.
    
    Attributes:
        name (str): Identifier for the resource (e.g., "DRAM", "SRAM", "VALU", "GEMMCORE", "TRANS").
        base_latency (int): Base latency associated with accessing this resource.
        stall_signal (WarpState): Hardware stall signal routed when this resource blocks issue.
    """
    def __init__(self, name: str, base_latency: int = 1, stall_signal: WarpState = WarpState.WARP_COMPUTE_STALL):
        self.name = name
        self.base_latency = base_latency
        self.stall_signal = stall_signal

    def InstrRetireRoutine(self, *args, **kwargs):
        """
        Resource-side cleanup routine triggered by `Instruction.retire()` for each
        resource in `instruction.resources`.
        """
        pass

    def reset(self):
        """Reset resource state for a new simulation run."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', base_lat={self.base_latency})"


class SM:
    """
    Streaming Multiprocessor (SM) Hardware Container.
    
    Houses:
    - 1 `SRAMResource` (On-chip Scratchpad Memory for this SM).
    - 1 `SMSP` compute sub-partition (`self.smsps[0]`).
    - A pointer to the enclosing `pGPU`'s `DRAMResource`.
    
    Architectural Note:
        The current pGPU architecture supports **only 1 SMSP per SM** (`SMSPS_PER_SM = 1`) due to ISA and SM Simulator limitations.
        The `SM` class serves as a structural abstraction decoupling the `SMSP` from
        `SRAMResource` ownership and providing an extension point for future multi-SMSP models.
    """
    SMSPS_PER_SM: int = 1

    def __init__(
        self,
        sm_id: int = 0,
        num_smsps: int = 1,
        num_warps_per_smsp: int = 4,
        sram: Optional["SRAMResource"] = None,
        dram: Optional["DRAMResource"] = None,
        smsps: Optional[Sequence["SMSP"]] = None,
    ):
        from pgpu.arch.memory import SRAMResource
        from pgpu.arch.smsp import SMSP

        if num_smsps != self.SMSPS_PER_SM:
            raise ValueError(
                f"The pGPU architecture currently supports only {self.SMSPS_PER_SM} SMSP per SM "
                f"(got num_smsps={num_smsps})."
            )
        if smsps is not None and len(smsps) != self.SMSPS_PER_SM:
            raise ValueError(
                f"The pGPU architecture currently supports only {self.SMSPS_PER_SM} SMSP per SM "
                f"(got len(smsps)={len(smsps)})."
            )

        self.sm_id = sm_id
        self.sram: SRAMResource = sram if sram is not None else SRAMResource(name=f"SRAM_SM{sm_id}")
        self.dram: Optional["DRAMResource"] = dram

        if smsps is not None:
            self.smsps: List[SMSP] = list(smsps)
        else:
            self.smsps = [SMSP(num_warps=num_warps_per_smsp, smsp_id=0, sm_id=sm_id)]

        self._wire_smsps()

    @property
    def smsp(self) -> "SMSP":
        """Return the single SMSP belonging to this SM."""
        return self.smsps[0]

    def _wire_smsps(self, base_warp_offset: int = 0, total_device_warps: Optional[int] = None) -> int:
        """Bind `self.sram` and `self.dram` pointers to the child SMSP."""
        sm_total_warps = sum(smsp.num_warps for smsp in self.smsps)
        tot_warps = total_device_warps if total_device_warps is not None else sm_total_warps
        curr_offset = base_warp_offset
        for idx, smsp in enumerate(self.smsps):
            smsp.sm_id = self.sm_id
            smsp.smsp_id = idx
            smsp.bind_memory(dram=self.dram, sram=self.sram)
            smsp.set_warp_offset(warp_id_offset=curr_offset, total_warps=tot_warps)
            curr_offset += smsp.num_warps
        return curr_offset

    def bind_dram(self, dram: "DRAMResource", base_warp_offset: int = 0, total_device_warps: Optional[int] = None) -> int:
        """Bind the device DRAMResource to this SM and propagate to the child SMSP."""
        self.dram = dram
        return self._wire_smsps(base_warp_offset=base_warp_offset, total_device_warps=total_device_warps)

    def reset_compute(self) -> None:
        """Reset child SMSP compute pipelines without wiping SRAM allocations/data."""
        for smsp in self.smsps:
            smsp.reset()

    def __repr__(self) -> str:
        return f"SM(sm_id={self.sm_id}, num_smsps={len(self.smsps)}, sram={self.sram.name})"


class pGPU:
    """
    Top-Level pGPU Device Container & Host Interface.
    
    Hierarchy:
    - 1 `DRAMResource` owned at the device level (`self.dram`).
    - 1 `SM` (`self.sm`), owning 1 `SRAMResource` and 1 `SMSP` (`self.smsp`).
    
    Architectural Note:
        Only **1 SM per pGPU** (`SMS_PER_GPU = 1`) and **1 SMSP per SM** (`SMSPS_PER_SM = 1`)
        are currently supported in the architecture. The `pGPU` -> `SM` -> `SMSP` hierarchy
        decouples memory ownership (`DRAMResource` on `pGPU`, `SRAMResource` on `SM`) from
        the `SMSP` execution pipeline and provides a decorative container for future extension.
    """
    SMS_PER_GPU: int = 1
    SMSPS_PER_SM: int = 1

    def __init__(
        self,
        name: str = "pGPU",
        num_sms: int = 1,
        num_smsps_per_sm: int = 1,
        num_warps_per_smsp: int = 4,
        dram: Optional["DRAMResource"] = None,
        sms: Optional[Sequence[SM]] = None,
    ):
        from pgpu.arch.memory import DRAMResource

        if num_sms != self.SMS_PER_GPU:
            raise ValueError(
                f"The pGPU architecture currently supports only {self.SMS_PER_GPU} SM per pGPU "
                f"(got num_sms={num_sms})."
            )
        if num_smsps_per_sm != self.SMSPS_PER_SM:
            raise ValueError(
                f"The pGPU architecture currently supports only {self.SMSPS_PER_SM} SMSP per SM "
                f"(got num_smsps_per_sm={num_smsps_per_sm})."
            )
        if sms is not None and len(sms) != self.SMS_PER_GPU:
            raise ValueError(
                f"The pGPU architecture currently supports only {self.SMS_PER_GPU} SM per pGPU "
                f"(got len(sms)={len(sms)})."
            )

        self.name = name
        self.dram: DRAMResource = dram if dram is not None else DRAMResource(name="DRAM")
        self.resources: List[AbstractResource] = [self.dram]

        if sms is not None:
            self.sms: List[SM] = list(sms)
        else:
            self.sms = [
                SM(
                    sm_id=0,
                    num_smsps=self.SMSPS_PER_SM,
                    num_warps_per_smsp=num_warps_per_smsp,
                    dram=self.dram,
                )
            ]

        self._wire_hierarchy()
        
        # Register the architectural parameter buffer in DRAM (0..PARAM_BUFFER_SIZE-1)
        if "__param_buffer__" not in self.dram.allocations:
            self.dram.alloc("__param_buffer__", size=PARAM_BUFFER_SIZE)

    def _wire_hierarchy(self) -> None:
        """Wire shared DRAM to all SMs/SMSPs and configure device-wide warp numbering."""
        total_warps = sum(smsp.num_warps for sm in self.sms for smsp in sm.smsps)
        curr_offset = 0
        self.resources = [self.dram]
        for idx, sm in enumerate(self.sms):
            sm.sm_id = idx
            curr_offset = sm.bind_dram(self.dram, base_warp_offset=curr_offset, total_device_warps=total_warps)
            self.resources.append(sm.sram)
            for smsp in sm.smsps:
                self.resources.extend([smsp.valu, smsp.gemm, smsp.trans])

    @property
    def sm(self) -> SM:
        """Convenience accessor for SM 0."""
        return self.sms[0]

    @property
    def smsp(self) -> "SMSP":
        """Convenience accessor for SM 0, SMSP 0."""
        return self.sms[0].smsps[0]

    @property
    def total_warps(self) -> int:
        """Total number of warps across all SMs and SMSPs in the device."""
        return sum(smsp.num_warps for sm in self.sms for smsp in sm.smsps)

    # ========================================================================
    # Host-Side DRAM Allocation & Directional Copy APIs
    # ========================================================================
    def dram_alloc(self, name: str, size: int) -> "MemoryAllocation":
        """
        Allocate a buffer of `size` 32-bit words in the device's shared DRAMResource.
        Only creates the buffer; data transfer is performed separately via `dram_copy`.
        """
        if not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError(f"dram_alloc expects a positive integer word size, got {size!r}.")
        return self.dram.alloc(name=name, size=int(size))

    # Alias
    alloc_dram = dram_alloc

    def dram_free(self, name: str) -> None:
        """Free a named allocation in the device's shared DRAMResource."""
        self.dram.free(name)

    free_dram = dram_free

    def dram_copy(
        self,
        alloc_or_addr: Union["MemoryAllocation", int],
        host_array: Optional[np.ndarray] = None,
        direction: Union[CopyDirection, str] = CopyDirection.H2D,
        size: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Transfer data between a host NumPy array and an allocated buffer in `self.dram`.
        
        Args:
            alloc_or_addr: Target `MemoryAllocation` handle or integer starting word address.
            host_array: Source NumPy array (required for H2D; optional destination buffer for D2H).
            direction: `CopyDirection.H2D` ("H2D") or `CopyDirection.D2H` ("D2H").
            size: Number of words to copy (defaults to allocation size or `len(host_array)`).
            
        Returns:
            For "D2H": the populated 1D `np.float32` NumPy array.
            For "H2D": `None`.
        """
        from pgpu.arch.memory import MemoryAllocation

        if isinstance(direction, CopyDirection):
            dir_str = direction.value
        elif isinstance(direction, str):
            dir_str = direction.strip().upper()
        else:
            raise ValueError(f"Invalid copy direction: {direction!r}. Expected 'H2D' or 'D2H'.")

        if dir_str not in ("H2D", "D2H"):
            raise ValueError(f"Invalid copy direction '{dir_str}'. Expected 'H2D' or 'D2H'.")

        if isinstance(alloc_or_addr, MemoryAllocation):
            base_addr = alloc_or_addr.base_addr
            alloc_size = alloc_or_addr.size
        elif isinstance(alloc_or_addr, (int, np.integer)):
            base_addr = int(alloc_or_addr)
            alloc_size = size
        else:
            raise TypeError(f"Expected MemoryAllocation or int word address, got {type(alloc_or_addr).__name__}.")

        if dir_str == "H2D":
            if host_array is None:
                raise ValueError("`host_array` must be provided when direction='H2D'.")
            flat = np.asarray(host_array, dtype=np.float32).ravel()
            num_words = int(size) if size is not None else len(flat)
            if num_words > len(flat):
                raise ValueError(f"Requested H2D copy of {num_words} words, but host_array has {len(flat)} elements.")
            addrs = np.arange(base_addr, base_addr + num_words, dtype=np.int64)
            self.dram.validate_allocated_addresses(addrs)
            for i in range(num_words):
                self.dram.memory[base_addr + i] = float(flat[i])
            return None
        else:
            # D2H
            num_words = int(size) if size is not None else (
                len(host_array.ravel()) if host_array is not None else alloc_size
            )
            if num_words is None:
                raise ValueError("`size` must be specified for D2H copy when passing a raw integer address.")
            addrs = np.arange(base_addr, base_addr + num_words, dtype=np.int64)
            self.dram.validate_allocated_addresses(addrs)
            out = np.array([self.dram.memory[base_addr + i] for i in range(num_words)], dtype=np.float32)
            if host_array is not None:
                host_array.ravel()[:num_words] = out
                return host_array
            return out

    # Alias
    memcpy = dram_copy

    # ========================================================================
    # Kernel Launch Interface
    # ========================================================================
    def launch(self, compiled_kernel: Any, *args: Any) -> int:
        """
        Launch an SPMD Kernel on `device` with dynamic arguments.
        
        1. Validates argument count against compiled kernel parameters.
        2. Writes argument values (DRAM allocation base addresses or numeric scalars)
           into the device's DRAM parameter buffer.
        3. Registers in-kernel SRAM allocations with each SM's SRAMResource,
           and writes their runtime SRAM base addresses into the parameter buffer.
        4. Executes the kernel across all SMs and SMSPs.
        5. Invokes driver epilogue cleanup to clear SRAM allocations.
        Returns total elapsed simulation cycles.
        """
        from pgpu.arch.memory import MemoryAllocation
        from pgpu.sw.driver import cleanup_sram

        arg_names = getattr(compiled_kernel, "arg_names", [])
        if len(args) != len(arg_names):
            raise TypeError(
                f"Kernel expected {len(arg_names)} arguments ({arg_names}), got {len(args)}."
            )

        # 1. Ensure parameter buffer is allocated in device DRAM
        if "__param_buffer__" not in self.dram.allocations:
            self.dram.alloc("__param_buffer__", size=PARAM_BUFFER_SIZE)

        # 2. Extract values for kernel arguments
        param_values: List[float] = []
        for arg in args:
            if isinstance(arg, MemoryAllocation):
                param_values.append(float(arg.base_addr))
            elif isinstance(arg, (int, float, np.integer, np.floating)):
                param_values.append(float(arg))
            else:
                raise TypeError(f"Unsupported kernel argument type {type(arg)}: {arg!r}")

        # 3. Clean and map SRAM allocations across all SMs, and extract their base addresses
        cleanup_sram(self)
        for sm in self.sms:
            if hasattr(compiled_kernel, "sram_interface") and compiled_kernel.sram_interface is not None:
                compiled_kernel.sram_interface.apply_to_sram(sm.sram)

        # Append runtime SRAM tile base addresses to the parameter values
        sram_bindings = getattr(compiled_kernel, "sram_bindings", [])
        for name, _ in sram_bindings:
            if name in self.sm.sram.allocations:
                param_values.append(float(self.sm.sram.allocations[name].base_addr))
            else:
                raise KeyError(f"SRAM allocation '{name}' not found on device.")

        # 4. Populate DRAM parameter buffer
        for slot_idx, val in enumerate(param_values):
            self.dram.memory[PARAM_BUFFER_BASE + slot_idx] = float(val)

        # 5. Run compute state on all SMSPs
        max_cycles = 0
        for sm in self.sms:
            for smsp in sm.smsps:
                smsp.reset()
                cycles = smsp.run(compiled_kernel)
                if cycles > max_cycles:
                    max_cycles = cycles

        # 6. Epilogue: cleanup SRAM allocations via driver function
        cleanup_sram(self)

        return max_cycles

    run = launch

    def add_resource(self, resource: AbstractResource, count: int = 1):
        """Add one or more instances of a hardware resource to the GPU device."""
        if isinstance(resource, AbstractResource):
            for _ in range(count):
                self.resources.append(resource)
        else:
            raise TypeError("Resource must be an instance of AbstractResource")

    def get_resources(self) -> List[AbstractResource]:
        """Return the list of configured hardware resources."""
        return self.resources

    def get_resource_by_name(self, name: str) -> Optional[AbstractResource]:
        """Find and return the first resource matching the given name."""
        for res in self.resources:
            if res.name == name:
                return res
        return None

    def reset_all(self):
        """Reset state across all hardware resources and SMSPs."""
        for res in self.resources:
            res.reset()
        for sm in self.sms:
            sm.reset_compute()

    def __repr__(self) -> str:
        return f"pGPU(name='{self.name}', num_sms={len(self.sms)}, total_warps={self.total_warps})"
