'''
Defines the following:
- AbstractResource: Base class for all GPU hardware resources (Memory, Execution Units)
- pGPU: Class for building a GPU device from resources

Author: Arjun Vadakkeveedu (arjunmenonv@alumni.iitm.ac.in)
Refactored & Extended: September 2026
'''

from typing import List, Optional
from pgpu.arch.smsp import WarpState


class AbstractResource:
    """
    Abstract base class for a hardware resource in the pGPU simulator.
    
    Attributes:
        name (str): Identifier for the resource (e.g., "DRAM", "SRAM", "VALU", "MMA", "SFU").
        base_latency (int): Base latency associated with accessing this resource.
        next_available_cycle (int): Cycle when the resource port becomes free to accept 
                                    the next instruction.
    """
    def __init__(self, name: str, base_latency: int = 1, stall_signal: WarpState = WarpState.WARP_COMPUTE_STALL):
        self.name = name
        self.base_latency = base_latency
        self.stall_signal = stall_signal
        self.next_available_cycle = 0

    def is_available(self, current_cycle: int) -> bool:
        """Check if the resource port is available at current_cycle."""
        return current_cycle >= self.next_available_cycle

    def get_actual_start(self, issue_cycle: int) -> int:
        """Return the earliest cycle when an instruction can actually issue on this resource."""
        return max(issue_cycle, self.next_available_cycle)

    def reserve(self, issue_cycle: int, duration: int) -> int:
        """
        Reserve the resource port for an operation starting at issue_cycle.
        
        Args:
            issue_cycle (int): Cycle when the instruction attempts to issue.
            duration (int): Execution latency of the operation in cycles.
            
        Returns:
            int: The actual start cycle of the operation.
        """
        actual_start = self.get_actual_start(issue_cycle)
        self.next_available_cycle = actual_start + duration
        return actual_start

    def InstrRetireRoutine(self, *args, **kwargs):
        """
        Resource-side cleanup routine triggered by `Instruction.retire()` for each
        resource in `instruction.resources` (LLVM MachineScheduler style).
        Stateless execution units default to a no-op; stateful resources (such as
        MemoryResource) override this to pop in-flight tokens (e.g., InFlightStore).
        """
        pass

    def reset(self):
        """Reset resource port state for a new simulation run."""
        self.next_available_cycle = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', base_lat={self.base_latency}, next_avail={self.next_available_cycle})"


class pGPU:
    """
    Container class representing a pGPU device constructed from hardware resources.
    """
    def __init__(self, name: str):
        self.name = name
        self.resources: List[AbstractResource] = []

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
        """Reset state across all hardware resources."""
        for res in self.resources:
            res.reset()

    def __repr__(self) -> str:
        return f"pGPU(name='{self.name}', num_resources={len(self.resources)})"
