import json
from collections import defaultdict
from typing import Dict, List, Any, Optional

from pgpu.arch.gpu_device import pGPU
from pgpu.sw.kernel import CompiledKernel
from pgpu.arch.isa import Instruction

class KernelTracer:
    """
    Automated tracer that hooks into a pGPU device and generates Perfetto-compatible
    trace events and a standalone HTML ASM viewer.
    """
    def __init__(self, output_base: str = "trace", target_warp: int = 0):
        self.output_base = output_base
        self.target_warp = target_warp
        self.events: List[dict] = []
        self.trace_records: List[dict] = []
        self._active_regions: Dict[int, set] = defaultdict(set)
        self._trace_regions: Dict[str, tuple] = {}

    def attach(self, dev: pGPU, kernel: CompiledKernel):
        """
        Attach the tracer to a pGPU device and extract trace_regions from the kernel.
        """
        self._trace_regions = getattr(kernel, "trace_regions", {})
        
        # Inject hook to all SMSPs
        for sm in dev.sms:
            for smsp in sm.smsps:
                smsp.trace_hook = self._hook

    def _hook(self, warp_id: int, pc: int, clone: Instruction, issue_cycle: int):
        # 1. Update active regions (only if PC is valid for regions)
        # PC is set to -1 for driver-injected context switch instructions
        if pc != -1:
            for region_name, (start_pc, end_pc) in self._trace_regions.items():
                is_inside = (start_pc <= pc <= end_pc)
                was_inside = region_name in self._active_regions[warp_id]

                if is_inside and not was_inside:
                    # Entered region
                    self._active_regions[warp_id].add(region_name)
                    self.events.append({
                        "name": region_name,
                        "cat": "warp_execution",
                        "ph": "B",
                        "pid": warp_id,
                        "tid": "Execution",
                        "ts": issue_cycle,
                    })
                elif not is_inside and was_inside:
                    # Exited region
                    self._active_regions[warp_id].remove(region_name)
                    self.events.append({
                        "name": region_name,
                        "cat": "warp_execution",
                        "ph": "E",
                        "pid": warp_id,
                        "tid": "Execution",
                        "ts": issue_cycle,
                    })
        
        # 2. Log resource usage for each instruction within active regions
        for region_name in self._active_regions[warp_id]:
            dur = max(clone.completion_time - issue_cycle, 1)
            op_name = clone.opcode.name.lower().replace("_", ".")
            res_name = clone.op_class.name
            
            self.events.append({
                "name": op_name,
                "cat": f"resource_{res_name}",
                "ph": "X",
                "pid": warp_id,
                "tid": res_name,
                "ts": issue_cycle,
                "dur": dur,
                "args": {"label": region_name, "pc": pc}
            })

        # 3. Collect HTML trace viewer records for the target warp
        if warp_id == self.target_warp:
            self.trace_records.append({
                "opcode":           clone.opcode.value,
                "op_class":         clone.op_class.value,
                "dest":             list(clone.dest),
                "srcs":             list(clone.srcs),
                "pred":             clone.pred,
                "pred_inv":         clone.pred_inv,
                "imm":              clone.imm if not isinstance(clone.imm, str) else None,
                "issue_cycle":      issue_cycle,
                "completion_cycle": clone.completion_time,
            })

    def export(self):
        """Export the collected trace events to Perfetto JSON and HTML viewer."""
        # Close any active regions at the end
        if self.events:
            last_ts = max((e.get("ts", 0) + e.get("dur", 0)) for e in self.events)
        else:
            last_ts = 0

        for warp_id, regions in self._active_regions.items():
            for region_name in list(regions):
                self.events.append({
                    "name": region_name,
                    "cat": "warp_execution",
                    "ph": "E",
                    "pid": warp_id,
                        "tid": "Execution",
                    "ts": last_ts,
                })
        

        # Add metadata events to name the process brackets nicely
        for w_id in set(self._active_regions.keys()).union(set(e.get("pid") for e in self.events if "pid" in e)):
            self.events.insert(0, {
                "name": "process_name",
                "ph": "M",
                "pid": w_id,
                "args": {"name": f"Warp {w_id}"}
            })

        json_path = f"{self.output_base}.json"
        with open(json_path, "w") as f:
            json.dump({"traceEvents": self.events, "displayTimeUnit": "ns"}, f, indent=2)
        print(f"Kernel trace exported to {json_path} (Perfetto format)")

        html_path = f"{self.output_base}.html"
        from pgpu.devtools.objdump import export_html
        export_html(self.trace_records, warp_id=self.target_warp, output_path=html_path)
