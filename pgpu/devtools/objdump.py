"""
pGPU objdump -- Kernel Disassembler & Instruction Tracer

Usage:
    # Standard disassembly (default demo kernel)
    python3 pgpu/devtools/objdump.py

    # Disassemble a specific build function
    python3 pgpu/devtools/objdump.py pgpu/examples/vector_add.py -k build_vector_add_kernel

    # Auto-wrap a raw kernel function via build_kernel
    python3 pgpu/devtools/objdump.py pgpu/examples/vector_add.py -k vector_add_kernel

    # With register live ranges (narrow = only used regs, wide = all 64+8 regs)
    python3 pgpu/devtools/objdump.py pgpu/examples/vector_add.py -k build_vector_add_kernel --live --mode narrow

    # Trace mode: run the simulator, capture the instruction stream for warp 0
    python3 pgpu/devtools/objdump.py pgpu/examples/vector_add.py -k build_vector_add_kernel --trace

    # Trace warp 2, with live ranges, export HTML viewer
    python3 pgpu/devtools/objdump.py pgpu/examples/vector_add.py -k build_vector_add_kernel --trace --warp 2 --live --html trace_out.html
    # Then open: python3 -m webbrowser trace_out.html
"""

import sys
import os
import json
from typing import Union, List, Optional

from pgpu.sw.kernel import CompiledKernel, WarpKernel, KernelProgram
from pgpu.arch.isa import Instruction, OpCode, OpClass, RegisterFile

# ─────────────────────────────────────────────────────────────────
# Live-range analysis
# ─────────────────────────────────────────────────────────────────

def compute_live_ranges(instructions: List[Instruction]):
    """
    Standard backward liveness analysis.
    Returns (live_in, live_out): one set per instruction.
    """
    n = len(instructions)
    live_in  = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]

    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            instr = instructions[i]

            new_out: set = set()
            if instr.opcode == OpCode.JMP:
                target_pc = int(instr.imm) if instr.imm is not None else i
                if 0 <= target_pc < n:
                    new_out |= live_in[target_pc]
                if instr.pred is not None and i + 1 < n:   # conditional → falls through too
                    new_out |= live_in[i + 1]
            elif i + 1 < n:
                new_out |= live_in[i + 1]
            live_out[i] = new_out

            uses = set(instr.srcs)
            if instr.pred is not None:
                uses.add(RegisterFile.PRED_BASE_ID + instr.pred)
            defs = set(instr.dest)
            new_in = uses | (live_out[i] - defs)
            if live_in[i] != new_in:
                live_in[i] = new_in
                changed = True

    return live_in, live_out


# ─────────────────────────────────────────────────────────────────
# Core objdump renderer
# ─────────────────────────────────────────────────────────────────

def _reg_label(r: int) -> str:
    if r < RegisterFile.NUM_VECTOR_REGS:
        return f"R{r}"
    if r == RegisterFile.RZ_ID:
        return "RZ"
    if r == RegisterFile.SR_LANEID_ID:
        return "SR_LANEID"
    if r == RegisterFile.SR_WARPID_ID:
        return "SR_WARPID"
    if r == RegisterFile.SR_NUMWARPS_ID:
        return "SR_NUMWARPS"
    if RegisterFile.PRED_BASE_ID <= r < RegisterFile.PRED_BASE_ID + RegisterFile.NUM_PRED_REGS:
        return f"P{r - RegisterFile.PRED_BASE_ID}"
    return f"R{r}"


def _state_char(r: int, instr: Instruction, live_in_i: set, live_out_i: set) -> str:
    """Return the nvdisasm-style state character for register `r` at instruction `i`."""
    if r in instr.dest:                               return "W"
    if r in instr.srcs or (instr.pred is not None and r == RegisterFile.PRED_BASE_ID + instr.pred): return "R"
    if r in live_in_i or r in live_out_i:             return "|"
    return "."


def _hex_group(regs: List[int], instr: Instruction,
               live_in_i: set, live_out_i: set) -> str:
    """
    Encode a group of up to 8 registers as a 2-digit hex number.
    Bit 7 (MSB) = first register in group, bit 0 (LSB) = last.
    A register is 'live' (bit=1) if it is W, R, or | at this instruction.
    """
    val = 0
    for bit, r in enumerate(regs):
        ch = _state_char(r, instr, live_in_i, live_out_i)
        if ch != ".":
            val |= (1 << (7 - bit))
    return f"{val:02X}"


def _format_live_row_narrow(i: int, instr: Instruction,
                            live_in, live_out,
                            v_groups: List[List[int]],
                            p_group: List[int]) -> str:
    """
    Compact hex row: one 2-digit hex per group of 8 vector registers,
    plus one for the 8 predicate registers.
    Example: C0 FE 00 00 00 00 00 00 | 80 | add R0, R1, R2
    """
    num_v_alive = sum(1 for r in live_in[i] if r < RegisterFile.NUM_VECTOR_REGS)
    num_p_alive = sum(1 for r in live_in[i]
                      if RegisterFile.PRED_BASE_ID <= r < RegisterFile.PRED_BASE_ID + RegisterFile.NUM_PRED_REGS)
    alive_str = f"{num_v_alive:2d}/64 {num_p_alive}/8"

    # We center the 2-digit hex string in a 6-character column to match the R0-7 headers
    v_str = " ".join(_hex_group(grp, instr, live_in[i], live_out[i]).center(6) for grp in v_groups)
    p_str = _hex_group(p_group, instr, live_in[i], live_out[i]).center(4)
    return f" {i:03d} | {alive_str} | {v_str} | {p_str} | {instr.disassemble()}"


def _format_live_row_wide(i: int, instr: Instruction,
                          live_in, live_out,
                          v_groups: List[List[int]],
                          p_group: List[int]) -> str:
    """
    Character-per-register row, grouped in blocks of 8 (no intra-group spaces).
    Example: WR|..... ||...... ........ ........ | ........ | add R0, R1, R2
    """
    num_v_alive = sum(1 for r in live_in[i] if r < RegisterFile.NUM_VECTOR_REGS)
    num_p_alive = sum(1 for r in live_in[i]
                      if RegisterFile.PRED_BASE_ID <= r < RegisterFile.PRED_BASE_ID + RegisterFile.NUM_PRED_REGS)
    alive_str = f"{num_v_alive:2d}/64 {num_p_alive}/8"

    v_str = " ".join(
        "".join(_state_char(r, instr, live_in[i], live_out[i]) for r in grp)
        for grp in v_groups
    )
    p_str = "".join(_state_char(r, instr, live_in[i], live_out[i]) for r in p_group)
    return f" {i:03d} | {alive_str} | {v_str} | {p_str} | {instr.disassemble()}"


def _live_range_groups():
    """Return (v_groups, p_group): v_groups is 8 lists of 8 vector reg IDs each."""
    v_regs = list(range(RegisterFile.NUM_VECTOR_REGS))
    v_groups = [v_regs[k:k+8] for k in range(0, 64, 8)]
    p_group = list(range(RegisterFile.PRED_BASE_ID,
                         RegisterFile.PRED_BASE_ID + RegisterFile.NUM_PRED_REGS))
    return v_groups, p_group


def objdump(kernel: Union[CompiledKernel, WarpKernel, KernelProgram],
            reg_live_range: bool = False,
            mode: str = "narrow") -> str:
    """
    Return the disassembly of `kernel` as a string.

    Args:
        kernel:         CompiledKernel, WarpKernel, or KernelProgram.
        reg_live_range: If True, print nvdisasm-style register live range columns.
        mode:           Live-range rendering mode (mirrors nvdisasm -lrm):
                          "narrow" (default) — one 2-digit hex per group of 8 regs.
                                               Shows all 64+8 registers compactly.
                          "wide"             — one char (W/R/|/.) per register,
                                               grouped in blocks of 8 with no intra-
                                               group spaces. Wider but more readable.
    """
    instructions: List[Instruction] = list(kernel)
    output: List[str] = []

    if not reg_live_range:
        for i, instr in enumerate(instructions):
            output.append(f"{i:03d} | {instr.disassemble()}")
        if isinstance(kernel, CompiledKernel):
            output.append('driver_call("<sm.sram.allocations.clear()>")')
        return "\n".join(output)

    # ── Live-range mode ──────────────────────────────────────────
    live_in, live_out = compute_live_ranges(instructions)
    v_groups, p_group = _live_range_groups()

    if mode == "narrow":
        # Header: group labels R0-7, R8-15, …, R56-63 | P0-7
        grp_headers = [f"R{k}-{k+7}" for k in range(0, 64, 8)]
        hdr_v = " ".join(f"{h:^6}" for h in grp_headers)   # Center in 6-char columns
        header = f" PC  | Alive (V/P) | {hdr_v} | P0-7 | Instruction"
        output.append(header)
        output.append("-" * len(header))
        for i, instr in enumerate(instructions):
            output.append(_format_live_row_narrow(i, instr, live_in, live_out, v_groups, p_group))

    else:  # wide
        # Header: R0-7, R8-15, … each occupying 8 chars
        grp_headers = [f"R{k}-{k+7}" for k in range(0, 64, 8)]
        hdr_v = " ".join(f"{h:<8}" for h in grp_headers)
        header = f" PC  | Alive (V/P) | {hdr_v} | P0-7    | Instruction"
        output.append(header)
        output.append("-" * len(header))
        for i, instr in enumerate(instructions):
            output.append(_format_live_row_wide(i, instr, live_in, live_out, v_groups, p_group))

    if isinstance(kernel, CompiledKernel):
        output.append('driver_call("<sm.sram.allocations.clear()>")')

    return "\n".join(output)


# ─────────────────────────────────────────────────────────────────
# Trace mode: run the simulator and capture per-warp instruction stream
# ─────────────────────────────────────────────────────────────────

def run_trace(kernel: CompiledKernel, target_warp: int = 0,
              launch_args: Optional[tuple] = None):
    """
    Execute the kernel through pGPU and collect the per-warp instruction trace.

    The tracer hooks into the SMSP's trace_hook callback after every instruction issue.
    `launch_args` are passed through to pGPU.launch; if None, zeros are used for all
    kernel arguments (sufficient to capture the instruction stream structure).

    Returns a list of compact dicts:
        { opcode, op_class, dest, srcs, pred, pred_inv, imm,
          issue_cycle, completion_cycle }
    """
    import numpy as np
    from pgpu.arch.gpu_device import pGPU

    trace_records: List[dict] = []

    def _hook(warp_id: int, pc: int, clone: Instruction, issue_cycle: int):
        if warp_id != target_warp:
            return
        trace_records.append({
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

    # Build a pGPU device, inject the hook on all SMSPs, then launch
    dev = pGPU(num_warps_per_smsp=4)
    for sm in dev.sms:
        for smsp in sm.smsps:
            smsp.trace_hook = _hook

    # Prepare zero-filled dummy args for all kernel parameters.
    # We allocate 32-word buffers (one warp-width) per arg so DRAM bounds checks pass
    # even when the kernel accesses lane-strided addresses inside the trace run.
    if launch_args is None:
        num_args = len(kernel.arg_names) if hasattr(kernel, "arg_names") and kernel.arg_names else 0
        if num_args > 0:
            dummy_allocs = [dev.dram_alloc(f"_trace_arg_{i}", 32) for i in range(num_args)]
            dev.launch(kernel, *dummy_allocs)
        else:
            dev.launch(kernel)
    else:
        dev.launch(kernel, *launch_args)

    return trace_records


def format_trace(records: List[dict], reg_live_range: bool = False,
                 mode: str = "narrow") -> str:
    """Render a captured trace as a human-readable string."""
    # Reconstruct Instruction objects for disassembly / liveness
    instrs: List[Instruction] = []
    for r in records:
        instrs.append(Instruction(
            opcode=OpCode(r["opcode"]),
            op_class=OpClass(r["op_class"]),
            dest=r["dest"],
            srcs=r["srcs"],
            pred=r["pred"],
            pred_inv=r["pred_inv"],
            imm=r["imm"],
        ))

    output: List[str] = []
    if not reg_live_range:
        for i, (instr, rec) in enumerate(zip(instrs, records)):
            issue = rec.get("issue_cycle", "?")
            comp  = rec.get("completion_cycle", "?")
            output.append(f"{i:04d} | [{issue:>7} → {comp:>7}] | {instr.disassemble()}")
        return "\n".join(output)

    live_in, live_out = compute_live_ranges(instrs)

    all_used: set = set()
    for i, instr in enumerate(instrs):
        all_used |= live_in[i] | live_out[i]
        all_used |= set(instr.dest) | set(instr.srcs)
        if instr.pred is not None:
            all_used.add(instr.pred)

    if mode == "wide":
        v_regs = list(range(RegisterFile.NUM_VECTOR_REGS))
        p_regs = list(range(RegisterFile.PRED_BASE_ID,
                            RegisterFile.PRED_BASE_ID + RegisterFile.NUM_PRED_REGS))
    else:
        v_regs = sorted(r for r in all_used if r < RegisterFile.NUM_VECTOR_REGS)
        p_regs = sorted(r for r in all_used
                        if RegisterFile.PRED_BASE_ID <= r < RegisterFile.PRED_BASE_ID + RegisterFile.NUM_PRED_REGS)

    hdr_v = " ".join(f"R{r}" for r in v_regs)
    hdr_p = " ".join(f"P{r - RegisterFile.PRED_BASE_ID}" for r in p_regs)
    header = f" IDX  | Issue → Comp  | Alive (V/P) | {hdr_v} | {hdr_p} | Instruction"
    output.append(header)
    output.append("-" * len(header))

    for i, (instr, rec) in enumerate(zip(instrs, records)):
        issue = rec.get("issue_cycle", "?")
        comp  = rec.get("completion_cycle", "?")
        timing = f"[{issue:>7} → {comp:>7}]"
        live_row = _format_live_row(i, instr, live_in, live_out, v_regs, p_regs)
        # Splice timing column between IDX and Alive
        output.append(f" {i:04d} | {timing} |" + live_row[6:])

    return "\n".join(output)


# ─────────────────────────────────────────────────────────────────
# HTML viewer export
# ─────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>pGPU Instruction Trace – Warp {warp_id}</title>
<style>
  body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; margin: 0; padding: 0; }}
  h1 {{ background: #252526; padding: 10px 18px; margin: 0; font-size: 14px; color: #cccccc;
        border-bottom: 1px solid #3c3c3c; }}
  #toolbar {{ background: #2d2d2d; padding: 6px 18px; border-bottom: 1px solid #3c3c3c;
              display: flex; gap: 12px; align-items: center; font-size: 12px; }}
  #toolbar label {{ color: #9cdcfe; }}
  #search {{ background: #3c3c3c; border: 1px solid #555; color: #d4d4d4;
             padding: 3px 8px; border-radius: 3px; width: 260px; font-size: 12px; }}
  #grid {{ display: grid; grid-template-columns: 60px 190px 90px auto; font-size: 12px;
           overflow-y: auto; height: calc(100vh - 82px); }}
  .hdr {{ position: sticky; top: 0; background: #252526; color: #9cdcfe;
          padding: 4px 8px; border-bottom: 1px solid #3c3c3c; font-weight: bold; }}
  .cell {{ padding: 2px 8px; border-bottom: 1px solid #2a2a2a; white-space: pre; }}
  .row-even {{ background: #1e1e1e; }}
  .row-odd  {{ background: #252526; }}
  .op-VALU     {{ color: #9cdcfe; }}
  .op-DRAM     {{ color: #f0a070; }}
  .op-SRAM     {{ color: #c586c0; }}
  .op-GEMMCORE {{ color: #4ec9b0; }}
  .op-TRANS    {{ color: #dcdcaa; }}
  .op-CTRL     {{ color: #808080; }}
  .timing {{ color: #ce9178; }}
  .idx    {{ color: #858585; }}
</style>
</head>
<body>
<h1>pGPU Instruction Trace &mdash; Warp {warp_id} &nbsp;|&nbsp; {n_instrs} instructions &nbsp;|&nbsp; total cycles: {total_cycles}</h1>
<div id="toolbar">
  <label>Filter: <input id="search" type="text" placeholder="opcode / mnemonic …"></label>
  <label style="color:#9cdcfe">Timing: <input id="chk-timing" type="checkbox" checked></label>
</div>
<div id="grid">
  <div class="hdr">IDX</div>
  <div class="hdr">Issue → Comp</div>
  <div class="hdr">Class</div>
  <div class="hdr">Instruction</div>
</div>
<script>
// Trace: each entry is [asm_string, op_class_name, issue_cycle, comp_cycle]
// Disassembly is pre-computed by Python – single source of truth.
const TRACE = {trace_json};

const grid = document.getElementById("grid");
let rows = [];

function render() {{
  rows.forEach(r => r.forEach(c => c.remove()));
  rows = [];
  const q = document.getElementById("search").value.toLowerCase();
  const showTiming = document.getElementById("chk-timing").checked;

  TRACE.forEach(([asm, cls, issue, comp], i) => {{
    if (q && !asm.toLowerCase().includes(q)) return;
    const parity = i % 2 === 0 ? "row-even" : "row-odd";

    const mk = (text, extra = "") => {{
      const d = document.createElement("div");
      d.className = "cell " + parity + " " + extra;
      d.textContent = text;
      return d;
    }};

    const cIdx    = mk(String(i).padStart(4, "0"), "idx");
    const cTiming = mk(showTiming ? (issue + " → " + (comp ?? "?")) : "", "timing");
    const cClass  = mk(cls, "op-" + cls);
    const cAsm    = mk(asm, "op-" + cls);

    [cIdx, cTiming, cClass, cAsm].forEach(c => grid.appendChild(c));
    rows.push([cIdx, cTiming, cClass, cAsm]);
  }});
}}

document.getElementById("search").addEventListener("input", render);
document.getElementById("chk-timing").addEventListener("change", render);
render();
</script>
</body>
</html>
"""


def export_html(records: List[dict], warp_id: int, output_path: str) -> None:
    """
    Write a self-contained HTML trace viewer to *output_path*.

    Trace format is [asm_string, op_class_name, issue_cycle, comp_cycle] —
    disassembly is performed here in Python (single source of truth),
    not duplicated in JavaScript.
    """
    compact = []
    for r in records:
        instr = Instruction(
            opcode=OpCode(r["opcode"]),
            op_class=OpClass(r["op_class"]),
            dest=r["dest"],
            srcs=r["srcs"],
            pred=r["pred"],
            pred_inv=r["pred_inv"],
            imm=r["imm"],
        )
        cls_name = OpClass(r["op_class"]).name  # e.g. "VALU", "DRAM", "SRAM" …
        compact.append([instr.disassemble(), cls_name,
                        r["issue_cycle"], r["completion_cycle"]])

    total_cycles = records[-1]["completion_cycle"] if records else 0

    html = _HTML_TEMPLATE.format(
        warp_id=warp_id,
        n_instrs=len(records),
        total_cycles=total_cycles,
        trace_json=json.dumps(compact),
    )

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Trace viewer saved → {output_path}")
    print(f"Open with:  python3 -m webbrowser '{os.path.abspath(output_path)}'")

# ─────────────────────────────────────────────────────────────────

def _load_kernel(file_path: str, func_name: str):
    import importlib.util, inspect
    from pgpu.sw.kernel import build_kernel, CompiledKernel
    import os

    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"Error: File '{abs_path}' not found.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_pgpu_dyn_module", abs_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_pgpu_dyn_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, func_name):
        print(f"Error: Function '{func_name}' not found in '{file_path}'")
        sys.exit(1)

    func = getattr(module, func_name)
    first_param = next(iter(inspect.signature(func).parameters), None)

    if first_param in ("kb", "builder", "kernel_builder"):
        print(f"Auto-wrapping raw kernel function '{func_name}' via build_kernel()...")
        return build_kernel(func)
    return func()

# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="pGPU objdump – disassemble a compiled kernel",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("file_path", nargs="?",
                        help="Path to the Python file containing the kernel function")
    parser.add_argument("-k", metavar="FUNC",
                        help="Name of the build function (or raw kernel function)")
    parser.add_argument("--live", action="store_true",
                        help="Print nvdisasm-style register live ranges")
    parser.add_argument("--mode", default="narrow", choices=["wide", "narrow"],
                        help="'narrow' (default) = compact 2-digit hex per 8-reg group | 'wide' = one char per reg")
    args = parser.parse_args()

    if args.file_path and args.k:
        kernel = _load_kernel(args.file_path, args.k)
    else:
        from pgpu.examples.vector_add import build_vector_add_kernel
        kernel = build_vector_add_kernel()

    print(objdump(kernel, reg_live_range=args.live, mode=args.mode))
