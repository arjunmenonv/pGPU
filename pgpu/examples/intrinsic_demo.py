"""Simple demo kernel using device intrinsics.

We allocate a DRAM buffer and fill it with a constant using ``memset``.
We also invoke a fast transcendental intrinsic (e.g. ``exp``) on a single
scalar value to illustrate ``emit_fast_trans``.
"""
from pgpu.sw.kernel import build_kernel
from pgpu.sw.intrinsics import memset, emit_fast_trans
from pgpu.devtools.kernel_tracer import KernelTracer
from pgpu.arch.gpu_device import pGPU

def demo_kernel(kb, r_buf_base):
    kb.trace_start("IntrinsicDemo")
    
    # Fill the buffer with 3.14 using the memset intrinsic
    buf_base = 0
    buf_size = 64
    
    memset(kb, target=buf_base, size=buf_size, value=3.14)
    
    # IMPORTANT: Since memset is cooperative across all warps, we must sync 
    # to ensure all warps have finished their stores before anyone reads the buffer!
    kb.asm("sync")
    
    # We only want one thread (Warp 0, Lane 0) to do the scalar work to avoid redundant/overlapping stores
    r0 = kb.alloc_reg("r0")
    r_addr = kb.alloc_reg("addr")
    p_lead = kb.alloc_pred("is_lead")
    
    # p_lead = (SR_WARPID == 0) & (SR_LANEID == 0)
    kb.asm("add", r0, kb.sr_warpid, kb.sr_laneid)
    kb.asm("set.imm", r_addr, 0.0)
    kb.asm("cmp.eq", p_lead, r0, r_addr) # r_addr happens to hold 0.0, which acts as 0
    
    kb.asm("dram.ld", r0, r_addr, pred=p_lead)
    
    # Compute exp fast
    emit_fast_trans(kb, "exp", r0, r0, n=8, pred=p_lead)
    
    # Store result back
    kb.asm("dram.st", r_addr, r0, pred=p_lead)
    
    kb.trace_end("IntrinsicDemo")
    return kb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Show live-range view")
    parser.add_argument("--mode", default="wide", choices=["wide", "narrow"], help="objdump view mode")
    args = parser.parse_args()
    
    buf_size = 64
    warp_write_size = 32
    num_req_warps = max(4, (buf_size + warp_write_size - 1) // warp_write_size)
    
    dev = pGPU(num_warps_per_smsp=num_req_warps)
    alloc = dev.dram_alloc("buf", buf_size)
    
    # Build the kernel
    kernel = build_kernel(demo_kernel)
    
    # Attach tracer
    tracer = KernelTracer("intrinsic_demo_trace")
    tracer.attach(dev, kernel)
    
    # Run the kernel
    dev.launch(kernel, alloc)
    
    # Export artifacts
    tracer.export()
    
    if args.live:
        from pgpu.devtools.objdump import objdump
        print(objdump(kernel, reg_live_range=True, mode=args.mode))
    else:
        from pgpu.devtools.objdump import objdump
        print(objdump(kernel))
