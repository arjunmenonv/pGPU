import unittest
import numpy as np
from pgpu.arch.gpu_device import pGPU, SM, CopyDirection
from pgpu.arch.smsp import SMSP
from pgpu.sw.kernel import WarpKernel
from pgpu.arch.memory import DRAMResource, SRAMResource
from pgpu.sw.intrinsics import warp_reduce

class TestWarpReduction(unittest.TestCase):
    def test_warp_reduce_add(self):
        self._run_reduction("add", np.sum)

    def test_warp_reduce_max(self):
        self._run_reduction("max", np.max)

    def test_warp_reduce_min(self):
        self._run_reduction("min", np.min)

    def _run_reduction(self, op_name, np_op):
        dram = DRAMResource(name="DRAM")
        sm0_sram = SRAMResource(name="SRAM_0")
        smsp0 = SMSP(num_warps=4)
        sm0 = SM(sm_id=0, sram=sm0_sram, smsps=[smsp0])
        device = pGPU(name="pGPU", dram=dram, sms=[sm0])

        in_buf = device.dram_alloc("in_buf", 32)
        out_buf = device.dram_alloc("out_buf", 1)

        np.random.seed(42)
        input_data = np.random.randn(32).astype(np.float32)
        device.dram_copy(in_buf, input_data, direction=CopyDirection.H2D)

        def reduce_kernel(kb: WarpKernel):
            s_buf = kb.sram.alloc("s_buf", 32)
            kb.trace_start(f"WarpReduce_{op_name}")

            p_w0_only = kb.alloc_pred("p_w0_only")
            kb.asm("cmp.eq", p_w0_only, kb.sr_warpid, kb.rz)
            kb.asm("jmp", "end_kernel", pred=p_w0_only, pred_inv=True)

            with kb.reg_scope("load_scope") as scope:
                r_dram_base = scope.alloc("dram_base")
                r_sram_base = scope.alloc("sram_base")
                r_dram_addr = scope.alloc("dram_addr")
                r_sram_addr = scope.alloc("sram_addr")
                r_val = scope.alloc("val")

                kb.asm("set.imm", r_dram_base, float(in_buf.base_addr))
                kb.asm("set.imm", r_sram_base, float(s_buf.base_addr))

                r_32_val = scope.alloc("thirtytwo_val")
                kb.asm("set.imm", r_32_val, 32.0)
                r_global_tid_load = scope.alloc("global_tid_load")
                kb.asm("fma", r_global_tid_load, kb.sr_warpid, r_32_val, kb.sr_laneid)
                
                p_w0 = scope.alloc_pred("p_w0")
                kb.asm("cmp.lt", p_w0, r_global_tid_load, r_32_val)

                kb.asm("add", r_dram_addr, r_dram_base, kb.sr_laneid)
                kb.asm("dram.ld", r_val, r_dram_addr, pred=p_w0)

                kb.asm("add", r_sram_addr, r_sram_base, kb.sr_laneid)
                kb.asm("sram.st", r_sram_addr, r_val, pred=p_w0)

            # Wait for all lanes to finish writing to SRAM before reduction begins
            kb.asm("sync.warp")

            # Perform reduction
            
            warp_reduce(kb, s_buf, op=op_name)

            with kb.reg_scope("store_scope") as scope:
                p_lead = scope.alloc_pred("p_lead")
                r_32 = scope.alloc("thirtytwo")
                r_global_tid = scope.alloc("global_tid")
                kb.asm("set.imm", r_32, 32.0)
                kb.asm("fma", r_global_tid, kb.sr_warpid, r_32, kb.sr_laneid)
                kb.asm("cmp.eq", p_lead, r_global_tid, kb.rz)

                r_res = scope.alloc("res")
                r_sram_base = scope.alloc("sram_base")
                
                kb.asm("set.imm", r_sram_base, float(s_buf.base_addr))
                kb.asm("sram.ld", r_res, r_sram_base, pred=p_lead)

                r_out_base = scope.alloc("out_base")
                kb.asm("set.imm", r_out_base, float(out_buf.base_addr))
                kb.asm("dram.st", r_out_base, r_res, pred=p_lead)

            kb.trace_end(f"WarpReduce_{op_name}")
            kb.label("end_kernel")

        from pgpu.sw.kernel import build_kernel
        kernel = build_kernel(reduce_kernel, num_warps=4, debug=False)

        # --- TRACER INJECTION ---
        from pgpu.devtools.kernel_tracer import KernelTracer
        tracer = KernelTracer(f"warp_reduce_{op_name}_trace")
        tracer.attach(device, kernel)

        cycles = device.launch(kernel)

        # --- TRACER EXPORT ---
        tracer.export()

        out_data = device.dram_copy(out_buf, direction=CopyDirection.D2H)
        expected = np_op(input_data).astype(np.float32)

        np.testing.assert_allclose(out_data[0], expected, rtol=1e-5)
        print(f"Reduction ({op_name}) computed perfectly! Output: {out_data[0]}, Expected: {expected}")
        print("Number of cycles taken:", cycles)

if __name__ == "__main__":
    unittest.main()
