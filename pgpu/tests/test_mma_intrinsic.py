import unittest
import numpy as np
from pgpu.arch.gpu_device import pGPU, SM, CopyDirection
from pgpu.arch.smsp import SMSP
from pgpu.sw.kernel import WarpKernel
from pgpu.arch.memory import DRAMResource, SRAMResource
from pgpu.sw.intrinsics import memset, mma_8x16_8x8

class TestMMAIntrinsic(unittest.TestCase):
    def test_mma_8x16_8x8(self):
        # 1. Setup hardware
        dram = DRAMResource(name="DRAM")
        sm0_sram = SRAMResource(name="SRAM_0")
        smsp0 = SMSP(num_warps=1)
        sm0 = SM(sm_id=0, sram=sm0_sram, smsps=[smsp0])
        device = pGPU(name="pGPU", dram=dram, sms=[sm0])
        out_buf = device.dram_alloc("out_buf", 64)
        
        val_a = 2.0
        val_b = 3.0
        val_c = 1.0
        
        def mma_test_kernel(kb: WarpKernel):
            # Allocate SRAM for A (128), B (128), C (64)
            a_sram = kb.sram.alloc("A", 128)
            b_sram = kb.sram.alloc("B", 128)
            c_sram = kb.sram.alloc("C", 64)
            
            # Initialize SRAM with memset
            memset(kb, a_sram, value=val_a)
            memset(kb, b_sram, value=val_b)
            memset(kb, c_sram, value=val_c)
            
            kb.asm("sync")
            
            kb.trace_start("MMA_Kernel")
            with kb.reg_scope("mma_exec") as scope:
                d0 = scope.alloc("d0")
                d1 = scope.alloc("d1")
                
                # Invoke the intrinsic
                mma_8x16_8x8(kb, a_sram, b_sram, c_sram, (d0, d1))
                
                # Compute output coordinates to write D out to DRAM
                r_row = scope.alloc("row")
                r_col = scope.alloc("col")
                r_thresh = scope.alloc("thresh")
                r_one = scope.alloc("one")
                
                kb.asm("set.imm", r_row, 0.0)
                kb.asm("set.imm", r_one, 1.0)
                
                for thresh in [8.0, 16.0, 24.0]:
                    p_ge = scope.alloc_pred(f"p_ge_{int(thresh)}")
                    kb.asm("set.imm", r_thresh, thresh)
                    kb.asm("cmp.lt", p_ge, kb.sr_laneid, r_thresh)
                    kb.asm("add", r_row, r_row, r_one, pred=p_ge, pred_inv=True)
                    
                r_eight = scope.alloc("eight")
                r_row_x_8 = scope.alloc("row_x_8")
                kb.asm("set.imm", r_eight, 8.0)
                kb.asm("mul", r_row_x_8, r_row, r_eight)
                kb.asm("sub", r_col, kb.sr_laneid, r_row_x_8)
                
                # c_base = row * 8 + col
                r_c_base = scope.alloc("c_base")
                kb.asm("fma", r_c_base, r_row, r_eight, r_col)
                
                # Write D out to DRAM
                r_dram_base = scope.alloc("dram_base")
                kb.asm("set.imm", r_dram_base, float(out_buf.base_addr))
                
                r_tmp_addr = scope.alloc("tmp_addr")
                
                # Write Q00 (d0)
                kb.asm("add", r_tmp_addr, r_dram_base, r_c_base)
                kb.asm("dram.st", r_tmp_addr, d0)
                
                # Write Q10 (d1) (offset +32)
                r_32 = scope.alloc("thirtytwo")
                kb.asm("set.imm", r_32, 32.0)
                kb.asm("add", r_tmp_addr, r_tmp_addr, r_32)
                kb.asm("dram.st", r_tmp_addr, d1)
            kb.trace_end("MMA_Kernel")
                
        # 3. Build Kernel
        from pgpu.sw.kernel import build_kernel
        kernel = build_kernel(mma_test_kernel, num_warps=1, debug=False)

        # --- TRACER INJECTION ---
        from pgpu.devtools.kernel_tracer import KernelTracer
        tracer = KernelTracer("mma_8x16_8x8_trace")
        tracer.attach(device, kernel)
        
        # 4. Launch kernel
        cycles = device.launch(kernel)
        
        # --- TRACER EXPORT ---
        tracer.export()
        
        # 5. Fetch and verify results
        d_out = device.dram_copy(out_buf, direction=CopyDirection.D2H)
        
        # Equivalent NumPy implementation
        A = np.full((8, 16), val_a, dtype=np.float32)
        B = np.full((8, 16), val_b, dtype=np.float32)
        C = np.full((8, 8), val_c, dtype=np.float32)
        
        expected_D = A @ B.T + C
        expected_D_flat = expected_D.flatten()
        
        np.testing.assert_allclose(d_out, expected_D_flat, rtol=1e-5)
        print("MMA Intrinsic successfully computed D = A @ B.T + C!")
        print("Num cycles taken:", cycles)

if __name__ == "__main__":
    unittest.main()
