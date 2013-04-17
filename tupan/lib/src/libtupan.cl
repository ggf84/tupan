#if __OPENCL_VERSION__ <= CL_VERSION_1_1
    #ifdef cl_khr_fp64
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
#endif

#include "smoothing.c"
#include "universal_kepler_solver.c"

#include "phi_kernel.cl"
#include "acc_kernel.cl"
#include "acc_jerk_kernel.cl"
#include "tstep_kernel.cl"
#include "pnacc_kernel.cl"
#include "nreg_kernels.cl"
#include "sakura_kernel.cl"

