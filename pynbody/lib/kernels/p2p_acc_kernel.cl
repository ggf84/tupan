#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_amd_fp64
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#ifdef cl_amd_printf
    #pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#ifdef DOUBLE
typedef double REAL;
typedef double2 REAL2;
typedef double4 REAL4;
#else
typedef float REAL;
typedef float2 REAL2;
typedef float4 REAL4;
#endif


REAL4 p2p_acc_kernel_core(REAL4 acc, REAL4 bi, REAL4 bj, REAL mj)
{
    REAL4 dr;
    dr.x = bi.x - bj.x;                                              // 1 FLOPs
    dr.y = bi.y - bj.y;                                              // 1 FLOPs
    dr.z = bi.z - bj.z;                                              // 1 FLOPs
    dr.w = bi.w + bj.w;                                              // 1 FLOPs
    REAL ds2 = 0.5 * dr.w;                                           // 1 FLOPs
    ds2 += dr.z * dr.z + dr.y * dr.y + dr.x * dr.x;                  // 6 FLOPs
    REAL rinv = rsqrt(ds2);                                          // 2 FLOPs
    rinv = (ds2 ? rinv:0);
    REAL mr3inv = mj * rinv;                                         // 1 FLOPs
    rinv *= rinv;                                                    // 1 FLOPs
    acc.w -= mr3inv;                                                 // 1 FLOPs
    mr3inv *= rinv;                                                  // 1 FLOPs
    acc.x -= mr3inv * dr.x;                                          // 2 FLOPs
    acc.y -= mr3inv * dr.y;                                          // 2 FLOPs
    acc.z -= mr3inv * dr.z;                                          // 2 FLOPs
    return acc;
}


__kernel void p2p_acc_kernel(__global const REAL4 *ipos,
                             __global const REAL4 *jpos,
                             __global const REAL *jmass,
                             __global REAL4 *iacc,
                             const uint ni,
                             const uint nj,
                             __local REAL4 *sharedPos,
                             __local REAL *sharedMass)
{
    uint tid = get_local_id(0);
    uint gid = get_global_id(0) * UNROLL_SIZE_I;
    uint localDim = get_local_size(0);

    REAL4 myPos[UNROLL_SIZE_I];
    REAL4 myAcc[UNROLL_SIZE_I];
    for (uint ii = 0; ii < UNROLL_SIZE_I; ++ii) {
        myPos[ii] = (gid + ii < ni) ? ipos[gid + ii] : ipos[ni-1];
        myAcc[ii] = (REAL4){0.0, 0.0, 0.0, 0.0};
    }

    uint tile;
    uint numTiles = ((nj + localDim - 1) / localDim) - 1;
    for (tile = 0; tile < numTiles; ++tile) {

        uint jdx = min(tile * localDim + tid, nj-1);
        sharedPos[tid] = jpos[jdx];
        sharedMass[tid] = jmass[jdx];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = 0; j < localDim; ) {
            for (uint jj = j; jj < j + UNROLL_SIZE_J; ++jj) {
               REAL4 otherPos = sharedPos[jj];
               REAL otherMass = sharedMass[jj];
               for (uint ii = 0; ii < UNROLL_SIZE_I; ++ii) {
                   myAcc[ii] = p2p_acc_kernel_core(myAcc[ii],
                                                   myPos[ii],
                                                   otherPos,
                                                   otherMass);
               }
            }
            j += UNROLL_SIZE_J;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint jdx = min(tile * localDim + tid, nj-1);
    sharedPos[tid] = jpos[jdx];
    sharedMass[tid] = jmass[jdx];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint j = 0; j < nj - (tile * localDim); ++j) {
        REAL4 otherPos = sharedPos[j];
        REAL otherMass = sharedMass[j];
        for (uint ii = 0; ii < UNROLL_SIZE_I; ++ii) {
                myAcc[ii] = p2p_acc_kernel_core(myAcc[ii],
                                                myPos[ii],
                                                otherPos,
                                                otherMass);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint ii = 0; ii < UNROLL_SIZE_I; ++ii) {
        if (gid + ii < ni) {
            iacc[gid + ii] = myAcc[ii];
        }
    }
}

