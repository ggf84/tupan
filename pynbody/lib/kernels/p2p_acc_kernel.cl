#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_amd_fp64
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
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


#include"p2p_acc_kernel_core.h"


__kernel void p2p_acc_kernel(const uint ni,
                             const uint nj,
                             __global const REAL4 *ipos,
                             __global const REAL4 *ivel,
                             __global const REAL4 *jpos,
                             __global const REAL4 *jvel,
                             __global REAL4 *iacc,
                             __local REAL4 *sharedPos,
                             __local REAL4 *sharedVel)
{
    uint tid = get_local_id(0);
    uint gid = get_global_id(0) * IUNROLL;
    uint localDim = get_local_size(0);

    REAL4 myPos[IUNROLL];
    REAL4 myVel[IUNROLL];
    REAL4 myAcc[IUNROLL];
    for (uint ii = 0; ii < IUNROLL; ++ii) {
        myPos[ii] = (gid + ii < ni) ? ipos[gid + ii] : ipos[ni-1];
        myVel[ii] = (gid + ii < ni) ? ivel[gid + ii] : ivel[ni-1];
        myAcc[ii] = (REAL4){0.0, 0.0, 0.0, 0.0};
    }

    uint tile;
    uint numTiles = ((nj + localDim - 1) / localDim) - 1;
    for (tile = 0; tile < numTiles; ++tile) {

        uint jdx = min(tile * localDim + tid, nj-1);
        sharedPos[tid] = jpos[jdx];
        sharedVel[tid] = jvel[jdx];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = 0; j < localDim; ) {
            for (uint jj = j; jj < j + JUNROLL; ++jj) {
               REAL4 otherPos = sharedPos[jj];
               REAL4 otherVel = sharedVel[jj];
               for (uint ii = 0; ii < IUNROLL; ++ii) {
                   myAcc[ii] = p2p_acc_kernel_core(myAcc[ii],
                                                   myPos[ii],
                                                   myVel[ii],
                                                   otherPos,
                                                   otherVel);
               }
            }
            j += JUNROLL;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint jdx = min(tile * localDim + tid, nj-1);
    sharedPos[tid] = jpos[jdx];
    sharedVel[tid] = jvel[jdx];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint j = 0; j < nj - (tile * localDim); ++j) {
        REAL4 otherPos = sharedPos[j];
        REAL4 otherVel = sharedVel[j];
        for (uint ii = 0; ii < IUNROLL; ++ii) {
            myAcc[ii] = p2p_acc_kernel_core(myAcc[ii],
                                            myPos[ii],
                                            myVel[ii],
                                            otherPos,
                                            otherVel);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint ii = 0; ii < IUNROLL; ++ii) {
        if (gid + ii < ni) {
            iacc[gid + ii] = myAcc[ii];
        }
    }
}   // Output shape: ({ni},4)

