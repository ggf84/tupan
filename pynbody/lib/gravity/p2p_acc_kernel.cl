#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
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



inline REAL4
p2p_acc_kernel_main_loop(const uint nj,
                         __global const REAL4 *jpos,
                         __global const REAL *jmass,
                         REAL4 myPos,
                         REAL myMass,
                         __local REAL4 *sharedPos,
                         __local REAL *sharedMass)
{
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);

    REAL4 myAcc = (REAL4){0.0, 0.0, 0.0, 0.0};

    uint tile;
    uint numTiles = ((nj + lsize - 1) / lsize) - 1;
    for (tile = 0; tile < numTiles; ++tile) {

        uint jdx = min(tile * lsize + lid, nj-1);
        sharedPos[lid] = jpos[jdx];
        sharedMass[lid] = jmass[jdx];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = 0; j < lsize; ) {
            for (uint jj = j; jj < j + JUNROLL; ++jj) {
               REAL4 otherPos = sharedPos[jj];
               REAL otherMass = sharedMass[jj];
               myAcc = p2p_acc_kernel_core(myAcc,
                                           myPos,
                                           myMass,
                                           otherPos,
                                           otherMass);
            }
            j += JUNROLL;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint jdx = min(tile * lsize + lid, nj-1);
    sharedPos[lid] = jpos[jdx];
    sharedMass[lid] = jmass[jdx];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint j = 0; j < nj - (tile * lsize); ++j) {
        REAL4 otherPos = sharedPos[j];
        REAL otherMass = sharedMass[j];
        myAcc = p2p_acc_kernel_core(myAcc,
                                    myPos,
                                    myMass,
                                    otherPos,
                                    otherMass);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    return myAcc;
}


__kernel void p2p_acc_kernel(const uint ni,
                             const uint nj,
                             __global const REAL4 *ipos,
                             __global const REAL *imass,
                             __global const REAL4 *jpos,
                             __global const REAL *jmass,
                             __global REAL4 *iacc,
                             __local REAL4 *sharedPos,
                             __local REAL *sharedMass)
{
    uint gid = 1*get_global_id(0);

    uint i = (gid+0 < ni) ? (gid+0) : (ni-1);
    iacc[i] = p2p_acc_kernel_main_loop(nj,
                                        jpos, jmass,
                                        ipos[i], imass[i],
                                        sharedPos, sharedMass);

//    uint ii = (gid+1 < ni) ? (gid+1) : (ni-1);
//    iacc[ii] = p2p_acc_kernel_main_loop(nj,
//                                        jpos, jmass,
//                                        ipos[ii], imass[ii],
//                                        sharedPos, sharedMass);

//    uint iii = (gid+2 < ni) ? (gid+2) : (ni-1);
//    iacc[iii] = p2p_acc_kernel_main_loop(nj,
//                                        jpos, jmass,
//                                        ipos[iii], imass[iii],
//                                        sharedPos, sharedMass);

//    uint iiii = (gid+3 < ni) ? (gid+3) : (ni-1);
//    iacc[iiii] = p2p_acc_kernel_main_loop(nj,
//                                        jpos, jmass,
//                                        ipos[iiii], imass[iiii],
//                                        sharedPos, sharedMass);
}   // Output shape: ({ni},4)




/******************************************************************************/


/*
__kernel void p2p_acc_kernel(const uint ni,
                             const uint nj,
                             __global const REAL4 *ipos,
                             __global const REAL *imass,
                             __global const REAL4 *jpos,
                             __global const REAL *jmass,
                             __global REAL4 *iacc,
                             __local REAL4 *sharedPos,
                             __local REAL *sharedMass)
{
    uint tid = get_local_id(0);
    uint gid = get_global_id(0) * IUNROLL;
    uint localDim = get_local_size(0);

    REAL4 myPos[IUNROLL];
    REAL myMass[IUNROLL];
    REAL4 myAcc[IUNROLL];
    for (uint ii = 0; ii < IUNROLL; ++ii) {
        myPos[ii] = (gid + ii < ni) ? ipos[gid + ii] : ipos[ni-1];
        myMass[ii] = (gid + ii < ni) ? imass[gid + ii] : imass[ni-1];
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
            for (uint jj = j; jj < j + JUNROLL; ++jj) {
               REAL4 otherPos = sharedPos[jj];
               REAL otherMass = sharedMass[jj];
               for (uint ii = 0; ii < IUNROLL; ++ii) {
                   myAcc[ii] = p2p_acc_kernel_core(myAcc[ii],
                                                   myPos[ii],
                                                   myMass[ii],
                                                   otherPos,
                                                   otherMass);
               }
            }
            j += JUNROLL;
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
        for (uint ii = 0; ii < IUNROLL; ++ii) {
            myAcc[ii] = p2p_acc_kernel_core(myAcc[ii],
                                            myPos[ii],
                                            myMass[ii],
                                            otherPos,
                                            otherMass);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint ii = 0; ii < IUNROLL; ++ii) {
        if (gid + ii < ni) {
            iacc[gid + ii] = myAcc[ii];
        }
    }
}   // Output shape: ({ni},4)
*/

