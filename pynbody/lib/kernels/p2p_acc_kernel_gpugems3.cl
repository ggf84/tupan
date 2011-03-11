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

#define WRAP(x,m) (((x)<m)?(x):(x-m))


REAL4 p2p_acc_kernel_core(REAL4 acc, REAL4 bip, REAL4 biv, REAL4 bjp, REAL4 bjv)
{
    REAL4 dr;
    dr.x = bip.x - bjp.x;                                            // 1 FLOPs
    dr.y = bip.y - bjp.y;                                            // 1 FLOPs
    dr.z = bip.z - bjp.z;                                            // 1 FLOPs
    dr.w = bip.w + bjp.w;                                            // 1 FLOPs
    REAL4 dv;
    dv.x = biv.x - bjv.x;                                            // 1 FLOPs
    dv.y = biv.y - bjv.y;                                            // 1 FLOPs
    dv.z = biv.z - bjv.z;                                            // 1 FLOPs
    dv.w = biv.w + bjv.w;                                            // 1 FLOPs

    REAL dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;              // 5 FLOPs
    REAL dv2 = dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;              // 5 FLOPs

    REAL rinv = rsqrt(dr2 + dr.w);                                   // 3 FLOPs
    rinv = (dr2 ? rinv:0);
    REAL r3inv = rinv * rinv * rinv;                                 // 2 FLOPs

    REAL e = 0.5 * dv2 + dv.w * rinv;                                // 3 FLOPs
    acc.w += (e * e * e) / (dv.w * dv.w);                            // 5 FLOPs

    r3inv *= bjv.w;                                                  // 1 FLOPs
    acc.x -= r3inv * dr.x;                                           // 2 FLOPs
    acc.y -= r3inv * dr.y;                                           // 2 FLOPs
    acc.z -= r3inv * dr.z;                                           // 2 FLOPs
    return acc;
}   // Total flop count: 38


__kernel void p2p_acc_kernel_gpugems3(const uint ni,
                                      const uint nj,
                                      __global const REAL4 *ipos,
                                      __global const REAL4 *ivel,
                                      __global const REAL4 *jpos,
                                      __global const REAL4 *jvel,
                                      __global REAL4 *iacc,
                                      __local REAL4 *sharedPos,
                                      __local REAL4 *sharedVel)
{
    uint gid = get_global_id(0) * IUNROLL;
    uint tidx = get_local_id(0);
    uint tidy = get_local_id(1);
    uint grpidx = get_group_id(0);// * IUNROLL;
    uint grpidy = get_group_id(1);// * IUNROLL;
    uint grpDimx = get_num_groups(0);// * IUNROLL;
    uint localDimx = get_local_size(0);
    uint localDimy = get_local_size(1);

    REAL4 myPos[IUNROLL];
    REAL4 myVel[IUNROLL];
    REAL4 myAcc[IUNROLL];
    for (uint ii = 0; ii < IUNROLL; ++ii) {
        myPos[ii] = (gid + ii < ni) ? ipos[gid + ii] : ipos[ni-1];
        myVel[ii] = (gid + ii < ni) ? ivel[gid + ii] : ivel[ni-1];
        myAcc[ii] = (REAL4){0.0, 0.0, 0.0, 0.0};
    }

    uint tile;
    uint numTiles = ((nj + localDimx*localDimy - 1) / (localDimx*localDimy));
    for(tile = grpidy; tile < numTiles-1 + grpidy; ++tile) {

        uint sdx = tidx + localDimx * tidy;
        uint jdx = WRAP(grpidx + tile, grpDimx) * localDimx + tidx;
        sharedPos[sdx] = (jdx < nj) ? jpos[jdx] : (REAL4){0.0, 0.0, 0.0, 0.0};
        sharedVel[sdx] = (jdx < nj) ? jvel[jdx] : (REAL4){0.0, 0.0, 0.0, 0.0};

        barrier(CLK_LOCAL_MEM_FENCE);
        ulong k = 0;
        for(uint j = 0; j < localDimx; ) {
            for (uint jj = 0; jj < JUNROLL; ++jj) {
                uint kdx = localDimx * tidy + k++;
                REAL4 otherPos = sharedPos[kdx];
                REAL4 otherVel = sharedVel[kdx];
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

    uint sdx = tidx + localDimx * tidy;
    uint jdx = WRAP(grpidx + tile, grpDimx) * localDimx + tidx;
    sharedPos[sdx] = (jdx < nj) ? jpos[jdx] : (REAL4){0.0, 0.0, 0.0, 0.0};
    sharedVel[sdx] = (jdx < nj) ? jvel[jdx] : (REAL4){0.0, 0.0, 0.0, 0.0};

    barrier(CLK_LOCAL_MEM_FENCE);
    ulong k = 0;
    for(uint j = 0; j < (numTiles * localDimy - tile) * localDimx; ++j) {
        uint kdx = localDimx * tidy + k++;
        REAL4 otherPos = sharedPos[kdx];
        REAL4 otherVel = sharedVel[kdx];
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

