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


REAL4 p2p_acc_kernel_core(REAL4 acc, REAL4 bi, REAL4 bj, REAL mj)
{
    REAL4 dr;
    dr.x = bi.x - bj.x;                                              // 1 FLOPs
    dr.y = bi.y - bj.y;                                              // 1 FLOPs
    dr.z = bi.z - bj.z;                                              // 1 FLOPs
    dr.w = bi.w + bj.w;                                              // 1 FLOPs
    REAL ds2 = dr.w;
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
}   // Total flop count: 22


__kernel void p2p_acc_kernel_gpugems3(const uint ni,
                                      const uint nj,
                                      __global const REAL4 *ipos,
                                      __global const REAL4 *jpos,
                                      __global const REAL *jmass,
                                      __global REAL4 *iacc,
                                      __local REAL4 *sharedPos,
                                      __local REAL *sharedMass)
{
    uint gid = get_global_id(0) * IUNROLL;
    uint tidx = get_local_id(0);
    uint tidy = get_local_id(1);
    uint grpidx = get_group_id(0) * IUNROLL;
    uint grpidy = get_group_id(1) * IUNROLL;
    uint grpDimx = get_num_groups(0) * IUNROLL;
    uint localDimx = get_local_size(0);
    uint localDimy = get_local_size(1);

    REAL4 myPos[IUNROLL];
    REAL4 myAcc[IUNROLL];
    for (uint ii = 0; ii < IUNROLL; ++ii) {
        myPos[ii] = (gid + ii < ni) ? ipos[gid + ii] : ipos[ni-1];
        myAcc[ii] = (REAL4){0.0, 0.0, 0.0, 0.0};
    }

    uint tile;
    uint numTiles = ((nj + localDimx*localDimy - 1) / (localDimx*localDimy));
    for(tile = grpidy; tile < numTiles-1 + grpidy; ++tile) {

        uint sdx = tidx + localDimx * tidy;
        uint jdx = WRAP(grpidx + tile, grpDimx) * localDimx + tidx;
        sharedPos[sdx] = (jdx < nj) ? jpos[jdx] : (REAL4){0.0, 0.0, 0.0, 0.0};
        sharedMass[sdx] = (jdx < nj) ? jmass[jdx] : (REAL)0.0;

        barrier(CLK_LOCAL_MEM_FENCE);
        ulong k = 0;
        for(uint j = 0; j < localDimx; ) {
            for (uint jj = 0; jj < JUNROLL; ++jj) {
                uint kdx = localDimx * tidy + k++;
                REAL4 otherPos = sharedPos[kdx];
                REAL otherMass = sharedMass[kdx];
                for (uint ii = 0; ii < IUNROLL; ++ii) {
                    myAcc[ii] = p2p_acc_kernel_core(myAcc[ii],
                                                    myPos[ii],
                                                    otherPos,
                                                    otherMass);
                }
            }
            j += JUNROLL;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint sdx = tidx + localDimx * tidy;
    uint jdx = WRAP(grpidx + tile, grpDimx) * localDimx + tidx;
    sharedPos[sdx] = (jdx < nj) ? jpos[jdx] : (REAL4){0.0, 0.0, 0.0, 0.0};
    sharedMass[sdx] = (jdx < nj) ? jmass[jdx] : (REAL)0.0;

    barrier(CLK_LOCAL_MEM_FENCE);
    ulong k = 0;
    for(uint j = 0; j < (numTiles * localDimy - tile) * localDimx; ++j) {
        uint kdx = localDimx * tidy + k++;
        REAL4 otherPos = sharedPos[kdx];
        REAL otherMass = sharedMass[kdx];
        for (uint ii = 0; ii < IUNROLL; ++ii) {
            myAcc[ii] = p2p_acc_kernel_core(myAcc[ii],
                                            myPos[ii],
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

