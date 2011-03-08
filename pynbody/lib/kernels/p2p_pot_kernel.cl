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

/*
XXX: dr.w foi transferido para o argumento da sqrt p/ evitar somar a contribuicao da propria particula quando dr.w != 0.0.
XXX: verificar perda de performance e fazer reajustes necessarios.
*/
REAL p2p_pot_kernel_core(REAL pot, REAL4 bi, REAL4 bj, REAL mj)
{
    REAL4 dr;
    dr.x = bi.x - bj.x;                                              // 1 FLOPs
    dr.y = bi.y - bj.y;                                              // 1 FLOPs
    dr.z = bi.z - bj.z;                                              // 1 FLOPs
    dr.w = bi.w + bj.w;                                              // 1 FLOPs
    REAL ds2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;              // 5 FLOPs
    REAL rinv = rsqrt(ds2 + dr.w);                                   // 3 FLOPs
    pot -= mj * (ds2 ? rinv:0);                                      // 2 FLOPs
    return pot;
}   // Total flop count: 14


__kernel void p2p_pot_kernel(const uint ni,
                             const uint nj,
                             __global const REAL4 *ipos,
                             __global const REAL4 *jpos,
                             __global const REAL *jmass,
                             __global REAL *ipot,
                             __local REAL4 *sharedPos,
                             __local REAL *sharedMass)
{
    uint tid = get_local_id(0);
    uint gid = get_global_id(0) * IUNROLL;
    uint localDim = get_local_size(0);

    REAL4 myPos[IUNROLL];
    REAL myPot[IUNROLL];
    for (uint ii = 0; ii < IUNROLL; ++ii) {
        myPos[ii] = (gid + ii < ni) ? ipos[gid + ii] : ipos[ni-1];
        myPot[ii] = 0.0;
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
                   myPot[ii] = p2p_pot_kernel_core(myPot[ii],
                                                   myPos[ii],
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
            myPot[ii] = p2p_pot_kernel_core(myPot[ii],
                                            myPos[ii],
                                            otherPos,
                                            otherMass);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint ii = 0; ii < IUNROLL; ++ii) {
        if (gid + ii < ni) {
            ipot[gid + ii] = myPot[ii];
        }
    }
}   // Output shape: ({ni},)

