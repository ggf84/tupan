inline REAL
p2p_phi_kernel_core(REAL phi, REAL4 bi, REAL mi, REAL4 bj, REAL mj)
{
    REAL4 dr;
    dr.x = bi.x - bj.x;                                              // 1 FLOPs
    dr.y = bi.y - bj.y;                                              // 1 FLOPs
    dr.z = bi.z - bj.z;                                              // 1 FLOPs
    dr.w = bi.w + bj.w;                                              // 1 FLOPs
    REAL dr2 = dr.z * dr.z + (dr.y * dr.y + dr.x * dr.x);            // 5 FLOPs
    REAL rinv = rsqrt(dr2 + dr.w);                                   // 3 FLOPs
    phi -= mj * ((dr2 > 0) ? rinv:0);                                // 2 FLOPs
    return phi;
}   // Total flop count: 14

