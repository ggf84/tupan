inline REAL4
p2p_acc_kernel_core(REAL4 acc, REAL4 bi, REAL mi, REAL4 bj, REAL mj)
{
    REAL4 dr;
    dr.x = bi.x - bj.x;                                              // 1 FLOPs
    dr.y = bi.y - bj.y;                                              // 1 FLOPs
    dr.z = bi.z - bj.z;                                              // 1 FLOPs
    dr.w = bi.w + bj.w;                                              // 1 FLOPs
    REAL dr2 = dr.z * dr.z + (dr.y * dr.y + dr.x * dr.x);            // 5 FLOPs
    REAL rinv = rsqrt(dr2 + dr.w);                                   // 3 FLOPs
    REAL r3inv = rinv = ((dr2 > 0) ? rinv:0);
    r3inv *= rinv * rinv;                                            // 2 FLOPs
    acc.w += (mi + mj) * r3inv;                                      // 3 FLOPs
    r3inv *= mj;                                                     // 1 FLOPs
    acc.x -= r3inv * dr.x;                                           // 2 FLOPs
    acc.y -= r3inv * dr.y;                                           // 2 FLOPs
    acc.z -= r3inv * dr.z;                                           // 2 FLOPs
    return acc;
}   // Total flop count: 24

