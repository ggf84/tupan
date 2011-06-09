inline REAL4
p2p_acc_kernel_core(REAL4 acc, REAL4 bip, REAL4 biv, REAL4 bjp, REAL4 bjv)
{
    REAL4 dr, dv;
    dr.x = bip.x - bjp.x;                                            // 1 FLOPs
    dr.y = bip.y - bjp.y;                                            // 1 FLOPs
    dr.z = bip.z - bjp.z;                                            // 1 FLOPs
    dr.w = bip.w + bjp.w;                                            // 1 FLOPs
    dv.x = biv.x - bjv.x;                                            // 1 FLOPs
    dv.y = biv.y - bjv.y;                                            // 1 FLOPs
    dv.z = biv.z - bjv.z;                                            // 1 FLOPs
    dv.w = biv.w + bjv.w;                                            // 1 FLOPs
    REAL dr2 = dr.z * dr.z + (dr.y * dr.y + dr.x * dr.x);            // 5 FLOPs
    REAL dv2 = dv.z * dv.z + (dv.y * dv.y + dv.x * dv.x);            // 5 FLOPs
    REAL rinv = rsqrt(dr2 + dr.w);                                   // 3 FLOPs
    REAL r3inv = rinv = ((dr2 > 0) ? rinv:0);
    r3inv *= rinv;                                                   // 1 FLOPs

    REAL e = 0.0 * dv2;                                              // 1 FLOPs
    e += dv.w * rinv;                                                // 2 FLOPs
    acc.w += e * r3inv;                                              // 2 FLOPs

//    REAL e = 2.0 * rinv;                                             // 1 FLOPs
//    e -= dv2 / dv.w;                                                 // 2 FLOPs
//    acc.w += dv.w * e * e * e;                                       // 2 FLOPs

    r3inv *= bjv.w * rinv;                                           // 2 FLOPs
    acc.x -= r3inv * dr.x;                                           // 2 FLOPs
    acc.y -= r3inv * dr.y;                                           // 2 FLOPs
    acc.z -= r3inv * dr.z;                                           // 2 FLOPs
    return acc;
}   // Total flop count: 35

