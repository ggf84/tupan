#ifndef BIOS_KERNEL_H
#define BIOS_KERNEL_H

#include"common.h"
#include"smoothing.h"
#include"universal_kepler_solver.h"


/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * *  BIOS - BInary-based n-bOdy Solver  * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */


inline REAL
get_phi(const REAL m,
        const REAL x,
        const REAL y,
        const REAL z,
        const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r = smoothed_inv_r1(r2, eps2);                          // 3 FLOPs
    return m * inv_r;                                                // 1 FLOPs
}


inline REAL3
get_acc(const REAL m,
        const REAL x,
        const REAL y,
        const REAL z,
        const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r3 = smoothed_inv_r3(r2, eps2);                         // 4 FLOPs
    REAL m_r3 = m * inv_r3;                                          // 1 FLOPs
    REAL3 a;
    a.x = -m_r3 * x;                                                 // 1 FLOPs
    a.y = -m_r3 * y;                                                 // 1 FLOPs
    a.z = -m_r3 * z;                                                 // 1 FLOPs
    return a;
}


inline void
leapfrog(const REAL dt,
         const REAL4 r0,
         const REAL4 v0,
         REAL4 *r1,
         REAL4 *v1)
{
    REAL4 r = r0;
    REAL4 v = v0;
    REAL dt_2 = dt / 2;

    r.x += v.x * dt_2;
    r.y += v.y * dt_2;
    r.z += v.z * dt_2;

    REAL3 a = get_acc(r.w, r.x, r.y, r.z, v.w);

    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;

    r.x += v.x * dt_2;
    r.y += v.y * dt_2;
    r.z += v.z * dt_2;

    *r1 = r;
    *v1 = v;
}


inline void
TTL_core(const REAL h,
         const REAL u0,
         const REAL k0,
         REAL4 *r0,
         REAL4 *v0,
         REAL *t)
{
    REAL4 r = *r0;
    REAL4 v = *v0;

    REAL dt0 = h / (2*u0);
    *t += dt0;
    r.x += v.x * dt0;
    r.y += v.y * dt0;
    r.z += v.z * dt0;

    REAL3 a12 = get_acc(r.w, r.x, r.y, r.z, v.w);
    REAL u12 = get_phi(r.w, r.x, r.y, r.z, v.w);
    REAL dt12 = h / u12;

    v.x += a12.x * dt12;
    v.y += a12.y * dt12;
    v.z += a12.z * dt12;

    REAL k1 = (v.x * v.x + v.y * v.y + v.z * v.z)/2;
    REAL u1 = u0 + (k1 - k0);

    REAL dt1 = h / (2*u1);
    *t += dt1;
    r.x += v.x * dt1;
    r.y += v.y * dt1;
    r.z += v.z * dt1;

    *r0 = r;
    *v0 = v;
}

#ifdef DOUBLE
    #define TOLERANCE 2.3283064365386962891E-10     // sqrt(2^-64)
#else
    #define TOLERANCE 1.52587890625E-5              // sqrt(2^-32)
#endif
inline void
TTL(const REAL dt0,
    const REAL4 r0,
    const REAL4 v0,
    REAL4 *r1,
    REAL4 *v1)
{
    int i;
    int err = 0;
    int nsteps = 1;
    REAL t = 0;
    REAL4 r = r0;
    REAL4 v = v0;

    do {
        err = 0;
        REAL dt = dt0 / nsteps;
        for (i = 0; i < nsteps; ++i) {

            REAL mij = r.w;
            REAL r2 = (r.x * r.x + r.y * r.y + r.z * r.z);
            REAL rv = (r.x * v.x + r.y * v.y + r.z * v.z);
            REAL v2 = (v.x * v.x + v.y * v.y + v.z * v.z);

            REAL3 ret = smoothed_inv_r1r2r3(r2, v.w);
            REAL inv_r = ret.x;
            REAL inv_r2 = ret.y;
            REAL inv_r3 = ret.z;

            REAL3 a;
            REAL mij_r3 = -mij * inv_r3;
            a.x = mij_r3 * r.x;
            a.y = mij_r3 * r.y;
            a.z = mij_r3 * r.z;
            REAL a2 = (a.x * a.x + a.y * a.y + a.z * a.z);

            REAL k0 = v2/2;
            REAL u0 = mij * inv_r;
            REAL u0dot = (v.x * a.x + v.y * a.y + v.z * a.z);
            REAL u0ddot = a2 - mij * inv_r3 * (v2 + - 3 * (rv * rv) * inv_r2);

            REAL h = (((((0) + u0ddot) * dt / 3) + u0dot) * dt / 2 + u0) * dt;

            TTL_core(h, u0, k0, &r, &v, &t);

            REAL dts = (i+1) * dt;
            if (fabs(fabs(t) - dts)/dts > TOLERANCE) {
                err = -1;
                t = 0;
                r = r0;
                v = v0;
                nsteps += (nsteps+1)/2;
                i = nsteps;
            }
        }
    } while (err != 0);

    *r1 = r;
    *v1 = v;
}


inline void
twobody_solver(const REAL dt,
               const REAL4 r0,
               const REAL4 v0,
               REAL4 *r1,
               REAL4 *v1)
{

//    leapfrog(dt, r0, v0, r1, v1);
//    TTL(dt, r0, v0, r1, v1);
    universal_kepler_solver(dt, r0, v0, r1, v1);
}


//
// bios_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL8
bios_kernel_core(REAL8 iposvel,
                 const REAL4 ri, const REAL4 vi,
                 const REAL4 rj, const REAL4 vj,
                 const REAL Mtot, const REAL dt)
{
    REAL4 r0;
    r0.x = ri.x - rj.x;                                              // 1 FLOPs
    r0.y = ri.y - rj.y;                                              // 1 FLOPs
    r0.z = ri.z - rj.z;                                              // 1 FLOPs
    r0.w = ri.w + rj.w;                                              // 1 FLOPs
    REAL4 v0;
    v0.x = vi.x - vj.x;                                              // 1 FLOPs
    v0.y = vi.y - vj.y;                                              // 1 FLOPs
    v0.z = vi.z - vj.z;                                              // 1 FLOPs
    v0.w = vi.w + vj.w;                                              // 1 FLOPs

    REAL r2 = r0.x * r0.x + r0.y * r0.y + r0.z * r0.z;

    if (r2 == 0) {
        return iposvel;
    }

    REAL4 r1, v1;
//    twobody_solver(dt, r0, v0, &r1, &v1);                            // ? FLOPS

    REAL inv_r3 = smoothed_inv_r3(r2, v0.w);                           // 4 FLOPs
    REAL gamma = dt * dt * r0.w * inv_r3;

//    if (4 * r0.w < r2 * sqrt(r2)) {
    if (4096 * gamma < 1) {
        leapfrog(dt, r0, v0, &r1, &v1);
    } else {
        universal_kepler_solver(dt, r0, v0, &r1, &v1);
    }

    REAL mimj = (ri.w * rj.w);                                       // 1 FLOPs
    REAL mu = mimj / r0.w;                                           // 1 FLOPs

    REAL Mmij = -(r0.w - Mtot) / Mtot;                               // 2 FLOPs
    REAL mdt = Mmij * dt;                                            // 1 FLOPs

    r0.x += v0.x * mdt;                                              // 2 FLOPs
    r0.y += v0.y * mdt;                                              // 2 FLOPs
    r0.z += v0.z * mdt;                                              // 2 FLOPs

    iposvel.s0 += mu * (r1.x - r0.x);                                // 3 FLOPs
    iposvel.s1 += mu * (r1.y - r0.y);                                // 3 FLOPs
    iposvel.s2 += mu * (r1.z - r0.z);                                // 3 FLOPs
    iposvel.s3  = 0;
    iposvel.s4 += mu * (v1.x - v0.x);                                // 3 FLOPs
    iposvel.s5 += mu * (v1.y - v0.y);                                // 3 FLOPs
    iposvel.s6 += mu * (v1.z - v0.z);                                // 3 FLOPs
    iposvel.s7  = 0;

    return iposvel;
}
// Total flop count: 36 + ?




#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL8
bios_kernel_accum(REAL8 myPosVel,
                  const REAL8 myData,
                  const REAL Mtot,
                  const REAL dt,
                  uint j_begin,
                  uint j_end,
                  __local REAL8 *sharedJData
                 )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        myPosVel = bios_kernel_core(myPosVel, myData.lo, myData.hi,
                                    sharedJData[j].lo, sharedJData[j].hi,
                                    Mtot, dt);
    }
    return myPosVel;
}


inline REAL8
bios_kernel_main_loop(const REAL8 myData,
                      const uint nj,
                      __global const REAL8 *jdata,
                      const REAL Mtot,
                      const REAL dt,
                      __local REAL8 *sharedJData
                     )
{
    uint lsize = get_local_size(0);

    REAL8 myPosVel = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[1];
        e[0] = async_work_group_copy(sharedJData, jdata + tile * lsize, nb, 0);
        wait_group_events(1, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            myPosVel = bios_kernel_accum(myPosVel, myData,
                                         Mtot, dt, j, j + JUNROLL,
                                         sharedJData);
        }
        myPosVel = bios_kernel_accum(myPosVel, myData,
                                     Mtot, dt, j, nb,
                                     sharedJData);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    myPosVel.s0 /= myData.s3;
    myPosVel.s1 /= myData.s3;
    myPosVel.s2 /= myData.s3;
    myPosVel.s3 /= myData.s3;
    myPosVel.s4 /= myData.s3;
    myPosVel.s5 /= myData.s3;
    myPosVel.s6 /= myData.s3;
    myPosVel.s7 /= myData.s3;

    return myPosVel;
}


__kernel void bios_kernel(const uint ni,
                          __global const REAL8 *idata,
                          const uint nj,
                          __global const REAL8 *jdata,
                          const REAL Mtot,
                          const REAL dt,
                          __global REAL8 *iposvel,
                          __local REAL8 *sharedJData
                         )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iposvel[i] = bios_kernel_main_loop(idata[i],
                                       nj, jdata,
                                       Mtot, dt,
                                       sharedJData);
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_bios_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    REAL Mtot, dt;
    PyObject *_idata = NULL;
    PyObject *_jdata = NULL;
    PyObject *_output = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOIOddO!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOIOffO!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_idata,
                                      &nj, &_jdata,
                                      &Mtot,
                                      &dt,
                                      &PyArray_Type, &_output))
        return NULL;

    // i-data
    PyObject *_idata_arr = PyArray_FROM_OTF(_idata, typenum, NPY_IN_ARRAY);
    REAL *idata_ptr = (REAL *)PyArray_DATA(_idata_arr);

    // j-data
    PyObject *_jdata_arr = PyArray_FROM_OTF(_jdata, typenum, NPY_IN_ARRAY);
    REAL *jdata_ptr = (REAL *)PyArray_DATA(_jdata_arr);

    // output-array
    PyObject *ret = PyArray_FROM_OTF(_output, typenum, NPY_INOUT_ARRAY);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, i8, j, j8;
    for (i = 0; i < ni; ++i) {
        i8 = 8*i;
        REAL8 iposvel = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
        REAL4 ri = {idata_ptr[i8  ], idata_ptr[i8+1],
                    idata_ptr[i8+2], idata_ptr[i8+3]};
        REAL4 vi = {idata_ptr[i8+4], idata_ptr[i8+5],
                    idata_ptr[i8+6], idata_ptr[i8+7]};
        for (j = 0; j < nj; ++j) {
            j8 = 8*j;
            REAL4 rj = {jdata_ptr[j8  ], jdata_ptr[j8+1],
                        jdata_ptr[j8+2], jdata_ptr[j8+3]};
            REAL4 vj = {jdata_ptr[j8+4], jdata_ptr[j8+5],
                        jdata_ptr[j8+6], jdata_ptr[j8+7]};
            iposvel = bios_kernel_core(iposvel, ri, vi, rj, vj, Mtot, dt);
        }
        ret_ptr[i8  ] = iposvel.s0 / ri.w;
        ret_ptr[i8+1] = iposvel.s1 / ri.w;
        ret_ptr[i8+2] = iposvel.s2 / ri.w;
        ret_ptr[i8+3] = iposvel.s3 / ri.w;
        ret_ptr[i8+4] = iposvel.s4 / ri.w;
        ret_ptr[i8+5] = iposvel.s5 / ri.w;
        ret_ptr[i8+6] = iposvel.s6 / ri.w;
        ret_ptr[i8+7] = iposvel.s7 / ri.w;
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(_idata_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(_jdata_arr);

    // Decrement the reference counts for ret-objects
    Py_DECREF(ret);

    Py_INCREF(Py_None);
    return Py_None;
}

#endif  // __OPENCL_VERSION__


#endif  // BIOS_KERNEL_H

