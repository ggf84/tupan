#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#define TSTEP_IMPLEMENT_STRUCT(N)							\
	typedef struct concat(tstep_data, N) {					\
		concat(real_t, N) w2_a, w2_b;						\
		concat(real_t, N) rx, ry, rz, vx, vy, vz, e2, m;	\
	} concat(Tstep_Data, N);

TSTEP_IMPLEMENT_STRUCT(1)
#if SIMD > 1
TSTEP_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Tstep_Data1 Tstep_Data;


static inline vec(Tstep_Data)
tstep_kernel_core(vec(Tstep_Data) ip, Tstep_Data jp, const real_t eta)
// flop count: 44
{
	vec(real_t) rx = ip.rx - jp.rx;
	vec(real_t) ry = ip.ry - jp.ry;
	vec(real_t) rz = ip.rz - jp.rz;
	vec(real_t) vx = ip.vx - jp.vx;
	vec(real_t) vy = ip.vy - jp.vy;
	vec(real_t) vz = ip.vz - jp.vz;
	vec(real_t) e2 = ip.e2 + jp.e2;
	vec(real_t) m = ip.m + jp.m;
	vec(real_t) r2 = rx * rx + ry * ry + rz * rz;
	vec(real_t) rv = rx * vx + ry * vy + rz * vz;
	vec(real_t) v2 = vx * vx + vy * vy + vz * vz;

	vec(real_t) inv_r1;
	vec(real_t) inv_r2 = smoothed_inv_r2_inv_r1(r2, e2, &inv_r1);	// flop count: 4

	vec(real_t) m_r1 = m * inv_r1;

	vec(real_t) a = (vec(real_t))(2);
	vec(real_t) b = (1 + a / 2) * inv_r2;

	vec(real_t) w2 = (v2 + a * m_r1) * inv_r2;
	vec(real_t) gamma = (w2 + b * m_r1) * inv_r2;
	gamma *= (eta * rsqrt(w2));
	w2 -= gamma * rv;

	w2 = select((vec(real_t))(0), w2, (r2 > 0));

	ip.w2_a += w2;
	ip.w2_b = fmax(w2, ip.w2_b);
	return ip;
}


#endif	// __TSTEP_KERNEL_COMMON_H__
