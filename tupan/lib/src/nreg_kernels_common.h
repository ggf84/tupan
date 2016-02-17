#ifndef __NREG_KERNELS_COMMON_H__
#define __NREG_KERNELS_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
static inline void
p2p_nreg_Xkernel_core(auto &ip, auto &jp, const auto dt)
// flop count: 53
{
	auto rx = ip.rx - jp.rx;
	auto ry = ip.ry - jp.ry;
	auto rz = ip.rz - jp.rz;
	auto vx = ip.vx - jp.vx;
	auto vy = ip.vy - jp.vy;
	auto vz = ip.vz - jp.vz;
	auto e2 = ip.e2 + jp.e2;

	rx += vx * dt;
	ry += vy * dt;
	rz += vz * dt;

	auto r2 = rx * rx + ry * ry + rz * rz;

	decltype(r2) inv_r1;
	auto inv_r3 = smoothed_inv_r3_inv_r1(r2, e2, &inv_r1);	// flop count: 5

	{	// i-particle
		auto m_r3 = jp.m * inv_r3;
		ip.drx += jp.m * rx;
		ip.dry += jp.m * ry;
		ip.drz += jp.m * rz;
		ip.ax -= m_r3 * rx;
		ip.ay -= m_r3 * ry;
		ip.az -= m_r3 * rz;
		ip.u += jp.m * inv_r1;
	}
	{	// j-particle
		auto m_r3 = ip.m * inv_r3;
		jp.drx -= ip.m * rx;
		jp.dry -= ip.m * ry;
		jp.drz -= ip.m * rz;
		jp.ax += m_r3 * rx;
		jp.ay += m_r3 * ry;
		jp.az += m_r3 * rz;
		jp.u += ip.m * inv_r1;
	}
}


static inline void
p2p_nreg_Vkernel_core(auto &ip, auto &jp, const auto dt)
// flop count: 33
{
	auto vx = ip.vx - jp.vx;
	auto vy = ip.vy - jp.vy;
	auto vz = ip.vz - jp.vz;
	auto ax = ip.ax - jp.ax;
	auto ay = ip.ay - jp.ay;
	auto az = ip.az - jp.az;

	vx += ax * dt;
	vy += ay * dt;
	vz += az * dt;

	auto v2 = vx * vx + vy * vy + vz * vz;

	ip.k += jp.m * v2;
	ip.dvx += jp.m * vx;
	ip.dvy += jp.m * vy;
	ip.dvz += jp.m * vz;

	jp.k += ip.m * v2;
	jp.dvx -= ip.m * vx;
	jp.dvy -= ip.m * vy;
	jp.dvz -= ip.m * vz;
}
#endif

// ----------------------------------------------------------------------------

#define NREG_X_IMPLEMENT_STRUCT(N)							\
	typedef struct concat(nreg_x_data, N) {					\
		concat(real_t, N) drx, dry, drz, ax, ay, az, u;		\
		concat(real_t, N) rx, ry, rz, vx, vy, vz, e2, m;	\
	} concat(Nreg_X_Data, N);

NREG_X_IMPLEMENT_STRUCT(1)
#if SIMD > 1
NREG_X_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Nreg_X_Data1 Nreg_X_Data;


static inline vec(Nreg_X_Data)
nreg_Xkernel_core(vec(Nreg_X_Data) ip, Nreg_X_Data jp, const real_t dt)
// flop count: 37
{
	vec(real_t) rx = ip.rx - jp.rx;
	vec(real_t) ry = ip.ry - jp.ry;
	vec(real_t) rz = ip.rz - jp.rz;
	vec(real_t) vx = ip.vx - jp.vx;
	vec(real_t) vy = ip.vy - jp.vy;
	vec(real_t) vz = ip.vz - jp.vz;
	vec(real_t) e2 = ip.e2 + jp.e2;

	rx += vx * dt;
	ry += vy * dt;
	rz += vz * dt;

	vec(real_t) r2 = rx * rx + ry * ry + rz * rz;

	vec(real_t) inv_r1;
	vec(real_t) m_r3 = jp.m * smoothed_inv_r3_inv_r1(r2, e2, &inv_r1);	// flop count: 6
	inv_r1 = select((vec(real_t))(0), inv_r1, (r2 > 0));
	m_r3 = select((vec(real_t))(0), m_r3, (r2 > 0));

	ip.drx += jp.m * rx;
	ip.dry += jp.m * ry;
	ip.drz += jp.m * rz;
	ip.u += jp.m * inv_r1;
	ip.ax -= m_r3 * rx;
	ip.ay -= m_r3 * ry;
	ip.az -= m_r3 * rz;
	return ip;
}


#define NREG_V_IMPLEMENT_STRUCT(N)						\
	typedef struct concat(nreg_v_data, N) {				\
		concat(real_t, N) dvx, dvy, dvz, k;				\
		concat(real_t, N) vx, vy, vz, ax, ay, az, m;	\
	} concat(Nreg_V_Data, N);

NREG_V_IMPLEMENT_STRUCT(1)
#if SIMD > 1
NREG_V_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Nreg_V_Data1 Nreg_V_Data;


static inline vec(Nreg_V_Data)
nreg_Vkernel_core(vec(Nreg_V_Data) ip, Nreg_V_Data jp, const real_t dt)
// flop count: 25
{
	vec(real_t) vx = ip.vx - jp.vx;
	vec(real_t) vy = ip.vy - jp.vy;
	vec(real_t) vz = ip.vz - jp.vz;
	vec(real_t) ax = ip.ax - jp.ax;
	vec(real_t) ay = ip.ay - jp.ay;
	vec(real_t) az = ip.az - jp.az;

	vx += ax * dt;
	vy += ay * dt;
	vz += az * dt;

	vec(real_t) v2 = vx * vx + vy * vy + vz * vz;

	ip.k += jp.m * v2;
	ip.dvx += jp.m * vx;
	ip.dvy += jp.m * vy;
	ip.dvz += jp.m * vz;
	return ip;
}


#endif	// __NREG_KERNELS_COMMON_H__
