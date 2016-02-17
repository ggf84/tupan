#ifndef __SNP_CRK_KERNEL_COMMON_H__
#define __SNP_CRK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

#define _3_2 (3/(real_t)(2))


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
static inline void
p2p_snp_crk_kernel_core(auto &ip, auto &jp)
// flop count: 128
{
	auto rx = ip.rx - jp.rx;
	auto ry = ip.ry - jp.ry;
	auto rz = ip.rz - jp.rz;
	auto vx = ip.vx - jp.vx;
	auto vy = ip.vy - jp.vy;
	auto vz = ip.vz - jp.vz;
	auto ax = ip.ax - jp.ax;
	auto ay = ip.ay - jp.ay;
	auto az = ip.az - jp.az;
	auto jx = ip.jx - jp.jx;
	auto jy = ip.jy - jp.jy;
	auto jz = ip.jz - jp.jz;
	auto e2 = ip.e2 + jp.e2;
	auto r2 = rx * rx + ry * ry + rz * rz;
	auto rv = rx * vx + ry * vy + rz * vz;
	auto v2 = vx * vx + vy * vy + vz * vz;
	auto va = vx * ax + vy * ay + vz * az;
	auto ra = rx * ax + ry * ay + rz * az;
	auto rj = rx * jx + ry * jy + rz * jz;

	decltype(r2) inv_r2;
	auto inv_r3 = smoothed_inv_r3_inv_r2(r2, e2, &inv_r2);	// flop count: 5

	auto alpha = rv * inv_r2;
	auto alpha2 = alpha * alpha;
	auto beta = 3 * ((v2 + ra) * inv_r2 + alpha2);
	auto gamma = (3 * va + rj) * inv_r2 + alpha * (beta - 4 * alpha2);

	alpha *= 3;
	gamma *= 3;

	vx -= alpha * rx;
	vy -= alpha * ry;
	vz -= alpha * rz;

	alpha *= 2;
	ax -= alpha * vx;
	ay -= alpha * vy;
	az -= alpha * vz;
	ax -= beta * rx;
	ay -= beta * ry;
	az -= beta * rz;

	alpha *= _3_2;
	beta *= 3;
	jx -= alpha * ax;
	jy -= alpha * ay;
	jz -= alpha * az;
	jx -= beta * vx;
	jy -= beta * vy;
	jz -= beta * vz;
	jx -= gamma * rx;
	jy -= gamma * ry;
	jz -= gamma * rz;

	{	// i-particle
		auto m_r3 = jp.m * inv_r3;
		ip.sx -= m_r3 * ax;
		ip.sy -= m_r3 * ay;
		ip.sz -= m_r3 * az;
		ip.cx -= m_r3 * jx;
		ip.cy -= m_r3 * jy;
		ip.cz -= m_r3 * jz;
	}
	{	// j-particle
		auto m_r3 = ip.m * inv_r3;
		jp.sx += m_r3 * ax;
		jp.sy += m_r3 * ay;
		jp.sz += m_r3 * az;
		jp.cx += m_r3 * jx;
		jp.cy += m_r3 * jy;
		jp.cz += m_r3 * jz;
	}
}
#endif

// ----------------------------------------------------------------------------

#define SNP_CRK_IMPLEMENT_STRUCT(N)												\
	typedef struct concat(snp_crk_data, N) {									\
		concat(real_t, N) sx, sy, sz, cx, cy, cz;								\
		concat(real_t, N) rx, ry, rz, vx, vy, vz, ax, ay, az, jx, jy, jz, e2, m;\
	} concat(Snp_Crk_Data, N);

SNP_CRK_IMPLEMENT_STRUCT(1)
#if SIMD > 1
SNP_CRK_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Snp_Crk_Data1 Snp_Crk_Data;


static inline vec(Snp_Crk_Data)
snp_crk_kernel_core(vec(Snp_Crk_Data) ip, Snp_Crk_Data jp)
// flop count: 115
{
	vec(real_t) rx = ip.rx - jp.rx;
	vec(real_t) ry = ip.ry - jp.ry;
	vec(real_t) rz = ip.rz - jp.rz;
	vec(real_t) vx = ip.vx - jp.vx;
	vec(real_t) vy = ip.vy - jp.vy;
	vec(real_t) vz = ip.vz - jp.vz;
	vec(real_t) ax = ip.ax - jp.ax;
	vec(real_t) ay = ip.ay - jp.ay;
	vec(real_t) az = ip.az - jp.az;
	vec(real_t) jx = ip.jx - jp.jx;
	vec(real_t) jy = ip.jy - jp.jy;
	vec(real_t) jz = ip.jz - jp.jz;
	vec(real_t) e2 = ip.e2 + jp.e2;
	vec(real_t) r2 = rx * rx + ry * ry + rz * rz;
	vec(real_t) rv = rx * vx + ry * vy + rz * vz;
	vec(real_t) v2 = vx * vx + vy * vy + vz * vz;
	vec(real_t) va = vx * ax + vy * ay + vz * az;
	vec(real_t) ra = rx * ax + ry * ay + rz * az;
	vec(real_t) rj = rx * jx + ry * jy + rz * jz;

	vec(real_t) inv_r2;
	vec(real_t) m_r3 = jp.m * smoothed_inv_r3_inv_r2(r2, e2, &inv_r2);	// flop count: 6
	inv_r2 = select((vec(real_t))(0), inv_r2, (r2 > 0));
	m_r3 = select((vec(real_t))(0), m_r3, (r2 > 0));

	vec(real_t) alpha = rv * inv_r2;
	vec(real_t) alpha2 = alpha * alpha;
	vec(real_t) beta = 3 * ((v2 + ra) * inv_r2 + alpha2);
	vec(real_t) gamma = (3 * va + rj) * inv_r2 + alpha * (beta - 4 * alpha2);

	alpha *= 3;
	gamma *= 3;

	vx -= alpha * rx;
	vy -= alpha * ry;
	vz -= alpha * rz;

	alpha *= 2;
	ax -= alpha * vx;
	ay -= alpha * vy;
	az -= alpha * vz;
	ax -= beta * rx;
	ay -= beta * ry;
	az -= beta * rz;

	alpha *= _3_2;
	beta *= 3;
	jx -= alpha * ax;
	jy -= alpha * ay;
	jz -= alpha * az;
	jx -= beta * vx;
	jy -= beta * vy;
	jz -= beta * vz;
	jx -= gamma * rx;
	jy -= gamma * ry;
	jz -= gamma * rz;

	ip.sx -= m_r3 * ax;
	ip.sy -= m_r3 * ay;
	ip.sz -= m_r3 * az;
	ip.cx -= m_r3 * jx;
	ip.cy -= m_r3 * jy;
	ip.cz -= m_r3 * jz;
	return ip;
}


#endif	// __SNP_CRK_KERNEL_COMMON_H__
