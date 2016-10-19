#ifndef __ACC_JRK_KERNEL_COMMON_H__
#define __ACC_JRK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<size_t TILE>
struct Acc_Jrk_Data_SoA {
	real_t m[TILE];
	real_t e2[TILE];
	real_t rx[TILE];
	real_t ry[TILE];
	real_t rz[TILE];
	real_t vx[TILE];
	real_t vy[TILE];
	real_t vz[TILE];
	real_t ax[TILE];
	real_t ay[TILE];
	real_t az[TILE];
	real_t jx[TILE];
	real_t jy[TILE];
	real_t jz[TILE];
};

template<size_t TILE, typename T = Acc_Jrk_Data_SoA<TILE>>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<T> part(ntiles);
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		p.m[kk] = __m[k];
		p.e2[kk] = __e2[k];
		p.rx[kk] = __rdot[(0*NDIM+0)*n + k];
		p.ry[kk] = __rdot[(0*NDIM+1)*n + k];
		p.rz[kk] = __rdot[(0*NDIM+2)*n + k];
		p.vx[kk] = __rdot[(1*NDIM+0)*n + k];
		p.vy[kk] = __rdot[(1*NDIM+1)*n + k];
		p.vz[kk] = __rdot[(1*NDIM+2)*n + k];
	}
	return part;
}

template<size_t TILE, typename PART>
void commit(const uint_t n, const PART& part, real_t __adot[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__adot[(0*NDIM+0)*n + k] = p.ax[kk];
		__adot[(0*NDIM+1)*n + k] = p.ay[kk];
		__adot[(0*NDIM+2)*n + k] = p.az[kk];
		__adot[(1*NDIM+0)*n + k] = p.jx[kk];
		__adot[(1*NDIM+1)*n + k] = p.jy[kk];
		__adot[(1*NDIM+2)*n + k] = p.jz[kk];
	}
}

template<size_t TILE>
struct P2P_acc_jrk_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 56
		decltype(jp.m) rx, ry, rz, vx, vy, vz;
		decltype(jp.m) inv_r3, im_r3, jm_r3;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto rr = ip.e2[i] + jp.e2[j];
				rx[j] = ip.rx[i] - jp.rx[j];
				ry[j] = ip.ry[i] - jp.ry[j];
				rz[j] = ip.rz[i] - jp.rz[j];
				vx[j] = ip.vx[i] - jp.vx[j];
				vy[j] = ip.vy[i] - jp.vy[j];
				vz[j] = ip.vz[i] - jp.vz[j];

				rr += rx[j] * rx[j] + ry[j] * ry[j] + rz[j] * rz[j];

				inv_r3[j] = rsqrt(rr);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto inv_r2 = inv_r3[j] * inv_r3[j];
				inv_r3[j] *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx[j] * vx[j] + ry[j] * vy[j] + rz[j] * vz[j];

				auto q1 = inv_r2 * (s1);
				vx[j] += q1 * rx[j];
				vy[j] += q1 * ry[j];
				vz[j] += q1 * rz[j];

				im_r3[j] = ip.m[i] * inv_r3[j];
				jm_r3[j] = jp.m[j] * inv_r3[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				jp.ax[j] += im_r3[j] * rx[j];
				jp.ay[j] += im_r3[j] * ry[j];
				jp.az[j] += im_r3[j] * rz[j];
				jp.jx[j] += im_r3[j] * vx[j];
				jp.jy[j] += im_r3[j] * vy[j];
				jp.jz[j] += im_r3[j] * vz[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				ip.ax[i] -= jm_r3[j] * rx[j];
				ip.ay[i] -= jm_r3[j] * ry[j];
				ip.az[i] -= jm_r3[j] * rz[j];
				ip.jx[i] -= jm_r3[j] * vx[j];
				ip.jy[i] -= jm_r3[j] * vy[j];
				ip.jz[i] -= jm_r3[j] * vz[j];
			}
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 43
		decltype(p.m) rx, ry, rz, vx, vy, vz;
		decltype(p.m) inv_r3, im_r3;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto rr = p.e2[i] + p.e2[j];
				rx[j] = p.rx[i] - p.rx[j];
				ry[j] = p.ry[i] - p.ry[j];
				rz[j] = p.rz[i] - p.rz[j];
				vx[j] = p.vx[i] - p.vx[j];
				vy[j] = p.vy[i] - p.vy[j];
				vz[j] = p.vz[i] - p.vz[j];

				rr   += rx[j] * rx[j] + ry[j] * ry[j] + rz[j] * rz[j];

				inv_r3[j] = rsqrt(rr);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				inv_r3[j] = (i == j) ? (0):(inv_r3[j]);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto inv_r2 = inv_r3[j] * inv_r3[j];
				inv_r3[j] *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx[j] * vx[j] + ry[j] * vy[j] + rz[j] * vz[j];

				auto q1 = inv_r2 * (s1);
				vx[j] += q1 * rx[j];
				vy[j] += q1 * ry[j];
				vz[j] += q1 * rz[j];

				im_r3[j] = p.m[i] * inv_r3[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				p.ax[j] += im_r3[j] * rx[j];
				p.ay[j] += im_r3[j] * ry[j];
				p.az[j] += im_r3[j] * rz[j];
				p.jx[j] += im_r3[j] * vx[j];
				p.jy[j] += im_r3[j] * vy[j];
				p.jz[j] += im_r3[j] * vz[j];
			}
		}
	}
};
#endif

// ----------------------------------------------------------------------------

#define ACC_JRK_IMPLEMENT_STRUCT(N)				\
	typedef struct concat(acc_jrk_data, N) {	\
		concat(real_t, N) m, e2;				\
		concat(real_t, N) rdot[2][NDIM];		\
		concat(real_t, N) adot[2][NDIM];		\
	} concat(Acc_Jrk_Data, N);

ACC_JRK_IMPLEMENT_STRUCT(1)
#if SIMD > 1
ACC_JRK_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Acc_Jrk_Data1 Acc_Jrk_Data;


static inline vec(Acc_Jrk_Data)
acc_jrk_kernel_core(vec(Acc_Jrk_Data) ip, vec(Acc_Jrk_Data) jp)
// flop count: 43
{
	vec(real_t) rdot[2][NDIM];
	#pragma unroll
	for (uint_t kdot = 0; kdot < 2; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			rdot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	vec(real_t) e2 = ip.e2 + jp.e2;

	vec(real_t) rr = rdot[0][0] * rdot[0][0]
			+ rdot[0][1] * rdot[0][1]
			+ rdot[0][2] * rdot[0][2];
	vec(real_t) rv = rdot[0][0] * rdot[1][0]
			+ rdot[0][1] * rdot[1][1]
			+ rdot[0][2] * rdot[1][2];

	vec(real_t) s1 = rv;

	vec(real_t) inv_r2;
	vec(real_t) m_r3 = jp.m * smoothed_inv_r3_inv_r2(rr, e2, &inv_r2);	// flop count: 6
	inv_r2 = select((vec(real_t))(0), inv_r2, (vec(int_t))(rr > 0));
	m_r3 = select((vec(real_t))(0), m_r3, (vec(int_t))(rr > 0));

	inv_r2 *= -3;

	const vec(real_t) q1 = inv_r2 * (s1);

	#pragma unroll
	for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
		rdot[1][kdim] += q1 * rdot[0][kdim];
	}

	#pragma unroll
	for (uint_t kdot = 0; kdot < 2; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			ip.adot[kdot][kdim] -= m_r3 * rdot[kdot][kdim];
		}
	}
	return ip;
}


#endif	// __ACC_JRK_KERNEL_COMMON_H__
