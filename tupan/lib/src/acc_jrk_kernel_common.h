#ifndef __ACC_JRK_KERNEL_COMMON_H__
#define __ACC_JRK_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Acc_Jrk {

template<size_t TILE>
struct Acc_Jrk_Data_SoA {
	real_t __ALIGNED__ m[TILE];
	real_t __ALIGNED__ e2[TILE];
	real_t __ALIGNED__ rx[TILE];
	real_t __ALIGNED__ ry[TILE];
	real_t __ALIGNED__ rz[TILE];
	real_t __ALIGNED__ vx[TILE];
	real_t __ALIGNED__ vy[TILE];
	real_t __ALIGNED__ vz[TILE];
	real_t __ALIGNED__ ax[TILE];
	real_t __ALIGNED__ ay[TILE];
	real_t __ALIGNED__ az[TILE];
	real_t __ALIGNED__ jx[TILE];
	real_t __ALIGNED__ jy[TILE];
	real_t __ALIGNED__ jz[TILE];
};

template<size_t TILE>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<Acc_Jrk_Data_SoA<TILE>> part(ntiles);
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
		for (size_t i = 0; i < TILE; ++i) {
			auto iax = ip.ax[i];
			auto iay = ip.ay[i];
			auto iaz = ip.az[i];
			auto ijx = ip.jx[i];
			auto ijy = ip.jy[i];
			auto ijz = ip.jz[i];
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];
				auto vx = ip.vx[i] - jp.vx[j];
				auto vy = ip.vy[i] - jp.vy[j];
				auto vz = ip.vz[i] - jp.vz[j];

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
				auto inv_r2 = inv_r3 * inv_r3;
				inv_r3 *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx * vx + ry * vy + rz * vz;

				auto q1 = inv_r2 * (s1);
				vx += q1 * rx;
				vy += q1 * ry;
				vz += q1 * rz;

				auto im_r3 = ip.m[i] * inv_r3;
				jp.ax[j] += im_r3 * rx;
				jp.ay[j] += im_r3 * ry;
				jp.az[j] += im_r3 * rz;
				jp.jx[j] += im_r3 * vx;
				jp.jy[j] += im_r3 * vy;
				jp.jz[j] += im_r3 * vz;

				auto jm_r3 = jp.m[j] * inv_r3;
				iax -= jm_r3 * rx;
				iay -= jm_r3 * ry;
				iaz -= jm_r3 * rz;
				ijx -= jm_r3 * vx;
				ijy -= jm_r3 * vy;
				ijz -= jm_r3 * vz;
			}
			ip.ax[i] = iax;
			ip.ay[i] = iay;
			ip.az[i] = iaz;
			ip.jx[i] = ijx;
			ip.jy[i] = ijy;
			ip.jz[i] = ijz;
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 43
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = p.e2[i] + p.e2[j];
				auto rx = p.rx[i] - p.rx[j];
				auto ry = p.ry[i] - p.ry[j];
				auto rz = p.rz[i] - p.rz[j];
				auto vx = p.vx[i] - p.vx[j];
				auto vy = p.vy[i] - p.vy[j];
				auto vz = p.vz[i] - p.vz[j];

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
				inv_r3 = (rr > ee) ? (inv_r3):(0);
				auto inv_r2 = inv_r3 * inv_r3;
				inv_r3 *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx * vx + ry * vy + rz * vz;

				auto q1 = inv_r2 * (s1);
				vx += q1 * rx;
				vy += q1 * ry;
				vz += q1 * rz;

				auto im_r3 = p.m[i] * inv_r3;

				p.ax[j] += im_r3 * rx;
				p.ay[j] += im_r3 * ry;
				p.az[j] += im_r3 * rz;
				p.jx[j] += im_r3 * vx;
				p.jy[j] += im_r3 * vy;
				p.jz[j] += im_r3 * vz;
			}
		}
	}
};

}	// namespace Acc_Jrk
#else


// ----------------------------------------------------------------------------


typedef struct acc_jrk_data {
	union {
		real_tn m;
		real_t _m[SIMD];
	};
	union {
		real_tn e2;
		real_t _e2[SIMD];
	};
	union {
		real_tn rx;
		real_t _rx[SIMD];
	};
	union {
		real_tn ry;
		real_t _ry[SIMD];
	};
	union {
		real_tn rz;
		real_t _rz[SIMD];
	};
	union {
		real_tn vx;
		real_t _vx[SIMD];
	};
	union {
		real_tn vy;
		real_t _vy[SIMD];
	};
	union {
		real_tn vz;
		real_t _vz[SIMD];
	};
	union {
		real_tn ax;
		real_t _ax[SIMD];
	};
	union {
		real_tn ay;
		real_t _ay[SIMD];
	};
	union {
		real_tn az;
		real_t _az[SIMD];
	};
	union {
		real_tn jx;
		real_t _jx[SIMD];
	};
	union {
		real_tn jy;
		real_t _jy[SIMD];
	};
	union {
		real_tn jz;
		real_t _jz[SIMD];
	};
} Acc_Jrk_Data;


typedef struct acc_jrk_data_soa {
	union {
		real_tn m[NLANES];
		real_t _m[NLANES * SIMD];
	};
	union {
		real_tn e2[NLANES];
		real_t _e2[NLANES * SIMD];
	};
	union {
		real_tn rx[NLANES];
		real_t _rx[NLANES * SIMD];
	};
	union {
		real_tn ry[NLANES];
		real_t _ry[NLANES * SIMD];
	};
	union {
		real_tn rz[NLANES];
		real_t _rz[NLANES * SIMD];
	};
	union {
		real_tn vx[NLANES];
		real_t _vx[NLANES * SIMD];
	};
	union {
		real_tn vy[NLANES];
		real_t _vy[NLANES * SIMD];
	};
	union {
		real_tn vz[NLANES];
		real_t _vz[NLANES * SIMD];
	};
	union {
		real_tn ax[NLANES];
		real_t _ax[NLANES * SIMD];
	};
	union {
		real_tn ay[NLANES];
		real_t _ay[NLANES * SIMD];
	};
	union {
		real_tn az[NLANES];
		real_t _az[NLANES * SIMD];
	};
	union {
		real_tn jx[NLANES];
		real_t _jx[NLANES * SIMD];
	};
	union {
		real_tn jy[NLANES];
		real_t _jy[NLANES * SIMD];
	};
	union {
		real_tn jz[NLANES];
		real_t _jz[NLANES * SIMD];
	};
} Acc_Jrk_Data_SoA;


static inline void
read_Acc_Jrk_Data(
	uint_t base,
	uint_t lid,
	Acc_Jrk_Data *p,
	uint_t n,
	global const real_t __m[],
	global const real_t __e2[],
	global const real_t __rdot[])
{
	for (uint_t k = 0, kk = base + lid;
				k < SIMD;
				k += 1, kk += WGSIZE) {
		if (kk < n) {
			p->_m[k] = __m[kk];
			p->_e2[k] = __e2[kk];
			p->_rx[k] = (__rdot+(0*NDIM+0)*n)[kk];
			p->_ry[k] = (__rdot+(0*NDIM+1)*n)[kk];
			p->_rz[k] = (__rdot+(0*NDIM+2)*n)[kk];
			p->_vx[k] = (__rdot+(1*NDIM+0)*n)[kk];
			p->_vy[k] = (__rdot+(1*NDIM+1)*n)[kk];
			p->_vz[k] = (__rdot+(1*NDIM+2)*n)[kk];
		}
	}
}


static inline void
simd_shuff_Acc_Jrk_Data(Acc_Jrk_Data *p)
{
	shuff(p->m, SIMD);
	shuff(p->e2, SIMD);
	shuff(p->rx, SIMD);
	shuff(p->ry, SIMD);
	shuff(p->rz, SIMD);
	shuff(p->vx, SIMD);
	shuff(p->vy, SIMD);
	shuff(p->vz, SIMD);
	shuff(p->ax, SIMD);
	shuff(p->ay, SIMD);
	shuff(p->az, SIMD);
	shuff(p->jx, SIMD);
	shuff(p->jy, SIMD);
	shuff(p->jz, SIMD);
}


#endif	// __cplusplus
#endif	// __ACC_JRK_KERNEL_COMMON_H__
