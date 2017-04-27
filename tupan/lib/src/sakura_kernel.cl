#include "sakura_kernel_common.h"


static inline void
p2p_sakura_kernel_core(
	const real_t dt,
	const int_t flag,
	uint_t lane,
	Sakura_Data *jp,
	local Sakura_Data_SoA *ip)
// flop count: 41 + ??
{
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
		for (uint_t k = 0; k < 1; ++k) {
			real_t m = ip->m[i] + jp->m;
			real_t e2 = ip->e2[i] + jp->e2;
			real_t r0x = ip->rx[i] - jp->rx;
			real_t r0y = ip->ry[i] - jp->ry;
			real_t r0z = ip->rz[i] - jp->rz;
			real_t v0x = ip->vx[i] - jp->vx;
			real_t v0y = ip->vy[i] - jp->vy;
			real_t v0z = ip->vz[i] - jp->vz;

			real_t r1x = r0x;
			real_t r1y = r0y;
			real_t r1z = r0z;
			real_t v1x = v0x;
			real_t v1y = v0y;
			real_t v1z = v0z;
			evolve_twobody(
				dt, flag, m, e2,
				r0x, r0y, r0z, v0x, v0y, v0z,
				&r1x, &r1y, &r1z, &v1x, &v1y, &v1z
			);	// flop count: ??

			real_t inv_m = 1 / m;
			real_t drx = r0x - r1x;
			real_t dry = r0y - r1y;
			real_t drz = r0z - r1z;
			real_t dvx = v0x - v1x;
			real_t dvy = v0y - v1y;
			real_t dvz = v0z - v1z;

			real_t imu = ip->m[i] * inv_m;
			jp->drx += imu * drx;
			jp->dry += imu * dry;
			jp->drz += imu * drz;
			jp->dvx += imu * dvx;
			jp->dvy += imu * dvy;
			jp->dvz += imu * dvz;

			real_t jmu = jp->m * inv_m;
			ip->drx[i] += jmu * drx;
			ip->dry[i] += jmu * dry;
			ip->drz[i] += jmu * drz;
			ip->dvx[i] += jmu * dvx;
			ip->dvy[i] += jmu * dvy;
			ip->dvz[i] += jmu * dvz;

			simd_shuff_Sakura_Data(jp);
		}
	}
}


static inline void
sakura_kernel_core(
	const real_t dt,
	const int_t flag,
	uint_t lane,
	Sakura_Data *jp,
	local Sakura_Data_SoA *ip)
// flop count: 28 + ??
{
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
		for (uint_t k = 0; k < 1; ++k) {
			real_t m = ip->m[i] + jp->m;
			real_t e2 = ip->e2[i] + jp->e2;
			real_t r0x = ip->rx[i] - jp->rx;
			real_t r0y = ip->ry[i] - jp->ry;
			real_t r0z = ip->rz[i] - jp->rz;
			real_t v0x = ip->vx[i] - jp->vx;
			real_t v0y = ip->vy[i] - jp->vy;
			real_t v0z = ip->vz[i] - jp->vz;

			real_t r1x = r0x;
			real_t r1y = r0y;
			real_t r1z = r0z;
			real_t v1x = v0x;
			real_t v1y = v0y;
			real_t v1z = v0z;
			evolve_twobody(
				dt, flag, m, e2,
				r0x, r0y, r0z, v0x, v0y, v0z,
				&r1x, &r1y, &r1z, &v1x, &v1y, &v1z
			);	// flop count: ??

			real_t inv_m = 1 / m;
			real_t drx = r0x - r1x;
			real_t dry = r0y - r1y;
			real_t drz = r0z - r1z;
			real_t dvx = v0x - v1x;
			real_t dvy = v0y - v1y;
			real_t dvz = v0z - v1z;

//			real_t imu = ip->m[i] * inv_m;
//			jp->drx += imu * drx;
//			jp->dry += imu * dry;
//			jp->drz += imu * drz;
//			jp->dvx += imu * dvx;
//			jp->dvy += imu * dvy;
//			jp->dvz += imu * dvz;

			real_t jmu = jp->m * inv_m;
			ip->drx[i] += jmu * drx;
			ip->dry[i] += jmu * dry;
			ip->drz[i] += jmu * drz;
			ip->dvx[i] += jmu * dvx;
			ip->dvy[i] += jmu * dvy;
			ip->dvz[i] += jmu * dvz;

			simd_shuff_Sakura_Data(jp);
		}
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
sakura_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[],
	global real_t __jdrdot[])
{
	local Sakura_Data_SoA _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t ii = 1 * WGSIZE * grp;
				ii < ni;
				ii += 1 * WGSIZE * ngrps) {
		Sakura_Data ip = {{0}};
		read_Sakura_Data(
			ii, lid, &ip,
			ni, __im, __ie2, __irdot
		);
		barrier(CLK_LOCAL_MEM_FENCE);
		_ip[warp].m[lane] = ip.m;
		_ip[warp].e2[lane] = ip.e2;
		_ip[warp].rx[lane] = ip.rx;
		_ip[warp].ry[lane] = ip.ry;
		_ip[warp].rz[lane] = ip.rz;
		_ip[warp].vx[lane] = ip.vx;
		_ip[warp].vy[lane] = ip.vy;
		_ip[warp].vz[lane] = ip.vz;
		_ip[warp].drx[lane] = ip.drx;
		_ip[warp].dry[lane] = ip.dry;
		_ip[warp].drz[lane] = ip.drz;
		_ip[warp].dvx[lane] = ip.dvx;
		_ip[warp].dvy[lane] = ip.dvy;
		_ip[warp].dvz[lane] = ip.dvz;
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint_t jj = 0;
					jj < nj;
					jj += 1 * WGSIZE) {
			Sakura_Data jp = {{0}};
			read_Sakura_Data(
				jj, lid, &jp,
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				p2p_sakura_kernel_core(dt, flag, lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			for (uint_t k = 0, kk = jj + lid;
						k < 1;
						k += 1, kk += WGSIZE) {
				if (kk < nj) {
					atomic_fadd(&(__jdrdot+(0*NDIM+0)*nj)[kk], jp._drx[k]);
					atomic_fadd(&(__jdrdot+(0*NDIM+1)*nj)[kk], jp._dry[k]);
					atomic_fadd(&(__jdrdot+(0*NDIM+2)*nj)[kk], jp._drz[k]);
					atomic_fadd(&(__jdrdot+(1*NDIM+0)*nj)[kk], jp._dvx[k]);
					atomic_fadd(&(__jdrdot+(1*NDIM+1)*nj)[kk], jp._dvy[k]);
					atomic_fadd(&(__jdrdot+(1*NDIM+2)*nj)[kk], jp._dvz[k]);
				}
			}
		}

		ip.drx = _ip[warp].drx[lane];
		ip.dry = _ip[warp].dry[lane];
		ip.drz = _ip[warp].drz[lane];
		ip.dvx = _ip[warp].dvx[lane];
		ip.dvy = _ip[warp].dvy[lane];
		ip.dvz = _ip[warp].dvz[lane];
		for (uint_t k = 0, kk = ii + lid;
					k < 1;
					k += 1, kk += WGSIZE) {
			if (kk < ni) {
				(__idrdot+(0*NDIM+0)*ni)[kk] -= ip._drx[k];
				(__idrdot+(0*NDIM+1)*ni)[kk] -= ip._dry[k];
				(__idrdot+(0*NDIM+2)*ni)[kk] -= ip._drz[k];
				(__idrdot+(1*NDIM+0)*ni)[kk] -= ip._dvx[k];
				(__idrdot+(1*NDIM+1)*ni)[kk] -= ip._dvy[k];
				(__idrdot+(1*NDIM+2)*ni)[kk] -= ip._dvz[k];
			}
		}
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
sakura_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[])
{
	// ------------------------------------------------------------------------
	const uint_t nj = ni;
	global const real_t *__jm = __im;
	global const real_t *__je2 = __ie2;
	global const real_t *__jrdot = __irdot;
	global real_t *__jdrdot = __idrdot;
	// ------------------------------------------------------------------------

	local Sakura_Data_SoA _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t ii = 1 * WGSIZE * grp;
				ii < ni;
				ii += 1 * WGSIZE * ngrps) {
		Sakura_Data ip = {{0}};
		read_Sakura_Data(
			ii, lid, &ip,
			ni, __im, __ie2, __irdot
		);
		barrier(CLK_LOCAL_MEM_FENCE);
		_ip[warp].m[lane] = ip.m;
		_ip[warp].e2[lane] = ip.e2;
		_ip[warp].rx[lane] = ip.rx;
		_ip[warp].ry[lane] = ip.ry;
		_ip[warp].rz[lane] = ip.rz;
		_ip[warp].vx[lane] = ip.vx;
		_ip[warp].vy[lane] = ip.vy;
		_ip[warp].vz[lane] = ip.vz;
		_ip[warp].drx[lane] = ip.drx;
		_ip[warp].dry[lane] = ip.dry;
		_ip[warp].drz[lane] = ip.drz;
		_ip[warp].dvx[lane] = ip.dvx;
		_ip[warp].dvy[lane] = ip.dvy;
		_ip[warp].dvz[lane] = ip.dvz;
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint_t jj = 0;
					jj < nj;
					jj += 1 * WGSIZE) {
			Sakura_Data jp = {{0}};
			read_Sakura_Data(
				jj, lid, &jp,
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				sakura_kernel_core(dt, flag, lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

//			for (uint_t k = 0, kk = jj + lid;
//						k < 1;
//						k += 1, kk += WGSIZE) {
//				if (kk < nj) {
//					atomic_fadd(&(__jdrdot+(0*NDIM+0)*nj)[kk], jp._drx[k]);
//					atomic_fadd(&(__jdrdot+(0*NDIM+1)*nj)[kk], jp._dry[k]);
//					atomic_fadd(&(__jdrdot+(0*NDIM+2)*nj)[kk], jp._drz[k]);
//					atomic_fadd(&(__jdrdot+(1*NDIM+0)*nj)[kk], jp._dvx[k]);
//					atomic_fadd(&(__jdrdot+(1*NDIM+1)*nj)[kk], jp._dvy[k]);
//					atomic_fadd(&(__jdrdot+(1*NDIM+2)*nj)[kk], jp._dvz[k]);
//				}
//			}
		}

		ip.drx = _ip[warp].drx[lane];
		ip.dry = _ip[warp].dry[lane];
		ip.drz = _ip[warp].drz[lane];
		ip.dvx = _ip[warp].dvx[lane];
		ip.dvy = _ip[warp].dvy[lane];
		ip.dvz = _ip[warp].dvz[lane];
		for (uint_t k = 0, kk = ii + lid;
					k < 1;
					k += 1, kk += WGSIZE) {
			if (kk < ni) {
				(__idrdot+(0*NDIM+0)*ni)[kk] -= ip._drx[k];
				(__idrdot+(0*NDIM+1)*ni)[kk] -= ip._dry[k];
				(__idrdot+(0*NDIM+2)*ni)[kk] -= ip._drz[k];
				(__idrdot+(1*NDIM+0)*ni)[kk] -= ip._dvx[k];
				(__idrdot+(1*NDIM+1)*ni)[kk] -= ip._dvy[k];
				(__idrdot+(1*NDIM+2)*ni)[kk] -= ip._dvz[k];
			}
		}
	}
}

