#include "sakura_kernel_common.h"


static inline void
sakura_kernel_core(
	uint_t ilid,
	uint_t jlid,
	local Sakura_Data *ip,
	local Sakura_Data *jp,
	const real_t dt,
	const int_t flag)
// flop count: 27 + ??
{
	for (uint_t ii = 0; ii < LMSIZE; ii += WGSIZE) {
		uint_t i = ii + ilid;
		real_t im = ip->m[i];
		real_t iee = ip->e2[i];
		real_t irx = ip->rx[i];
		real_t iry = ip->ry[i];
		real_t irz = ip->rz[i];
		real_t ivx = ip->vx[i];
		real_t ivy = ip->vy[i];
		real_t ivz = ip->vz[i];
//		real_t idrx = ip->drx[i];
//		real_t idry = ip->dry[i];
//		real_t idrz = ip->drz[i];
//		real_t idvx = ip->dvx[i];
//		real_t idvy = ip->dvy[i];
//		real_t idvz = ip->dvz[i];
		for (uint_t jj = 0; jj < LMSIZE; jj += WGSIZE) {
			uint_t j = jj + jlid;
			real_t m = im + jp->m[j];
			real_t e2 = iee + jp->e2[j];
			real_t r0x = irx - jp->rx[j];
			real_t r0y = iry - jp->ry[j];
			real_t r0z = irz - jp->rz[j];
			real_t v0x = ivx - jp->vx[j];
			real_t v0y = ivy - jp->vy[j];
			real_t v0z = ivz - jp->vz[j];

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
			real_t drx = r1x - r0x;
			real_t dry = r1y - r0y;
			real_t drz = r1z - r0z;
			real_t dvx = v1x - v0x;
			real_t dvy = v1y - v0y;
			real_t dvz = v1z - v0z;

			real_t imu = im * inv_m;
			jp->drx[j] -= imu * drx;
			jp->dry[j] -= imu * dry;
			jp->drz[j] -= imu * drz;
			jp->dvx[j] -= imu * dvx;
			jp->dvy[j] -= imu * dvy;
			jp->dvz[j] -= imu * dvz;

//			real_t jmu = jp->m[j] * inv_m;
//			idrx += jmu * drx;
//			idry += jmu * dry;
//			idrz += jmu * drz;
//			idvx += jmu * dvx;
//			idvy += jmu * dvy;
//			idvz += jmu * dvz;
		}
//		ip->drx[i] = idrx;
//		ip->dry[i] = idry;
//		ip->drz[i] = idrz;
//		ip->dvx[i] = idvx;
//		ip->dvy[i] = idvy;
//		ip->dvz[i] = idvz;
	}
}


static inline void
sakura_kernel_impl(
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
	global real_t __jdrdot[],
	local Sakura_Data *ip,
	local Sakura_Data *jp)
{
	uint_t lid = get_local_id(0);
	for (uint_t ii = LMSIZE * 1 * get_group_id(0);
				ii < ni;
				ii += LMSIZE * 1 * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LMSIZE * 1), (ni - ii));
		for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
			uint_t k = kk + lid;
			ip->m[k] = (real_t1)(0);
			ip->e2[k] = (real_t1)(0);
			ip->rx[k] = (real_t1)(0);
			ip->ry[k] = (real_t1)(0);
			ip->rz[k] = (real_t1)(0);
			ip->vx[k] = (real_t1)(0);
			ip->vy[k] = (real_t1)(0);
			ip->vz[k] = (real_t1)(0);
			ip->drx[k] = (real_t1)(0);
			ip->dry[k] = (real_t1)(0);
			ip->drz[k] = (real_t1)(0);
			ip->dvx[k] = (real_t1)(0);
			ip->dvy[k] = (real_t1)(0);
			ip->dvz[k] = (real_t1)(0);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		async_work_group_copy(ip->_m, __im+ii, iN, 0);
		async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vx, __irdot+(1*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vy, __irdot+(1*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vz, __irdot+(1*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_drx, __idrdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_dry, __idrdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_drz, __idrdot+(0*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_dvx, __idrdot+(1*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_dvy, __idrdot+(1*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_dvz, __idrdot+(1*NDIM+2)*ni+ii, iN, 0);
		for (uint_t jj = 0;
					jj < nj;
					jj += LMSIZE * 1) {
			uint_t jN = min((uint_t)(LMSIZE * 1), (nj - jj));
			for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
				uint_t k = kk + lid;
				jp->m[k] = (real_t1)(0);
				jp->e2[k] = (real_t1)(0);
				jp->rx[k] = (real_t1)(0);
				jp->ry[k] = (real_t1)(0);
				jp->rz[k] = (real_t1)(0);
				jp->vx[k] = (real_t1)(0);
				jp->vy[k] = (real_t1)(0);
				jp->vz[k] = (real_t1)(0);
				jp->drx[k] = (real_t1)(0);
				jp->dry[k] = (real_t1)(0);
				jp->drz[k] = (real_t1)(0);
				jp->dvx[k] = (real_t1)(0);
				jp->dvy[k] = (real_t1)(0);
				jp->dvz[k] = (real_t1)(0);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			async_work_group_copy(jp->_m, __jm+jj, jN, 0);
			async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
			async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vx, __jrdot+(1*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vy, __jrdot+(1*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vz, __jrdot+(1*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_drx, __jdrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_dry, __jdrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_drz, __jdrdot+(0*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_dvx, __jdrdot+(1*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_dvy, __jdrdot+(1*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_dvz, __jdrdot+(1*NDIM+2)*nj+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			for (uint_t k = 0; k < 1; ++k) {
//				#pragma unroll
				for (uint_t l = 0; l < WGSIZE; ++l) {
					sakura_kernel_core(l, lid, jp, ip, dt, flag);
//					sakura_kernel_core((lid + l) % WGSIZE, lid, jp, ip, dt, flag);
				}
				for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
					uint_t i = kk + lid;
					shuff(ip->m[i], 1);
					shuff(ip->e2[i], 1);
					shuff(ip->rx[i], 1);
					shuff(ip->ry[i], 1);
					shuff(ip->rz[i], 1);
					shuff(ip->vx[i], 1);
					shuff(ip->vy[i], 1);
					shuff(ip->vz[i], 1);
					shuff(ip->drx[i], 1);
					shuff(ip->dry[i], 1);
					shuff(ip->drz[i], 1);
					shuff(ip->dvx[i], 1);
					shuff(ip->dvy[i], 1);
					shuff(ip->dvz[i], 1);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		async_work_group_copy(__idrdot+(0*NDIM+0)*ni+ii, ip->_drx, iN, 0);
		async_work_group_copy(__idrdot+(0*NDIM+1)*ni+ii, ip->_dry, iN, 0);
		async_work_group_copy(__idrdot+(0*NDIM+2)*ni+ii, ip->_drz, iN, 0);
		async_work_group_copy(__idrdot+(1*NDIM+0)*ni+ii, ip->_dvx, iN, 0);
		async_work_group_copy(__idrdot+(1*NDIM+1)*ni+ii, ip->_dvy, iN, 0);
		async_work_group_copy(__idrdot+(1*NDIM+2)*ni+ii, ip->_dvz, iN, 0);
	}
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
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
	local Sakura_Data _ip;
	local Sakura_Data _jp;

	sakura_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		dt, flag,
		__idrdot, __jdrdot,
		&_ip, &_jp
	);

	sakura_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		dt, flag,
		__jdrdot, __idrdot,
		&_jp, &_ip
	);
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
sakura_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[])
{
	local Sakura_Data _ip;
	local Sakura_Data _jp;

	sakura_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		dt, flag,
		__idrdot, __idrdot,
		&_ip, &_jp
	);
}

