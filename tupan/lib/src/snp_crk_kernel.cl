#include "snp_crk_kernel_common.h"


kernel void
snp_crk_kernel(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __iadot[])
{
	uint_t lid = get_local_id(0);
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * SIMD < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= SIMD;
		i = (i+SIMD < ni) ? (i):(ni-SIMD);
		i *= (SIMD < ni);

		vec(Snp_Crk_Data) ip;
		ip.m = vec(vload)(0, __im + i);
		ip.e2 = vec(vload)(0, __ie2 + i);
		#pragma unroll
		for (uint_t kdot = 0; kdot < 4; ++kdot) {
			#pragma unroll
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				global const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = vec(vload)(0, ptr + i);
				ip.adot[kdot][kdim] = (real_tn)(0);
			}
		}

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; ((j + LSIZE) - 1) < nj; j += LSIZE) {
			Snp_Crk_Data jp;
			jp.m = __jm[j + lid];
			jp.e2 = __je2[j + lid];
			#pragma unroll
			for (uint_t kdot = 0; kdot < 4; ++kdot) {
				#pragma unroll
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					global const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j + lid];
					jp.adot[kdot][kdim] = 0;
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			local Snp_Crk_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = snp_crk_kernel_core(ip, _jp[k]);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			Snp_Crk_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			#pragma unroll
			for (uint_t kdot = 0; kdot < 4; ++kdot) {
				#pragma unroll
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					global const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
					jp.adot[kdot][kdim] = 0;
				}
			}
			ip = snp_crk_kernel_core(ip, jp);
		}

		#pragma unroll
		for (uint_t kdot = 0; kdot < 4; ++kdot) {
			#pragma unroll
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				global real_t *ptr = &__iadot[(kdot*NDIM+kdim)*ni];
				vec(vstore)(ip.adot[kdot][kdim], 0, ptr + i);
			}
		}
	}
}

