#include "acc_kernel_common.h"


kernel void
acc_kernel(
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

		vec(Acc_Data) ip;
		ip.m = vec(vload)(0, __im + i);
		ip.e2 = vec(vload)(0, __ie2 + i);
		for (uint_t kdot = 0; kdot < 1; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				global const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = vec(vload)(0, ptr + i);
				ip.adot[kdot][kdim] = (real_tn)(0);
			}
		}

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; ((j + LSIZE) - 1) < nj; j += LSIZE) {
			Acc_Data jp;
			jp.m = __jm[j + lid];
			jp.e2 = __je2[j + lid];
			for (uint_t kdot = 0; kdot < 1; ++kdot) {
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					global const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j + lid];
					jp.adot[kdot][kdim] = 0;
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			local Acc_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = acc_kernel_core(ip, _jp[k]);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			Acc_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (uint_t kdot = 0; kdot < 1; ++kdot) {
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					global const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
					jp.adot[kdot][kdim] = 0;
				}
			}
			ip = acc_kernel_core(ip, jp);
		}

		for (uint_t kdot = 0; kdot < 1; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				global real_t *ptr = &__iadot[(kdot*NDIM+kdim)*ni];
				vec(vstore)(ip.adot[kdot][kdim], 0, ptr + i);
			}
		}
	}
}

