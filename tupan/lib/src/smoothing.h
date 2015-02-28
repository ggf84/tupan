#ifndef __SMOOTHING_H__
#define __SMOOTHING_H__

#include "common.h"


static inline real_tn
get_inv_r(
	real_tn const r2,
	real_tn const e2,
	int_tn const mask)
{
	real_tn inv_r = rsqrt(r2 + e2);
	return select((real_tn)(0), inv_r, mask);
}
// Total flop count: 2


static inline real_tn
smoothed_m_r1(
	real_tn const m,
	real_tn const r2,
	real_tn const e2,
	int_tn const mask)
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	return m * inv_r;
}
// Total flop count: 3


static inline real_tn
smoothed_m_r2(
	real_tn const m,
	real_tn const r2,
	real_tn const e2,
	int_tn const mask)
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	return m * inv_r * inv_r;
}
// Total flop count: 4


static inline real_tn
smoothed_m_r3(
	real_tn const m,
	real_tn const r2,
	real_tn const e2,
	int_tn const mask)
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	return m * inv_r * inv_r * inv_r;
}
// Total flop count: 5


static inline real_tn
smoothed_m_r1_inv_r2(
	real_tn const m,
	real_tn const r2,
	real_tn const e2,
	int_tn const mask,
	real_tn *inv_r2)
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	*inv_r2 = inv_r * inv_r;
	return m * inv_r;
}
// Total flop count: 4


static inline real_tn
smoothed_m_r3_inv_r2(
	real_tn const m,
	real_tn const r2,
	real_tn const e2,
	int_tn const mask,
	real_tn *inv_r2)
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	*inv_r2 = inv_r * inv_r;
	return m * inv_r * *inv_r2;
}
// Total flop count: 5


static inline real_tn
smoothed_m_r1_m_r3(
	real_tn const m,
	real_tn const r2,
	real_tn const e2,
	int_tn const mask,
	real_tn *m_r3)
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	real_tn m_r1 = m * inv_r;
	*m_r3 = inv_r * inv_r * m_r1;
	return m_r1;
}
// Total flop count: 5


static inline real_tn
smoothed_inv_r1r2(
	real_tn const r2,
	real_tn const e2,
	int_tn const mask,
	real_tn *inv_r2)
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	*inv_r2 = inv_r * inv_r;
	return inv_r;
}
// Total flop count: 3


#endif	// __SMOOTHING_H__
