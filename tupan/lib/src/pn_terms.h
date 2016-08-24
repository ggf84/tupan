#ifndef __PN_TERMS_H__
#define __PN_TERMS_H__

#include "common.h"

// defines some fractions used in PN expressions
#define _1_2 (1/(real_t)(2))
#define _1_4 (1/(real_t)(4))
#define _3_2 (3/(real_t)(2))
#define _3_4 (3/(real_t)(4))
#define _4_5 (4/(real_t)(5))
#define _5_2 (5/(real_t)(2))
#define _5_4 (5/(real_t)(4))
#define _9_2 (9/(real_t)(2))
#define _13_12 (13/(real_t)(12))
#define _15_2 (15/(real_t)(2))
#define _15_4 (15/(real_t)(4))
#define _15_8 (15/(real_t)(8))
#define _17_2 (17/(real_t)(2))
#define _24_7 (24/(real_t)(7))
#define _27_4 (27/(real_t)(4))
#define _35_16 (35/(real_t)(16))
#define _39_2 (39/(real_t)(2))
#define _41_16 (41/(real_t)(16))
#define _43_2 (43/(real_t)(2))
#define _44_15 (44/(real_t)(15))
#define _48_35 (48/(real_t)(35))
#define _52_3 (52/(real_t)(3))
#define _52_15 (52/(real_t)(15))
#define _55_4 (55/(real_t)(4))
#define _56_15 (56/(real_t)(15))
#define _57_4 (57/(real_t)(4))
#define _63_4 (63/(real_t)(4))
#define _69_2 (69/(real_t)(2))
#define _81_8 (81/(real_t)(8))
#define _83_8 (83/(real_t)(8))
#define _90_7 (90/(real_t)(7))
#define _91_2 (91/(real_t)(2))
#define _91_8 (91/(real_t)(8))
#define _95_12 (95/(real_t)(12))
#define _123_32 (123/(real_t)(32))
#define _123_64 (123/(real_t)(64))
#define _132_35 (132/(real_t)(35))
#define _137_8 (137/(real_t)(8))
#define _152_21 (152/(real_t)(21))
#define _152_35 (152/(real_t)(35))
#define _171_2 (171/(real_t)(2))
#define _177_4 (177/(real_t)(4))
#define _180_7 (180/(real_t)(7))
#define _184_21 (184/(real_t)(21))
#define _191_4 (191/(real_t)(4))
#define _204_35 (204/(real_t)(35))
#define _205_2 (205/(real_t)(2))
#define _207_8 (207/(real_t)(8))
#define _225_2 (225/(real_t)(2))
#define _229_4 (229/(real_t)(4))
#define _243_4 (243/(real_t)(4))
#define _246_35 (246/(real_t)(35))
#define _259_4 (259/(real_t)(4))
#define _269_4 (269/(real_t)(4))
#define _283_2 (283/(real_t)(2))
#define _288_5 (288/(real_t)(5))
#define _292_35 (292/(real_t)(35))
#define _307_8 (307/(real_t)(8))
#define _311_4 (311/(real_t)(4))
#define _334_35 (334/(real_t)(35))
#define _348_5 (348/(real_t)(5))
#define _357_4 (357/(real_t)(4))
#define _357_8 (357/(real_t)(8))
#define _372_5 (372/(real_t)(5))
#define _375_4 (375/(real_t)(4))
#define _383_2 (383/(real_t)(2))
#define _415_8 (415/(real_t)(8))
#define _454_15 (454/(real_t)(15))
#define _455_8 (455/(real_t)(8))
#define _471_8 (471/(real_t)(8))
#define _479_8 (479/(real_t)(8))
#define _534_35 (534/(real_t)(35))
#define _545_3 (545/(real_t)(3))
#define _547_3 (547/(real_t)(3))
#define _565_4 (565/(real_t)(4))
#define _582_5 (582/(real_t)(5))
#define _615_64 (615/(real_t)(64))
#define _654_35 (654/(real_t)(35))
#define _684_5 (684/(real_t)(5))
#define _696_5 (696/(real_t)(5))
#define _723_4 (723/(real_t)(4))
#define _732_35 (732/(real_t)(35))
#define _744_5 (744/(real_t)(5))
#define _854_15 (854/(real_t)(15))
#define _939_4 (939/(real_t)(4))
#define _984_35 (984/(real_t)(35))
#define _1028_21 (1028/(real_t)(21))
#define _1068_35 (1068/(real_t)(35))
#define _1113_8 (1113/(real_t)(8))
#define _1252_35 (1252/(real_t)(35))
#define _1308_35 (1308/(real_t)(35))
#define _1336_35 (1336/(real_t)(35))
#define _1432_35 (1432/(real_t)(35))
#define _1746_5 (1746/(real_t)(5))
#define _1768_105 (1768/(real_t)(105))
#define _1954_5 (1954/(real_t)(5))
#define _2069_8 (2069/(real_t)(8))
#define _2056_21 (2056/(real_t)(21))
#define _2224_21 (2224/(real_t)(21))
#define _2864_35 (2864/(real_t)(35))
#define _2864_105 (2864/(real_t)(105))
#define _2872_21 (2872/(real_t)(21))
#define _3172_21 (3172/(real_t)(21))
#define _3568_105 (3568/(real_t)(105))
#define _3992_105 (3992/(real_t)(105))
#define _4328_105 (4328/(real_t)(105))
#define _4888_105 (4888/(real_t)(105))
#define _5056_105 (5056/(real_t)(105))
#define _5752_105 (5752/(real_t)(105))
#define _5812_105 (5812/(real_t)(105))
#define _6224_105 (6224/(real_t)(105))
#define _6388_105 (6388/(real_t)(105))
#define _10048_105 (10048/(real_t)(105))
#define _13576_105 (13576/(real_t)(105))


static inline void
p2p_pn2(
	const real_tn im,
	const real_tn jm,
	const real_tn inv_r,
	const real_tn iv2,
	const real_tn jv2,
	const real_tn ivjv,
	const real_tn niv,
	const real_tn njv,
	const real_tn njv2,
	const CLIGHT clight,
	PN *pn2)
{
	// Include ~1/c^2 terms (6+2 == 8 terms)
	pn2->a = - iv2
			 - 2 * jv2
			 + 4 * ivjv
			 + _3_2 * njv2
			 + inv_r * ( + 5 * im
						 + 4 * jm );

	pn2->b = + 4 * niv
			 - 3 * njv;

	pn2->a *= clight.inv2;
	pn2->b *= clight.inv2;
}
// 11+3+2 == 16 FLOPs


static inline void
p2p_pn4(
	const real_tn im,
	const real_tn jm,
	const real_tn im2,
	const real_tn jm2,
	const real_tn imjm,
	const real_tn inv_r,
	const real_tn inv_r2,
	const real_tn iv2,
	const real_tn jv2,
	const real_tn jv4,
	const real_tn ivjv,
	const real_tn nv,
	const real_tn niv,
	const real_tn njv,
	const real_tn niv2,
	const real_tn njv2,
	const real_tn nivnjv,
	const CLIGHT clight,
	PN *pn4)
{
	// Include ~1/c^4 terms (21+10 == 31 terms)
	pn4->a = + 2 * ( - jv4
					 + ivjv * ( + 2 * jv2
								- ivjv ) )
			 + njv2 * ( + _3_2 * iv2
						+ _9_2 * jv2
						- 6 * ivjv
						- _15_8 * njv2 )
			 - inv_r2 * ( + _57_4 * im2
						  + 9 * jm2
						  + _69_2 * imjm )
			 + inv_r * ( + im * ( - _15_4 * iv2
								  + _5_4 * jv2
								  - _5_2 * ivjv
								  + _39_2 * niv2
								  - 39 * nivnjv
								  + _17_2 * njv2 )
						 + jm * ( + 4 * jv2
								  - 8 * ivjv
								  + 2 * niv2
								  - 4 * nivnjv
								  - 6 * njv2 ) );

	pn4->b = + inv_r * ( + im * ( - _63_4 * niv
								  + _55_4 * njv )
						 - jm * 2 * ( + niv
									  + njv ) )
			 + iv2 * njv
			 - ivjv * nv * 4
			 + jv2 * ( + 4 * niv
					   - 5 * njv )
			 + njv2 * ( - 6 * niv
						+ _9_2 * njv );

	pn4->a *= clight.inv4;
	pn4->b *= clight.inv4;
}
// 46+24+2 == 72 FLOPs


static inline void
p2p_pn5(
	const real_tn im,
	const real_tn jm,
	const real_tn inv_r,
	const real_tn v2,
	const real_tn nv,
	const CLIGHT clight,
	PN *pn5)
{
	// Include ~1/c^5 terms (3+3 == 6 terms)
	pn5->a = + nv * ( + inv_r * ( - 6 * im
								  + _52_3 * jm )
					  + 3 * v2 );

	pn5->b = + inv_r * ( + 2 * im
						 - 8 * jm )
			 - v2;

	real_tn m_c5r = _4_5 * (im * inv_r) * clight.inv5;
	pn5->a *= m_c5r;
	pn5->b *= m_c5r;
}
// 7+5+4 == 16 FLOPs


static inline void
p2p_pn6(
	const real_tn im,
	const real_tn jm,
	const real_tn im2,
	const real_tn jm2,
	const real_tn imjm,
	const real_tn inv_r,
	const real_tn inv_r2,
	const real_tn v2,
	const real_tn iv2,
	const real_tn jv2,
	const real_tn jv4,
	const real_tn ivjv,
	const real_tn nv,
	const real_tn nvnv,
	const real_tn niv,
	const real_tn njv,
	const real_tn niv2,
	const real_tn njv2,
	const real_tn nivnjv,
	const CLIGHT clight,
	PN *pn6)
{
	// Include ~1/c^6 terms (66+37 == 103 terms)
	pn6->a = + njv2 * ( + 3 * ivjv * ( + ivjv
									   - 4 * jv2 )
						+ _3_2 * jv2 * ( + iv2
										 + 5 * jv2 )
						+ njv2 * ( + _15_2 * ( + ivjv
											   - jv2
											   - _1_4 * iv2 )
								   + _35_16 * njv2 ) )
			 + 2 * jv2 * ( + ivjv * ( + 2 * jv2
									  - ivjv )
						   - jv4 )
			 + im * inv_r * ( + niv * ( + njv * ( + 244 * ivjv
												  - _205_2 * iv2
												  - _283_2 * jv2
												  + _383_2 * njv2 )
										+ niv * ( + _229_4 * ( + iv2
															   + jv2
															   - 2 * ivjv )
												  - _723_4 * njv2
												  + niv * ( + _171_2 * ( + njv
																		 - _1_4 * niv ) ) ) )
							  + njv2 * ( + _191_4 * iv2
										 + _259_4 * jv2
										 - _225_2 * ivjv
										 - _455_8 * njv2 )
							  + ivjv * ( + _91_2 * iv2
										 + 43 * jv2
										 - _177_4 * ivjv )
							  - _91_8 * iv2 * ( + iv2
												+ 2 * jv2 )
							  - _81_8 * jv4 )
			 + jm * inv_r * 4 * ( + jv4
								  + njv * ( + niv * ( + ivjv
													  - jv2 )
											+ njv * ( + 3 * ( + ivjv
															  - jv2 )
													  - _3_2 * niv2
													  + njv * ( + 3 * niv
																+ _3_2 * njv ) ) )
								  + ivjv * ( + ivjv
											 - 2 * jv2 ) )
			 + jm2 * inv_r2 * ( - niv2
								+ 2 * nivnjv
								+ _43_2 * njv2
								+ 18 * ivjv
								- 9 * jv2 )
			 + imjm * inv_r2 * ( + _415_8 * niv2
								 - _375_4 * nivnjv
								 + _1113_8 * njv2
								 + 18 * iv2
								 + PI2 * ( + _123_64 * v2
										   - _615_64 * nvnv )
								 + 33 * ( + ivjv
										  - _1_2 * jv2 ) )
			 + im2 * inv_r2 * ( - _2069_8 * niv2
								+ 543 * nivnjv
								- _939_4 * njv2
								+ _471_8 * iv2
								+ _357_8 * ( + jv2
											 - 2 * ivjv ) )
			 + inv_r * inv_r2 * ( + 16 * jm * jm2
								  + im2 * jm * ( + _547_3
												 - _41_16 * PI2 )
								  - _13_12 * im * im2
								  + im * jm2 * ( + _545_3
												 - _41_16 * PI2 ) );

	pn6->b = + njv * ( + jv2 * ( + iv2
								 - 7 * jv2 )
					   + 2 * ivjv * ( + 4 * jv2
									  - ivjv )
					   + njv * ( + 6 * niv * ( + ivjv
											   - 2 * jv2 )
								 + njv * ( + 6 * ( + 2 * jv2
												   - ivjv
												   - _1_4 * iv2 )
										   + njv * ( + _15_2 * ( + niv
																 - _3_4 * njv ) ) ) ) )
			 + 4 * niv * ( + jv4
						   - ivjv * jv2 )
			 + jm * inv_r * ( + njv * ( + 4 * ( + ivjv
												- jv2
												- _1_2 * niv2 )
										+ njv * ( + 2 * ( + 4 * niv
														  + njv ) ) )
							  + 2 * niv * ( + ivjv
											- jv2 ) )
			 + im * inv_r * ( + niv * ( + _207_8 * iv2
										+ _81_8 * jv2
										- 36 * ivjv
										- _269_4 * njv2
										+ niv * ( + _565_4 * njv
												  - _243_4 * niv ) )
							  + njv * ( + _83_8 * jv2
										+ _27_4 * ivjv
										- _137_8 * iv2
										- _95_12 * njv2 ) )
			 + inv_r2 * ( + jm2 * ( + 4 * niv
									+ 5 * njv )
						  + im2 * ( + _311_4 * niv
									- _357_4 * njv )
						  + imjm * ( + _479_8 * njv
									 - _307_8 * niv
									 + _123_32 * PI2 * nv ) );

	pn6->a *= clight.inv6;
	pn6->b *= clight.inv6;
}
// ??+??+2 == ??? FLOPs


static inline void
p2p_pn7(
	const real_tn im,
	const real_tn jm,
	const real_tn im2,
	const real_tn jm2,
	const real_tn imjm,
	const real_tn inv_r,
	const real_tn inv_r2,
	const real_tn v2,
	const real_tn iv2,
	const real_tn jv2,
	const real_tn iv4,
	const real_tn jv4,
	const real_tn ivjv,
	const real_tn nv,
	const real_tn nvnv,
	const real_tn niv,
	const real_tn njv,
	const real_tn niv2,
	const real_tn njv2,
	const real_tn nivnjv,
	const CLIGHT clight,
	PN *pn7)
{
	// Include ~1/c^7 terms (40+25 == 65 terms)
	pn7->a = + im2 * inv_r2 * ( + _3992_105 * niv
								- _4328_105 * njv )
			 + imjm * inv_r * inv_r2 * ( - _13576_105 * niv
										 + _2872_21 * njv )
			 + jm2 * inv_r * inv_r2 * ( - _3172_21 * nv )
			 + im * inv_r * ( + niv * ( + 48 * niv2
										- _4888_105 * iv2
										+ _2056_21 * ivjv
										- _1028_21 * jv2 )
							  + nivnjv * ( - _696_5 * niv
										   + _744_5 * njv )
							  + njv * ( - _288_5 * njv2
										+ _5056_105 * iv2
										- _2224_21 * ivjv
										+ _5812_105 * jv2 ) )
			 + jm * inv_r * ( + niv * ( - _582_5 * niv2
										- _2864_35 * ivjv
										+ _1432_35 * jv2 )
							  + nivnjv * ( + _1746_5 * niv
										   - _1954_5 * njv )
							  + _3568_105 * nv * iv2
							  + njv * ( + 158 * njv2
										- _5752_105 * jv2
										+ _10048_105 * ivjv ) )
			 + ( + nv * ( - 56 * nvnv * nvnv
						  - _246_35 * iv4 )
				 + niv * ( + v2 * ( + 60 * niv2
									- 180 * nivnjv
									+ 174 * njv2 )
						   + ivjv * ( + _1068_35 * ( + iv2
													 - ivjv )
									  + _984_35 * jv2 )
						   - _534_35 * iv2 * jv2
						   - _204_35 * jv4 )
				 + njv * ( - 54 * njv2 * v2
						   + ivjv * ( - _984_35 * iv2
									  + _180_7 * ivjv
									  - _732_35 * jv2 )
						   + _90_7 * iv2 * jv2
						   + _24_7 * jv4 ) );

	pn7->b = - im2 * inv_r2 * _184_21
			 + imjm * inv_r2 * _6224_105
			 + jm2 * inv_r2 * _6388_105
			 + im * inv_r * ( + _52_15 * niv2
							  - _56_15 * nivnjv
							  - _44_15 * njv2
							  - _132_35 * iv2
							  + _152_35 * ivjv
							  - _48_35 * jv2 )
			 + jm * inv_r * ( + _454_15 * niv2
							  - _372_5 * nivnjv
							  + _854_15 * njv2
							  - _152_21 * iv2
							  + _2864_105 * ivjv
							  - _1768_105 * jv2 )
			 + ( + 60 * nvnv * nvnv
				 + v2 * ( - _348_5 * niv2
						  + _684_5 * nivnjv
						  - 66 * njv2 )
				 + _334_35 * iv4
				 + ivjv * ( - _1336_35 * iv2
							+ _1308_35 * ivjv
							- _1252_35 * jv2 )
				 + _654_35 * iv2 * jv2
				 + _292_35 * jv4 );

	real_tn m_c7r = (im * inv_r) * clight.inv7;
	pn7->a *= m_c7r;
	pn7->b *= m_c7r;
}
// ??+??+4 == ??? FLOPs


static inline PN
p2p_pnterms(
	const real_tn im,
	const real_tn ivx,
	const real_tn ivy,
	const real_tn ivz,
	const real_tn jm,
	const real_tn jvx,
	const real_tn jvy,
	const real_tn jvz,
	const real_tn rx,
	const real_tn ry,
	const real_tn rz,
	const real_tn vx,
	const real_tn vy,
	const real_tn vz,
	const real_tn v2,
	const real_tn inv_r,
	const real_tn inv_r2,
	const CLIGHT clight)
{
	PN pn = PN_Init(0, 0);
	PN pn1 = PN_Init(0, 0);
	PN pn2 = PN_Init(0, 0);
	PN pn3 = PN_Init(0, 0);
	PN pn4 = PN_Init(0, 0);
	PN pn5 = PN_Init(0, 0);
	PN pn6 = PN_Init(0, 0);
	PN pn7 = PN_Init(0, 0);

	if (clight.order >= 1) {
		// XXX: not implemented.
		if (clight.order >= 2) {
			real_tn iv2 = ivx * ivx + ivy * ivy + ivz * ivz;
			real_tn jv2 = jvx * jvx + jvy * jvy + jvz * jvz;
			real_tn ivjv = ivx * jvx + ivy * jvy + ivz * jvz;
			real_tn niv = rx * ivx + ry * ivy + rz * ivz;
			real_tn njv = rx * jvx + ry * jvy + rz * jvz;
			niv *= inv_r;
			njv *= inv_r;
			real_tn njv2 = njv * njv;
			p2p_pn2(
				im, jm, inv_r,
				iv2, jv2, ivjv,
				niv, njv, njv2,
				clight, &pn2);	// 16 FLOPs
			if (clight.order >= 3) {
				// XXX: not implemented.
				if (clight.order >= 4) {
					real_tn im2 = im * im;
					real_tn jm2 = jm * jm;
					real_tn imjm = im * jm;
					real_tn jv4 = jv2 * jv2;
					real_tn niv2 = niv * niv;
					real_tn nivnjv = niv * njv;
					real_tn nv = rx * vx + ry * vy + rz * vz;
					nv *= inv_r;
					p2p_pn4(
						im, jm, im2, jm2, imjm, inv_r, inv_r2,
						iv2, jv2, jv4, ivjv, nv, niv, njv,
						niv2, njv2, nivnjv, clight, &pn4);	// 72 FLOPs
					if (clight.order >= 5) {
						p2p_pn5(im, jm, inv_r, v2, nv, clight, &pn5);	// 16 FLOPs
						if (clight.order >= 6) {
							real_tn nvnv = nv * nv;
							p2p_pn6(
								im, jm, im2, jm2, imjm, inv_r, inv_r2,
								v2, iv2, jv2, jv4, ivjv, nv, nvnv,
								niv, njv, niv2, njv2, nivnjv, clight, &pn6);	// ??? FLOPs
							if (clight.order >= 7) {
								real_tn iv4 = iv2 * iv2;
								p2p_pn7(
									im, jm, im2, jm2, imjm, inv_r, inv_r2,
									v2, iv2, jv2, iv4, jv4, ivjv,
									nv, nvnv, niv, njv, niv2, njv2, nivnjv,
									clight, &pn7);	// ??? FLOPs
							}
						}
					}
				}
			}
		}
	}

	// Form the 213 post-Newtonian terms
	// ----> ((((((65    + 103  ) + 6    ) + 31   ) + 0    ) + 8    ) + 0    )
	pn.a += ((((((pn7.a + pn6.a) + pn5.a) + pn4.a) + pn3.a) + pn2.a) + pn1.a);
	pn.b += ((((((pn7.b + pn6.b) + pn5.b) + pn4.b) + pn3.b) + pn2.b) + pn1.b);

	real_tn m_r2 = jm * inv_r2;
	pn.a *= inv_r;
	pn.a *= m_r2;
	pn.b *= m_r2;
	return pn;
}
// ??+??+?? == ??? FLOPs


#endif	// __PN_TERMS_H__
