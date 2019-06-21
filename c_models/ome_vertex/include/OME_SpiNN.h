/*
 * IHC_AN.h
 *
 *  Created on: 1 Nov 2016
 *      Author: rjames
        robert.james@manchester.ac.uk
 */

#ifndef OME_SpiNN_H_
#define OME_SpiNN_H_

#define REAL double //float
#define REAL_CONST(x) x//x##f


#define MAX_CHIPX 1//255
#define MAX_CHIPY 1//255
#define MAX_COREID 16
#define SEED_SEL_SIZE 1024

#define SEGSIZE 8//16//96//

#define MAX_SIGNAL_S 1

#define TIMER2_CONF        0x82
#define TIMER2_LOAD        0

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define ABS(x) (((x)<0) ? -(x) : (x))
#define SIGN(x) (((x)<0) ? -(1.0f) : (1.0f))

typedef union
{
	uint32_t u;
	float f;
} uint_float_union;


#endif /* IHC_AN_H_ */
