/*
 * IHC_AN.h
 *
 *  Created on: 1 Nov 2016
 *      Author: rjames
        robert.james@manchester.ac.uk
 */

#ifndef OME_SpiNN_H_
#define OME_SpiNN_H_

#define REAL double

#define SEGSIZE 8

#define MAX_SIGNAL_S 1

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

struct parameters_struct {
    uint total_ticks;
    uint COREID;
    uint NUM_DRNL;
    uint NUM_MACK;
    uint FS;
    uint NUM_BFS;
    uint KEY;
    uint R2S_KEY;
    uint TS;
    REAL SHB1;
    REAL SHB2;
    REAL SHB3;
    REAL SHA1;
    REAL SHA2;
    REAL SHA3;
    uint DATA;
} parameters_struct;


#endif /* IHC_AN_H_ */
