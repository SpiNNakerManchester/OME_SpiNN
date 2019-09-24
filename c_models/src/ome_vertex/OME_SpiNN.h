/*
 * Copyright (c) 2019-2020 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OME_SpiNN_H_
#define OME_SpiNN_H_


#define REAL double

//! \brief params region data format
typedef struct parameters_struct{
    uint32_t total_ticks;
    int seg_size;
    uint key;
    uint PADDING;
    REAL dt;
} parameters_struct;

//! \brief filter coeffs struct
typedef struct filter_coeffs_struct{
    REAL shb1;
    REAL shb2;
    REAL shb3;
    REAL sha1;
    REAL sha2;
    REAL sha3;
} filter_coeffs_struct;

//! \brief input data struct
typedef struct data_struct{
    REAL * in_data;
} data_struct;

//! \brief concha params
typedef struct concha_params_struct{
    REAL concha_gain_scalar;
    REAL ear_canal_gain_scalar;
} concha_params_struct;

//! \brief data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    FILTER_COEFFS,
    DATA,
    CONCHA_PARAMS,
    PROFILER,
    PROVENANCE
} regions;

//! \brief provenance data items
typedef enum provenance_items {
    B0, B1, B2, A0, A1, A2
} provenance_items;

//! \brief callback priorities
typedef enum callback_priorities {
    SDP_PRIORITY = 1, DMA_TRANSFER_DONE_PRIORITY = 1, TIMER_TICK_PRIORITY = 0
} callback_priorities;

#define DMA_TAG 0

#define SEGSIZE 8

#define MAX_SIGNAL_S 1

#define STAPES_SCALAR 5e-7

#define STAPES_1 10.0

#define CONCHA_H 3000.0

#define CONCHA_1 1500.0

#define EAR_CANAL_L 3000.0

#define EAR_CANAL_H 3800.0

// ABS HOW, given this has no spike input?!
#define A_RATT 1.0 //TODO: this should be determined by spiking input

// argument to avoid callback api
#define OME_FILLER_ARG 0

#define max(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define ABS(x) (((x)<0) ? -(x) : (x))
#define SIGN(x) (((x)<0) ? -(1.0f) : (1.0f))

typedef union {
	uint32_t u;
	float f;
} uint_float_union;

#endif /* OME_SpiNN_H_ */
