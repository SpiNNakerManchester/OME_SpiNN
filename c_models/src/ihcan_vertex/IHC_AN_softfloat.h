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

#ifndef IHC_AN_softfloat_H_
#define IHC_AN_softfloat_H_

#include "spin1_api.h"

//! \brief
#define RECIP_BETA 1.0 / 400.0

//! \brief
#define GAMMA 100.0f

//! \brief
#define ECA 0.066

//! \brief
#define POWER 3.0f

//!*********************************** cilia constants *************//

//! \brief
#define CILIA_TC 0.00012

//! \brief
#define CILIA_FILTER_B1 1.0

//! \brief
#define CILIA_C 0.05f

//! \brief
#define CILIA_U0 0.3e-9f

//! \brief
#define CILIA_U1 1e-9f

//! \brief
#define CILIA_G_MAX 6e-9f

//! \brief
#define CILIA_GA 0.1e-9f

//! \brief
#define CILIA_ET 0.1f

//! \brief
#define CILIA_GK 2.1e-8f

//! \brief
#define CILIA_EK -0.08f

//! \brief
#define CILIA_RPC 0.04f

//! ********************* inner hair cell constants ************** //

//! \brief represents the capacitance of the hair cell 5e-12 farads
#define HAIR_CELL_CAPACITANCE 5e-12f

//! \brief the maximum calcium conductance for a channel in the model
#define GMAXCA 20e-9

//! \brief calcium channel time constant for low hair cells
#define TAU_CA_HSR 1.0 / 500e-6

//! \brief the maximum number of neurotransmitters possible at a synapse
#define MAXIMUM_NUM_NEUROTRANSMITTERS_AT_SYNAPSE 4.0f

//! \brief calcium channel time constant for medium hair cells
#define TAU_CA_MSR 1.0 / 350e-6

//! \brief calcium channel time constant for high hair cells
#define TAU_CA_LSR 1.0 / 200e-6

//! \brief hair cell membrane time constant tauM
#define DT_TAU_M 5e-5

//! \brief synaptic cleft neurotransmitter loss rate
#define SYNAPSE_CLEFT_LOSS_RATE 150.0f

//! \brief rate of neurotransmitter replacement to pre synapse from hair cell
#define PRE_SYNAPSE_REPLACEMENT_RATE_HAIR_CELL 15.0f

//! \brief rate of neurotransmitter replacement to presynapse from reuptake
//! store
#define PRE_SYNAPSE_REPLACEMENT_RATE_RE_UP_TAKE 300.0f

//! \brief rate of neurotransmitter from the synaptic cleft to the reuptake
//! store
#define SYNAPSE_CLEFT_RATE_TO_RE_UP_TAKE_STORE 300.0f

//! \brief  refractory period for ihc in seconds
#define IHC_REFRACTORY_PERIOD 7.5e-4

//! *********************** random named constants *****************//

//! \brief argument to avoid callback api
#define IHCAN_FILLER_ARG 0

//! \brief word to byte conversion
#define WORD_TO_BYTE_CONVERSION 4

//! \brief seed size
#define N_SEED_ELEMENTS 4

//! \brief data spec regions
typedef enum regions {
    SYSTEM = 0,
    PARAMS = 1,
    CILIA_PARAMS = 2,
    INNER_EAR_PARAMS = 3,
    DT_BASED_PARAMS = 4,
    RANDOM_SEEDS = 5,
    NEURON_RECORDING = 6,
    SDRAM_EDGE_ADDRESS = 7,
    PROFILER = 8,
    PROVENANCE = 9,
} regions;

//! \brief recording regions
typedef enum recording_regions {
    SPIKE_RECORDING_REGION_ID = 0,
    SPIKE_PROBABILITY_REGION_ID = 1,
} recording_regions;

typedef enum priorities {
    MC_PACKET_PRIORITY = -1, USER = 0, TIMER = 0, SDP_PRIORITY = 1, DMA = 0
} priorities;

//! \brief provenance data items locations
typedef enum extra_provenance_data_region_entries{
    N_SIMULATION_TICKS = 0,
    SEG_INDEX = 1,
    DATA_READ_COUNT = 2,
    DATA_WRITE_COUNT_SPIKES = 3,
    DATA_WRITE_COUNT_SPIKE_PROB = 4,
    MC_RX_COUNT = 5,
    MC_TRANSMISSION_COUNT = 6
} extra_provenance_data_region_entries;

//! \brief the data items in sdram from the params region
typedef struct parameters_struct{
    int resampling_factor;
    int number_fibres;
    int seg_size;
    int number_of_sdram_buffers;
    int num_lsr;
    int num_msr;
    int num_hsr;
    uint my_key;
} parameters_struct;

//! \brief elements based off dt
typedef struct dt_params_struct{
    float dt;
    float z;
}dt_params_struct;

//! \brief sdram edge data from sdram
typedef struct sdram_out_buffer_param{
    double* sdram_base_address;
} sdram_out_buffer_param;

//! \brief synapse params
typedef struct synapse_params_struct{
    float refrac_period;
    float ydt;
    float xdt;
    float ldt;
    float rdt;
} synapse_params_struct;

//! \brief cilia constants struct params
typedef struct cilia_constants_struct{
    float recips0;
    float recips1;
} cilia_constants_struct;

//! \brief inner ear params
typedef struct inner_ear_param_struct{
    float an_cleft_lsr;
    float an_cleft_msr;
    float an_cleft_hsr;
    float an_avail_lsr;
    float an_avail_msr;
    float an_avail_hsr;
    float an_repro_lsr;
    float an_repro_msr;
    float an_repro_hsr;
    float ihcv;
    float m_ica_curr;
    float ekp;
    float ca_curr_lsr;
    float ca_curr_msr;
    float ca_curr_hsr;
    float r_max_recip;
} inner_ear_param_struct;

#endif /* IHC_AN_H_ */
