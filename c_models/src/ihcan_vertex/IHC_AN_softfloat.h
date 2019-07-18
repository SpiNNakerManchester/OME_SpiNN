/*
 * IHC_AN.h
 *
 *  Created on: 1 Nov 2016
 *      Author: rjames
 */

#ifndef IHC_AN_softfloat_H_
#define IHC_AN_softfloat_H_

#define RDM_MAX UINT32_MAX + 1.0f
#define R_MAX_RECIP 1.0f / RDM_MAX

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
#define CILIA_C 0.3f

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


//! *********************** random named constants *****************//

// \brief argument to avoid callback api
#define IHCAN_FILLER_ARG 0

//! \brief word to byte conversion
#define WORD_TO_BYTE_CONVERSION 4

//! \brief seed size
#define N_SEED_ELEMENTS 4

//! ***************************  random unnamed constants ***********//

#define RANDOM_1 5e-12f
#define RANDOM_2 20e-9
#define RANDOM_3 1.0 / 500e-6
#define RANDOM_4 4.0f
#define RANDOM_5 1.0 / 350e-6
#define RANDOM_6 1.0 / 200e-6
#define RANDOM_7 5e-5
#define RANDOM_8 150.0f
#define RANDOM_9 15.0f
#define RANDOM_10 300.0f
#define RANDOM_11 7.5e-4

//! \brief data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    CILIA_PARAMS,
    INNER_EAR_PARAMS,
    RANDOM_SEEDS,
    RECORDING,
    SDRAM_EDGE_ADDRESS,
    PROFILER,
    PROVENANCE,
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
    float dt;
    float dt_spikes;
    float dt_segment;
    float z;
} parameters_struct;

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
    float *m;
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
} inner_ear_param_struct;


#endif /* IHC_AN_H_ */
