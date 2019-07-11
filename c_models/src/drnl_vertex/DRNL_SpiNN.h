#ifndef DRNL_spiNN_h_
#define DRNL_spiNN_h_

#include "spin1_api.h"

//data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    FILTER_PARAMS,
    RECORDING,
    PROFILER,
    SDRAM_EDGE_ADDRESS,
    SYNAPSE_PARAMS,
    POPULATION_TABLE,
    SYNAPTIC_MATRIX,
    SYNAPSE_DYNAMICS,
    CONNECTOR_BUILDER,
    DIRECT_MATRIX
} regions;

typedef enum priorities {
    SPIKE_CHECK_PRIORITY = 1, PROCESS_HANDLER_PRIORITY = 1,
    APP_END_PRIORITY = 2, DATA_WRITE_PRIORITY = 0, COUNT_TICKS_PRIORITY = 0,
    MC_PACKET_PRIORITY = -1, SDP_PRIORITY = 1, DMA_TRANSFER_DONE_PRIORITY = 0
} priorities;

typedef enum synapse_type_indices {
    EXCITATORY = 0, INHIBITORY = 1
} synapse_type_indices;

// recording region
#define MOC_RECORDING_REGION 0

// random named constants
#define BITS_IN_WORD 32

// argument to avoid callback api
#define DRNL_FILLER_ARG 1

// moc stuff
#define MOC_BUFFER_SIZE 10
#define INCOMING_SPIKE_BUFFER_SIZE 256

//linear pathway. unknown what each of these are for.
#define LIN_GAIN 200.0
#define A 30e4
#define C 0.25k

// max of 2 numbers
#define max(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

// absolute value
#define absolute_value(x) (((x)<0) ? -(x) : (x))

// which sign it is
#define find_sign(x) (((x)<0) ? -(1.0) : (1.0))

// union from float and uint32. for transmission and reception
typedef union {
	uint u;
	float f;
} uint_float_union;

// params from the parameter region in sdram
typedef struct parameters_struct{
   int data_size;
   uint key;
   uint ome_data_key;
   int is_recording;
   int seq_size;
   int n_buffers_in_sdram;
   int n_synapse_types;
   uint moc_resample_factor;
   double moc_dec_1;
   double moc_dec_2;
   double moc_dec_3;
   double moc_factor_1;
   double ctbm;
   double receip_ctbm;
   double disp_thresh;
} parameters_struct;

// params from the filter params region in sdram
typedef struct filter_params_struct{
   double la1;
   double la2;
   double lb0;
   double lb1;
   double nla1;
   double nla2;
   double nlb0;
   double nlb1;
} filter_params_struct;

// sdram edge data from sdram
typedef struct sdram_out_buffer_param{
    double* sdram_base_address;
    int sdram_edge_size;
} sdram_out_buffer_param;

#endif /* DRNL_spiNN_h_ */
