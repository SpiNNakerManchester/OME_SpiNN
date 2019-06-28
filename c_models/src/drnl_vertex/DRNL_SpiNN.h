#ifndef IHC_AN_softfloat_H_
#define IHC_AN_softfloat_H_

//data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    FILTER_PARAMS,
    RECORDING,
    PROFILER,
    SDRAM_EDGE_ADDRESS
}regions;

typedef enum priorities {
    SPIKE_CHECK_PRIORITY = 1, PROCESS_HANDLER_PRIORITY = 1,
    APP_END_PRIORITY = 2, DATA_WRITE_PRIORITY = 0, COUNT_TICKS_PRIORITY = 0,
    MC_PACKET_PRIORITY = -1, SDP_PRIORITY = 1, DMA_TRANSFER_DONE_PRIORITY = 0
} priorities;

#define REAL double
#define REAL_CONST(x) x

// recording region
#define MOC_RECORDING_REGION 0

// random named constants
#define BITS_IN_WORD 32

// argument to avoid callback api
#define FILLER_ARG 0

// sampling freq
#define SAMPLING_FREQUENCY 44100.0

// converter constant
#define MOC_RESAMPLE_FACTOR_CONVERTER 1000.0

// moc stuff
#define MOC_BUFFER_SIZE 10
#define MOC_DELAY 1
#define MOC_DELAY_ARRAY_LEN 500

//?????????
#define RATE_TO_ATTENTUATION_FACTOR 6e2

//linear pathway. unknown what each of these are for.
#define LIN_GAIN 200.0
#define A 30e4
#define C 0.25k
#define CTBM 1e-9 * pow(10.0, 32.0 / 20.0)
#define RECIP_CTBM 1.0 / CTBM
#define DISP_THRESH CTBM / A

// tau stuff..... unknown of human names
#define MOC_TAU_0 0.055
#define MOC_TAU_1 0.4
#define MOC_TAU_2 1
#define MOC_TAU_WEIGHT 0.9

// max of 2 numbers
#define max(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

// absolute value
#define ABS(x) (((x)<0) ? -(x) : (x))

// which sign it is
#define SIGN(x) (((x)<0) ? -(1.0) : (1.0))

// union from float and uint32. for transmission and reception
typedef union {
	uint32_t u;
	float f;
} uint_float_union;

// table data
typedef struct key_mask_table {
    uint32_t key;
    uint32_t mask;
    uint32_t conn_index;
} key_mask_table_entry;

// last neuron ingo
typedef struct last_neuron_info_t{
    uint32_t e_index;
    uint32_t w_index;
    uint32_t id_shift;
} last_neuron_info_t;

// params from the parameter region in sdram
typedef struct parameters_struct{
   uint32_t data_size;
   uint32_t key;
   uint32_t fs;
   uint32_t ome_data_key;
   uint32_t is_recording;
   uint32_t seq_size;
   uint32_t n_buffers_in_sdram;
   uint32_t moc_conn_lut;
} parameters_struct;

// params from the filter params region in sdram
typedef struct filter_params_struct{
   REAL la1;
   REAL la2;
   REAL lb0;
   REAL lb1;
   REAL nla1;
   REAL nla2;
   REAL nlb0;
   REAL nlb1;
} filter_params_struct;

// sdram edge data from sdram
typedef struct sdram_out_buffer_param{
    REAL* sdram_base_address;
    uint32_t sdram_edge_size;
} sdram_out_buffer_param;

#endif /* IHC_AN_H_ */
