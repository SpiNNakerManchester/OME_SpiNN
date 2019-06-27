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

#define REAL double
#define REAL_CONST(x) x


#define MAX_CHIPX 1
#define MAX_CHIPY 1
#define MAX_COREID 16
#define SEED_SEL_SIZE 1024

// random named constants
#define BITS_IN_WORD 32

// sampling freq
#define SAMPLING_FREQUENCY 44100.0

// moc stuff
#define MOC_BUFFER_SIZE 10
#define MOC_DELAY 1
#define MOC_DELAY_ARRAY_LEN 500

#define MAX_SIGNAL_S 1

// timer 2 stuff
#define TIMER2_CONF        0x82
#define TIMER2_LOAD        0

//?????????
#define RATE_TO_ATTENTUATION_FACTOR 6e2

//linear pathway
#define LIN_GAIN 200.0
#define A 30e4
#define C 0.25k
#define CTBM 1e-9 * pow(10.0, 32.0 / 20.0)
#define RECIP_CTBM 1.0 / CTBM
#define DISP_THRESH CTBM / A

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

typedef union
{
	uint32_t u;
	float f;
} uint_float_union;

typedef struct key_mask_table {
    uint32_t key;
    uint32_t mask;
    uint32_t conn_index;
} key_mask_table_entry;

typedef struct last_neuron_info_t{
    uint32_t e_index;
    uint32_t w_index;
    uint32_t id_shift;
} last_neuron_info_t;

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

typedef struct sdram_out_buffer_param{
    address_t sdram_base_address;
    uint32_t sdram_edge_size;
} sdram_out_buffer_param;



#endif /* IHC_AN_H_ */
