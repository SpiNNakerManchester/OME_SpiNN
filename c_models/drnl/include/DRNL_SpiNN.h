/*
 * IHC_AN.h
 *
 *  Created on: 1 Nov 2016
 *      Author: rjames
 */

#ifndef IHC_AN_softfloat_H_
#define IHC_AN_softfloat_H_

#define REAL double //float
#define REAL_CONST(x) x//x##f


#define MAX_CHIPX 1//255
#define MAX_CHIPY 1//255
#define MAX_COREID 16
//#define SEED_SEL_SIZE (((MAX_CHIPX << 8) | MAX_CHIPY) << 8) | (MAX_COREID << 3)   
#define SEED_SEL_SIZE 1024

#define SEGSIZE 8//16//96//100
#define MOC_BUFFER_SIZE 10//min sim 10ms 1kHz MOC FS

#define SAMPLING_FREQUENCY 44100.
#define MOC_DELAY 1 //32 //3.2ms * 10 as timestep is 0.1ms
#define MOC_DELAY_ARRAY_LEN 500

#define MAX_SIGNAL_S 1

#define TIMER2_CONF        0x82
#define TIMER2_LOAD        0

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define ABS(x) (((x)<0) ? -(x) : (x))
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

typedef struct{
    uint32_t e_index;
    uint32_t w_index;
    uint32_t id_shift;
}last_neuron_info_t;

typedef struct {
   //uint32_t N_TICKS;
   uint32_t DATA_SIZE;
   uint32_t OMECOREID;
   uint32_t COREID;
   uint32_t OMEAPPID;
   uint32_t OME_KEY;
   uint32_t KEY;
   uint32_t NUM_IHCAN;
   uint32_t CF;
   uint32_t DELAY;
   uint32_t FS;
   uint32_t OME_DATA_KEY;
   uint32_t  IS_RECORDING;
   REAL L_A1;
   REAL L_A2;
   REAL L_B0;
   REAL L_B1;
   REAL NL_A1;
   REAL NL_A2;
   REAL NL_B0;
   REAL NL_B1;
   uint32_t MOC_CONN_LUT;
}parameters_struct;

#endif /* IHC_AN_H_ */
