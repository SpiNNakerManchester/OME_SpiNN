/*
 ============================================================================
 Name        : SpiNNakEar_OME.c
 Author      : Robert James
 Version     : 1.0
 Description : Outer and middle ear model for use in SpiNNakEar system
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdfix.h>
#include "OME_SpiNN.h"
#include "spin1_api.h"
#include "math.h"
#include "complex.h"
#include "random.h"
#include "stdfix-exp.h"
#include "log.h"
#include <data_specification.h>
#include <profiler.h>
#include <profile_tags.h>
#include <debug.h>

//#define PROFILE

//=========GLOBAL VARIABLES============//
REAL m_pi;
REAL fs;
REAL dt;
REAL max_rate;
uint chipID;
uint test_DMA;
uint seg_index;
uint read_switch;
uint write_switch;
uint processing;
uint index_x;
uint index_y;
uint timer_tick_period;
uint final_ack;

bool app_complete=false;

uint_float_union MC_union;

REAL concha_l;
REAL concha_h;
REAL conchaG;
REAL ear_canal_l;
REAL ear_canal_h;
REAL ear_canal_g;
REAL stapes_h;
REAL stapes_l;
REAL stapes_scalar;
REAL ar_tau;
REAL ar_delay;
REAL ar_rate_threshold;
REAL rate_to_attenuation_factor;
REAL bf_length;

REAL concha_q;
REAL concha_j;
REAL concha_k;
REAL concha_l;

REAL concha_gain_scalar;
REAL recip_concha_filter_a0;
REAL ear_canal_q;
REAL ear_canal_j;
REAL ear_canal_k;
REAL ear_canal_l;
REAL ear_canal_gain_scalar;
REAL recip_ear_canal_filter_a0;
REAL a_ratt;
REAL w_n;
REAL stapes_hp_order;
REAL sf;
REAL stapes_lp_b;
REAL stapes_tau;
REAL past_stapes_disp;

REAL concha_filter_b[3];
REAL concha_filter_a[3];
REAL ear_canal_filter_b[3];
REAL ear_canal_filter_a[3];
REAL stapes_hp_b[3];
REAL stapes_hp_a[3];
REAL stapes_lp_a[2];
REAL past_input[2];
REAL past_concha[2];
REAL past_ear_canal_input[2];
REAL past_ear_canal[2];
REAL past_stapes_input[2];
REAL past_stapes[2];

int start_count_process;
int end_count_process;
int start_count_read;
int end_count_read;
int start_count_write;
int end_count_write;
int start_count_full;
int end_count_full;

uint sync_count = 0;
uint read_ticks = 0;
bool first_tick = true;

REAL *dtcm_buffer_a;
REAL *dtcm_buffer_b;
uint *dtcm_buffer_x;

REAL *sdram_in_buffer;
REAL *sdramout_buffer;
REAL *profile_buffer;

//data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    RECORDING,
    PROFILER
}regions;

// The general parameters
parameters_struct parameters;

// The mask which indicates the sequence number
uint sequence_mask;

// the core ID given by the placement software
uint placement_coreID;

// MC key for data transmission
uint key;

//ready to send key used for setup comms.
uint r2s_key;

//time scale factor
uint time_scale;

//number of connected DRNL instances
uint num_drnls;

//number of connected acknowledgment tree instances
uint num_macks;

//model fs
uint sampling_frequency;

//application initialisation
bool app_init(void)
{
    //pi reference for filter constant calcs
    m_pi = acos(-1.0);
	seg_index = 0;
	read_switch = 0;
	write_switch = 0;

	log_info("[core %d] -----------------------\n", spin1_get_core_id());
	log_info("[core %d] starting simulation\n", spin1_get_core_id());
	//obtain data spec
	address_t data_address = data_specification_get_data_address();

    // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM, data_address),
            APPLICATION_NAME_HASH, NULL, NULL, NULL, 1, 0)) {
        return false;
    }

    address_t parameters_address =
        data_specification_get_region(PARAMS, data_address);
    spin1_memcpy(
        &parameters, data_specification_get_region(PARAMS, data_address),
         sizeof(parameters_struct));

    // Get the size of the data in words
    total_ticks = parameters.total_ticks;
    log_info("total_ticks=%d", parameters.total_ticks);

    // Get a pointer to the input data buffer
    sdram_in_buffer = (REAL *) &(params_enum[DATA + 7]);
    log_info("sdram_in_buffer=0x%08x", (uint)sdram_in_buffer);
    
    //obtain this core ID from the host placement perspective
    placement_coreID = params.COREID;

    // Get the key to send the data with
    key = params.KEY;
    log_info("OME-->DRNL key=%d\n", key);

    r2s_key = params.R2S_KEY;
    log_info("r2s key=%d", r2s_key);

    time_scale = params.TS;
    log_info("time_scale=%d",time_scale);

    //Get number of child DRNL vertices
    num_drnls = params.NUM_DRNL;
    log_info("num drnls=%d\n", num_drnls);

    //get number of multi-cast acknowledges
    num_macks = params.NUM_MACK;
    log_info("num macks=%d\n", num_macks);

    //Get sampling frequency
    sampling_frequency = params.FS;

    fs= (double)sampling_frequency;
    dt=(1.0 / fs);
    timer_tick_period = (uint)(1e6 * ((REAL)SEGSIZE / fs) * time_scale);

    log_info("timer period=%d", (uint)timer_tick_period);
    log_info("fs=%d", sampling_frequency);
    final_ack = 0;

	// Allocate buffers
	//input double buffers
	dtcm_buffer_a = (REAL *) sark_alloc(SEGSIZE, sizeof(REAL));
	dtcm_buffer_b = (REAL *) sark_alloc(SEGSIZE, sizeof(REAL));
	dtcm_buffer_x = (uint *) sark_alloc(SEGSIZE, sizeof(uint));

	if(dtcm_buffer_a == NULL || dtcm_buffer_b == NULL
	        || dtcm_buffer_x== NULL || sdram_in_buffer == NULL) {
	        
		test_DMA = FALSE;
		log_info(
		    "[core %d] error - cannot allocate buffer\n",
		    spin1_get_core_id());
		return false;
	}
	else {
        test_DMA = TRUE;
        // initialise buffers
        for (uint i = 0; i < SEGSIZE; i++) {
            dtcm_buffer_a[i] = 0;
            dtcm_buffer_b[i] = 0;
        }
        for (uint i = 0; i < SEGSIZE; i++) {
            dtcm_buffer_x[i] = 0;
        }

        log_info(
            "[core %d] data spec output buffer tag= %d\n",
             spin1_get_core_id (), (uint) placement_coreID);

        //============MODEL INITIALISATION================//
        bf_length = params.NUM_BFS;
        concha_l = 1500.0;
        concha_h = 3000.0;
        conchaG = 5.0;
        ear_canal_l = 3000.0;
        ear_canal_h = 3800.0;
        ear_canal_g = 5.0;
        stapes_h = 700.0;
        stapes_l = 10.0;
        stapes_scalar = 5e-7;
        ar_tau = 0.2;
        ar_delay = 0.0085;
        ar_rate_threshold = 100.0;
        rate_to_attenuation_factor = 0.1 / bf_length;

        concha_q = (REAL)m_pi * (REAL)dt * (REAL)(concha_h - concha_l);
        concha_j = 1.0 / (1.0 + (1.0 / tan(concha_q)));
        concha_k = 
            (2.0 * cos((REAL)m_pi * (REAL)dt * (REAL)(concha_h + concha_l))) / 
            ((1.0 + tan(concha_q)) * cos(concha_q));
        concha_l = (tan(concha_q) - 1.0) / (tan(concha_q) + 1.0);
        concha_gain_scalar = pow(10.0, conchaG / 20.0);

        concha_filter_b[0] = concha_j;
        concha_filter_b[1] = 0.0;
        concha_filter_b[2] = -1.0 * concha_j;
        concha_filter_a[0] = 1.0;
        concha_filter_a[1] = -1.0 * concha_k;
        concha_filter_a[2] = -1.0 * concha_l;
        recip_concha_filter_a0 = 1.0 / concha_filter_a[0];

        ear_canal_q = m_pi * dt * (ear_canal_h - ear_canal_l);
        ear_canal_j = 1.0/(1.0+ (1.0/tan(ear_canal_q)));
        ear_canal_k = (
            2.0 * cos(m_pi * dt * (ear_canal_h + ear_canal_l))) / (
                (1.0 + tan(ear_canal_q)) * cos(ear_canal_q));
        ear_canal_l = (tan(ear_canal_q) - 1.0) / (tan(ear_canal_q) + 1.0);
        ear_canal_gain_scalar = pow(10.0, ear_canal_g / 20.0);

        ear_canal_filter_b[0] = ear_canal_j;
        ear_canal_filter_b[1] = 0.0;
        ear_canal_filter_b[2] = -ear_canal_j;
        ear_canal_filter_a[0] = 1.0;
        ear_canal_filter_a[1] = -ear_canal_k;
        ear_canal_filter_a[2] = -ear_canal_l;
        recip_ear_canal_filter_a0 = 1.0/ear_canal_filter_a[0];

        //stapes filter coeffs pre generated due to butterworth calc code 
        // overflow
        stapes_hp_b[0] = params.SHB1;
        stapes_hp_b[1] = params.SHB2;
        stapes_hp_b[2] = params.SHB3;
        stapes_hp_a[0] = params.SHA1;
        stapes_hp_a[1] = params.SHA2;
        stapes_hp_a[2] = params.SHA3;

        stapes_tau = 1.0 / (2 * (REAL)m_pi * stapes_l);
        stapes_lp_a[0] = 1.0;
        stapes_lp_a[1] = dt / stapes_tau -1.0;
        stapes_lp_b = 1.0 + stapes_lp_a[1];

        //--Initialise recurring variables--/
        past_input[0] = 0.0;
        past_input[1] = 0.0;
        past_concha[0] = 0.0;
        past_concha[1] = 0.0;

        past_ear_canal_input[0] = 0.0;
        past_ear_canal_input[1] = 0.0;
        past_ear_canal[0] = 0.0;
        past_ear_canal[1] = 0.0;

        a_ratt = 1.0; //TODO: this should be determined by spiking input

        past_stapes_input[0] = 0.0;
        past_stapes_input[1] = 0.0;
        past_stapes[0] = 0.0;
        past_stapes[1] = 0.0;

        past_stapes_disp = 0.0;

    #ifdef PROFILE
        // Setup profiler
        profiler_init(
            data_specification_get_region(1, data_address));
    #endif
        log_info("init complete");
	}
    return true;
}

void app_end(uint null_a, uint null_b) {
    app_complete = true;
    log_info("All data has been sent seg_index=%d", seg_index);
    app_done();
    simulation_ready_to_read();
    log_info("spinn_exit %\n");
}
//DMA read
void data_read(uint null_a, uint null_b) {
	REAL *dtcm_buffer_in;
	//read from DMA and copy into DTCM
	if(read_ticks < parameters.total_ticks && test_DMA == TRUE) {
	    #ifdef PROFILE
            if (seg_index==0) {
                profiler_write_entry_disable_irq_fiq(
                    PROFILER_ENTER | PROFILER_DMA_READ);
            }
        #endif
	    read_ticks++;
		//assign receive buffer
		if (!read_switch) {
			dtcm_buffer_in=dtcm_buffer_a;
			read_switch=1;
		}
		else {
			dtcm_buffer_in=dtcm_buffer_b;
			read_switch=0;
		}

		spin1_dma_transfer(
		    DMA_READ, &sdram_in_buffer[seg_index * SEGSIZE], dtcm_buffer_in,
		    DMA_READ, SEGSIZE * sizeof(REAL));
	}
   	// stop if desired number of ticks reached
    else if (read_ticks >= parameters.total_ticks && !app_complete) {
        spin1_schedule_callback(app_end, NULL, NULL, 2);
    }
}

void process_chan(REAL *in_buffer)
{
    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_ENTER | PROFILER_TIMER);
    #endif

	uint i;
	uint j;
	uint k;
	uint si = 0;
	REAL concha;
	REAL ear_canal_input;
	REAL ear_canal_res;
	REAL ear_canal_output;
	REAL ar_output;
	REAL stapes_velocity;
	REAL stapes_displacement;
	REAL filter_1;
	REAL diff;
	REAL sub;

    #ifdef PRINT
        log_info(
            "[core %d] segment %d (offset=%d) starting processing\n",
            spin1_get_core_id(), seg_index, segment_offset);
    #endif

	for(i=0; i < SEGSIZE; i++){
		//concha
        filter_1 = (
            concha_filter_b[0] * in_buffer[i] + concha_filter_b[1]
            * past_input[0] + concha_filter_b[2] * past_input[1]);
        concha = (
            concha_filter_a[0] * filter_1 - concha_filter_a[1] * past_concha[0]
            - concha_filter_a[2] * past_concha[1]);

		//update vars
		past_input[1] = past_input[0];
		past_input[0] = in_buffer[i];

		past_concha[1] = past_concha[0];
		past_concha[0] = concha;

		ear_canal_input = concha_gain_scalar * concha + in_buffer[i];

		//ear canal
		filter_1 =
		    ear_canal_filter_b[0] * ear_canal_input + ear_canal_filter_b[1]
		    * past_ear_canal_input[0] + ear_canal_filter_b[2]
		    * past_ear_canal_input[1];

        ear_canal_res =
            ear_canal_filter_a[0] * filter_1 - ear_canal_filter_a[1]
            * past_ear_canal[0] - ear_canal_filter_a[2] * past_ear_canal[1];

		//update vars
		past_ear_canal_input[1] = past_ear_canal_input[0];
		past_ear_canal_input[0] = ear_canal_input;

		past_ear_canal[1] = past_ear_canal[0];
		past_ear_canal[0] = ear_canal_res;

		ear_canal_output = 
		    ear_canal_gain_scalar * ear_canal_res + ear_canal_input;

		//Acoustic Reflex
		ar_output= a_ratt * stapes_scalar * ear_canal_output;
        filter_1 =
            stapes_hp_b[0] * ar_output + stapes_hp_b[1] * past_stapes_input[0]
			+ stapes_hp_b[2] * past_stapes_input[1];
        stapes_velocity =
            stapes_hp_a[0] * filter_1 - stapes_hp_a[1] * past_stapes[0]
            - stapes_hp_a[2] * past_stapes[1];

		//update vars
		past_stapes_input[1] = past_stapes_input[0];
		past_stapes_input[0] = ar_output;
		past_stapes[1] = past_stapes[0];
		past_stapes[0] = stapes_velocity;

		//stapes displacement
		filter_1 = stapes_lp_b * stapes_velocity;
		sub = stapes_lp_a[1] * past_stapes_disp;
        stapes_displacement = filter_1 - sub;
		//update vars
		past_stapes_disp = stapes_displacement;

        //assign output to float/uint union
		MC_union.f = stapes_displacement;

		//transmit uint output as MC with payload to all DRNLs
        spin1_send_mc_packet(key, MC_union.u, WITH_PAYLOAD);
	}

    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
    #endif
}

void transfer_handler(uint tid, uint ttag) {
	if (ttag==DMA_READ) {
	    //increment segment index
		seg_index++;

		//choose current available buffers
		if(!read_switch) {
			process_chan(dtcm_buffer_b);
		}
		else {
			process_chan(dtcm_buffer_a);
		}
	}
	else {
		log_info("[core %d] invalid %d DMA tag!\n", spin1_get_core_id(), ttag);
	}
}

void sync_check(uint mc_key, uint null) {
    sync_count++;
    log_info(
        "ack mc packet received from key=%d sync_count=%d seg_index=%d\n",
        mc_key-2, sync_count, seg_index);
}

void app_done ()
{
    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(
            PROFILER_EXIT | PROFILER_DMA_READ);
        profiler_finalise();
    #endif
	log_info("b0:%k", (accum)stapes_hp_b[0]);
    log_info("b1:%k", (accum)stapes_hp_b[1]);
    log_info("b2:%k", (accum)stapes_hp_b[2]);
	log_info("a0:%k", (accum)stapes_hp_a[0]);
	log_info("a1:%k", (accum)stapes_hp_a[1]);
    log_info("a2:%k", (accum)stapes_hp_a[2]);

  // report simulation time
  log_info(
    "[core %d] simulation lasted %d ticks\n",
     spin1_get_core_id(), spin1_get_simulation_time());

  // say goodbye
  log_info("[core %d] stopping simulation\n", spin1_get_core_id());
}

void c_main() {
    // Get core and chip IDs
    chipID = spin1_get_chip_id ();

    if (app_init()) {
        //set timer tick
        spin1_set_timer_tick(timer_tick_period);
        //setup callbacks
        //process channel once data input has been read to DTCM
        spin1_callback_on(DMA_TRANSFER_DONE, transfer_handler, 1);
        //reads from DMA to DTCM every tick
        spin1_callback_on(TIMER_TICK, data_read, 0);
        //start/end of simulation synchronisation callback
        simulation_run();
    }
}

