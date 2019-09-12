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
#include <simulation.h>
#include <profiler.h>
#include <debug.h>

//=========GLOBAL VARIABLES============//
uint seg_index = 0;
uint read_switch = 0;

//! \brief the conversion between uint32_t and floats
uint_float_union multicast_union;

//! \brief ??????????
REAL stapes_lp_b;

//! \brief ??????????
REAL past_stapes_disp = 0.0;

//! \brief filter for b
REAL concha_filter_b[3];

//! \brief filter for a
REAL concha_filter_a[3];

//! \brief  ear canal filters
REAL ear_canal_filter_b[3];

//! \brief ear cancel filters
REAL ear_canal_filter_a[3];

//! \brief ?????
REAL stapes_hp_b[3];

//! \brief ?????
REAL stapes_hp_a[3];

//! \brief ?????
REAL stapes_lp_a[2];

//! \brief previous inputs
REAL past_input[2];

//! \brief previous concha
REAL past_concha[2];

//! \brief past canal input
REAL past_ear_canal_input[2];

//! \brief past ear canal
REAL past_ear_canal[2];

//! \brief past ??????
REAL past_stapes_input[2];

//! \brief ?????????
REAL past_stapes[2];

// *************** SIM PARAMS **************** //

//! \brief what tick we're on
uint32_t read_ticks = UINT32_MAX;

//! \brief infinite run pointer
static uint32_t infinite_run;

//! \brief simulation timer tick (based on its time step)
uint32_t time_to_reach;

//! \brief dtcm buffers
REAL *dtcm_buffer_a;
REAL *dtcm_buffer_b;
uint *dtcm_buffer_x;

//! input data buffer
REAL *sdram_in_buffer;

// The sdram parameter structs
parameters_struct parameters;
filter_coeffs_struct filter_coeffs;
data_struct data;
concha_params_struct concha_params;

//! \brief stores provenance data
//! \param[in] provenance_region: the sdram location for the prov data.
void _store_provenance_data(address_t provenance_region) {
    log_debug("writing other provenance data");

    // store the data into the provenance data region
    filter_coeffs_struct *prov_struct =
        (filter_coeffs_struct*) provenance_region;
    prov_struct->shb1 = stapes_hp_b[0];
    prov_struct->shb2 = stapes_hp_b[1];
    prov_struct->shb3 = stapes_hp_b[2];
    prov_struct->sha1 = stapes_hp_a[0];
    prov_struct->sha2 = stapes_hp_a[1];
    prov_struct->sha3 = stapes_hp_a[2];


    log_info("b0:%F", stapes_hp_b[0]);
    log_info("b1:%F", stapes_hp_b[1]);
    log_info("b2:%F", stapes_hp_b[2]);
	log_info("a0:%F", stapes_hp_a[0]);
	log_info("a1:%F", stapes_hp_a[1]);
    log_info("a2:%F", stapes_hp_a[2]);

    log_debug("finished other provenance data");
}

//! \brief DMA read every timer tick to get input data.
//! \param[in] unused_a: forced by api
//! \param[in] unused_b: forced by api
//! \return none
void data_read(uint unused_a, uint unused_b) {
    use(unused_a);
    use(unused_b);

	REAL *dtcm_buffer_in;
	read_ticks++;

	// check if there's anything to do
	if (read_ticks >= parameters.total_ticks ||
            read_ticks >= time_to_reach) {

        simulation_handle_pause_resume(NULL);
        #ifdef PROFILE
            profiler_write_entry_disable_irq_fiq(
                PROFILER_EXIT | PROFILER_DMA_READ);
            profiler_finalise();
        #endif
        simulation_ready_to_read();
    } else {
        //read from DMA and copy into DTCM if there is still stuff to read
        log_debug(
            "read ticks %d, total ticks %d",
            read_ticks, parameters.total_ticks);

        #ifdef PROFILE
            if (seg_index == 0) {
                profiler_write_entry_disable_irq_fiq(
                    PROFILER_ENTER | PROFILER_DMA_READ);
            }
        #endif

        //assign receive buffer
        if (!read_switch) {
            dtcm_buffer_in = dtcm_buffer_a;
            read_switch = 1;
        }
        else {
            dtcm_buffer_in = dtcm_buffer_b;
            read_switch = 0;
        }

        // set off a dma
        spin1_dma_transfer(
            DMA_TAG, &sdram_in_buffer[seg_index * parameters.seg_size],
            dtcm_buffer_in, DMA_READ, parameters.seg_size * sizeof(REAL));
    }
}

//! \brief having completed a dma, read in seq size elements and process to
//! input for drnls and send to said verts.
//! \param[in] in_buffer: the buffer to read in from
//! \return None
void process_chan(REAL *in_buffer) {
    log_debug("done dma");
    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_ENTER | PROFILER_TIMER);
    #endif

	REAL concha;
	REAL ear_canal_input;
	REAL ear_canal_res;
	REAL ear_canal_output;
	REAL ar_output;
	REAL stapes_velocity;
	REAL stapes_displacement;
	REAL filter_1;
	REAL sub;

	for (int i = 0; i < parameters.seg_size; i++){
	    log_info("in buffer %d is %F", i, in_buffer[i]);
	}

	for (int i = 0; i < parameters.seg_size; i++){
		// concha
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

		ear_canal_input = (
		    concha_params.concha_gain_scalar * concha + in_buffer[i]);
		log_debug(" ear canal input = %F", ear_canal_input);

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

		ear_canal_output = (
		    concha_params.ear_canal_gain_scalar * ear_canal_res +
		    ear_canal_input);

		log_debug(" ear canal output = %F", ear_canal_output);

		//Acoustic Reflex
		ar_output = A_RATT * STAPES_SCALAR * ear_canal_output;
		log_debug(" ar_output = %F", ar_output);

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
        multicast_union.f = stapes_displacement;

		//transmit uint output as MC with payload to all DRNLs
		log_debug(
		    "i=%d %u",
		    ((seg_index - 1) * parameters.seg_size) + i, multicast_union.u);
        spin1_send_mc_packet(parameters.key, multicast_union.u, WITH_PAYLOAD);
	}

    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
    #endif
}

//! \brief handles dma complete.
//! \param[in] unused_a: forced by api
//! \param[in] unused_b: forced by api
//! \return none
void transfer_handler(uint unused0, uint unused1) {
    use(unused0);
    use(unused1);

    //increment segment index
    seg_index++;

    //choose current available buffers
    if (!read_switch) {
        process_chan(dtcm_buffer_b);
    }
    else {
        process_chan(dtcm_buffer_a);
    }
}


//! \brief application initialisation
//! \return bool saying if the init was successful or not
bool app_init(uint32_t *timer_period)
{
	//obtain data spec
	data_specification_metadata_t *data_address =
	    data_specification_get_data_address();

    // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM, data_address),
            APPLICATION_NAME_HASH, timer_period, &time_to_reach,
            &infinite_run, &read_ticks,
            SDP_PRIORITY, DMA_TRANSFER_DONE_PRIORITY)) {
        return false;
    }

     // sort out provenance data
    simulation_set_provenance_function(
        _store_provenance_data,
        data_specification_get_region(PROVENANCE, data_address));

    // get params from sdram
    spin1_memcpy(
        &parameters, data_specification_get_region(PARAMS, data_address),
         sizeof(parameters_struct));

    log_debug("dt is %F", parameters.dt);
    log_debug("total_ticks=%d", parameters.total_ticks);
    log_debug("OME-->DRNL key=%d\n", parameters.key);

    // Get a pointer to the input data buffer
    sdram_in_buffer =
        (REAL *) data_specification_get_region(DATA, data_address);

	// Allocate buffers
	//input double buffers
	dtcm_buffer_a = (REAL *) sark_alloc(parameters.seg_size, sizeof(REAL));
	dtcm_buffer_b = (REAL *) sark_alloc(parameters.seg_size, sizeof(REAL));
	dtcm_buffer_x = (uint *) sark_alloc(parameters.seg_size, sizeof(uint));

	if (dtcm_buffer_a == NULL || dtcm_buffer_b == NULL
	        || dtcm_buffer_x== NULL || sdram_in_buffer == NULL) {
		log_error("error - cannot allocate buffer\n");
		return false;
	}

    // initialise buffers
    for (int i = 0; i < parameters.seg_size; i++) {
        dtcm_buffer_a[i] = 0;
        dtcm_buffer_b[i] = 0;
    }
    for (int i = 0; i < parameters.seg_size; i++) {
        dtcm_buffer_x[i] = 0;
    }

    spin1_memcpy(
        &concha_params,
        data_specification_get_region(CONCHA_PARAMS, data_address),
        sizeof(concha_params_struct));

    log_info(
        "concha gain scalar = %k, ear canal gain scalar = %k",
        (accum) concha_params.concha_gain_scalar,
        (accum) concha_params.ear_canal_gain_scalar);


    log_info(
        "concha gain scalar = %F, ear canal gain scalar = %F",
        concha_params.concha_gain_scalar, concha_params.ear_canal_gain_scalar);

    // ear speicfics
    double m_pi = acos(-1.0);
    REAL concha_q = m_pi * parameters.dt * (REAL)(CONCHA_H - CONCHA_1);
    REAL concha_j = 1.0 / (1.0 + (1.0 / tan(concha_q)));

    log_info("concha_j %k", (accum) concha_j);

    log_debug(" concha Q %F", concha_q);
    log_debug("pi is %F", m_pi);

    REAL concha_k =
        (2.0 * cos(m_pi * parameters.dt * (REAL)(CONCHA_H + CONCHA_1))) /
        ((1.0 + tan(concha_q)) * cos(concha_q));
    REAL concha_l = (tan(concha_q) - 1.0) / (tan(concha_q) + 1.0);

    concha_filter_b[0] = concha_j;
    concha_filter_b[1] = 0.0;
    concha_filter_b[2] = -1.0 * concha_j;
    concha_filter_a[0] = 1.0;
    concha_filter_a[1] = -1.0 * concha_k;
    concha_filter_a[2] = -1.0 * concha_l;

    log_info(
        "filter b0 %F, b1 %F, b2 %F, a0 %F, a1 %F, a2 %F",
        concha_filter_b[0], concha_filter_b[1], concha_filter_b[2],
        concha_filter_a[0], concha_filter_a[1], concha_filter_a[2]);

    log_info(
        "filter b0 %k, b1 %k, b2 %k, a0 %k, a1 %k, a2 %k",
        (accum)concha_filter_b[0], (accum)concha_filter_b[1], (accum)concha_filter_b[2],
        (accum)concha_filter_a[0], (accum)concha_filter_a[1], (accum)concha_filter_a[2]);

    REAL ear_canal_q = m_pi * parameters.dt * (EAR_CANAL_H - EAR_CANAL_L);
    REAL ear_canal_j = 1.0 / (1.0 + (1.0 / tan(ear_canal_q)));
    REAL ear_canal_k =
        (2.0 * cos(m_pi * parameters.dt * (EAR_CANAL_H + EAR_CANAL_L))) / (
            (1.0 + tan(ear_canal_q)) * cos(ear_canal_q));
    REAL ear_canal_l = (tan(ear_canal_q) - 1.0) / (tan(ear_canal_q) + 1.0);

    ear_canal_filter_b[0] = ear_canal_j;
    ear_canal_filter_b[1] = 0.0;
    ear_canal_filter_b[2] = -ear_canal_j;
    ear_canal_filter_a[0] = 1.0;
    ear_canal_filter_a[1] = -ear_canal_k;
    ear_canal_filter_a[2] = -ear_canal_l;

    //stapes filter coeffs pre generated due to butterworth calc code
    // overflow
    spin1_memcpy(
        &filter_coeffs,
        data_specification_get_region(FILTER_COEFFS, data_address),
        sizeof(filter_coeffs_struct));

    stapes_hp_b[0] = filter_coeffs.shb1;
    stapes_hp_b[1] = filter_coeffs.shb2;
    stapes_hp_b[2] = filter_coeffs.shb3;
    stapes_hp_a[0] = filter_coeffs.sha1;
    stapes_hp_a[1] = filter_coeffs.sha2;
    stapes_hp_a[2] = filter_coeffs.sha3;

    log_info(
        "shb1 %F, shb2 %F shb3 %F, sha1 %F, sha2 %F sha3 %F",
        stapes_hp_b[0], stapes_hp_b[1], stapes_hp_b[2], stapes_hp_a[0],
        stapes_hp_a[1], stapes_hp_a[2]);

    REAL stapes_tau = 1.0 / (2 * m_pi * STAPES_1);
    stapes_lp_a[0] = 1.0;
    stapes_lp_a[1] = parameters.dt / stapes_tau -1.0;
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

    past_stapes_input[0] = 0.0;
    past_stapes_input[1] = 0.0;
    past_stapes[0] = 0.0;
    past_stapes[1] = 0.0;

    #ifdef PROFILE
        // Setup profiler
        profiler_init(
            data_specification_get_region(1, data_address));
    #endif

    // successful init
    log_info("init complete");
    return true;
}


//! \brief entrance method
void c_main() {

    uint32_t timer_period;
    if (app_init(&timer_period)) {

        //set timer tick
        spin1_set_timer_tick(timer_period);

        //setup callbacks
        //process channel once data input has been read to DTCM
        simulation_dma_transfer_done_callback_on(DMA_TAG, transfer_handler);
        //reads from DMA to DTCM every tick
        spin1_callback_on(TIMER_TICK, data_read, TIMER_TICK_PRIORITY);
        //start/end of simulation synchronisation callback
        simulation_run();
    }
}
