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

//! Inner Hair Cell + Auditory Nerve model for use in SpiNNakEar  system
#include "IHC_AN_softfloat.h"
#include "spin1_api.h"
#include "random.h"
#include "stdfix-exp.h"
#include "log.h"
#include "bit_field.h"
#include "neuron/neuron_recording.h"
#include <data_specification.h>
#include <profiler.h>
#include <simulation.h>
#include <debug.h>

//****************** provenance data ***************************//

//! \brief how many spikes sent
int spike_count = 0;

//! \brief how many reads done
int data_read_count = 0;

//! \brief how many writes done for spikes
int data_write_count_spikes = 0;

//! \brief how many writes done for spikes prob
int data_write_count_spikes_prob = 0;

//! \brief how many mc packets received
int mc_rx_count = 0;

//! \brief state variable for which sdram buffer to read
int seg_index = 0;

// ****************** globals ******************************//

//! \brief sim ticks
static uint32_t simulation_ticks = 0;

//! \brief infinite run pointer
static uint32_t infinite_run;

//! \brief time / ticks done
uint32_t time;

//! \brief simulation timer tick (based on its time step)
uint32_t time_pointer;

//********************* switch **************//

//! \brief bool state for switching dtcm input buffers
uint read_switch = 0;

//! \random number generator seed
mars_kiss64_seed_t local_seed;

//! \brief refrac
uint *refrac;

//**************** buffers **********************//
//! \brief input buffers
double *dtcm_buffer_a;
double *dtcm_buffer_b;

//! \brief sdram edge buffer
double *sdram_in_buffer;

//************************* cilia **************** //

//! \brief ???????
double cilia_filter_b2;

//! \brief ?????????
double cilia_filter_a1;

//! \brief ??????????
float cilia_dt_cap;

//! \brief ??????????
float dt_tau_m;

//! ********************** param structs ***************** //

//! \brief struct holding params from the param region
parameters_struct parameters;

//! \brief struct holding params determined from built in constants
synapse_params_struct synapse_params;

//! \brief cilia params
cilia_constants_struct cilia_params;

//! \brief inner ear params
inner_ear_param_struct inner_ear_params;

//! \brief dt based params
dt_params_struct dt_params;

//! *********************** recurring values ******************//

//! \brief ????
double past_cilia_disp = 0.0;

//! \brief ??????
float ihcv_now;

//! \brief ????????
float m_ica_curr;

//! \brief ?????????
float dt_spikes;

//! \brief recording flags
uint32_t recording_flags;

//! ****************************** arrays ******************//

//! \brief ????????
float *ca_curr;

//! \brief ??????????
float *an_cleft;

//! \brief ???????????
float *an_avail;

//! \brief ?????????
float *an_repro;

//! \brief ?????????
float *rec_tau_ca;

//! \brief ?????????
float *g_max_ca;

//! \brief ????????
float *ca_th;

//! \brief ????????
uint *refact;

//! \brief ?????????
float *synapse_m;

//! \brief tracker for when recording finishes putting spike data into sdram.
//! Switches write buffers and handles profiling if needed
void record_finished_spikes(void) {
    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
    #endif
    data_write_count_spikes++;
}

//! \brief tracker for when recording finishes putting spike prob data into
//! sdram. Switches write buffers and handles profiling if needed
void record_finished_spikes_prob(void) {
    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
    #endif
    data_write_count_spikes_prob++;
}

// processes multicast packets
//! \param[in] mc_key: the multicast key
//! \param[in] payload: the payload of the MC packet
//! \return None
void data_read(uint mc_key, uint payload) {

    use(mc_key);
    use(payload);

    // measure time between each call of this function (should approximate
    // the global clock in OME)
    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(
            PROFILER_ENTER | PROFILER_TIMER);
    #endif

    double *dtcm_buffer_in;

    //read from DMA and copy into DTCM
    //assign receive buffer
    if (!read_switch) {
        dtcm_buffer_in = dtcm_buffer_a;
        read_switch = 1;
    } else {
        dtcm_buffer_in = dtcm_buffer_b;
        read_switch = 0;
    }
    spin1_dma_transfer(
        DMA_READ,
        &sdram_in_buffer[
            (seg_index & (parameters.number_of_sdram_buffers - 1)) *
            parameters.seg_size],
        dtcm_buffer_in, DMA_READ, parameters.seg_size * sizeof(double));

    data_read_count ++;
}

//! \brief Main segment processing loop
//select correct output buffer type
void process_chan(double *in_buffer) {

	for (int i = 0; i < parameters.seg_size; i++) {

        //==========cilia_params filter===============//
        double filter_1 = (
            CILIA_FILTER_B1 * in_buffer[i] + cilia_filter_b2 * past_cilia_disp);

        double cilia_disp = filter_1 - cilia_filter_a1 * past_cilia_disp;
        past_cilia_disp = cilia_disp * CILIA_C;

        //===========Apply Scaler============//
        float utconv = past_cilia_disp;

        //=========Apical Conductance========//
        float ex1 = expk((accum)( -(utconv - CILIA_U1) * cilia_params.recips1));
        float ex2 = expk((accum)( -(utconv - CILIA_U0) * cilia_params.recips0));
        float guconv = (CILIA_GA + (CILIA_G_MAX / (1.0f + ex2 * (1.0f + ex1))));

        //========Receptor Potential=========//
        ihcv_now += (
            ((-guconv * (ihcv_now - CILIA_ET)) -
             (CILIA_GK * (ihcv_now - inner_ear_params.ekp))) * cilia_dt_cap);

        //================mICa===============//
        float ex3 = expk((accum) - GAMMA * ihcv_now);
        float mi_ca_inf = 1.0f / (1.0f + ex3 * RECIP_BETA);
        m_ica_curr += (mi_ca_inf - m_ica_curr) * dt_tau_m;

        //================ICa================//
        float mica_pow_conv = m_ica_curr;
        for (float k = 0; k < POWER - 1; k++) {
            mica_pow_conv *= m_ica_curr;
        }
        //============Fibres=============//
        for (int j = 0; j < parameters.number_fibres; j++) {

            //======Synaptic Ca========//
            float i_ca = g_max_ca[j] * mica_pow_conv * (ihcv_now - ECA);
            float sub1 = i_ca * dt_params.dt;
            float sub2 = (ca_curr[j] * dt_params.dt) * rec_tau_ca[j];
            ca_curr[j] += sub1 - sub2;

            //invert Ca
            float pos_ca_curr = -1.0f * ca_curr[j];
            if (i % parameters.resampling_factor == 0) {

                //=====Vesicle Release Rate MAP_BS=====//
                float ca_curr_pow = pos_ca_curr * dt_params.z;
                for (float k = 0.0; k < POWER - 1; k++) {
                    ca_curr_pow *= pos_ca_curr * dt_params.z;
                }

                //=====Release Probability=======//
                float release_prob = ca_curr_pow * dt_spikes;
                float m_q = synapse_m[j] - an_avail[j];
                if (m_q < 0.0f) {
                    m_q = 0.0f;
                }

                //===========Ejected============//
                float release_prob_pow = 1.0f;
                for (float k = 0; k < an_avail[j]; k++) {
                    release_prob_pow =
                        release_prob_pow * (1.0f - release_prob);
                }

                float probability = 1.0f - release_prob_pow;
                if (refrac[j] > 0) {
                    refrac[j] --;
                }

                float ejected;
                bool spiked;
                if (probability > (
                        (float) mars_kiss64_seed(local_seed) *
                        inner_ear_params.r_max_recip)) {
                    ejected = 1.0f;
                    if (refrac[j] <= 0) {
                        log_info(
                            "will spike with key %d", parameters.my_key | j);
                        spiked = TRUE;
                        spin1_send_mc_packet(
                            parameters.my_key | j, 0, NO_PAYLOAD);

                        refrac[j]= (uint) (
                            synapse_params.refrac_period + (
                                ((float) mars_kiss64_seed(local_seed) *
                                inner_ear_params.r_max_recip) *
                                synapse_params.refrac_period)
                                 + 0.5f);
                    } else {
                        spiked = FALSE;
                    }
                } else {
                    ejected = 0.0f;
                    spiked = FALSE;
                }

                //=========Reprocessed=========//
                float repro_rate = synapse_params.xdt;
                float x_pow = 1.0f;
                float y_pow = 1.0f;

                for (float k = 0.0; k < m_q; k++) {
                    y_pow *= (1.0f - synapse_params.ydt);
                }
                for (float k = 0; k < an_repro[j]; k++) {
                    x_pow *= (1.0f - repro_rate);
                }

                probability = 1.0f - x_pow;
                float reprocessed;
                if (probability > (
                        (float) mars_kiss64_seed(local_seed) *
                        inner_ear_params.r_max_recip)) {
                    reprocessed = 1.0f;
                } else {
                    reprocessed = 0.0f;
                }

                //========Replenish==========//
                probability = 1.0f - y_pow;
                float replenish;
                if (probability > (
                        (float) mars_kiss64_seed(local_seed) *
                        inner_ear_params.r_max_recip)) {
                    replenish = 1.0f;
                } else {
                    replenish = 0.0f;
                }

                //==========Update Variables=========//
                an_avail[j] = an_avail[j] + replenish + reprocessed - ejected;
                float re_uptake_and_lost =
                    (synapse_params.rdt + synapse_params.ldt) * an_cleft[j];
                float re_uptake = synapse_params.rdt * an_cleft[j];
                an_cleft[j] = an_cleft[j] + ejected - re_uptake_and_lost;
                an_repro[j] = an_repro[j] + re_uptake - reprocessed;

                //=======write output value to buffer to go to SDRAM ========//
                if (spiked) {
                    neuron_recording_set_spike((j * parameters.seg_size) + i);
                }
                neuron_recording_set_float_recorded_param(
                    SPIKE_PROBABILITY_REGION_ID, (j * parameters.seg_size) + i,
                    ca_curr_pow);
                }
        }
    }
	// set off the record to sdram
	neuron_recording_matrix_record(time);
	neuron_recording_spike_record(time, SPIKE_RECORDING_REGION_ID);
}

//! \brief interface for when dma transfer is successful
//! \param[in] tid:  forced by api
//! \param[in] ttag:  forced by api
//! \return None
void transfer_handler(uint tid, uint ttag) {
    use(tid);
    use(ttag);

    //increment segment index
    seg_index ++;

    //choose current available buffers
    if (!read_switch) {
        process_chan(dtcm_buffer_b);
    } else {
        process_chan(dtcm_buffer_a);
    }
}

//! \brief timer control
//! \param[in] null_a:  forced by api
//! \param[in] null_b:  forced by api
//! \return None
void count_ticks(uint null_a, uint null_b) {
    use(null_a);
    use(null_b);

    time++;

    neuron_recording_do_timestep_update(time);

    // If a fixed number of simulation ticks are specified and these have passed
    if (infinite_run != TRUE && time >= simulation_ticks) {

        // handle the pause and resume functionality
        neuron_recording_finalise();
        simulation_handle_pause_resume(NULL);

         // Subtract 1 from the time so this tick gets done again on the next
        // run
        time -= 1;

        simulation_ready_to_read();
        return;
    }
}

void _store_provenance_data(address_t provenance_region) {
    log_debug("writing other provenance data");

    // store the data into the provenance data region
    provenance_region[N_SIMULATION_TICKS] = simulation_ticks;
    provenance_region[SEG_INDEX] = seg_index;
    provenance_region[DATA_READ_COUNT] = data_read_count;
    provenance_region[DATA_WRITE_COUNT_SPIKES] = data_write_count_spikes;
    provenance_region[DATA_WRITE_COUNT_SPIKE_PROB] =
        data_write_count_spikes_prob;
    provenance_region[MC_RX_COUNT] = mc_rx_count;
    provenance_region[MC_TRANSMISSION_COUNT] = spike_count;

    log_debug("finished other provenance data");
}

//application initialisation
//! \param[in] timer_period: the pointer for the timer period
//! \return bool stating if the init was successful
bool app_init(uint32_t *timer_period)
{
	log_info("starting init \n");

	//obtain data spec
	data_specification_metadata_t *data_address =
	    data_specification_get_data_address();
	// Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM, data_address),
            APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
            &infinite_run, &time_pointer, SDP_PRIORITY, DMA)) {
        return false;
    }

    // sort out provenance data
    simulation_set_provenance_function(
        _store_provenance_data,
        data_specification_get_region(PROVENANCE, data_address));

    // get parameters region
    spin1_memcpy(
        &parameters, data_specification_get_region(PARAMS, data_address),
        sizeof(parameters_struct));

    // get cilia params
    spin1_memcpy(
        &cilia_params,
        data_specification_get_region(CILIA_PARAMS, data_address),
        sizeof(cilia_constants_struct));

    // get inner ear params
    spin1_memcpy(
        &inner_ear_params,
        data_specification_get_region(INNER_EAR_PARAMS, data_address),
        sizeof(inner_ear_param_struct));

    // get dt params
    spin1_memcpy(
        &dt_params,
        data_specification_get_region(DT_BASED_PARAMS, data_address),
        sizeof(dt_params_struct));

    // set the current ihcv to the start point
    ihcv_now = inner_ear_params.ihcv;
    m_ica_curr = inner_ear_params.m_ica_curr;

    // factor to perform vesicle model resampling by
    log_info("AN key=%d", parameters.my_key);
    log_info("n_lsr=%d", parameters.num_lsr);
    log_info("n_msr=%d", parameters.num_msr);
    log_info("n_hsr=%d", parameters.num_hsr);

    #ifdef PROFILE
        // configure timer 2 for profiling
        profiler_init(
            data_specification_get_region(PROFILER, data_address));
    #endif

	//********  Allocate buffers in DTCM **********//

	// input buffers
	dtcm_buffer_a = (double *) sark_alloc (
	    parameters.seg_size, sizeof(double));
	dtcm_buffer_b = (double *) sark_alloc (
	    parameters.seg_size, sizeof(double));

    // verify buffers were actually initialised
	if (dtcm_buffer_a == NULL || dtcm_buffer_b == NULL) {
		log_error("error - cannot allocate buffers");
		return false;
	}

	// set up recording
	if (!neuron_recording_initialise(
            data_specification_get_region(NEURON_RECORDING, data_address),
            &recording_flags, parameters.number_fibres * parameters.seg_size)) {
        log_error("failed to set up recording");
        return false;
    }

    // ********** initialize sections of DTCM **************/

    // input buffers
    for (int i = 0; i < parameters.seg_size; i++) {
        dtcm_buffer_a[i] = 0.0;
        dtcm_buffer_b[i] = 0.0;
    }

    //******************** array defs off num fibres **************//
    refact = spin1_malloc(parameters.number_fibres * sizeof(uint));
    ca_curr = spin1_malloc(parameters.number_fibres * sizeof(float));
    an_cleft = spin1_malloc(parameters.number_fibres * sizeof(float));
    an_avail = spin1_malloc(parameters.number_fibres * sizeof(float));
    an_repro = spin1_malloc(parameters.number_fibres * sizeof(float));
    rec_tau_ca = spin1_malloc(parameters.number_fibres * sizeof(float));
    g_max_ca = spin1_malloc(parameters.number_fibres * sizeof(float));
    ca_th = spin1_malloc(parameters.number_fibres * sizeof(float));
    synapse_m = spin1_malloc(parameters.number_fibres * sizeof(float));

    // verify buffers were actually initialised
	if (refact == NULL || ca_curr == NULL || an_cleft == NULL ||
	        an_avail == NULL || an_repro == NULL || ca_th == NULL ||
	        synapse_m == NULL || rec_tau_ca == NULL ||
	        g_max_ca == NULL) {
		log_error("cannot allocate params based off number of fibres\n");
		return false;
	}

    //*********************** sdram edge data ******************//
    sdram_out_buffer_param sdram_params;
    spin1_memcpy(
        &sdram_params,
        data_specification_get_region(SDRAM_EDGE_ADDRESS, data_address),
        sizeof(sdram_out_buffer_param));
	sdram_in_buffer = sdram_params.sdram_base_address;
    log_info("sdram in buffer @ 0x%08x\n", (uint) sdram_in_buffer);

	//****************MODEL INITIALISATION******************//

    //initialise random number generator
	spin1_memcpy(
        &local_seed,
        data_specification_get_region(RANDOM_SEEDS, data_address),
        sizeof(mars_kiss64_seed_t));
    validate_mars_kiss64_seed(local_seed);
    //local_seed[0] = 0;
    //local_seed[1] = 0;
    //local_seed[2] = 0;
    //local_seed[3] = 1;

	//initialise cilia
	cilia_filter_b2 = (double) dt_params.dt / CILIA_TC - 1.0;
	cilia_filter_a1 = (double) dt_params.dt / CILIA_TC;
	cilia_dt_cap = dt_params.dt / HAIR_CELL_CAPACITANCE;

	//==========Recurring Values=================//
	for (int i=0; i < parameters.num_lsr; i++) {
		ca_curr[i] = inner_ear_params.ca_curr_lsr;
		an_cleft[i] = inner_ear_params.an_cleft_lsr;
		an_avail[i] = inner_ear_params.an_avail_lsr;
		an_repro[i] = inner_ear_params.an_repro_lsr;
		refrac[i] = 0;
		g_max_ca[i] = GMAXCA;
		rec_tau_ca[i] = TAU_CA_LSR;
		synapse_m[i] = MAXIMUM_NUM_NEUROTRANSMITTERS_AT_SYNAPSE;
	}

	for (int i = 0; i < parameters.num_msr; i++) {
		ca_curr[i + parameters.num_lsr] = inner_ear_params.ca_curr_msr;
		an_cleft[i + parameters.num_lsr] = inner_ear_params.an_cleft_msr;
		an_avail[i + parameters.num_lsr] = inner_ear_params.an_avail_msr;
		an_repro[i + parameters.num_lsr] = inner_ear_params.an_repro_msr;
		refrac[i + parameters.num_lsr] = 0;
		g_max_ca[i + parameters.num_lsr] = GMAXCA;
		rec_tau_ca[i + parameters.num_lsr] = TAU_CA_MSR;
		synapse_m[i + parameters.num_lsr] =
		    MAXIMUM_NUM_NEUROTRANSMITTERS_AT_SYNAPSE;
	}

	for (int i = 0; i < parameters.num_hsr; i++) {
		ca_curr[i + parameters.num_lsr + parameters.num_msr] =
		    inner_ear_params.ca_curr_hsr;
		an_cleft[i + parameters.num_lsr + parameters.num_msr] =
		    inner_ear_params.an_cleft_hsr;
		an_avail[i + parameters.num_lsr + parameters.num_msr] =
		    inner_ear_params.an_avail_hsr;
		an_repro[i + parameters.num_lsr + parameters.num_msr] =
		    inner_ear_params.an_repro_hsr;
		refrac[i + parameters.num_lsr + parameters.num_msr] = 0;
		g_max_ca[i + parameters.num_lsr + parameters.num_msr] = GMAXCA;
		rec_tau_ca[i + parameters.num_lsr + parameters.num_msr] = TAU_CA_HSR;
		synapse_m[i + parameters.num_lsr + parameters.num_msr] =
		    MAXIMUM_NUM_NEUROTRANSMITTERS_AT_SYNAPSE;
	}

	//=========initialise the pre synapse params========//
	dt_tau_m = dt_params.dt / DT_TAU_M;

	//=======initialise the synapse params=======//
	dt_spikes = (float) parameters.resampling_factor * dt_params.dt;
	synapse_params.ldt = SYNAPSE_CLEFT_LOSS_RATE * dt_spikes;
	synapse_params.ydt = PRE_SYNAPSE_REPLACEMENT_RATE_HAIR_CELL * dt_spikes;
	if (synapse_params.ydt > 1.0f) {
		synapse_params.ydt = 1.0f;
	}
	synapse_params.xdt = SYNAPSE_CLEFT_RATE_TO_RE_UP_TAKE_STORE * dt_spikes;
	synapse_params.rdt = PRE_SYNAPSE_REPLACEMENT_RATE_RE_UP_TAKE * dt_spikes;
	synapse_params.refrac_period = (IHC_REFRACTORY_PERIOD / dt_spikes);

    return true;
}

//! \brief c main
void c_main()
{
    // Get core and chip IDs
    uint32_t timer_period;
    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    if (app_init(&timer_period)) {

        // Set timer tick (in microseconds)
        log_info("setting timer tick callback for %d microseconds",
        timer_period);
        spin1_set_timer_tick(timer_period);

        //setup callbacks
        //process channel once data input has been read to DTCM
        simulation_dma_transfer_done_callback_on(DMA_READ, transfer_handler);

        //reads from DMA to DTCM every MC packet received
        spin1_callback_on (MC_PACKET_RECEIVED, data_read, MC_PACKET_PRIORITY);
        spin1_callback_on (TIMER_TICK, count_ticks, TIMER);

        simulation_run();
    }
}
