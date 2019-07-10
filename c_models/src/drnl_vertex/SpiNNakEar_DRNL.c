/*
 ============================================================================
 Name        : SpiNNakEar_DRNL.c
 Author      : Robert James
 Version     : 1.0
 Description : Dual Resonance Non-Linear filterbank cochlea model for use in
               SpiNNakEar system
 ============================================================================
 */

//#include "stdfix-full-iso.h"
#include "DRNL_SpiNN.h"
#include "spin1_api.h"
#include "math.h"
#include "log.h"
#include <data_specification.h>
#include <profiler.h>
#include <simulation.h>
#include <recording.h>
#include <debug.h>
#include "neuron/synapses.c"
#include "neuron/spike_processing.c"
#include "neuron/population_table/population_table_binary_search_impl.c"
#include "neuron/direct_synapses.c"
#include "neuron/plasticity/synapse_dynamics_static_impl.c"
#include "neuron/structural_plasticity/synaptogenesis_dynamics_static_impl.c"


//#define PROFILE

//=========GLOBAL VARIABLES============//

// params from the params region
parameters_struct parameters;

// params from the filter params region
filter_params_struct filter_params;

// params
double max_rate;
uint test_dma;
uint seg_index;
uint read_switch;
uint write_switch;
uint processing;
uint index_x;
uint index_y;
uint mc_seg_idx;
uint_float_union MC_union;
uint moc_buffer_index = 0;
uint moc_i = 0;
uint moc_write_switch = 0;
uint moc_resample_factor;
uint moc_sample_count = 0;
uint mc_tx_count = 0;
bool app_complete = false;

double moc;
double moc_now_1;
double moc_now_2;
double moc_now_4;
double moc_dec_1;
double moc_dec_2;
double moc_dec_3;
double moc_factor_1;
double moc_factor_2;
double moc_factor_3;

double lin_x1;
double lin_y1[2];
double lin_y2[2];

double nlin_x1a;
double nlin_y1a[2];
double nlin_y2a[2];
double nlin_x1b;
double nlin_y1b[2];
double nlin_y2b[2];

uint rx_any_spikes = 0;
uint moc_changed = 0;

// the buffers
float *dtcm_buffer_a;
float *dtcm_buffer_b;
double *dtcm_buffer_x;
double *dtcm_buffer_y;
double *dtcm_buffer_moc_x;
double *dtcm_buffer_moc_y;

// sdram edge buffer.
double *sdram_out_buffer;

// multicast bits
double moc_spike_weight = 0;
uint32_t moc_seg_output_n_bytes;

// recording interface demands
uint is_recording;
uint32_t recording_flags;

// simulation interface demands
static uint32_t simulation_ticks = 0;
uint32_t time;

//! \brief interface for getting the weight for a given synapse type
//! \param[in] synapse_type_index the synapse type (e.g. exc. or inh.)
//! \param[in] neuron_index the index of the neuron
//! \param[in] weights_this_timestep weight inputs to be added
//! \return None
void neuron_add_inputs(
        index_t synapse_type_index, index_t neuron_index,
        input_t weights_this_timestep) {
    use(neuron_index);
    if (synapse_type_index == EXCITATORY){
        moc_spike_weight += weights_this_timestep;
    }
    else if (synapse_type_index == INHIBITORY) {
        moc_spike_weight -= weights_this_timestep;
    }
    else {
        log_error("don't recognise this synapse type.");
    }
}

//! \brief Initialises the recording parts of the model
//! \return True if recording initialisation is successful, false otherwise
static bool initialise_recording() {
    address_t address = data_specification_get_data_address();
    address_t recording_region = data_specification_get_region(
            RECORDING, address);

    bool success = recording_initialize(recording_region, &recording_flags);
    return success;
}

void data_write(uint arg_1, uint arg_2)
{
	if (arg_1 == 0 && arg_2 == 0){
	    _setup_synaptic_dma_read();
	    return;
	}

	double *dtcm_buffer_out;
	double *dtcm_buffer_moc;
	uint out_index;
	
	if (test_dma == TRUE) {
		if (!write_switch) {
			out_index = index_x;
			dtcm_buffer_out = dtcm_buffer_x;
		}
		else {
			out_index = index_y;
			dtcm_buffer_out = dtcm_buffer_y;
		}

        spin1_dma_transfer(
            DMA_WRITE, &sdram_out_buffer[out_index],
            dtcm_buffer_out, DMA_WRITE,
            parameters.seq_size * sizeof(double));

        //flip write buffers
        write_switch = !write_switch;

		if (moc_i>=MOC_BUFFER_SIZE) {

		    if (!moc_write_switch){
		        dtcm_buffer_moc=dtcm_buffer_moc_x;
		    }
		    else{
		        dtcm_buffer_moc=dtcm_buffer_moc_y;
		    }

            recording_record(
                MOC_RECORDING_REGION, dtcm_buffer_moc, moc_seg_output_n_bytes);

            //flip moc_write buffers
            moc_write_switch = !moc_write_switch;
            moc_i = 0;
        }
	}
}

uint process_chan(
        double *out_buffer, float *in_buffer, double *moc_out_buffer) {
	uint segment_offset =
	    parameters.seq_size * (
	        (seg_index - 1) & (parameters.n_buffers_in_sdram - 1));

	uint i;
	double linout1;
	double linout2;
	double nonlinout1a;
	double non_linout_2a;
	double non_linout_1b;
	double non_linout_2b;
	double abs_x;
	double compressed_non_lin;
	double filter_1;

	//TODO: change MOC method to a synapse model
	for (i = 0; i < parameters.seq_size; i++) {
		//Linear Path
        filter_1 =
            filter_params.lb0 * in_buffer[i] +
            filter_params.lb1 * lin_x1;
        linout1 =
            filter_1 - filter_params.la1 * lin_y1[1] -
            filter_params.la2 * lin_y1[0];

		lin_x1 = in_buffer[i];
		lin_y1[0] = lin_y1[1];
		lin_y1[1] = linout1;

        filter_1 =
            LIN_GAIN * filter_params.lb0 * linout1 +
            filter_params.lb1 * lin_y1[0];
        linout2 =
            filter_1 - filter_params.la1 * lin_y2[1] -
            filter_params.la2 * lin_y2[0];

		lin_y2[0] = lin_y2[1];
		lin_y2[1] = linout2;

		//non-linear path
		//stage 1
        filter_1 =
            filter_params.nlb0 * in_buffer[i] +
            filter_params.nlb1 * nlin_x1a;
        nonlinout1a =
            filter_1 - filter_params.nla1 * nlin_y1a[1] -
            filter_params.nla2 * nlin_y1a[0];

		nlin_x1a = in_buffer[i];
		nlin_y1a[0] = nlin_y1a[1];
		nlin_y1a[1] = nonlinout1a;

        filter_1 =
            filter_params.nlb0 * nonlinout1a +
            filter_params.nlb1 * nlin_y1a[0];
        non_linout_2a =
            filter_1 - filter_params.nla1 * nlin_y2a[1] -
            filter_params.nla2 * nlin_y2a[0];

		nlin_y2a[0] = nlin_y2a[1];
		nlin_y2a[1] = non_linout_2a;

		//MOC efferent effects
        moc_now_1 = moc_now_1 * moc_dec_1 + moc_spike_weight * moc_factor_1;
        moc_now_2 = moc_now_2 * moc_dec_2 + moc_spike_weight * moc_factor_2;
        moc_now_4 = moc_now_4 * moc_dec_3 + moc_spike_weight * moc_factor_3;

        moc = 1.0 / (1 + moc_now_1 + moc_now_2 + moc_now_4);

        if (moc > 1.0) {
            log_error("out of bounds moc_n%d", moc);
        }

        if (moc < 0.0) {
            log_error("out of bounds moc_n%d", moc);
        }
		non_linout_2a *= moc;

		//stage 2
		abs_x = absolute_value(non_linout_2a);

		if (abs_x < DISP_THRESH) {
			compressed_non_lin = A * non_linout_2a;
		}
		else {
			compressed_non_lin =
			    find_sign(non_linout_2a) * CTBM * (double) expk(
			        C * logk((accum)(A * (abs_x * RECIP_CTBM))));
		}

		//stage 3
        filter_1 =
            filter_params.nlb0 * compressed_non_lin +
            filter_params.nlb1 * nlin_x1b;
        non_linout_1b =
            filter_1 - filter_params.nla1 * nlin_y1b[1] -
            filter_params.nla2 * nlin_y1b[0];

		nlin_x1b = compressed_non_lin;
		nlin_y1b[0] = nlin_y1b[1];
		nlin_y1b[1] = non_linout_1b;

        filter_1 =
            filter_params.nlb0 * non_linout_1b +
            filter_params.nlb1 * nlin_y1b[0];
        non_linout_2b =
            filter_1 - filter_params.nla1 * nlin_y2b[1] -
            filter_params.nla2 * nlin_y2b[0];

		nlin_y2b[0] = nlin_y2b[1];
		nlin_y2b[1] = non_linout_2b;

		//save to buffer
		out_buffer[i] = linout2 + non_linout_2b;

		//if recording MOC
		moc_sample_count++;
		if (moc_sample_count == moc_resample_factor){
		    moc_out_buffer[moc_i] = moc;
		    if (moc != 1.0){
		        moc_changed = 1;
		    }
		    moc_i++;
		    moc_sample_count = 0;
		}
	}
	return segment_offset;
}

void app_end(uint null_a, uint null_b)
{
    use(null_a);
	use(null_b);

    recording_finalise();
    app_complete = true;
    simulation_ready_to_read();
}

void process_handler(uint null_a, uint null_b)
{
    use(null_a);
	use(null_b);

    double *dtcm_moc;
    seg_index++;

    if (!moc_write_switch){
        dtcm_moc = dtcm_buffer_moc_x;
    }
    else {
        dtcm_moc = dtcm_buffer_moc_y;
    }

    //choose current buffers
    if (!read_switch && !write_switch){
        index_x = process_chan(dtcm_buffer_x, dtcm_buffer_b, dtcm_moc);
    } else if (!read_switch && write_switch){
        index_y = process_chan(dtcm_buffer_y, dtcm_buffer_b, dtcm_moc);
    } else if (read_switch && !write_switch){
        index_x = process_chan(dtcm_buffer_x, dtcm_buffer_a, dtcm_moc);
    } else{
        index_y = process_chan(dtcm_buffer_y, dtcm_buffer_a, dtcm_moc);
    }
    spin1_trigger_user_event(DRNL_FILLER_ARG, DRNL_FILLER_ARG);
}

void write_complete(uint tid, uint ttag) {
    use(tid);
	use(ttag);

	// profiler
    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
    #endif

    //send MC packet to connected IHC/AN models
    mc_tx_count++;
    while (!spin1_send_mc_packet(parameters.key, DRNL_FILLER_ARG, NO_PAYLOAD)) {
        spin1_delay_us(1);
    }
}

void data_read(uint mc_key, uint payload) {
    if (mc_key == parameters.ome_data_key) {
        //payload is OME output value
        //convert payload to float
        MC_union.u = payload;

        //collect the next segment of samples and copy into DTCM
        if (test_dma == TRUE) {
            mc_seg_idx++;

            #ifdef PROFILE
            if ( mc_seg_idx >= parameters.seq_size){
                profiler_write_entry_disable_irq_fiq(
                    PROFILER_ENTER | PROFILER_TIMER);
            }
            #endif

            //assign receive buffer
            if (!read_switch) {
                dtcm_buffer_a[mc_seg_idx-1] = MC_union.f;

                //completed filling a segment of input values
                if (mc_seg_idx >= parameters.seq_size) {
                    mc_seg_idx = 0;
                    read_switch = 1;
                    spin1_schedule_callback(
                        process_handler, DRNL_FILLER_ARG, DRNL_FILLER_ARG,
                        PROCESS_HANDLER_PRIORITY);
                }
            }
            else {
                dtcm_buffer_b[mc_seg_idx-1] = MC_union.f;

                //completed filling a segment of input values
                if (mc_seg_idx >= parameters.seq_size)
                {
                    mc_seg_idx = 0;
                    read_switch = 0;
                    spin1_schedule_callback(
                        process_handler, DRNL_FILLER_ARG, DRNL_FILLER_ARG,
                        PROCESS_HANDLER_PRIORITY);
                }
            }
        }
    }
}

void app_done () {
    #ifdef PROFILE
	profiler_finalise();
    #endif
}

void count_ticks(uint null_a, uint null_b) {
    use(null_a);
	use(null_b);

    time++;
    moc_spike_weight = 0;
    if (time > simulation_ticks && !app_complete){
        spin1_schedule_callback(
            app_end, DRNL_FILLER_ARG, DRNL_FILLER_ARG, APP_END_PRIORITY);
    }
}

//application initialisation
bool app_init(uint32_t *timer_period) {
	seg_index = 0;
	read_switch = 0;
	write_switch = 0;

    //obtain data spec
	address_t data_address = data_specification_get_data_address();

    // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM, data_address),
            APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
            NULL, SDP_PRIORITY, DMA_TRANSFER_DONE_PRIORITY)) {
        return false;
    }

    spin1_memcpy(
        &parameters, data_specification_get_region(PARAMS, data_address),
        sizeof(parameters_struct));

    log_info("is rec:%d", parameters.is_recording);
    if (parameters.is_recording) {
        if (!initialise_recording()) {
            log_error("failed to set up recording");
            return false;
        }
    }

    //Get sampling frequency
    uint sampling_frequency = parameters.fs;
    double fs = (double) sampling_frequency;
	double dt = (1.0 / fs);

    moc_resample_factor = (fs / MOC_RESAMPLE_FACTOR_CONVERTER);
    log_info("moc resample factor =%d\n", moc_resample_factor);

	log_info("ome_data_key=%d\n", parameters.ome_data_key);

	//output results buffer (shared with child IHCANs)
    sdram_out_buffer_param sdram_params;
    spin1_memcpy(
        &sdram_params,
        data_specification_get_region(SDRAM_EDGE_ADDRESS, data_address),
        sizeof(sdram_out_buffer_param));
	sdram_out_buffer = sdram_params.sdram_base_address;

    log_info(
        "[core %d] sdram out buffer @ 0x%08x\n",
        spin1_get_core_id(), (uint) sdram_out_buffer);

	//DTCM input buffers
	dtcm_buffer_a = (float *) sark_alloc (parameters.seq_size, sizeof(float));
	dtcm_buffer_b = (float *) sark_alloc (parameters.seq_size, sizeof(float));

	//DTCM output buffers
	dtcm_buffer_x = (double *) sark_alloc (parameters.seq_size, sizeof(double));
	dtcm_buffer_y = (double *) sark_alloc (parameters.seq_size, sizeof(double));
    dtcm_buffer_moc_x = (double *) sark_alloc (MOC_BUFFER_SIZE, sizeof(double));
	dtcm_buffer_moc_y = (double *) sark_alloc (MOC_BUFFER_SIZE, sizeof(double));

    // if any of the buffers failed to be allocated, go boom
	if (dtcm_buffer_a == NULL || dtcm_buffer_b == NULL ||
	        dtcm_buffer_x == NULL ||dtcm_buffer_y == NULL ||
	        dtcm_buffer_moc_x == NULL || dtcm_buffer_moc_y == NULL ||
	        sdram_out_buffer == NULL) {

		test_dma = FALSE;
		log_info("error - cannot allocate 1 or more of the buffers\n");
		return false;
	}
	else {
		test_dma = TRUE;

		// initialize sections of DTCM, system RAM and SDRAM
		for (uint i = 0; i < parameters.seq_size; i++) {
			dtcm_buffer_a[i] = 0;
			dtcm_buffer_b[i] = 0;
		}

		for (uint i = 0; i < parameters.seq_size; i++) {
			dtcm_buffer_x[i] = 0;
			dtcm_buffer_y[i] = 0;
		}

        for (uint i = 0; i < MOC_BUFFER_SIZE; i++) {
            dtcm_buffer_moc_x[i] = 0;
			dtcm_buffer_moc_y[i] = 0;
		}

		for (uint i = 0; i < sdram_params.sdram_edge_size / sizeof(double);
		        i++) {
			sdram_out_buffer[i] = 0;
		}

        mc_seg_idx = 0;
        moc_seg_output_n_bytes = MOC_BUFFER_SIZE * sizeof(double);
	}

    spin1_memcpy(
        &filter_params,
        data_specification_get_region(FILTER_PARAMS, data_address),
        sizeof(filter_params_struct));

    log_info("lin a1: %k", (accum) filter_params.la1);
    log_info("lin a2: %k", (accum) filter_params.la2);
    log_info("lin b0: %k", (accum) filter_params.lb0);
    log_info("lin b1: %k", (accum) filter_params.lb1);
    log_info("nlin a1: %k", (accum) filter_params.nla1);
    log_info("nlin a2: %k", (accum) filter_params.nla2);
    log_info("nlin b0: %k", (accum) filter_params.nlb0);
    log_info("nlin b1: %k", (accum)  filter_params.nlb1);

	//starting values
	lin_x1 = 0.0;
	lin_y1[0] = 0.0;
	lin_y1[1] = 0.0;

	lin_y2[0] = 0.0;
	lin_y2[1] = 0.0;

	nlin_x1a = 0.0;
	nlin_y1a[0] = 0.0;
	nlin_y1a[1] = 0.0;

	nlin_y2a[0] = 0.0;
	nlin_y2a[1] = 0.0;

	nlin_x1b = 0.0;
	nlin_y1b[0] = 0.0;
	nlin_y1b[1] = 0.0;

	nlin_y2b[0] = 0.0;
	nlin_y2b[1] = 0.0;

	moc_now_1 = 0.0;
	moc_now_2 = 0.0;
	moc_now_4 = 0.0;

    moc_dec_1 = exp(- dt / MOC_TAU_0);
    moc_dec_2 = exp(- dt / MOC_TAU_1);
    moc_dec_3 = exp(- dt / MOC_TAU_2);

    moc_factor_1 = RATE_TO_ATTENTUATION_FACTOR * MOC_TAU_WEIGHT * dt;
    moc_factor_2 = 0.0;
    moc_factor_3 = 0.0;

    // Set up the synapses
    uint32_t *ring_buffer_to_input_buffer_left_shifts;
    if (!synapses_initialise(
            data_specification_get_region(SYNAPSE_PARAMS, data_address),
            1, parameters.n_synapse_types,
            &ring_buffer_to_input_buffer_left_shifts)) {
        return false;
    }

    // set up direct synapses
    address_t direct_synapses_address = NULL;
    if (!direct_synapses_initialise(
            data_specification_get_region(DIRECT_MATRIX, data_address),
            &direct_synapses_address)){
        return false;
    }

    // Set up the population table
    uint32_t row_max_n_words;
    if (!population_table_initialise(
            data_specification_get_region(POPULATION_TABLE, data_address),
            data_specification_get_region(SYNAPTIC_MATRIX, data_address),
            direct_synapses_address, &row_max_n_words)) {
        return false;
    }

    // Set up the synapse dynamics
    address_t synapse_dynamics_region_address =
        data_specification_get_region(SYNAPSE_DYNAMICS, data_address);
    synapse_dynamics_initialise(
        synapse_dynamics_region_address, 1, parameters.n_synapse_types,
        ring_buffer_to_input_buffer_left_shifts);

    // set up spike processing
    if (!spike_processing_initialise(
            row_max_n_words, MC_PACKET_PRIORITY, DATA_WRITE_PRIORITY,
            INCOMING_SPIKE_BUFFER_SIE)) {
        return false;
    }

#ifdef PROFILE
    profiler_init(data_specification_get_region(PROFILER, data_address));
#endif
    return true;
}

//! \brief
void c_main() {
    // Get core and chip IDs
    uint32_t timer_period;

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    if (app_init(&timer_period)) {
        // Set timer tick (in microseconds)
        log_info(
            "setting timer tick callback for %d microseconds", timer_period);
        spin1_set_timer_tick(timer_period);

        //setup callbacks
        //process channel once data input has been read to DTCM
        simulation_dma_transfer_done_callback_on(DMA_WRITE, write_complete);
        spin1_callback_on(MCPL_PACKET_RECEIVED, data_read, MC_PACKET_PRIORITY);
        spin1_callback_on(USER_EVENT, data_write, DATA_WRITE_PRIORITY);
        spin1_callback_on(TIMER_TICK, count_ticks, COUNT_TICKS_PRIORITY);

        simulation_run();
    }
}
