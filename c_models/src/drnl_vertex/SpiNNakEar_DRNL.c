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

//!  Dual Resonance Non-Linear filterbank cochlea model for use in
//! SpiNNakEar system

#include "DRNL_SpiNN.h"
#include "log.h"
#include <data_specification.h>
#include <profiler.h>
#include <simulation.h>
#include <debug.h>
#include "neuron/neuron_recording.h"
#include "neuron/synapses.c"
#include "neuron/spike_processing.c"
#include "neuron/population_table/population_table_binary_search_impl.c"
#include "neuron/direct_synapses.c"
#include "neuron/plasticity/synapse_dynamics_static_impl.c"
#include "neuron/structural_plasticity/synaptogenesis_dynamics_static_impl.c"
#include "neuron/bit_field_filter.h"

//#define PROFILE

//=========GLOBAL VARIABLES============//

// params from the params region
parameters_struct parameters;

// params from the float params region
double_parameters_struct double_params;

// params from the filter params region
filter_params_struct filter_params;

// params
double max_rate;
uint seg_index;
uint read_switch;
uint write_switch;
uint processing;
uint index_x;
uint index_y;
int mc_seg_idx;
int overall_sample_id = 0;
uint_float_union MC_union;

double moc;
double moc_now_1;
double moc_now_2;
double moc_now_3;
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

// the buffers
float *dtcm_buffer_a;
float *dtcm_buffer_b;
double *dtcm_buffer_x;
double *dtcm_buffer_y;

// sdram edge buffer.
double *sdram_out_buffer;

// multicast bits
double moc_spike_weight = 0;

// recording interface demands
uint32_t recording_flags;

// ****************** simulation interface demands *************//

//! \brief the number of ticks done so far
static uint32_t simulation_ticks = 0;

//! \brief time to reach
uint32_t time;

//! \brief infinite run pointer
static uint32_t infinite_run;

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
}

//! \brief write data to sdram edge
//! \param[in] arg_1: forced by api but used to distinguish between dmas
//! \param[in] arg_2: forced by api but used to distinguish between dmas
//! \return none
void data_write(uint arg_1, uint arg_2)
{
	if (arg_1 == 0 && arg_2 == 0){
	    setup_synaptic_dma_read();
	    return;
	}

	double *dtcm_buffer_out;
	uint out_index;

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
}

//! \brief processes a channel
//! \param[in] out_buffer: where to store results
//! \param[in] in_buffer: in data
//! \return segment offset
uint process_chan(double *out_buffer, float *in_buffer) {
	uint segment_offset =
	    parameters.seq_size * (
	        (seg_index - 1) & (parameters.n_buffers_in_sdram - 1));

	//TODO: change MOC method to a synapse model
	for (int i = 0; i < parameters.seq_size; i++) {
		//Linear Path
        double filter_1 =
            filter_params.lb0 * in_buffer[i] +
            filter_params.lb1 * lin_x1;
        double linout1 =
            filter_1 - filter_params.la1 * lin_y1[1] -
            filter_params.la2 * lin_y1[0];

		lin_x1 = in_buffer[i];
		lin_y1[0] = lin_y1[1];
		lin_y1[1] = linout1;

        filter_1 =
            LIN_GAIN * filter_params.lb0 * linout1 +
            filter_params.lb1 * lin_y1[0];
        double linout2 =
            filter_1 - filter_params.la1 * lin_y2[1] -
            filter_params.la2 * lin_y2[0];

		lin_y2[0] = lin_y2[1];
		lin_y2[1] = linout2;

		//non-linear path
		//stage 1
        filter_1 =
            filter_params.nlb0 * in_buffer[i] +
            filter_params.nlb1 * nlin_x1a;
        double nonlinout1a =
            filter_1 - filter_params.nla1 * nlin_y1a[1] -
            filter_params.nla2 * nlin_y1a[0];

		nlin_x1a = in_buffer[i];
		nlin_y1a[0] = nlin_y1a[1];
		nlin_y1a[1] = nonlinout1a;

        filter_1 =
            filter_params.nlb0 * nonlinout1a +
            filter_params.nlb1 * nlin_y1a[0];
        double non_linout_2a =
            filter_1 - filter_params.nla1 * nlin_y2a[1] -
            filter_params.nla2 * nlin_y2a[0];

		nlin_y2a[0] = nlin_y2a[1];
		nlin_y2a[1] = non_linout_2a;

		//MOC efferent effects
        moc_now_1 = moc_now_1 * moc_dec_1 + moc_spike_weight * moc_factor_1;
        moc_now_2 = moc_now_2 * moc_dec_2 + moc_spike_weight * moc_factor_2;
        moc_now_3 = moc_now_3 * moc_dec_3 + moc_spike_weight * moc_factor_3;

        moc = 1.0 / (1 + moc_now_1 + moc_now_2 + moc_now_3);

        if (moc > 1.0) {
            log_error("out of bounds moc_n%d", moc);
        }

        if (moc < 0.0) {
            log_error("out of bounds moc_n%d", moc);
        }
		// original moc att location
		//non_linout_2a *= moc;

		//stage 2
		double abs_x = absolute_value(non_linout_2a);
        double compressed_non_lin = 0.0;
		if (abs_x < double_params.disp_thresh) {
			compressed_non_lin = A * non_linout_2a;
		}
		else {
			compressed_non_lin =
			    find_sign(non_linout_2a) * double_params.ctbm * (double) expk(
			        C * logk((accum)(A * (abs_x * double_params.receip_ctbm))));
		}

		//stage 3
        filter_1 =
            filter_params.nlb0 * compressed_non_lin +
            filter_params.nlb1 * nlin_x1b;
        double non_linout_1b =
            filter_1 - filter_params.nla1 * nlin_y1b[1] -
            filter_params.nla2 * nlin_y1b[0];

		nlin_x1b = compressed_non_lin;
		nlin_y1b[0] = nlin_y1b[1];
		nlin_y1b[1] = non_linout_1b;

        filter_1 =
            filter_params.nlb0 * non_linout_1b +
            filter_params.nlb1 * nlin_y1b[0];
        double non_linout_2b =
            filter_1 - filter_params.nla1 * nlin_y2b[1] -
            filter_params.nla2 * nlin_y2b[0];

		nlin_y2b[0] = nlin_y2b[1];
		nlin_y2b[1] = non_linout_2b;

		//save to buffer
		//out_buffer[i] = linout2 + non_linout_2b;
		// changed moc att to channel output
		out_buffer[i] = (linout2 + non_linout_2b) * moc;

		neuron_recording_set_double_recorded_param(
		    MOC_RECORDING_REGION, 0, moc);
		neuron_recording_matrix_record(overall_sample_id);
        neuron_recording_do_timestep_update(overall_sample_id);
        overall_sample_id += 1;
	}
	return segment_offset;
}

//! \brief write data to sdram edge
//! \param[in] null_a: forced by api
//! \param[in] null_b: forced by api
//! \return none
void process_handler(uint null_a, uint null_b) {
    use(null_a);
	use(null_b);

    seg_index++;

    //choose current buffers
    if (!read_switch && !write_switch){
        index_x = process_chan(dtcm_buffer_x, dtcm_buffer_b);
    } else if (!read_switch && write_switch){
        index_y = process_chan(dtcm_buffer_y, dtcm_buffer_b);
    } else if (read_switch && !write_switch){
        index_x = process_chan(dtcm_buffer_x, dtcm_buffer_a);
    } else{
        index_y = process_chan(dtcm_buffer_y, dtcm_buffer_a);
    }
    spin1_trigger_user_event(DRNL_FILLER_ARG, DRNL_FILLER_ARG);
}

//! \brief write data to sdram edge
//! \param[in] tid: forced by api
//! \param[in] ttag: forced by api
//! \return none
void write_complete(uint tid, uint ttag) {
    use(tid);
	use(ttag);

	// profiler
    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
    #endif

    //send MC packet to connected IHC/AN models
    log_info("sending packet to ihcans");
    while (!spin1_send_mc_packet(
            parameters.key, DRNL_FILLER_ARG, NO_PAYLOAD)) {
        spin1_delay_us(1);
    }
}

//! \brief write data to sdram edge
//! \param[in] mc_key: key received
//! \param[in] payload: payload of packet
//! \return none
void data_read(uint mc_key, uint payload) {
    if (mc_key == parameters.ome_data_key) {
        //payload is OME output value
        //convert payload to float
        MC_union.u = payload;

        //collect the next segment of samples and copy into DTCM
        mc_seg_idx++;

        #ifdef PROFILE
        if (mc_seg_idx >= parameters.seq_size) {
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
    else {
        log_error("received packet which i dont expect");
    }
}

//! \brief write data to sdram edge
//! \param[in] tid: forced by api
//! \param[in] ttag: forced by api
//! \return none
void count_ticks(uint null_a, uint null_b) {
    use(null_a);
    use(null_b);

    time++;
    log_info("time %d, sim ticks %d", time, simulation_ticks);

    // make the synapses set off the neuron add input method
    synapses_do_timestep_update(time);

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

//application initialisation
//! \param[in] timer_period: the pointer for the timer period
//! \return bool stating if init was successful or not
static inline bool app_init(uint32_t *timer_period) {
	seg_index = 0;
	read_switch = 0;
	write_switch = 0;

    //obtain data spec
	data_specification_metadata_t *data_address =
	    data_specification_get_data_address();

    // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM, data_address),
            APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
            &infinite_run, &time, SDP_PRIORITY, DMA_TRANSFER_DONE_PRIORITY)) {
        return false;
    }

    // get params
    spin1_memcpy(
        &parameters, data_specification_get_region(PARAMS, data_address),
        sizeof(parameters_struct));

    // get double params
    spin1_memcpy(
        &double_params,
        data_specification_get_region(DOUBLE_PARAMS, data_address),
        sizeof(double_parameters_struct));

    log_info("moc resample factor =%d\n",  parameters.moc_resample_factor);

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

    // if any of the buffers failed to be allocated, go boom
	if (dtcm_buffer_a == NULL || dtcm_buffer_b == NULL ||
	        dtcm_buffer_x == NULL ||dtcm_buffer_y == NULL ||
	        sdram_out_buffer == NULL) {

		log_info("error - cannot allocate 1 or more of the buffers\n");
		return false;
	}
	else {
		// initialize sections of DTCM, system RAM and SDRAM
		for (int i = 0; i < parameters.seq_size; i++) {
			dtcm_buffer_a[i] = 0;
			dtcm_buffer_b[i] = 0;
		}

		for (int i = 0; i < parameters.seq_size; i++) {
			dtcm_buffer_x[i] = 0;
			dtcm_buffer_y[i] = 0;
		}

        // clear sdram space
        int steps = sdram_params.sdram_edge_size / sizeof(double);
		for (int i = 0; i < steps; i++) {
			sdram_out_buffer[i] = 0;
		}

        mc_seg_idx = 0;
	}

    spin1_memcpy(
        &filter_params,
        data_specification_get_region(FILTER_PARAMS, data_address),
        sizeof(filter_params_struct));

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
	moc_now_3 = 0.0;

    moc_dec_1 = double_params.moc_dec_1;
    moc_dec_2 = double_params.moc_dec_2;
    moc_dec_3 = double_params.moc_dec_3;

    moc_factor_1 = double_params.moc_factor_1;
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
            INCOMING_SPIKE_BUFFER_SIZE)) {
        return false;
    }

    log_info("initialising the bit field region");
    if (!bit_field_filter_initialise(
            data_specification_get_region(BIT_FIELD_FILTER, data_address))){
        return false;
    }

    if (!neuron_recording_initialise(
            data_specification_get_region(NEURON_RECORDING, data_address),
            &recording_flags, N_NEURONS)) {
        log_error("failed to set up recording");
        return false;
    }


#ifdef PROFILE
    profiler_init(data_specification_get_region(PROFILER, data_address));
#endif
    return true;
}

//! \brief entrance method
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
