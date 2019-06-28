/*
 ============================================================================
 Name        : SpiNNakEar_DRNL.c
 Author      : Robert James
 Version     : 1.0
 Description : Dual Resonance Non-Linear filterbank cochlea model for use in
               SpiNNakEar system
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdfix.h>
#include "DRNL_SpiNN.h"
#include "spin1_api.h"
#include "math.h"
#include "complex.h"
#include <random.h>
#include "stdfix-exp.h"
#include "log.h"
#include <data_specification.h>
#include <profiler.h>
#include <simulation.h>
#include <recording.h>
#include <debug.h>

//#define PROFILE

//=========GLOBAL VARIABLES============//

// params from the params region
parameters_struct parameters;

// params from the filter params region
filter_params_struct filter_params;

// params
REAL max_rate;
uint test_dma;
uint seg_index;
uint read_switch;
uint write_switch;
uint processing;
uint index_x;
uint index_y;
uint mc_seg_idx;
uint_float_union MC_union;
uint moc_spike_count = 0;
uint moc_buffer_index = 0;
uint moc_i = 0;
uint moc_write_switch = 0;
uint moc_resample_factor;
uint moc_sample_count = 0;
uint mc_tx_count = 0;
bool app_complete = false;

REAL moc;
REAL moc_now_1;
REAL moc_now_2;
REAL moc_now_4;
REAL moc_dec_1;
REAL moc_dec_2;
REAL moc_dec_3;
REAL moc_factor_1;
REAL moc_factor_2;
REAL moc_factor_3;

REAL lin_x1;
REAL lin_y1[2];
REAL lin_y2[2];

REAL nlin_x1a;
REAL nlin_y1a[2];
REAL nlin_y2a[2];
REAL nlin_x1b;
REAL nlin_y1b[2];
REAL nlin_y2b[2];

uint rx_any_spikes = 0;
uint moc_changed = 0;

// the buffers
float *dtcm_buffer_a;
float *dtcm_buffer_b;
REAL *dtcm_buffer_x;
REAL *dtcm_buffer_y;
REAL *dtcm_buffer_moc_x;
REAL *dtcm_buffer_moc_y;

// sdram edge buffer.
REAL *sdram_out_buffer;

//MOC count buffer
uint *moc_count_buffer;

// multicast bits
uint n_mocs;
uint n_conn_lut_words;
uint *moc_conn_lut;
uint32_t moc_seg_output_n_bytes;
static key_mask_table_entry *key_mask_table;
static last_neuron_info_t last_neuron_info;

// reocrding interface demands
uint is_recording;
uint32_t recording_flags;

// simulation interface demands
static uint32_t simulation_ticks = 0;
uint32_t time;


//! \brief Initialises the recording parts of the model
//! \return True if recording initialisation is successful, false otherwise
static bool initialise_recording() {
    address_t address = data_specification_get_data_address();
    address_t recording_region = data_specification_get_region(
            RECORDING, address);

    log_info("Recording starts at 0x%08x", recording_region);

    bool success = recording_initialize(recording_region, &recording_flags);
    log_info("Recording flags = 0x%08x", recording_flags);
    return success;
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
    REAL fs = (REAL) sampling_frequency;
	REAL dt = (1.0 / fs);

    moc_resample_factor = (fs / MOC_RESAMPLE_FACTOR_CONVERTER);
    log_info("moc resample factor =%d\n", moc_resample_factor);

	log_info("ome_data_key=%d\n", parameters.ome_data_key);

    uint *moc_conn_lut_address;
	moc_conn_lut_address = &parameters.moc_conn_lut;

	n_mocs = moc_conn_lut_address[0];
	log_info("n_mocs=%d\n", n_mocs);

    n_conn_lut_words = moc_conn_lut_address[1];

    // Allocate buffers
    uint n_key_mask_table_bytes = n_mocs * sizeof(key_mask_table_entry);
    key_mask_table =
        (key_mask_table_entry *) spin1_malloc(n_key_mask_table_bytes);

    uint n_conn_lut_bytes = n_conn_lut_words * 4;
    moc_conn_lut = (uint *)spin1_malloc(n_conn_lut_bytes);

    spin1_memcpy(moc_conn_lut, &(moc_conn_lut_address[2]),
        n_conn_lut_bytes);

    for (uint i = 0; i < n_conn_lut_words; i++) {
        log_info("conn_lut entry: 0x%x", moc_conn_lut[i]);
    }

    spin1_memcpy(
        key_mask_table,
        &(moc_conn_lut_address[2 + n_conn_lut_words]),
        n_key_mask_table_bytes);

    for (uint i = 0; i < n_mocs; i++) {
        log_info(
            "key: %d mask: 0x%x count:%d",
            key_mask_table[i].key, key_mask_table[i].mask,
            key_mask_table[i].conn_index);
    }

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
	dtcm_buffer_x = (REAL *) sark_alloc (parameters.seq_size, sizeof(REAL));
	dtcm_buffer_y = (REAL *) sark_alloc (parameters.seq_size, sizeof(REAL));
    dtcm_buffer_moc_x = (REAL *) sark_alloc (MOC_BUFFER_SIZE, sizeof(REAL));
	dtcm_buffer_moc_y = (REAL *) sark_alloc (MOC_BUFFER_SIZE, sizeof(REAL));

	moc_count_buffer = (uint *) sark_alloc (MOC_DELAY_ARRAY_LEN, sizeof(uint));

    // if any of the buffers failed to be allocated, go boom
	if (dtcm_buffer_a == NULL || dtcm_buffer_b == NULL ||
	        dtcm_buffer_x == NULL ||dtcm_buffer_y == NULL ||
	        dtcm_buffer_moc_x == NULL || dtcm_buffer_moc_y == NULL ||
	        sdram_out_buffer == NULL || moc_count_buffer == NULL) {

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

		for (uint i = 0; i < sdram_params.sdram_edge_size / sizeof(REAL);
		        i++) {
			sdram_out_buffer[i] = 0;
		}

		for (uint i = 0; i < MOC_DELAY_ARRAY_LEN ; i++) {
            moc_count_buffer[i] = 0;
		}

        mc_seg_idx = 0;
        moc_seg_output_n_bytes = MOC_BUFFER_SIZE * sizeof(REAL);
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

    moc_spike_count = 0;

#ifdef PROFILE
    profiler_init(data_specification_get_region(PROFILER, data_address));
#endif
    return true;
}

bool check_incoming_spike_id(uint spike){
    //find corresponding key_mask_index entry
    uint32_t imin = 0;
    uint32_t imax = n_mocs;

    while (imin < imax) {
        int imid = (imax + imin) >> 1;
        key_mask_table_entry entry = key_mask_table[imid];
        if ((spike & entry.mask) == entry.key) {
            uint neuron_id = spike & ~entry.mask;
            last_neuron_info.e_index = entry.conn_index;
            last_neuron_info.w_index = neuron_id / BITS_IN_WORD;
            last_neuron_info.id_shift =
                BITS_IN_WORD - 1 -(neuron_id % BITS_IN_WORD);
	        return (
	            moc_conn_lut[last_neuron_info.e_index +
	                         last_neuron_info.w_index] &
	            ((uint32_t)1 << last_neuron_info.id_shift));
        }
        else if (entry.key < spike) {

            // Entry must be in upper part of the table
            imin = imid + 1;
        } else {

            // Entry must be in lower part of the table
            imax = imid;
        }
    }
    log_info("rx spike: %u not in pop table!", spike);
    return false;
}

void update_moc_buffer(uint sc) {
    moc_count_buffer[moc_buffer_index] = sc;
    moc_buffer_index++;
    if (moc_buffer_index >= MOC_DELAY_ARRAY_LEN){
        moc_buffer_index = 0;
    }
}

uint get_current_moc_spike_count() {
    int index_diff = moc_buffer_index - MOC_DELAY;
    uint delayed_index;
    
    if (index_diff < 0) {
        //wrap around
        delayed_index = (MOC_DELAY_ARRAY_LEN - 1) + index_diff;
    }
    else{
        delayed_index = index_diff;
    }

    return moc_count_buffer[delayed_index];
}

void data_write(uint null_a, uint null_b)
{
	use(null_a);
	use(null_b);

	REAL *dtcm_buffer_out;
	REAL *dtcm_buffer_moc;
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
            parameters.seq_size * sizeof(REAL));

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

uint process_chan(REAL *out_buffer, float *in_buffer, REAL *moc_out_buffer) {
	uint segment_offset =
	    parameters.seq_size * (
	        (seg_index - 1) & (parameters.n_buffers_in_sdram - 1));

	uint i;
	REAL linout1;
	REAL linout2;
	REAL nonlinout1a;
	REAL non_linout_2a;
	REAL non_linout_1b;
	REAL non_linout_2b;
	REAL abs_x;
	REAL compressed_non_lin;
	REAL filter_1;

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
		REAL cur_moc_spike_count = (REAL) get_current_moc_spike_count();
		if (cur_moc_spike_count < 0) {
		    log_info("-ve moc_n%d", moc_spike_count);
		}

        moc_now_1 = moc_now_1 * moc_dec_1 + cur_moc_spike_count * moc_factor_1;
        moc_now_2 = moc_now_2 * moc_dec_2 + cur_moc_spike_count * moc_factor_2;
        moc_now_4 = moc_now_4 * moc_dec_3 + cur_moc_spike_count * moc_factor_3;

        moc = 1.0 / (1 + moc_now_1 + moc_now_2 + moc_now_4);

        if (moc > 1.0) {
            log_info("out of bounds moc_n%d", moc);
        }

        if (moc < 0.0) {
            log_info("out of bounds moc_n%d", moc);
        }
		non_linout_2a *= moc;

		//stage 2
		abs_x = ABS(non_linout_2a);

		if (abs_x < DISP_THRESH) {
			compressed_non_lin = A * non_linout_2a;
		}
		else {
			compressed_non_lin =
			    SIGN(non_linout_2a) * CTBM * (REAL) expk(
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
    log_info("total simulation ticks = %d", simulation_ticks);
    log_info("processed %d segments", seg_index);
    log_info("sent %d mc packets", mc_tx_count);
    log_info("spinn_exit\n");
    log_info("rx any spikes = %d", rx_any_spikes);
    log_info("moc changed = %d", moc_changed);
    app_complete = true;
    simulation_ready_to_read();
}

void process_handler(uint null_a, uint null_b)
{
    use(null_a);
	use(null_b);

    REAL *dtcm_moc;
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
    spin1_trigger_user_event(FILLER_ARG, FILLER_ARG);
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
    while (!spin1_send_mc_packet(parameters.key, FILLER_ARG, NO_PAYLOAD)) {
        spin1_delay_us(1);
    }
}

void spike_check(uint rx_key, uint null_a){
    use(null_a);
    if (check_incoming_spike_id(rx_key)){
        moc_spike_count++;
    }
}

void moc_spike_received(uint mc_key, uint null_a) {
    use(null_a);

    spin1_schedule_callback(
        spike_check, mc_key, FILLER_ARG, SPIKE_CHECK_PRIORITY);
    if (!rx_any_spikes){
        rx_any_spikes = 1;
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
                        process_handler, FILLER_ARG, FILLER_ARG,
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
                        process_handler, FILLER_ARG, FILLER_ARG,
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

    update_moc_buffer(moc_spike_count);
    moc_spike_count = 0;
    time++;
    if (time > simulation_ticks && !app_complete){
        spin1_schedule_callback(
            app_end, FILLER_ARG, FILLER_ARG, APP_END_PRIORITY);
    }
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
        spin1_callback_on(
            MC_PACKET_RECEIVED, moc_spike_received, MC_PACKET_PRIORITY);
        spin1_callback_on(USER_EVENT, data_write, DATA_WRITE_PRIORITY);
        spin1_callback_on(TIMER_TICK, count_ticks, COUNT_TICKS_PRIORITY);

        simulation_run();
    }
}
