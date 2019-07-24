/*
 ============================================================================
 Name        : SpiNNakEar_IHCAN.c
 Author      : Robert James
 Version     : 1.0
 Description : Inner Hair Cell + Auditory Nerve model for use in SpiNNakEar
               system
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdfix.h>
#include "AN_group_node.h"
#include "spin1_api.h"
#include "math.h"
#include "random.h"
#include "stdfix-exp.h"
#include "log.h"
#include <data_specification.h>
#include <simulation.h>
#include <debug.h>

//! \brief the number of times this core has fired
uint32_t spike_count = 0;

//! \brief the key map
static key_map_struct *key_mask_table;

//! \brief params
static params_struct parameters;

//************** simulation interface demands *******//
//! \brief how many ticks done
static uint32_t simulation_ticks = 0;

//! \brief time to run to
uint32_t time;

//! \brief infinite run pointer
static uint32_t infinite_run;

//! \brief locate new key and transmit
//! \param[in] spike: the key received
//! \param[in] null_a: forced by api.
void key_search_and_send(uint spike, uint null_a) {
    use(null_a);

    if (parameters.is_final_row){
        spike_count ++;
    }
    //search through ihc_keys for the rx key
    uint32_t imin = 0;
    uint32_t imax = parameters.n_children;
    uint32_t imid;
    key_mask_table_entry entry;
    while (imin < imax) {
        imid = (imax + imin) >> 1;
        entry = key_mask_table->entries[imid];
        if ((spike & entry.mask) == entry.key){
            int neuron_id = entry.offset + (spike & ~entry.mask);
            if(neuron_id >= parameters.n_atoms){
                log_error("incorrect neuron ID generated %d", neuron_id);
                rt_error(RTE_SWERR);
            }
            while(!spin1_send_mc_packet(
                    parameters.key | neuron_id, PARAM_FILLER, NO_PAYLOAD)){
                spin1_delay_us(1);
            }
            return;
        }
        else if (entry.key < spike) {
            // Entry must be in upper part of the table
            imin = imid + 1;
        } else {
            // Entry must be in lower part of the table
            imax = imid;
        }
    }
    log_error("key %d not found!\n", spike);
}

//! \brief mc packet reception
//! \param[in] mc_key: the key of the mc packet
//! \param[in] null: the forced api param.
void spike_rx(uint mc_key, uint null) {
    use(null);

    // only going to process and retransmit if i have a key
    if (parameters.has_key) {
        spin1_schedule_callback(
            key_search_and_send, mc_key, PARAM_FILLER, USER);
    }
}

//! \brief stores provenance data
//! \param[in] provenance_region: the sdram location for the prov data.
void _store_provenance_data(address_t provenance_region) {
    log_debug("writing other provenance data");

    // store the data into the provenance data region
    if (parameters.is_final_row) {
        provenance_region[N_SPIKES_TRANSMITTED] = spike_count;
    }

    log_debug("finished other provenance data");
}

//! \brief timer for mapping to auto pause and resume
//! \param[in] null_a: forced by api
//! \param[in] null_b: forced by api
void count_ticks(uint null_a, uint null_b){
    use(null_a);
    use(null_b);

    time++;
    // If a fixed number of simulation ticks are specified and these have passed
    if (infinite_run != TRUE && time >= simulation_ticks) {

        // handle the pause and resume functionality
        simulation_handle_pause_resume(NULL);

         // Subtract 1 from the time so this tick gets done again on the next
        // run
        time -= 1;
        simulation_ready_to_read();
        return;
    }

}


//! \brief application initialisation
//! \param[in] timer_period: pointer to the timer period
//! \return: bool saying if init was successful or not
bool app_init(uint32_t *timer_period) {
	//obtain data spec
	data_specification_metadata_t *data_address =
	    data_specification_get_data_address();
    log_info("data_address = %d\n", data_address);

        // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM, data_address),
            APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
            &infinite_run, &time, SDP_PRIORITY, DMA_TRANSFER_DONE_PRIORITY)) {
        return false;
    }

    //get parameters
    spin1_memcpy(
        &parameters, data_specification_get_region(PARAMS, data_address),
        sizeof(params_struct));

    //! print out params data
    log_info("n_ihcs = %d", parameters.n_children);
    log_info("an_key = %d", parameters.key);
    log_info("is_key = %d", parameters.has_key);
    log_info("is_final = %d", parameters.is_final_row);
    log_info("n_atoms = %d", parameters.n_atoms);

    // how many bytes represents the array of keys
    int n_ihc_key_bytes =
        parameters.n_children * sizeof(key_mask_table_entry);

    // read in array of keys
    log_info("n_ihc_key_bytes = %d", n_ihc_key_bytes);
    key_mask_table = (key_map_struct*) spin1_malloc(n_ihc_key_bytes);
    if (key_mask_table == NULL) {
        log_error("failed to allocate memory");
        return false;
    }
    spin1_memcpy(
        key_mask_table,  data_specification_get_region(KEY_MAP, data_address),
        n_ihc_key_bytes);

    // print key array
    for (int i = 0; i < parameters.n_children; i++) {
        log_info(
            "ihc key:0x%x mask:0x%x offset:%d",
            key_mask_table->entries[i].key, key_mask_table->entries[i].mask,
            key_mask_table->entries[i].offset);
    }

    // sort out provenance data
    simulation_set_provenance_function(
        _store_provenance_data,
        data_specification_get_region(PROVENANCE, data_address));

    return true;
}

//! \brief entrance method
void c_main() {
    // Load DTCM data
    uint32_t timer_period;

    // initialise the model
    if (!app_init(&timer_period)){
        log_error("failed to init the application. shutting down");
        rt_error(RTE_API);
    }

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    // Set timer tick (in microseconds)
    log_info("setting timer tick callback for %d microseconds", timer_period);
    spin1_set_timer_tick(timer_period);

    //setup callbacks
    spin1_callback_on(MC_PACKET_RECEIVED, spike_rx, MC);
    spin1_callback_on(TIMER_TICK, count_ticks, TIMER);


    simulation_run();
}
