/*
 ============================================================================
 Name        : SpiNNakEar_IHCAN.c
 Author      : Robert James
 Version     : 1.0
 Description : Inner Hair Cell + Auditory Nerve model for use in SpiNNakEar system
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

//data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    RECORDING,
    PROFILER}regions;

// The parameters to be read from memory
enum params {
    N_IHCS,
    AN_KEY,
    IS_KEY,
    IS_FINAL,
    N_ATOMS,
    IHC_KEYS
};

uint32_t an_key;
uint32_t is_key;
uint32_t n_ihcs;
uint32_t is_final;
uint32_t spike_count = 0;
uint32_t n_atoms;
bool final_r2s;
static key_mask_table_entry *key_mask_table;
uint32_t coreID,chipID;

static uint32_t simulation_ticks = 0;
uint32_t time;

bool app_complete = false;

//application initialisation
bool app_init(uint32_t *timer_period)
{
	io_printf (IO_BUF, "[core %d] -----------------------\n", coreID);
	io_printf (IO_BUF, "[core %d] an group starting simulation\n", coreID);
	//obtain data spec
	address_t data_address = data_specification_get_data_address();
    io_printf(IO_BUF,"data_address=%d\n",data_address);

        // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM, data_address),
            APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
            NULL, 1, 0)) {
        return false;
    }

    //get parameters
    address_t params = data_specification_get_region(
                                        PARAMS,data_address);

    n_ihcs = params[N_IHCS];
    io_printf(IO_BUF,"n_ihcs=%d\n",n_ihcs);

    uint32_t n_ihc_key_bytes =  n_ihcs * sizeof(key_mask_table_entry);
    io_printf(IO_BUF,"n_ihc_key_bytes=%d\n",n_ihc_key_bytes);
    key_mask_table = (key_mask_table_entry *)spin1_malloc(n_ihc_key_bytes);
    if (key_mask_table == NULL){
        io_printf (IO_BUF, "failed to allocate memory\n");
        app_end();
    }

    spin1_memcpy(key_mask_table, &params[IHC_KEYS],n_ihc_key_bytes);

    for (uint i=0;i<n_ihcs;i++){
        io_printf(IO_BUF,"ihc key:0x%x mask:0x%x offset:%d\n",key_mask_table[i].key,key_mask_table[i].mask,key_mask_table[i].offset);
    }
    an_key = params[AN_KEY];
    is_key = params[IS_KEY];
    is_final = params[IS_FINAL];
    n_atoms = params[N_ATOMS];
    io_printf(IO_BUF,"an_key=%d\n",an_key);
    io_printf(IO_BUF,"is_key=%d\n",is_key);
    io_printf(IO_BUF,"is_final=%d\n",is_final);
    io_printf(IO_BUF,"n_atoms=%d\n",n_atoms);
    final_r2s = 0;
}

void key_search_and_send(uint32_t spike,uint null){
//    io_printf(IO_BUF,"r\n");
    if (is_final) spike_count++;
    //search through ihc_keys for the rx key
    uint32_t imin = 0;
    uint32_t imax = n_ihcs;
    uint32_t imid;
    key_mask_table_entry entry;
    while (imin < imax) {
//    for (uint imid=0;imid<imax;imid++){
        imid = (imax + imin) >> 1;
        entry = key_mask_table[imid];
        if ((spike & entry.mask) == entry.key){
            uint32_t neuron_id = entry.offset+ (spike & ~entry.mask);
            if(neuron_id>=n_atoms){
                log_info("incorrect neuron ID generated %d",neuron_id);
                rt_error(RTE_SWERR);
            }
//            io_printf(IO_BUF,"id: %d txk: 0x%x\n",neuron_id,an_key|neuron_id);
            while(!spin1_send_mc_packet(an_key|neuron_id,0,NO_PAYLOAD)){
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
    io_printf(IO_BUF,"key %d not found!\n",(uint32_t)(spike & entry.mask));
}

//void user_event_callback(uint null,uint null){
//
//
//}

void spike_rx(uint32_t mc_key, uint null){
//    io_printf(IO_BUF,"k:%d\n",mc_key);
    if (is_key)spin1_schedule_callback(key_search_and_send,mc_key,NULL,1);
//    if (is_key){
//        key_search_and_send(mc_key,NULL);
//        }
}

void app_end()
{
//    if (!is_final){
//        while (!spin1_send_mc_packet(an_key, 0, WITH_PAYLOAD)) {
//            spin1_delay_us(1);
//        }
//    }
//    spin1_exit(0);
    log_info("total simulation ticks = %d",
          simulation_ticks);
    if(is_final)log_info("total spike count = %d",spike_count);
    app_complete = true;
    simulation_ready_to_read();
    io_printf (IO_BUF, "spinn_exit\n");
}

void mc_payload_rx_callback(uint null_a, uint null_b){
//    app_end();
    log_info("command from OME received %d",final_r2s);
    spin1_schedule_callback(app_end,NULL,NULL,2);
//    if(final_r2s)spin1_schedule_callback(app_end,NULL,NULL,2);
//    else final_r2s = 1;
}

void count_ticks(uint null_a, uint null_b){

    time++;
    if (time>simulation_ticks && !app_complete)spin1_schedule_callback(app_end,NULL,NULL,2);

}

void c_main()
{
    // Get core and chip IDs
    coreID = spin1_get_core_id ();
    chipID = spin1_get_chip_id ();
    // Load DTCM data
    uint32_t timer_period;

    // initialise the model
    if (!app_init(&timer_period)){
    rt_error(RTE_API);
    }

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    // Set timer tick (in microseconds)
    log_info("setting timer tick callback for %d microseconds",
          timer_period);
    spin1_set_timer_tick(timer_period);

    //setup callbacks
    spin1_callback_on (MC_PACKET_RECEIVED,spike_rx,-1);
//    spin1_callback_on (MCPL_PACKET_RECEIVED,mc_payload_rx_callback,-1);
    spin1_callback_on (TIMER_TICK,count_ticks,0);

//    spin1_start (SYNC_WAIT);
    simulation_run();

}
