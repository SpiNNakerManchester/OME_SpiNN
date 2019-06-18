/*
 ============================================================================
 Name        : SpiNNakEar_DRNL.c
 Author      : Robert James
 Version     : 1.0
 Description : Dual Resonance Non-Linear filterbank cochlea model for use in SpiNNakEar system
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
#include <profile_tags.h>
#include <simulation.h>
#include <recording.h>
#include <debug.h>

//#define PROFILE

//=========GLOBAL VARIABLES============//
REAL Fs,dt,max_rate;
uint coreID;
uint chipID;
uint test_DMA;
uint seg_index;
uint cbuff_numseg;
uint read_switch;
uint write_switch;
uint processing;
uint index_x;
uint index_y;
uint MC_seg_idx;
uint_float_union MC_union;
uint ack_rx=0;
uint moc_spike_count=0;
uint sample_counter=0;
uint moc_buffer_index = 0;
uint moc_i=0;
uint moc_write_switch = 0;
uint moc_resample_factor;
uint moc_sample_count = 0;
uint moc_seg_index = 0;
uint mc_tx_count = 0;
bool app_complete = false;

REAL cf,nlin_b0,nlin_b1,nlin_b2,nlin_a1,nlin_a2,
       lin_b0,lin_b1,lin_b2,lin_a1,lin_a2,lin_gain,
       a,ctBM,dispThresh,recip_ctBM,MOC,MOCnow1,
       MOCnow2,MOCnow3,MOCdec1,MOCdec2,MOCdec3,
       MOCfactor1,MOCfactor2,MOCfactor3,MOCspikeCount;


accum c;

REAL lin_x1;
REAL lin_y1[2],lin_y2[2];

REAL nlin_x1a;
REAL nlin_y1a[2],nlin_y2a[2];
REAL nlin_x1b;
REAL nlin_y1b[2],nlin_y2b[2];
REAL MOCtau[3],MOCtauweights[3];

int start_count_process;
int end_count_process;
int start_count_read;
int end_count_read;
int start_count_write;
int end_count_write;

uint sync_count=0;
uint rx_any_spikes = 0;
uint moc_changed = 0;

float *dtcm_buffer_a;
float *dtcm_buffer_b;
REAL *dtcm_buffer_x;
REAL *dtcm_buffer_y;
REAL *dtcm_buffer_moc_x;
REAL *dtcm_buffer_moc_y;

REAL *sdramout_buffer;

//MOC count buffer
uint *moc_count_buffer;

//data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    RECORDING,
    PROFILER}regions;
// The parameters to be read from memory
enum params {
    //N_TICKS,
    DATA_SIZE,
    OMECOREID,
    COREID,
    OMEAPPID,
    OME_KEY,
    KEY,
    NUM_IHCAN,
    CF,
    DELAY,
    FS,
    OME_DATA_KEY,
    IS_RECORDING,
    L_A1,
    L_A1_DOUBLE,
    L_A2,
    L_A2_DOUBLE,
    L_B0,
    L_B0_DOUBLE,
    L_B1,
    L_B1_DOUBLE,
    NL_A1,
    NL_A1_DOUBLE,
    NL_A2,
    NL_A2_DOUBLE,
    NL_B0,
    NL_B0_DOUBLE,
    NL_B1,
    NL_B1_DOUBLE,
    MOC_CONN_LUT
};

// The size of the remaining data to be sent
uint data_size;
// the core ID given by the placement software
uint placement_coreID;
uint ome_coreID;
uint ome_appID;
uint ome_key;
uint key;
uint mask;
uint num_ihcans;
uint drnl_cf;
uint delay;
uint sampling_frequency;
uint ome_data_key;
uint n_mocs;
uint n_conn_lut_words;
uint *moc_conn_lut;
uint is_recording;
uint32_t recording_flags;
uint32_t seg_output_n_bytes;
uint32_t moc_seg_output_n_bytes;
static key_mask_table_entry *key_mask_table;
static last_neuron_info_t last_neuron_info;

uint *moc_conn_lut_address;
uint n_samples_per_100us;

//uint32_t TOTAL_TICKS;
static uint32_t simulation_ticks=0;
uint32_t time;


//! \brief Initialises the recording parts of the model
//! \return True if recording initialisation is successful, false otherwise
static bool initialise_recording(){
    address_t address = data_specification_get_data_address();
    address_t recording_region = data_specification_get_region(
            RECORDING, address);

    log_info("Recording starts at 0x%08x", recording_region);

    bool success = recording_initialize(recording_region, &recording_flags);
    log_info("Recording flags = 0x%08x", recording_flags);
    return success;
}
//application initialisation
bool app_init(uint32_t *timer_period)
{
	seg_index=0;
	cbuff_numseg=4;
	read_switch=0;
	write_switch=0;

    //obtain data spec
	address_t data_address = data_specification_get_data_address();

    // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM, data_address),
            APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
            NULL, 1, 0)) {
        return false;
    }
    address_t params = data_specification_get_region(PARAMS, data_address);
    parameters_struct parameters;
    spin1_memcpy(&parameters, (parameters_struct*)data_specification_get_region(PARAMS, data_address), sizeof(parameters_struct));

	// Get the size of the data in words
    data_size = params[DATA_SIZE];
//    TOTAL_TICKS= data_size/SEGSIZE;
//    log_info("TOTAL_TICKS=%d",TOTAL_TICKS);

    //obtain ome core ID from the host placement perspective
    ome_coreID = params[OMECOREID];
    //obtain this core ID from the host placement perspective
    placement_coreID = params[COREID];

    //obtain ome application ID from the host placement perspective
    ome_appID = params[OMEAPPID];
    //key for synchronisation messages to be sent back to parent OME
    ome_key=params[OME_KEY];
    log_info("omekey:%d",ome_key);
    //key for synchronisation messages to be sent to child IHC/ANs
    key=params[KEY];
    log_info("key:%d",key);

    is_recording = params[IS_RECORDING];
    log_info("is rec:%d",is_recording);//not used, will always record
    if (!initialise_recording()) return false;

    //get the mask needed to extract comms protocol from MC keys
    mask = 3;
    //number of child IHC/ANs
    num_ihcans=params[NUM_IHCAN];
    //DRNL bandpass center frequency
    drnl_cf=params[CF];
    //transmission delay (not used)
    delay = params[DELAY];

    log_info("CF=%d\n",drnl_cf);
    //Get sampling frequency
    sampling_frequency = params[FS];
    Fs= (REAL)sampling_frequency;
	dt=(1.0/Fs);
	//calculate how many segments approx 1ms is
	n_samples_per_100us = 100e-6/dt;
    log_info("n_samples_per_100us=%d\n",n_samples_per_100us);
    moc_resample_factor = (Fs/1000.);
    log_info("moc resample factor =%d\n",moc_resample_factor);
	ome_data_key = params[OME_DATA_KEY];
    io_printf(IO_BUF,"ome_data_key=%d\n",ome_data_key);

	moc_conn_lut_address = &params[MOC_CONN_LUT];
	n_mocs = moc_conn_lut_address[0];
	io_printf(IO_BUF,"n_mocs=%d\n",n_mocs);
    n_conn_lut_words = moc_conn_lut_address[1];

    // Allocate buffers
    uint n_key_mask_table_bytes = n_mocs * sizeof(key_mask_table_entry);
    key_mask_table = (key_mask_table_entry *)spin1_malloc(n_key_mask_table_bytes);

    uint n_conn_lut_bytes = n_conn_lut_words * 4;
    moc_conn_lut = (uint *)spin1_malloc(n_conn_lut_bytes);

    spin1_memcpy(moc_conn_lut, &(moc_conn_lut_address[2]),
        n_conn_lut_bytes);

    for (uint i=0;i<n_conn_lut_words;i++){
        log_info("conn_lut entry: 0x%x",moc_conn_lut[i]);
    }

    spin1_memcpy(key_mask_table, &(moc_conn_lut_address[2+n_conn_lut_words]),
        n_key_mask_table_bytes);

        for (uint i=0;i<n_mocs;i++){
        log_info("key: %d mask: 0x%x count:%d",key_mask_table[i].key,key_mask_table[i].mask,key_mask_table[i].conn_index);
    }

	//output results buffer (shared with child IHCANs)
	//hack for smaller SDRAM intermediate circular buffers
	data_size=cbuff_numseg*SEGSIZE;

	sdramout_buffer = (REAL *) sark_xalloc (sv->sdram_heap,
					 data_size * sizeof(REAL),
					 placement_coreID,
					 ALLOC_LOCK);

    io_printf(IO_BUF,"[core %d] sdram out buffer @ 0x%08x\n", coreID,
               (uint) sdramout_buffer);

	//DTCM input buffers
	dtcm_buffer_a = (float *) sark_alloc (SEGSIZE, sizeof(float));
	dtcm_buffer_b = (float *) sark_alloc (SEGSIZE, sizeof(float));
	//DTCM output buffers
	dtcm_buffer_x = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_buffer_y = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
    dtcm_buffer_moc_x = (REAL *) sark_alloc (MOC_BUFFER_SIZE, sizeof(REAL));
	dtcm_buffer_moc_y = (REAL *) sark_alloc (MOC_BUFFER_SIZE, sizeof(REAL));

	moc_count_buffer = (uint *) sark_alloc (MOC_DELAY_ARRAY_LEN,sizeof(uint));

	if (dtcm_buffer_a == NULL ||dtcm_buffer_b == NULL ||dtcm_buffer_x == NULL ||dtcm_buffer_y == NULL 
		||dtcm_buffer_moc_x == NULL ||dtcm_buffer_moc_y == NULL 	||  sdramout_buffer == NULL || moc_count_buffer == NULL)
	{
		test_DMA = FALSE;
		io_printf (IO_BUF, "[core %d] error - cannot allocate buffer\n", coreID);
		return false;
	}
	else
	{
		test_DMA = TRUE;
		// initialize sections of DTCM, system RAM and SDRAM
		for (uint i = 0; i < SEGSIZE; i++)
		{
			dtcm_buffer_a[i]   = 0;
			dtcm_buffer_b[i]   = 0;
		}
		for (uint i = 0; i < SEGSIZE; i++)
		{
			dtcm_buffer_x[i]   = 0;
			dtcm_buffer_y[i]   = 0;
		}
        for (uint i = 0; i < MOC_BUFFER_SIZE; i++)
		{
            dtcm_buffer_moc_x[i]   = 0;
			dtcm_buffer_moc_y[i]   = 0;
		}
		for (uint i=0;i<data_size;i++)
		{
			sdramout_buffer[i]  = 0;
		}
		for (uint i=0;i<MOC_DELAY_ARRAY_LEN;i++)
		{
            moc_count_buffer[i] = 0;
		}
        MC_seg_idx=0;
        seg_output_n_bytes = SEGSIZE * sizeof(REAL);
        moc_seg_output_n_bytes = MOC_BUFFER_SIZE * sizeof(REAL);
	}

	
	//============MODEL INITIALISATION================//
    //REAL complex lin_z1,lin_z2,lin_z3,lin_tf,nlin_z1,nlin_z2,nlin_z3,nlin_tf;
    REAL rateToAttentuationFactor,lin_cf,nlBWp,nlBWq,linBWp,linBWq,linCFp,linCFq,
        nlin_bw,nlin_phi,nlin_theta,nlin_cos_theta,nlin_sin_theta,nlin_alpha,
        lin_bw,lin_phi,lin_theta,lin_cos_theta,lin_sin_theta,lin_alpha;

	//set center frequency
	//cf=(REAL)drnl_cf;

	//non-linear pathway
	/*nlBWq=180.0;//58.7147428141;//
	nlBWp=0.14;//0.0342341965;//
	nlin_bw=nlBWp * cf + nlBWq;
	nlin_phi=2.0 * M_PI * nlin_bw * dt;
	nlin_theta= 2.0 * M_PI * cf * dt;
	nlin_cos_theta= cos(nlin_theta);
	nlin_sin_theta= sin(nlin_theta);
	nlin_alpha= -exp(-nlin_phi) * nlin_cos_theta;
	nlin_a1= 2.0 * nlin_alpha;
	nlin_a2= exp(-2.0 * nlin_phi);
	nlin_z1 = (1.0 + nlin_alpha * nlin_cos_theta) - (nlin_alpha * nlin_sin_theta) * _Complex_I;
	nlin_z2 = (1.0 + nlin_a1 * nlin_cos_theta) - (nlin_a1 * nlin_sin_theta) * _Complex_I;
	nlin_z3 = (nlin_a2 * cos(2.0 * nlin_theta)) - (nlin_a2 * sin(2.0 * nlin_theta)) * _Complex_I;
	nlin_tf = (nlin_z2 + nlin_z3) / nlin_z1;
	nlin_b0 = cabs(nlin_tf);
	nlin_b1 = nlin_alpha * nlin_b0;*/
    nlin_a1 = parameters.NL_A1;
    nlin_a2 = parameters.NL_A2;
    nlin_b0 = parameters.NL_B0;
    nlin_b1 = parameters.NL_B1;

	//compression algorithm variables
	a=30e4;//5e4;
	c=0.25k;
	ctBM = 1e-9 * pow(10.0,32.0/20.0);
	recip_ctBM=1.0/ctBM;
	dispThresh=ctBM/a;

	//linear pathway
	lin_gain=200.0;
	/*linBWq=235.0;//76.65535867396389;//
	linBWp=0.2;//0.048905995;//
	lin_bw=linBWp * cf + linBWq;
	lin_phi=2.0 * M_PI * lin_bw * dt;
	linCFp=0.62;
	linCFq=266.0;
	lin_cf=linCFp*cf+linCFq;
	lin_theta= 2.0 * M_PI * lin_cf * dt;
	lin_cos_theta= cos(lin_theta);
	lin_sin_theta= sin(lin_theta);
	lin_alpha= -exp(-lin_phi) * lin_cos_theta;
	lin_a1= 2.0 * lin_alpha;
	lin_a2= exp(-2.0 * lin_phi);
	lin_z1 = (1.0 + lin_alpha * lin_cos_theta) - (lin_alpha * lin_sin_theta) * _Complex_I;
	lin_z2 = (1.0 + lin_a1 * lin_cos_theta) - (lin_a1 * lin_sin_theta) * _Complex_I;
	lin_z3 = (lin_a2 * cos(2.0 * lin_theta)) - (lin_a2 * sin(2.0 * lin_theta)) * _Complex_I;
	lin_tf = (lin_z2 + lin_z3) / lin_z1;
	lin_b0 = cabs(lin_tf);
	lin_b1 = lin_alpha * lin_b0;*/
	lin_a1 = parameters.L_A1;
	lin_a2 = parameters.L_A2;
	lin_b0 = parameters.L_B0;
	lin_b1 = parameters.L_B1;

    log_info("lin a1: %k",(accum)lin_a1);
    log_info("lin a2: %k",(accum)lin_a2);
    log_info("lin b0: %k",(accum)lin_b0);
    log_info("lin b1: %k",(accum)lin_b1);
    log_info("nlin a1: %k",(accum)nlin_a1);
    log_info("nlin a2: %k",(accum)nlin_a2);
    log_info("nlin b0: %k",(accum)nlin_b0);
    log_info("nlin b1: %k",(accum)nlin_b1);

	//starting values
	lin_x1=0.0;
	lin_y1[0]=0.0;
	lin_y1[1]=0.0;
	
	lin_y2[0]=0.0;
	lin_y2[1]=0.0;	

	nlin_x1a=0.0;
	nlin_y1a[0]=0.0;
	nlin_y1a[1]=0.0;

	nlin_y2a[0]=0.0;
	nlin_y2a[1]=0.0;

	nlin_x1b=0.0;
	nlin_y1b[0]=0.0;
	nlin_y1b[1]=0.0;
	
	nlin_y2b[0]=0.0;
	nlin_y2b[1]=0.0;

	rateToAttentuationFactor = 6e2;//4.25e3;//1e3;//10;//15;//

	MOCnow1=0.0;
	MOCnow2=0.0;
	MOCnow3=0.0;

	MOCtau[0] = 0.055;//0.05;//
	MOCtau[1] = 0.4;//0.3;//
    MOCtau[2] = 1;//100;//

    MOCtauweights[0] = 0.9;//0.7;//
    MOCtauweights[1] = 0.1;//0.3;//
    MOCtauweights[2] = 0;

    MOCdec1 = exp(- dt/MOCtau[0]);
    MOCdec2 = exp(- dt/MOCtau[1]);
    MOCdec3 = exp(- dt/MOCtau[2]);

//    MOCfactor1 = rateToAttentuationFactor * MOCtauweights[0] * dt;//0.01 * rateToAttentuationFactor * MOCtauweights[0] * dt;
//    MOCfactor2 = 0.;//0.01 * rateToAttentuationFactor * MOCtauweights[1] * dt;
    MOCfactor1 = rateToAttentuationFactor * MOCtauweights[0] * dt;
    MOCfactor2 = 0.;//rateToAttentuationFactor * MOCtauweights[1] * dt;
    MOCfactor3 = 0.;//0.01 * rateToAttentuationFactor * MOCtauweights[2] * dt;

    MOCspikeCount=0;

#ifdef PROFILE
    profiler_init(
        data_specification_get_region(1, data_address));
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
        if ((spike & entry.mask) == entry.key){
            uint neuron_id = spike & ~entry.mask;
            last_neuron_info.e_index = entry.conn_index;
            last_neuron_info.w_index = neuron_id/32;
            last_neuron_info.id_shift = 31-(neuron_id%32);
	        return(moc_conn_lut[last_neuron_info.e_index+last_neuron_info.w_index] & ((uint32_t)1 << last_neuron_info.id_shift));
        }
        else if (entry.key < spike) {

            // Entry must be in upper part of the table
            imin = imid + 1;
        } else {

            // Entry must be in lower part of the table
            imax = imid;
        }
    }
    log_info("rx spike: %u not in pop table!",spike);
    return false;
}

void update_moc_buffer(uint sc){
    moc_count_buffer[moc_buffer_index]=sc;
    moc_buffer_index++;
//    log_info("mbi%d",moc_buffer_index);
    if (moc_buffer_index >= MOC_DELAY_ARRAY_LEN) moc_buffer_index = 0;
}

uint get_current_moc_spike_count(){
    int index_diff = moc_buffer_index - MOC_DELAY;
    uint delayed_index;
    if (index_diff<0){
        //wrap around
        delayed_index = (MOC_DELAY_ARRAY_LEN-1) + index_diff;
    }
    else{
        delayed_index = index_diff;
    }

    return moc_count_buffer[delayed_index];
}

recording_complete_callback_t record_finished(void)
{
//    log_info("recording segment moc complete %d",moc_seg_index);
    moc_seg_index++;
}

void data_write(uint null_a, uint null_b)
{
	REAL *dtcm_buffer_out;
	REAL *dtcm_buffer_moc;
	uint out_index;
	
	if(test_DMA == TRUE)
	{
		if(!write_switch)
		{
			out_index=index_x;
			dtcm_buffer_out=dtcm_buffer_x;
		}
		else
		{
			out_index=index_y;
			dtcm_buffer_out=dtcm_buffer_y;
		}

        spin1_dma_transfer(DMA_WRITE,&sdramout_buffer[out_index],dtcm_buffer_out,DMA_WRITE,
		  						SEGSIZE*sizeof(REAL));
        //flip write buffers
        write_switch=!write_switch;

		if(moc_i>=MOC_BUFFER_SIZE){

		    if (!moc_write_switch)dtcm_buffer_moc=dtcm_buffer_moc_x;
		    else dtcm_buffer_moc=dtcm_buffer_moc_y;

            recording_record_and_notify(0, dtcm_buffer_moc,
                                    moc_seg_output_n_bytes,record_finished);
            //flip moc_write buffers
            moc_write_switch=!moc_write_switch;
            moc_i = 0;
        }
	}
}

uint process_chan(REAL *out_buffer,float *in_buffer,REAL *moc_out_buffer)
{
	uint segment_offset=SEGSIZE*((seg_index-1) & (cbuff_numseg-1));
	uint i;
	REAL linout1,linout2,nonlinout1a,nonlinout2a,nonlinout1b,nonlinout2b,abs_x,compressedNonlin;
	REAL filter_1;

	//TODO: change MOC method to a synapse model
	for(i=0;i<SEGSIZE;i++)
	{
		//Linear Path
        filter_1 = lin_b0 * in_buffer[i] + lin_b1 * lin_x1;
        linout1= filter_1 - lin_a1 * lin_y1[1] - lin_a2 * lin_y1[0];

		lin_x1=in_buffer[i];
		lin_y1[0]=lin_y1[1];
		lin_y1[1]=linout1;

        filter_1 = lin_gain*lin_b0 * linout1 + lin_b1 * lin_y1[0];
        linout2= filter_1 - lin_a1 * lin_y2[1] - lin_a2 * lin_y2[0];
		
		lin_y2[0]= lin_y2[1];
		lin_y2[1]= linout2;

		//non-linear path
		//stage 1
        filter_1 =  nlin_b0 * in_buffer[i] + nlin_b1 * nlin_x1a;
        nonlinout1a = filter_1 - nlin_a1 * nlin_y1a[1] - nlin_a2 * nlin_y1a[0];

		nlin_x1a=in_buffer[i];
		nlin_y1a[0]=nlin_y1a[1];
		nlin_y1a[1]=nonlinout1a;

        filter_1 = nlin_b0 * nonlinout1a + nlin_b1 * nlin_y1a[0];
        nonlinout2a = filter_1 - nlin_a1 * nlin_y2a[1] - nlin_a2 * nlin_y2a[0];

		nlin_y2a[0]= nlin_y2a[1];
		nlin_y2a[1]= nonlinout2a;

		//MOC efferent effects
		MOCspikeCount = (REAL)get_current_moc_spike_count();
		if (MOCspikeCount<0.)log_info("-ve moc_n%d",MOCspikeCount);
//		if (MOCspikeCount>0)log_info("moc_n%d",MOCspikeCount);
//		if (MOCspikeCount<3)MOCspikeCount=0;
        MOCnow1= MOCnow1* MOCdec1+ MOCspikeCount* MOCfactor1;
        MOCnow2= MOCnow2* MOCdec2+ MOCspikeCount* MOCfactor2;
        MOCnow3= MOCnow3* MOCdec3+ MOCspikeCount* MOCfactor3;
        /*MOCnow1= MOCnow1* MOCdec1+ MOCspikeCount*MOCspikeCount* MOCfactor1;
        MOCnow2= MOCnow2* MOCdec2+ MOCspikeCount*MOCspikeCount* MOCfactor2;
        MOCnow3= MOCnow3* MOCdec3+ MOCspikeCount*MOCspikeCount* MOCfactor3;*/
/*        MOCnow1= MOCnow1* MOCdec1+ MOCspikeCount*MOCspikeCount*MOCspikeCount* MOCfactor1;
        MOCnow2= MOCnow2* MOCdec2+ MOCspikeCount*MOCspikeCount*MOCspikeCount* MOCfactor2;
        MOCnow3= MOCnow3* MOCdec3+ MOCspikeCount*MOCspikeCount*MOCspikeCount* MOCfactor3;*/

/*        MOCnow1= MOCnow1* MOCdec1+ MOCspikeCount* (1-MOCdec1);
        MOCnow2= MOCnow2* MOCdec2+ MOCspikeCount* (1-MOCdec2);
        MOCnow3= MOCnow3* MOCdec3+ MOCspikeCount* (1-MOCdec3);*/
        // MOC= 1 when all MOCnow are zero
        // 0 < MOC < 1
        MOC= 1./(1+MOCnow1+MOCnow2+MOCnow3);
//        MOC= 1./(1+MOCnow1*MOCfactor1+MOCnow2*MOCfactor2+MOCnow3*MOCfactor3);
        if (MOC>1.) log_info("out of bounds moc_n%d",MOC);
        if (MOC<0.) log_info("out of bounds moc_n%d",MOC);
		nonlinout2a*=MOC;

		//stage 2
		abs_x= ABS(nonlinout2a);

		if(abs_x<dispThresh)
		{			
			compressedNonlin= a * nonlinout2a;
		}
		else
		{
			compressedNonlin=SIGN(nonlinout2a) * ctBM * (REAL)expk(c * logk((accum)(a*(abs_x*recip_ctBM))));
		}

		//stage 3
        filter_1 = nlin_b0 * compressedNonlin + nlin_b1 * nlin_x1b;
        nonlinout1b = filter_1 - nlin_a1 * nlin_y1b[1] - nlin_a2 * nlin_y1b[0];

		nlin_x1b=compressedNonlin;
		nlin_y1b[0]=nlin_y1b[1];
		nlin_y1b[1]=nonlinout1b;

        filter_1 = nlin_b0 * nonlinout1b + nlin_b1 * nlin_y1b[0];
        nonlinout2b = filter_1 - nlin_a1 * nlin_y2b[1] - nlin_a2 * nlin_y2b[0];

		nlin_y2b[0]= nlin_y2b[1];
		nlin_y2b[1]= nonlinout2b;

		//save to buffer
		out_buffer[i]=linout2 + nonlinout2b;
		//if recording MOC
		moc_sample_count++;
		if(moc_sample_count==moc_resample_factor){
		    moc_out_buffer[moc_i]=MOC;
//		    moc_out_buffer[moc_i]=out_buffer[i];
//          moc_out_buffer[moc_i]=MOCspikeCount;
		    if (MOC != 1.) moc_changed = 1;
		    moc_i++;
		    moc_sample_count=0;
		}
	}
	return segment_offset;
}

void app_end(uint null_a,uint null_b)
{
    recording_finalise();
    log_info("total simulation ticks = %d",
        simulation_ticks);
    log_info("processed %d segments",seg_index);
    log_info("sent %d mc packets",mc_tx_count);
    io_printf(IO_BUF,"spinn_exit\n");
    log_info("rx any spikes = %d",rx_any_spikes);
    log_info("moc changed = %d",moc_changed);
    app_complete=true;
    simulation_ready_to_read();
}

void process_handler(uint null_a,uint null_b)
{
        REAL *dtcm_moc;
		seg_index++;
		if (!moc_write_switch)dtcm_moc = dtcm_buffer_moc_x;
		else dtcm_moc = dtcm_buffer_moc_y;
		//choose current buffers
		if(!read_switch && !write_switch)
		{
			index_x=process_chan(dtcm_buffer_x,dtcm_buffer_b,dtcm_moc);
		}
		else if(!read_switch && write_switch)
		{
			index_y=process_chan(dtcm_buffer_y,dtcm_buffer_b,dtcm_moc);
		}
		else if(read_switch && !write_switch)
		{
			index_x=process_chan(dtcm_buffer_x,dtcm_buffer_a,dtcm_moc);
		}
		else
		{
			index_y=process_chan(dtcm_buffer_y,dtcm_buffer_a,dtcm_moc);
		}
        spin1_trigger_user_event(NULL,NULL);
}

void write_complete(uint tid, uint ttag)
{
    #ifdef PROFILE
    profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
    #endif
//    log_info("segment send complete %d",seg_index);
//    //flip write buffers
//    write_switch=!write_switch;
    //send MC packet to connected IHC/AN models
    mc_tx_count++;
    while (!spin1_send_mc_packet(key, 0, NO_PAYLOAD))
    {
        spin1_delay_us(1);
    }
}

void spike_check(uint32_t rx_key,uint null){
    if (check_incoming_spike_id(rx_key)){
//        io_printf(IO_BUF,"MOC spike from %u\n",rx_key);
        moc_spike_count++;
    }
}

void moc_spike_received(uint mc_key, uint null)
{
    spin1_schedule_callback(spike_check,mc_key,NULL,1);
    if (!rx_any_spikes)rx_any_spikes=1;
}


void data_read(uint mc_key, uint payload)
{
    if (mc_key == ome_data_key)
    {
        //payload is OME output value
        //convert payload to float
        MC_union.u = payload;
        //collect the next segment of samples and copy into DTCM
        if(test_DMA == TRUE)
        {
//            if(sample_counter>=n_samples_per_100us){
//	            update_moc_buffer(moc_spike_count);
//	            moc_spike_count = 0;
//	            sample_counter = 0;
//	        }
            MC_seg_idx++;
            #ifdef PROFILE
            if(MC_seg_idx>=SEGSIZE)profiler_write_entry_disable_irq_fiq
            (PROFILER_ENTER | PROFILER_TIMER);
            #endif
            //assign recieve buffer
            if(!read_switch)
            {
                dtcm_buffer_a[MC_seg_idx-1] = MC_union.f;
                //completed filling a segment of input values
                if(MC_seg_idx>=SEGSIZE)
                {
                    MC_seg_idx=0;
                    read_switch=1;
                    spin1_schedule_callback(process_handler,0,0,1);
                }
            }
            else
            {
                dtcm_buffer_b[MC_seg_idx-1] = MC_union.f;
                //completed filling a segment of input values
                if(MC_seg_idx>=SEGSIZE)
                {
                    MC_seg_idx=0;
                    read_switch=0;
                    spin1_schedule_callback(process_handler,0,0,1);
                }
            }
//            sample_counter++;
        }
    }
}

void app_done ()
{
    #ifdef PROFILE
	profiler_finalise();
    #endif
}

void count_ticks(uint null_a, uint null_b){

    update_moc_buffer(moc_spike_count);
    moc_spike_count = 0;
    time++;
    if (time>simulation_ticks && !app_complete)spin1_schedule_callback(app_end,NULL,NULL,2);
}

void c_main()
{
    // Get core and chip IDs
    coreID = spin1_get_core_id ();
    chipID = spin1_get_chip_id ();
    uint32_t timer_period;

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    if(app_init(&timer_period)){
        // Set timer tick (in microseconds)
        log_info("setting timer tick callback for %d microseconds",timer_period);
        spin1_set_timer_tick(timer_period);

        //setup callbacks
        //process channel once data input has been read to DTCM
        simulation_dma_transfer_done_callback_on(DMA_WRITE, write_complete);
        spin1_callback_on (MCPL_PACKET_RECEIVED,data_read,-1);
        spin1_callback_on (MC_PACKET_RECEIVED,moc_spike_received,-1);
        spin1_callback_on (USER_EVENT,data_write,0);
        spin1_callback_on (TIMER_TICK,count_ticks,0);

        simulation_run();
    }
}

