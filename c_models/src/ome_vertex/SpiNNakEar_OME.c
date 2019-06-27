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
REAL Fs,dt,max_rate;
uint coreID;
uint chipID;
uint test_DMA;
uint seg_index;
uint read_switch;
uint write_switch;
uint processing;
uint index_x;
uint index_y;
uint TOTAL_TICKS;
uint TIMER_TICK_PERIOD;
uint final_ack;

bool app_complete=false;

uint_float_union MC_union;

REAL conchaL,conchaH,conchaG,earCanalL,earCanalH,earCanalG,stapesH,stapesL,stapesScalar,
	ARtau,ARdelay,ARrateThreshold,rateToAttenuationFactor,BFlength;

REAL concha_q,concha_j,concha_k,concha_l;

REAL conchaGainScalar,recip_conchaFilter_a0,
	earCanal_q,earCanal_j,earCanal_k,earCanal_l,earCanalGainScalar,recip_earCanalFilter_a0,
	ARatt,Wn,stapesHP_order,sf,stapesLP_b,stapes_tau,past_stapesDisp;

REAL conchaFilter_b[3],conchaFilter_a[3],earCanalFilter_b[3],earCanalFilter_a[3],stapesHP_b[3],stapesHP_a[3],stapesLP_a[2],past_input[2],past_concha[2],past_earCanalInput[2],past_earCanal[2],past_stapesInput[2]
,past_stapes[2];


int start_count_process;
int end_count_process;
int start_count_read;
int end_count_read;
int start_count_write;
int end_count_write;
int start_count_full;
int end_count_full;

uint sync_count=0;
uint read_ticks;
bool first_tick = true;

REAL *dtcm_buffer_a;
REAL *dtcm_buffer_b;
uint *dtcm_buffer_x;

REAL *sdramin_buffer;
REAL *sdramout_buffer;
REAL *profile_buffer;

//data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    RECORDING,
    PROFILER}regions;

// The parameters to be read from memory
enum params_enum {
    DATA_SIZE = 0,
    COREID,
    NUM_DRNL,
    NUM_MACK,
    FS,
    NUM_BFS,
    KEY,
    R2S_KEY,
    TS,
    SHB1,
    SHB2,
    SHB3,
    SHA1,
    SHA2,
    SHA3,
    DATA
    };

struct parameters {
    uint DATA_SIZE;
    uint COREID;
    uint NUM_DRNL;
    uint NUM_MACK;
    uint FS;
    uint NUM_BFS;
    uint KEY;
    uint R2S_KEY;
    uint TS;
    REAL SHB1;
    REAL SHB2;
    REAL SHB3;
    REAL SHA1;
    REAL SHA2;
    REAL SHA3;
    uint DATA;
    };

// The general parameters
struct parameters params;

// The size of the remaining data to be sent
uint data_size;
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
//model Fs
uint sampling_frequency;

//application initialisation
bool app_init(void)
{
    m_pi = acos(-1.0);//pi reference for filter constant calcs
	seg_index=0;
	read_switch=0;
	write_switch=0;

	io_printf (IO_BUF, "[core %d] -----------------------\n", coreID);
	io_printf (IO_BUF, "[core %d] starting simulation\n", coreID);
	//obtain data spec
	address_t data_address = data_specification_get_data_address();

    // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM, data_address),
            APPLICATION_NAME_HASH, NULL, NULL,
            NULL, 1, 0)) {
        return false;
    }

    address_t parameters_address = data_specification_get_region(PARAMS, data_address);
    address_t params_enum = data_specification_get_region(PARAMS, data_address);

    struct parameters *ome_params = (struct parameters *) parameters_address;
        spin1_memcpy(&params, ome_params, sizeof(struct parameters));

    // Get the size of the data in words
    data_size = params.DATA_SIZE;
    read_ticks=0;
    TOTAL_TICKS= data_size/SEGSIZE;
    log_info("data_size=%d",data_size);
    log_info("TOTAL_TICKS=%d",TOTAL_TICKS);

    // Get a pointer to the input data buffer
    sdramin_buffer = (REAL *) &(params_enum[DATA+7]);//+7 to account for double types filter params
    log_info("sdramin_buffer=0x%08x",(uint)sdramin_buffer);
    //obtain this core ID from the host placement perspective
    placement_coreID = params.COREID;

    // Get the key to send the data with
    key = params.KEY;
    io_printf (IO_BUF, "OME-->DRNL key=%d\n",key);

    r2s_key=params.R2S_KEY;
    log_info("r2s key=%d",r2s_key);

    time_scale=params.TS;
    log_info("time_scale=%d",time_scale);

    //Get number of child DRNL vertices
    num_drnls=params.NUM_DRNL;
    io_printf (IO_BUF, "num drnls=%d\n",num_drnls);

    //get number of multi-cast acknowledges
    num_macks=params.NUM_MACK;
    io_printf (IO_BUF, "num macks=%d\n",num_macks);

    //Get sampling frequency
    sampling_frequency = params.FS;

    Fs= (double)sampling_frequency;
    dt=(1.0/Fs);
    TIMER_TICK_PERIOD = (uint)(1e6 * ((REAL)SEGSIZE/Fs) * time_scale);//scaled by simulation time scale

    log_info("timer period=%d",(uint)TIMER_TICK_PERIOD);
    log_info("Fs=%d",sampling_frequency);
    final_ack=0;

	// Allocate buffers
	//input double buffers
	dtcm_buffer_a = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_buffer_b = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
    //output buffer (uint ready for MC payload)
	dtcm_buffer_x = (uint *) sark_alloc (SEGSIZE, sizeof(uint));

	if(dtcm_buffer_a == NULL ||dtcm_buffer_b == NULL
	    || dtcm_buffer_x== NULL ||sdramin_buffer == NULL)
	{
		test_DMA = FALSE;
		io_printf (IO_BUF, "[core %d] error - cannot allocate buffer\n", coreID);
		return false;
	}
	else
	{
        test_DMA = TRUE;
        // initialise buffers
        for (uint i = 0; i < SEGSIZE; i++)
        {
            dtcm_buffer_a[i]   = 0;
            dtcm_buffer_b[i]   = 0;
        }
        for (uint i = 0; i < SEGSIZE; i++)
        {
            dtcm_buffer_x[i]   = 0;
        }

        io_printf (IO_BUF, "[core %d] data spec output buffer tag= %d\n", coreID,
           (uint) placement_coreID);

        //============MODEL INITIALISATION================//
        BFlength = params.NUM_BFS;
        conchaL=1500.0;
        conchaH=3000.0;
        conchaG=5.0;
        earCanalL=3000.0;
        earCanalH=3800.0;
        earCanalG=5.0;
        stapesH=700.0;
        stapesL=10.0;
        stapesScalar=5e-7;
        ARtau=0.2;
        ARdelay=0.0085;
        ARrateThreshold=100.0;
        rateToAttenuationFactor=0.1/BFlength;

        concha_q= (REAL)m_pi * (REAL)dt * (REAL)(conchaH - conchaL);
        concha_j= 1.0/(1.0+ (1.0/tan(concha_q)));
        concha_k= (2.0 * cos((REAL)m_pi * (REAL)dt * (REAL)(conchaH + conchaL)))
                    / ((1.0 + tan(concha_q)) * cos(concha_q));
        concha_l= (tan(concha_q) - 1.0)/(tan(concha_q) + 1.0);
        conchaGainScalar=pow(10.0,conchaG/20.0);

        conchaFilter_b[0]=concha_j;
        conchaFilter_b[1]=0.0;
        conchaFilter_b[2]=-1.0*concha_j;
        conchaFilter_a[0]=1.0;
        conchaFilter_a[1]=-1.0*concha_k;
        conchaFilter_a[2]=-1.0*concha_l;
        recip_conchaFilter_a0=1.0/conchaFilter_a[0];

        earCanal_q= m_pi * dt * (earCanalH - earCanalL);
        earCanal_j= 1.0/(1.0+ (1.0/tan(earCanal_q)));
        earCanal_k= (2.0 * cos(m_pi * dt * (earCanalH + earCanalL))) / ((1.0 + tan(earCanal_q)) * cos(earCanal_q));
        earCanal_l= (tan(earCanal_q) - 1.0)/(tan(earCanal_q) + 1.0);
        earCanalGainScalar=pow(10.0,earCanalG/20.0);

        earCanalFilter_b[0]=earCanal_j;
        earCanalFilter_b[1]=0.0;
        earCanalFilter_b[2]=-earCanal_j;
        earCanalFilter_a[0]=1.0;
        earCanalFilter_a[1]=-earCanal_k;
        earCanalFilter_a[2]=-earCanal_l;
        recip_earCanalFilter_a0=1.0/earCanalFilter_a[0];

        //stapes filter coeffs pre generated due to butterworth calc code overflow
        stapesHP_b[0] = params.SHB1;
        stapesHP_b[1] = params.SHB2;
        stapesHP_b[2] = params.SHB3;
        stapesHP_a[0] = params.SHA1;
        stapesHP_a[1] = params.SHA2;
        stapesHP_a[2] = params.SHA3;

        stapes_tau= 1.0/ (2 * (REAL)m_pi * stapesL);
        stapesLP_a[0]= 1.0;
        stapesLP_a[1]= dt/stapes_tau -1.0;
        stapesLP_b= 1.0 + stapesLP_a[1];

        //--Initialise recurring variables--/
        past_input[0]=0.0;
        past_input[1]=0.0;
        past_concha[0]=0.0;
        past_concha[1]=0.0;

        past_earCanalInput[0]=0.0;
        past_earCanalInput[1]=0.0;
        past_earCanal[0]=0.0;
        past_earCanal[1]=0.0;

        ARatt=1.0; //TODO: this should be determined by spiking input

        past_stapesInput[0]=0.0;
        past_stapesInput[1]=0.0;
        past_stapes[0]=0.0;
        past_stapes[1]=0.0;

        past_stapesDisp=0.0;

    #ifdef PROFILE
        // Setup profiler
        profiler_init(
            data_specification_get_region(1, data_address));
    #endif
        log_info("init complete");
	}
    return true;
}

void app_end(uint null_a,uint null_b)
{
    app_complete = true;
    log_info("All data has been sent seg_index=%d",seg_index);
    //send 10 times in case the network is congested and drops packets
//    for (uint i=0;i<10;i++){
//        while (!spin1_send_mc_packet(r2s_key|1, 0, WITH_PAYLOAD)) {
//            spin1_delay_us(1);
//        }
//        spin1_delay_us(10000);//10ms
//    }
//    spin1_exit(0);
    app_done ();
    simulation_ready_to_read();
    io_printf (IO_BUF, "spinn_exit %\n");

    /*if(final_ack==1)
    {
        spin1_exit(0);
        io_printf (IO_BUF, "spinn_exit %\n");
    }
    else
    {
        log_info("All data has been sent seg_index=%d",seg_index);
        while (!spin1_send_mc_packet(r2s_key|1, 0, WITH_PAYLOAD)) {
            spin1_delay_us(1);
        }
        final_ack=1;
//        //ROB HACK TO REMOVE SECOND ACKS
//        spin1_exit(0);
//        io_printf (IO_BUF, "spinn_exit %\n");
    }*/
}
//DMA read
void data_read(uint null_a, uint null_b)
{
	REAL *dtcm_buffer_in;
/*	if(test_DMA == TRUE && sync_count<num_macks && seg_index==0)
	{
	    if(first_tick==true){
		io_printf (IO_BUF, "sending r2s\n");
	        while (!spin1_send_mc_packet(r2s_key|1, 0, WITH_PAYLOAD)) {
                spin1_delay_us(1);
            }
            first_tick = false;
        }
	}*/
	//read from DMA and copy into DTCM
//	else if(read_ticks<TOTAL_TICKS && test_DMA == TRUE && sync_count==num_macks)
	if(read_ticks<TOTAL_TICKS && test_DMA == TRUE)
	{
	    #ifdef PROFILE
	    if(seg_index==0)profiler_write_entry_disable_irq_fiq(PROFILER_ENTER | PROFILER_DMA_READ);
        #endif
	    read_ticks++;
		//assign recieve buffer
		if(!read_switch)
		{
			dtcm_buffer_in=dtcm_buffer_a;
			read_switch=1;
		}
		else
		{
			dtcm_buffer_in=dtcm_buffer_b;
			read_switch=0;
		}

		spin1_dma_transfer(DMA_READ,&sdramin_buffer[seg_index*SEGSIZE], dtcm_buffer_in, DMA_READ,
			   SEGSIZE*sizeof(REAL));
	}
   	// stop if desired number of ticks reached
	/*else if (read_ticks >= (TOTAL_TICKS+3))//+ 3 ticks to account for OME+DRNL+IHCAN processing latency
	{

	    if(final_ack==1)//2nd time in this call therefore
	                    //child models have replied with finished processing acks
	    {
            spin1_schedule_callback(app_end,NULL,NULL,2);
	    }
	    else
	    {
            sync_count=0;
	        //send finished processing MC packet
            spin1_schedule_callback(app_end,NULL,NULL,2);
	    }
	}*/
//    else if (read_ticks>=TOTAL_TICKS && sync_count==num_macks) read_ticks++;//additional latency wait
//    else if (read_ticks>=TOTAL_TICKS) read_ticks++;//additional latency wait
    else if (read_ticks>=TOTAL_TICKS && !app_complete) spin1_schedule_callback(app_end,NULL,NULL,2);//additional latency wait
//    else if ((read_ticks>TOTAL_TICKS + 3) && !app_complete)spin1_schedule_callback(app_end,NULL,NULL,2);
}

void process_chan(REAL *in_buffer)
{
    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_ENTER | PROFILER_TIMER);
    #endif
	uint i,j,k;
	uint si=0;
	REAL concha,earCanalInput,earCanalRes,earCanalOutput,ARoutput,stapesVelocity,stapesDisplacement;
	REAL filter_1,diff;
	REAL sub;
#ifdef PRINT
	io_printf (IO_BUF, "[core %d] segment %d (offset=%d) starting processing\n", coreID,seg_index,segment_offset);
#endif

	for(i=0;i<SEGSIZE;i++)
	{
		//concha
        filter_1 = conchaFilter_b[0]*in_buffer[i]
                    + conchaFilter_b[1]*past_input[0]
                    + conchaFilter_b[2]*past_input[1];
        concha = conchaFilter_a[0]*filter_1
                    - conchaFilter_a[1]*past_concha[0]
                    - conchaFilter_a[2]*past_concha[1];

		//update vars
		past_input[1]=past_input[0];
		past_input[0]=in_buffer[i];

		past_concha[1]=past_concha[0];
		past_concha[0]=concha;

		earCanalInput= conchaGainScalar * concha + in_buffer[i];

		//ear canal
		filter_1 = earCanalFilter_b[0]*earCanalInput
				    + earCanalFilter_b[1]*past_earCanalInput[0]
				    + earCanalFilter_b[2]*past_earCanalInput[1];
        earCanalRes = earCanalFilter_a[0] * filter_1
                      - earCanalFilter_a[1]*past_earCanal[0]
                      - earCanalFilter_a[2]*past_earCanal[1];
		//update vars
		past_earCanalInput[1]=past_earCanalInput[0];
		past_earCanalInput[0]=earCanalInput;

		past_earCanal[1]=past_earCanal[0];
		past_earCanal[0]=earCanalRes;

		earCanalOutput= earCanalGainScalar * earCanalRes + earCanalInput;

		//Acoustic Reflex
		ARoutput= ARatt * stapesScalar * earCanalOutput;
        filter_1 = stapesHP_b[0] * ARoutput +
				    stapesHP_b[1] * past_stapesInput[0] +
				    stapesHP_b[2] * past_stapesInput[1];
        stapesVelocity = stapesHP_a[0] * filter_1
                         - stapesHP_a[1] * past_stapes[0]
				         - stapesHP_a[2] * past_stapes[1];
		//update vars
		past_stapesInput[1]= past_stapesInput[0];
		past_stapesInput[0]= ARoutput;
		past_stapes[1]= past_stapes[0];
		past_stapes[0]= stapesVelocity;

		//stapes displacement
		filter_1 = stapesLP_b * stapesVelocity;
		sub = stapesLP_a[1] * past_stapesDisp;
        stapesDisplacement = filter_1 - sub;
		//update vars
		past_stapesDisp=stapesDisplacement;

        //assign output to float/uint union
		MC_union.f = stapesDisplacement;
		//transmit uint output as MC with payload to all DRNLs
        spin1_send_mc_packet(key,MC_union.u, WITH_PAYLOAD);
	}

    #ifdef PROFILE
    profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
    #endif
}

void transfer_handler(uint tid, uint ttag)
{
	if (ttag==DMA_READ)
	{
	//increment segment index
		seg_index++;

		//choose current available buffers
		if(!read_switch)
		{
			process_chan(dtcm_buffer_b);
		}
		else
		{
			process_chan(dtcm_buffer_a);
		}
	}
	else
	{
		io_printf(IO_BUF,"[core %d] invalid %d DMA tag!\n",coreID,ttag);
	}
}

void sync_check(uint mc_key, uint null)
{
    sync_count++;
    io_printf (IO_BUF,"ack mcpacket recieved from key=%d sync_count=%d seg_index=%d\n",mc_key-2,sync_count,seg_index);
}

void app_done ()
{
#ifdef PROFILE
  profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_DMA_READ);
  profiler_finalise();
#endif
	log_info("b0:%k",(accum)stapesHP_b[0]);
    log_info("b1:%k",(accum)stapesHP_b[1]);
    log_info("b2:%k",(accum)stapesHP_b[2]);
	log_info("a0:%k",(accum)stapesHP_a[0]);
	log_info("a1:%k",(accum)stapesHP_a[1]);
    log_info("a2:%k",(accum)stapesHP_a[2]);
  // report simulation time
  io_printf (IO_BUF, "[core %d] simulation lasted %d ticks\n", coreID,
             spin1_get_simulation_time());

  // say goodbye
  io_printf (IO_BUF, "[core %d] stopping simulation\n", coreID);
}

void c_main()
{
    // Get core and chip IDs
    coreID = spin1_get_core_id ();
    chipID = spin1_get_chip_id ();

    if(app_init())
    {
        //set timer tick
        spin1_set_timer_tick (TIMER_TICK_PERIOD);
        //setup callbacks
        //process channel once data input has been read to DTCM
        spin1_callback_on (DMA_TRANSFER_DONE,transfer_handler,1);
        //reads from DMA to DTCM every tick
        spin1_callback_on (TIMER_TICK,data_read,0);
        //start/end of simulation syncronisation callback
//        log_info("setting up MC callback");
//        spin1_callback_on (MC_PACKET_RECEIVED,sync_check,-1);
//        spin1_start (SYNC_WAIT);
        simulation_run();
//        app_done ();
    }
}

