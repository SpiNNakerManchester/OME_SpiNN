// DO NOT EDIT! THIS FILE WAS GENERATED FROM src/DRNL_SpiNN_pipe_MC.c

/*
 ============================================================================
 Name        : IHC_AN_softfloat.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdfix.h>
#include "DRNL_SpiNN.h"
#include "spin1_api.h"
#include "math.h"
#include "complex.h"
#include "random.h"
#include "stdfix-exp.h"
#include "log.h"
#include <data_specification.h>
#include <profiler.h>
#include <profile_tags.h>
#include <simulation.h>
//#define PROFILE

//=========GLOBAL VARIABLES============//
REAL Fs,dt,max_rate;
uint coreID;
uint chipID;
uint test_DMA;
uint seg_index;
uint cbuff_index;
uint cbuff_numseg;
uint read_switch;
uint write_switch;
uint processing;
uint index_x;
uint index_y;
uint MC_seg_idx;
uint_float_union MC_union;
uint ack_rx=0;

/*REAL cf,lin_cf,nlin_b0,nlin_b1,nlin_b2,nlin_a1,nlin_a2,nlBWp,nlBWq,linBWp,linBWq,linCFp,linCFq,
    nlin_bw,nlin_phi,nlin_theta,nlin_cos_theta,nlin_sin_theta,nlin_alpha,
    lin_b0,lin_b1,lin_b2,lin_a1,lin_a2,lin_bw,lin_phi,lin_theta,lin_cos_theta,
    lin_sin_theta,lin_alpha,lin_gain,

a,ctBM,dispThresh,recip_ctBM,MOC,MOCnow1,MOCnow2,MOCnow3,MOCdec1,MOCdec2,MOCdec3,MOCfactor1,
MOCfactor2,MOCfactor3,rateToAttentuationFactor,MOCspikeCount;*/
//ctBM,dispThresh;

REAL cf,nlin_b0,nlin_b1,nlin_b2,nlin_a1,nlin_a2,
    lin_b0,lin_b1,lin_b2,lin_a1,lin_a2,lin_gain,

a,ctBM,dispThresh,recip_ctBM,MOC,MOCnow1,MOCnow2,MOCnow3,MOCdec1,MOCdec2,MOCdec3,MOCfactor1,
MOCfactor2,MOCfactor3,MOCspikeCount;

accum c;
//

REAL lin_x1;
REAL lin_y1[2],lin_y2[2];

REAL nlin_x1a;
REAL nlin_y1a[2],nlin_y2a[2];
REAL nlin_x1b;
REAL nlin_y1b[2],nlin_y2b[2];
REAL MOCtau[3],MOCtauweights[3];

//uint seed_selection[SEED_SEL_SIZE];//TODO:this needs to be moved to SDRAM

int start_count_process;
int end_count_process;
int start_count_read;
int end_count_read;
int start_count_write;
int end_count_write;

uint sync_count=0;

float *dtcm_buffer_a;
float *dtcm_buffer_b;
double *dtcm_buffer_x;
double *dtcm_buffer_y;
float *dtcm_profile_buffer;

float *sdramin_buffer;
double *sdramout_buffer;
float *profile_buffer;

//data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    RECORDING,
    PROFILER}regions;
// The parameters to be read from memory
enum params {
    DATA_SIZE = 0,
    OMECOREID,
    COREID,
    OMEAPPID,
    OME_KEY,
    KEY,
    NUM_IHCAN,
    CF,
    DELAY,
    FS
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

//application initialisation
void app_init(void)
{
	seg_index=0;
	cbuff_index=0;
	cbuff_numseg=3;
	read_switch=0;
	write_switch=0;

	//io_printf (IO_BUF, "[core %d] -----------------------\n", coreID);
	//io_printf (IO_BUF, "[core %d] starting simulation\n", coreID);

    //obtain data spec
	address_t data_address = data_specification_get_data_address();
    address_t params = data_specification_get_region(0, data_address);

	// Get the size of the data in words
    data_size = params[DATA_SIZE];

    //obtain ome core ID from the host placement perspective
    ome_coreID = params[OMECOREID];
    //obtain this core ID from the host placement perspective
    placement_coreID = params[COREID];

    //obtain ome application ID from the host placement perspective
    ome_appID = params[OMEAPPID];

    ome_key=params[OME_KEY];
    log_mini_info("%u%d", 9013,ome_key);  /* "omekey:%d"*/

    key=params[KEY];
    log_mini_info("%u%d", 9014,key);  /* "key:%d"*/

    //get the mask needed for comms protocol from MC keys
    mask = 3;//1;

    num_ihcans=params[NUM_IHCAN];

    drnl_cf=params[CF];

    delay = params[DELAY];

  //  log_info("delay=%d\n",delay);
    //log_info("key=%d\n",key);
    //log_info("ome key=%d\n",ome_key);
  //  log_info("mask=%d\n",mask);
    //log_info("data_size=%d",data_size);

    //log_info("num_ihcans=%d\n",num_ihcans);

    log_mini_info("%u%d", 9015,drnl_cf);  /* "CF=%d\n"*/

    //Get sampling frequency
    sampling_frequency = params[FS];

    Fs= (REAL)sampling_frequency;
	dt=(1.0/Fs);

    // Allocate buffers somewhere in SDRAM
	//output results buffer
	//hack for smaller SDRAM intermediate buffers
	data_size=cbuff_numseg*SEGSIZE;

	sdramout_buffer = (double *) sark_xalloc (sv->sdram_heap,
					 data_size * sizeof(double),
					 placement_coreID,
					 ALLOC_LOCK);

	/*profile_buffer = (REAL *) sark_xalloc (sv->sdram_heap,
					3 * ((uint)44100./SEGSIZE) *sizeof(REAL),
					placement_coreID|64,
					ALLOC_LOCK);*/
	
	// and a buffer in DTCM
	
	dtcm_buffer_a = (float *) sark_alloc (SEGSIZE, sizeof(float));
	dtcm_buffer_b = (float *) sark_alloc (SEGSIZE, sizeof(float));
	dtcm_buffer_x = (double *) sark_alloc (SEGSIZE, sizeof(double));
	dtcm_buffer_y = (double *) sark_alloc (SEGSIZE, sizeof(double));
	//dtcm_profile_buffer = (REAL *) sark_alloc (3*TOTAL_TICKS, sizeof(REAL));
	
	if (dtcm_buffer_a == NULL ||dtcm_buffer_b == NULL ||dtcm_buffer_x == NULL ||dtcm_buffer_y == NULL 
			||  sdramout_buffer == NULL )//||  dtcm_profile_buffer == NULL)
	/*if (sdramout_buffer == NULL || sdramin_buffer == NULL || dtcm_buffer_y == NULL 
			|| dtcm_buffer_a == NULL || dtcm_buffer_b == NULL || dtcm_profile_buffer == NULL ||dtcm_buffer_x == NULL)*/
	{
		test_DMA = FALSE;
		//io_printf (IO_BUF, "[core %d] error - cannot allocate buffer\n", coreID);
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
	
		for (uint i=0;i<data_size;i++)
		{
			sdramout_buffer[i]  = 0;
		}

        MC_seg_idx=0;
	}

	
	//============MODEL INITIALISATION================//
    double complex lin_z1,lin_z2,lin_z3,lin_tf,nlin_z1,nlin_z2,nlin_z3,nlin_tf;
    double rateToAttentuationFactor,lin_cf,nlBWp,nlBWq,linBWp,linBWq,linCFp,linCFq,
        nlin_bw,nlin_phi,nlin_theta,nlin_cos_theta,nlin_sin_theta,nlin_alpha,
        lin_bw,lin_phi,lin_theta,lin_cos_theta,lin_sin_theta,lin_alpha;

	//set center frequency
	//cf=4000.0;
	cf=(REAL)drnl_cf;

	//non-linear pathway
	nlBWq=180.0;
	nlBWp=0.14;
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
	nlin_b1 = nlin_alpha * nlin_b0;

	/*io_printf(IO_BUF,"b0:%k\n",(accum)nlin_b0);
    io_printf(IO_BUF,"b1:%k\n",(accum)nlin_b1);
	io_printf(IO_BUF,"a1:%k\n",(accum)nlin_a1);
	io_printf(IO_BUF,"a2:%k\n",(accum)nlin_a2);*/

	//compression algorithm variables
	a=30e4;//5e4;
	c=0.25k;
	ctBM = 1e-9 * pow(10.0,32.0/20.0);
	//ctBM=3.981071705534974e-08;
	recip_ctBM=1.0/ctBM;
	dispThresh=ctBM/a;

	//linear pathway
	lin_gain=200.0;
	linBWq=235.0;
	linBWp=0.2;
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
	lin_b1 = lin_alpha * lin_b0;

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

	//MOC=1.0; // TODO change this to be a model input
	rateToAttentuationFactor = 20e4;

	MOCnow1=0.0;
	MOCnow2=0.0;
	MOCnow3=0.0;

	MOCtau[0] = 0.055;
	MOCtau[1] = 0.4;
    MOCtau[2] = 1;

    MOCtauweights[0] = 0.9;
    MOCtauweights[1] = 0.1;
    MOCtauweights[2] = 0;

    MOCdec1 = exp(- dt/MOCtau[0]);
    MOCdec2 = exp(- dt/MOCtau[1]);
    MOCdec3 = exp(- dt/MOCtau[2]);

    MOCfactor1 = 0.01 * rateToAttentuationFactor * MOCtauweights[0] * dt;
    MOCfactor2 = 0.01 * rateToAttentuationFactor * MOCtauweights[1] * dt;
    MOCfactor3 = 0.01 * rateToAttentuationFactor * MOCtauweights[2] * dt;

    MOCspikeCount=0;

#ifdef PROFILE
    // configure timer 2 for profiling
    // enabled, free running, interrupt disabled, no pre-scale, 32 bit, free-running mode
    //tc[T2_CONTROL] = TIMER2_CONF;
        // Setup profiler
    profiler_init(
        data_specification_get_region(1, data_address));
#endif
    
}

void app_end(uint null_a,uint null_b)
{
    if( sync_count<num_ihcans)
    {
        //send end MC packet to child IHCANs
        //log_info("sending final packet to IHCANs\n");
        while (!spin1_send_mc_packet(key|1, 0, NO_PAYLOAD)) {
            spin1_delay_us(1);
        }
        //wait for acknowledgment from child IHCANs
        while(sync_count<num_ihcans)
        {
           spin1_delay_us(1);
        }
    }
    //log_info("all final acks received, sending final ack to OME and closing\n");
    //random delay to prevent network lock up
    //spin1_delay_us(delay);
    //send final ack packet back to parent OME
    while (!spin1_send_mc_packet(ome_key|2, 0, NO_PAYLOAD)) {
        spin1_delay_us(1);
    }
    //io_printf (IO_BUF, "spinn_exit %d\n",seg_index);
    //simulation_exit();
    spin1_exit (0);
}

void data_write(uint null_a, uint null_b)
{
	double *dtcm_buffer_out;
	uint out_index;
	
	if(test_DMA == TRUE)
	{
		if(!write_switch)
		{
			out_index=index_x;
			dtcm_buffer_out=dtcm_buffer_x;
#ifdef PRINT	
			io_printf (IO_BUF, "buff_x write\n");
#endif
		}
		else
		{
			out_index=index_y;
			dtcm_buffer_out=dtcm_buffer_y;
#ifdef PRINT
			io_printf (IO_BUF, "buff_y write\n");
#endif
		}
#ifdef PROFILE
  //start_count_write = tc[T2_COUNT];
#endif
		spin1_dma_transfer(DMA_WRITE,&sdramout_buffer[out_index],dtcm_buffer_out,DMA_WRITE,
		  						SEGSIZE*sizeof(double));
#ifdef PRINT
  log_mini_info("%u%d%d%08x", 9016, coreID,seg_index,(uint) &sdramout_buffer[out_index]);
  /* "[core %d] segment %d written @ 0x%08x\n"*/

  log_mini_info("%u%d%d%08x%08x", 9017, coreID,seg_index,(uint) &sdramout_buffer[out_index],(uint) &sdramout_buffer[out_index+(NUMFIBRES-1)*SEGSIZE+SEGSIZE-1]);
  /* "[core %d] segment %d written to @ 0x%08x - 0x%08x\n"*/
#endif
	}
}

uint process_chan(double *out_buffer,float *in_buffer)
{
	//uint segment_offset=SEGSIZE*(seg_index-1);
	uint segment_offset=SEGSIZE*(cbuff_index);
	uint i;		
	REAL linout1,linout2,nonlinout1a,nonlinout2a,nonlinout1b,nonlinout2b,abs_x,compressedNonlin;
	REAL filter_1;

	for(i=0;i<SEGSIZE;i++)
	{
	    //if(in_buffer[i]!=0.0)log_info("%k",(accum)in_buffer[i]);
		//Linear Path
		/*linout1= lin_b0 * in_buffer[i] + lin_b1 * lin_x1 -
				lin_a1 * lin_y1[1] - lin_a2 * lin_y1[0];		*/
        filter_1 = lin_b0 * in_buffer[i] + lin_b1 * lin_x1;
        linout1= filter_1 - lin_a1 * lin_y1[1] - lin_a2 * lin_y1[0];

		lin_x1=in_buffer[i];
		lin_y1[0]=lin_y1[1];
		lin_y1[1]=linout1;

		/*linout2= lin_b0 * linout1 + lin_b1 * lin_y1[0] -
				lin_a1 * lin_y2[1] - lin_a2 * lin_y2[0];*/
        /*MAP_BS update*/
        /*linout2= lin_gain*lin_b0 * linout1 + lin_b1 * lin_y1[0] -
				lin_a1 * lin_y2[1] - lin_a2 * lin_y2[0];*/
        filter_1 = lin_gain*lin_b0 * linout1 + lin_b1 * lin_y1[0];
        linout2= filter_1 - lin_a1 * lin_y2[1] - lin_a2 * lin_y2[0];
		
		lin_y2[0]= lin_y2[1];
		lin_y2[1]= linout2;

		//non-linear path
		//stage 1
		/*nonlinout1a= nlin_b0 * in_buffer[i] + nlin_b1 * nlin_x1a -
				     nlin_a1 * nlin_y1a[1] - nlin_a2 * nlin_y1a[0];*/
        filter_1 =  nlin_b0 * in_buffer[i] + nlin_b1 * nlin_x1a;
        nonlinout1a = filter_1 - nlin_a1 * nlin_y1a[1] - nlin_a2 * nlin_y1a[0];

		nlin_x1a=in_buffer[i];
		nlin_y1a[0]=nlin_y1a[1];
		nlin_y1a[1]=nonlinout1a;

		/*nonlinout2a= nlin_b0 * nonlinout1a + nlin_b1 * nlin_y1a[0] -
				     nlin_a1 * nlin_y2a[1] - nlin_a2 * nlin_y2a[0];*/
        filter_1 = nlin_b0 * nonlinout1a + nlin_b1 * nlin_y1a[0];
        nonlinout2a = filter_1 - nlin_a1 * nlin_y2a[1] - nlin_a2 * nlin_y2a[0];

		nlin_y2a[0]= nlin_y2a[1];
		nlin_y2a[1]= nonlinout2a;

		//MOC efferent effects
        MOCnow1= MOCnow1* MOCdec1+ MOCspikeCount* MOCfactor1;
        MOCnow2= MOCnow2* MOCdec2+ MOCspikeCount* MOCfactor2;
        MOCnow3= MOCnow3* MOCdec3+ MOCspikeCount* MOCfactor3;
        // MOC= 1 when all MOCnow are zero
        // 0 < MOC < 1
        MOC= 1./(1+MOCnow1+MOCnow2+MOCnow3);
		nonlinout2a*=MOC;
		//stage 2
		abs_x= ABS(nonlinout2a);

		if(abs_x<dispThresh)
		{			
			compressedNonlin= a * nonlinout2a;
		}
		else// if(abs_x>0.0)//compress
		{
			//compressedNonlin=SIGN(nonlinout2a) * ctBM * (REAL)expk(c * logk((accum)(a*(abs_x*recip_ctBM))));
			compressedNonlin= SIGN(nonlinout2a) * ctBM * exp((double)c * log((double)(a * (abs_x / ctBM))));
		}	
		/*else
		{
			compressedNonlin=0.0;
		}			*/

		//stage 3 
		/*nonlinout1b= nlin_b0 * compressedNonlin + nlin_b1 * nlin_x1b -
				 nlin_a1 * nlin_y1b[1] - nlin_a2 * nlin_y1b[0];*/
        filter_1 = nlin_b0 * compressedNonlin + nlin_b1 * nlin_x1b;
        nonlinout1b = filter_1 - nlin_a1 * nlin_y1b[1] - nlin_a2 * nlin_y1b[0];

		nlin_x1b=compressedNonlin;
		nlin_y1b[0]=nlin_y1b[1];
		nlin_y1b[1]=nonlinout1b;

		/*nonlinout2b= nlin_b0 * nonlinout1b + nlin_b1 * nlin_y1b[0] -
				nlin_a1 * nlin_y2b[1] - nlin_a2 * nlin_y2b[0];*/
        filter_1 = nlin_b0 * nonlinout1b + nlin_b1 * nlin_y1b[0];
        nonlinout2b = filter_1 - nlin_a1 * nlin_y2b[1] - nlin_a2 * nlin_y2b[0];

		nlin_y2b[0]= nlin_y2b[1];
		nlin_y2b[1]= nonlinout2b;

		//save to buffer
	    //out_buffer[i]=linout2*lin_gain + nonlinout2b;
		//MAP_BS update
		out_buffer[i]=linout2 + nonlinout2b;
		//out_buffer[i]=in_buffer[i];//compressedNonlin;//nonlinout2a;//linout2;//linout1;//nonlinout1a;//nlin_b0;//nonlinout1a;//nonlinout2b;//nonlinout1a;
	}
	//log_info("processing complete %d",seg_index);
	MOCspikeCount = 0;
	return segment_offset;
}

void process_handler(uint null_a,uint null_b)
{
    	#ifdef PROFILE
        //profiler_write_entry_disable_irq_fiq(PROFILER_ENTER | PROFILER_TIMER);
        #endif
    //increment segment index
		seg_index++;
	    //check circular buffer
		if(cbuff_index<cbuff_numseg-1)
		{    //increment circular buffer index
		    cbuff_index++;
		}
		else
		{
		    cbuff_index=0;
		}
		//choose current buffers
		if(!read_switch && !write_switch)
		{
			index_x=process_chan(dtcm_buffer_x,dtcm_buffer_b);

		}
		else if(!read_switch && write_switch)
		{
			index_y=process_chan(dtcm_buffer_y,dtcm_buffer_b);
		}
		else if(read_switch && !write_switch)
		{
			index_x=process_chan(dtcm_buffer_x,dtcm_buffer_a);
		}
		else
		{
			index_y=process_chan(dtcm_buffer_y,dtcm_buffer_a);
		}
    	spin1_trigger_user_event(NULL,NULL);
}

void transfer_handler(uint tid, uint ttag)
{
	/*if (ttag==DMA_READ)
	{
	//	log_info("DMA_READ tag");
	}
	else */
	if (ttag==DMA_WRITE)
	{
	    #ifdef PROFILE
        profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
        #endif
		//flip write buffers
		write_switch=!write_switch;
        //send MC packet to connected IHC/AN models
        //log_info("sending write complete packet %d",seg_index);
        while (!spin1_send_mc_packet(key, 0, NO_PAYLOAD))
        {
            spin1_delay_us(1);
        }
	}
	else
	{
		#ifdef PRINT
		io_printf(IO_BUF,"[core %d] invalid %d DMA tag!\n",coreID,ttag);
		#endif
	}
}

void command_received(uint mc_key, uint null)
{
    uint command = mc_key & mask;
    //log_info("command tx");

    //if(command == 0 && seg_index==0)//ready to send packet received from OME
    if(command == 1 && seg_index==0)//ready to send packet received from OME
    {
        if (sync_count<num_ihcans)//waiting for acknowledgement from child IHCANs
        {
            //log_info("sending r2s packet, ack count=%d, total ihcans=%d",sync_count,num_ihcans);
            //sending ready to send MC packet to connected IHCAN models
            while (!spin1_send_mc_packet(key|1, 0, NO_PAYLOAD))
            {
                spin1_delay_us(1);
            }
        }

        else if (sync_count==num_ihcans)
        {
            log_mini_info("%u", 9018);  /* "ack"*/
            //wait for random delay to prevent network lockup
            //spin1_delay_us(delay);
            //now all acknowledgments have been received from the child IHCAN models, send acknowledgement back to parent OME
            while (!spin1_send_mc_packet(ome_key|2, 0, NO_PAYLOAD))
            {
                spin1_delay_us(1);
            }
            sync_count=0;
        }
    }

   // else if (command == 0 && seg_index>0)
    else if (command == 1 && seg_index>0)
    {
        spin1_schedule_callback(app_end,NULL,NULL,2);
    }

    //else if (command == 1)//acknowledgement packet received from a child IHCAN
    else if (command == 2)//acknowledgement packet received from a child IHCAN
    {
        sync_count++;
       // log_info("ack mcpacket received sync_count=%d seg_index=%d\n",sync_count,seg_index);
    }
    else if (command == 0)//on receipt of an MOC spike increment MOCspikesCount
    {
        MOCspikeCount++;
    }
}
//uint check=0;
//DMA read
void data_read(uint mc_key, uint payload)
{
    //log_info("mcpacket recieved %d\n",seg_index);
    //payload is OME output value therefore next segment in input buffer memory is ready
    //convert payload to float
    MC_union.u = payload;
        //collect the next segment of samples and copy into DTCM
        if(test_DMA == TRUE)
        {
           // log_info("payload = %k",(accum)MC_union.f);
            //assign recieve buffer
            MC_seg_idx++;
            //if(MC_union.f!=0.0) check=1;
            #ifdef PROFILE
            //if(MC_seg_idx==1)profiler_write_entry_disable_irq_fiq(PROFILER_ENTER | PROFILER_TIMER);
            //if(MC_seg_idx>=SEGSIZE)profiler_write_entry_disable_irq_fiq(PROFILER_EXIT | PROFILER_TIMER);
            if(MC_seg_idx>=SEGSIZE)profiler_write_entry_disable_irq_fiq(PROFILER_ENTER | PROFILER_TIMER);
            #endif
            if(!read_switch)
            {
                dtcm_buffer_a[MC_seg_idx-1] = MC_union.f;
                //completed filling a segment of input values
                if(MC_seg_idx>=SEGSIZE)
                {
                    //log_info("rxa: %k",(accum)dtcm_buffer_a[MC_seg_idx-1]);
                    MC_seg_idx=0;
                    read_switch=1;
                    //if(check==1)log_info("non zero input received! %d",(accum)MC_union.f);
                    //check=0;
                    spin1_schedule_callback(process_handler,0,0,1);
                }
            }
            else
            {
                dtcm_buffer_b[MC_seg_idx-1] = MC_union.f;
                //completed filling a segment of input values
                if(MC_seg_idx>=SEGSIZE)
                {
                    //log_info("rxb: %k",(accum)dtcm_buffer_b[MC_seg_idx-1]);
                    MC_seg_idx=0;
                    read_switch=0;
                    //if(check==1)log_info("non zero input received! %d",(accum)MC_union.f);
                    //check=0;
                    spin1_schedule_callback(process_handler,0,0,1);
                }
            }

           // spin1_dma_transfer(DMA_READ,&sdramin_buffer[seg_index*SEGSIZE], dtcm_buffer_in, DMA_READ,
            //       SEGSIZE*sizeof(REAL));
            //spin1_dma_transfer(DMA_READ,&sdramin_buffer[cbuff_index*SEGSIZE], dtcm_buffer_in, DMA_READ,
             //      SEGSIZE*sizeof(REAL));
        }
   // }
}

void app_done ()
{
  // report simulation time
/*  io_printf (IO_BUF, "[core %d] simulation lasted %d ticks\n", coreID,
             spin1_get_simulation_time());*/

  //copy profile data
#ifdef PROFILE
	  profiler_finalise();
#endif
  
  // say goodbye
 // io_printf (IO_BUF, "sim exit\n");
 // io_printf (IO_BUF, "[core %d] -------------------\n", coreID);
}

void c_main()
{
  // Get core and chip IDs
  coreID = spin1_get_core_id ();
  chipID = spin1_get_chip_id ();
  //set timer tick
  //spin1_set_timer_tick (TIMER_TICK_PERIOD);
  app_init();
  //setup callbacks
  //process channel once data input has been read to DTCM
  spin1_callback_on (DMA_TRANSFER_DONE,transfer_handler,0);
  //simulation_dma_transfer_done_callback_on(DMA_READ,transfer_handler);

  //reads from DMA to DTCM every tick
  //spin1_callback_on (TIMER_TICK,data_read,-1);
  spin1_callback_on (MCPL_PACKET_RECEIVED,data_read,-1);
  spin1_callback_on (MC_PACKET_RECEIVED,command_received,-1);
  spin1_callback_on (USER_EVENT,data_write,0);

  //simulation_run();

  spin1_start (SYNC_WAIT);
  app_done ();
}

