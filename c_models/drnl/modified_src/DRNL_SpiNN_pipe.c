// DO NOT EDIT! THIS FILE WAS GENERATED FROM src/DRNL_SpiNN_pipe.c

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

#define TIMER_TICK_PERIOD 3600//2600//2300//REALTIME (23ms to process 100 44100Hz samples TODO: make this dependent on numfibres

#define TOTAL_TICKS 100//240//173//197
#define PROFILE
//#define LOOP_PROFILE
//#define PRINT

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

REAL cf,nlin_b0,nlin_b1,nlin_b2,nlin_a1,nlin_a2,nlBWp,nlBWq,linBWp,linBWq,linCFp,linCFq,	nlin_bw,nlin_phi,nlin_theta,nlin_cos_theta,nlin_sin_theta,nlin_alpha,

lin_b0,lin_b1,lin_b2,lin_a1,lin_a2,lin_bw,lin_phi,lin_theta,lin_cos_theta,lin_sin_theta,lin_alpha,
lin_gain,

a,ctBM,dispThresh,recip_ctBM,compressedNonlin,MOC;
//ctBM,dispThresh;

accum c;

REAL complex lin_z1,lin_z2,lin_z3,lin_tf,nlin_z1,nlin_z2,nlin_z3,nlin_tf;

REAL lin_x1;
REAL lin_y1[2],lin_y2[2];

REAL nlin_x1a;
REAL nlin_y1a[2],nlin_y2a[2];
REAL nlin_x1b;
REAL nlin_y1b[2],nlin_y2b[2];

//uint seed_selection[SEED_SEL_SIZE];//TODO:this needs to be moved to SDRAM

int start_count_process;
int end_count_process;
int start_count_read;
int end_count_read;
int start_count_write;
int end_count_write;

uint sync_count=0;

REAL *dtcm_buffer_a;
REAL *dtcm_buffer_b;
REAL *dtcm_buffer_x;
REAL *dtcm_buffer_y;
REAL *dtcm_profile_buffer;

REAL *sdramin_buffer;
REAL *sdramout_buffer;
REAL *profile_buffer;

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
    FS
};

// The size of the remaining data to be sent
uint data_size;
// Pointer to the start of the data still to be sent and confirmed
//REAL *data;
// The mask which indicates the sequence number
uint sequence_mask;
// the core ID given by the placement software
uint placement_coreID;
uint ome_coreID;
uint ome_appID;
uint ome_key;
uint key;
uint num_ihcans;
uint drnl_cf;
uint sampling_frequency;

//application initialisation
void app_init(void)
{

	seg_index=0;
	cbuff_index=0;
	cbuff_numseg=3;
	read_switch=0;
	write_switch=0;

	/* say hello */
	
	io_printf (IO_BUF, "[core %d] -----------------------\n", coreID);
	io_printf (IO_BUF, "[core %d] starting simulation\n", coreID);

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

    key=params[KEY];

    num_ihcans=params[NUM_IHCAN];

    drnl_cf=params[CF];

    log_mini_info("%u%d", 9001,key);  /* "key=%d\n"*/

    log_mini_info("%u%d", 9002,data_size);  /* "data_size=%d"*/

    //log_info("num_ihcans=%d\n",num_ihcans);

    log_mini_info("%u%d", 9003,drnl_cf);  /* "CF=%d\n"*/

    //Get sampling frequency
    sampling_frequency = params[FS];

    Fs= (REAL)sampling_frequency;
	dt=(1.0/Fs);

    // Allocate buffers somewhere in SDRAM
	//output results buffer
	//hack for smaller SDRAM intermediate buffers
	data_size=cbuff_numseg*SEGSIZE;

	sdramout_buffer = (REAL *) sark_xalloc (sv->sdram_heap,
					 data_size * sizeof(REAL),
					 placement_coreID,
					 ALLOC_LOCK);

	/*profile_buffer = (REAL *) sark_xalloc (sv->sdram_heap,
					3 * ((uint)44100./SEGSIZE) *sizeof(REAL),
					placement_coreID|64,
					ALLOC_LOCK);*/
	
	// and a buffer in DTCM
	
	dtcm_buffer_a = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_buffer_b = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_buffer_x = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_buffer_y = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_profile_buffer = (REAL *) sark_alloc (3*TOTAL_TICKS, sizeof(REAL));
	
	if (dtcm_buffer_a == NULL ||dtcm_buffer_b == NULL ||dtcm_buffer_x == NULL ||dtcm_buffer_y == NULL 
			||  sdramout_buffer == NULL ||  dtcm_profile_buffer == NULL)
	/*if (sdramout_buffer == NULL || sdramin_buffer == NULL || dtcm_buffer_y == NULL 
			|| dtcm_buffer_a == NULL || dtcm_buffer_b == NULL || dtcm_profile_buffer == NULL ||dtcm_buffer_x == NULL)*/
	{
		test_DMA = FALSE;
		io_printf (IO_BUF, "[core %d] error - cannot allocate buffer\n", coreID);
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

		for (uint i=0;i<3 * TOTAL_TICKS;i++)
		{
			dtcm_profile_buffer[i]  = 0;
			//profile_buffer[i]  = 0;
		}
		
		io_printf (IO_BUF, "[core %d] dtcm buffer a @ 0x%08x\n", coreID,
				   (uint) dtcm_buffer_a);
		io_printf (IO_BUF, "[core %d] sdram out buffer @ 0x%08x\n", coreID,
				   (uint) sdramout_buffer);
		io_printf (IO_BUF, "[core %d] sdram in buffer @ 0x%08x\n", coreID,
				   (uint) sdramin_buffer);	
		/*io_printf (IO_BUF, "[core %d] profile buffer @ 0x%08x\n", coreID,
				   (uint) profile_buffer);	*/
	}
	
	//============MODEL INITIALISATION================//

	//set center frequency
	//cf=4000.0;
	cf=(REAL)drnl_cf;

	//non-linear pathway
	nlBWq=180.0;
	nlBWp=0.14;
	nlin_bw=nlBWp * cf + nlBWq;
	nlin_phi=2.0 * (REAL)PI * nlin_bw * dt;
	nlin_theta= 2.0 * (REAL)PI * cf * dt;
	nlin_cos_theta= cos(nlin_theta);
	nlin_sin_theta= sin(nlin_theta);
	nlin_alpha= -exp(-nlin_phi) * nlin_cos_theta;
	nlin_a1= 2.0 * nlin_alpha;
	nlin_a2= exp(-2.0 * nlin_phi);
	nlin_z1 = (1.0 + nlin_alpha * nlin_cos_theta) - (nlin_alpha * nlin_sin_theta) * _Complex_I;
	nlin_z2 = (1.0 + nlin_a1 * nlin_cos_theta) - (nlin_a1 * nlin_sin_theta) * _Complex_I;
	nlin_z3 = (nlin_a2 * cos(2.0 * nlin_theta)) - (nlin_a2 * sin(2.0 * nlin_theta)) * _Complex_I;
	nlin_tf = (nlin_z2 + nlin_z3) / nlin_z1;
	nlin_b0 = cabsf(nlin_tf);
	nlin_b1 = nlin_alpha * nlin_b0;

	//compression algorithm variables
	a=5e4;
	c=0.25k;
	ctBM=3.981071705534974e-08;
	recip_ctBM=1.0/ctBM;
	dispThresh=ctBM/a;

	//linear pathway
	lin_gain=200.0;
	linBWq=235.0;
	linBWp=0.2;
	lin_bw=linBWp * cf + linBWq;
	lin_phi=2.0 * (REAL)PI * lin_bw * dt;
	lin_theta= 2.0 * (REAL)PI * cf * dt;
	lin_cos_theta= cos(lin_theta);
	lin_sin_theta= sin(lin_theta);
	lin_alpha= -exp(-lin_phi) * lin_cos_theta;
	lin_a1= 2.0 * lin_alpha;
	lin_a2= exp(-2.0 * lin_phi);
	lin_z1 = (1.0 + lin_alpha * lin_cos_theta) - (lin_alpha * lin_sin_theta) * _Complex_I;
	lin_z2 = (1.0 + lin_a1 * lin_cos_theta) - (lin_a1 * lin_sin_theta) * _Complex_I;
	lin_z3 = (lin_a2 * cos(2.0 * lin_theta)) - (lin_a2 * sin(2.0 * lin_theta)) * _Complex_I;
	lin_tf = (lin_z2 + lin_z3) / lin_z1;
	lin_b0 = cabsf(lin_tf);
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

	compressedNonlin=0.0;
	
	nlin_y2a[0]=0.0;
	nlin_y2a[1]=0.0;

	nlin_x1b=0.0;
	nlin_y1b[0]=0.0;
	nlin_y1b[1]=0.0;
	
	nlin_y2b[0]=0.0;
	nlin_y2b[1]=0.0;

	MOC=1.0; // TODO change this to be a model input	

	

#ifdef PROFILE
    // configure timer 2 for profiling
    // enabled, free running, interrupt disabled, no pre-scale, 32 bit, free-running mode
    tc[T2_CONTROL] = TIMER2_CONF;
#endif
    
}

void app_end(uint null_a,uint null_b)
{
    //send end MC packet to child IHCANs
    log_mini_info("%u", 9004);  /* "sending final packet to IHCANs\n"*/
    while (!spin1_send_mc_packet(key, 1, WITH_PAYLOAD)) {
        spin1_delay_us(1);
    }

    //wait for acknowledgment from child IHCANs
    while(sync_count<num_ihcans)
    {
       spin1_delay_us(1);
    }

    log_mini_info("%u", 9005);  /* "all final acks received, sending final ack to OME and closing\n"*/
    //send final ack packet back to parent OME
    while (!spin1_send_mc_packet(ome_key, 2, WITH_PAYLOAD)) {
        spin1_delay_us(1);
    }


    io_printf (IO_BUF, "spinn_exit %d\n",seg_index);
    spin1_exit (0);

}

void data_write(uint null_a, uint null_b)
{
	REAL *dtcm_buffer_out;
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
  start_count_write = tc[T2_COUNT];
#endif
		spin1_dma_transfer(DMA_WRITE,&sdramout_buffer[out_index],dtcm_buffer_out,DMA_WRITE,
		  						SEGSIZE*sizeof(REAL));
#ifdef PRINT
  log_mini_info("%u%d%d%08x", 9006, coreID,seg_index,(uint) &sdramout_buffer[out_index]);
  /* "[core %d] segment %d written @ 0x%08x\n"*/

  log_mini_info("%u%d%d%08x%08x", 9007, coreID,seg_index,(uint) &sdramout_buffer[out_index],(uint) &sdramout_buffer[out_index+(NUMFIBRES-1)*SEGSIZE+SEGSIZE-1]);
  /* "[core %d] segment %d written to @ 0x%08x - 0x%08x\n"*/
#endif
	}
}

//DMA read
void data_read(uint ticks, uint payload)
{

    //log_info("mcpacket recieved %d\n",seg_index);

    if (payload==1)
    {
        //OME has finished writing to SDRAM schedule end callback
        spin1_schedule_callback(app_end,NULL,NULL,2);
    }

    else if(payload==2)//acknowledgement packet received from a child IHCAN
    {
        sync_count++;
        log_mini_info("%u%d%d", 9008,sync_count,seg_index);  /* "ack mcpacket recieved sync_count=%d seg_index=%d\n"*/
    }

    else if (payload==3 && seg_index==0)//ready to send packet received from OME
    {

        if(sdramin_buffer==NULL)//if first time in this callback setup input buffer
        {
            log_mini_info("%u%d", 9009,coreID);  /* "[core %d] ready to send from OME received, setting up input buffer"*/
            sdramin_buffer = (REAL *) sark_tag_ptr (ome_coreID, 0);
            log_mini_info("%u%d%08x", 9010, coreID,(uint) sdramin_buffer);
            /* "[core %d] sdram in buffer @ 0x%08x\n"*/

            if(sdramin_buffer==NULL)//if initial input buffer setup fails
            {
                test_DMA = FALSE;
                //io_printf (IO_BUF, "[core %d] error - cannot allocate buffer, ending application\n", coreID);
                spin1_schedule_callback(app_end,NULL,NULL,2);
            }
        }

        if (sync_count<num_ihcans)//waiting for acknowledgement from child IHCANs
        {
            log_mini_info("%u%d%d", 9011,sync_count,num_ihcans);  /* "sending r2s packet, ack count=%d, total ihcans=%d"*/
            //sending ready to send MC packet to connected IHCAN models
            while (!spin1_send_mc_packet(key, 3, WITH_PAYLOAD))
            {
                spin1_delay_us(1);
            }
        }

        else if (sync_count==num_ihcans)
        {
            log_mini_info("%u", 9012);  /* "sending ack packet to OME\n"*/
            //now all acknowledgments have been received from the child IHCAN models, send acknowledgement back to parent OME
            while (!spin1_send_mc_packet(ome_key, 2, WITH_PAYLOAD))
            {
                spin1_delay_us(1);
            }
            sync_count=0;
        }
    }
    else //payload is 0 therefore next segment in input buffer memory is ready
    {
        #ifdef PROFILE
          start_count_read = tc[T2_COUNT];
        #endif

        REAL *dtcm_buffer_in;

        //read from DMA and copy into DTCM
        if(test_DMA == TRUE)
        {
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

            #ifdef PRINT
                    io_printf (IO_BUF, "[core %d] sdram DMA read @ 0x%08x (segment %d)\n", coreID,
                                  (uint) &sdramin_buffer[(cbuff_index)*SEGSIZE],seg_index+1);
            #endif

           // spin1_dma_transfer(DMA_READ,&sdramin_buffer[seg_index*SEGSIZE], dtcm_buffer_in, DMA_READ,
            //       SEGSIZE*sizeof(REAL));
            spin1_dma_transfer(DMA_READ,&sdramin_buffer[cbuff_index*SEGSIZE], dtcm_buffer_in, DMA_READ,
                   SEGSIZE*sizeof(REAL));
        }
    }
}


uint process_chan(REAL *out_buffer,REAL *in_buffer) 
{  
	//uint segment_offset=SEGSIZE*(seg_index-1);
	uint segment_offset=SEGSIZE*(cbuff_index);
	uint i;		
	REAL linout1,linout2,nonlinout1a,nonlinout2a,nonlinout1b,nonlinout2b,abs_x;

	for(i=0;i<SEGSIZE;i++)
	{
	/*	#ifdef PROFILE
		if(i==0)
		{
		  start_count_process = tc[T2_COUNT];
		}
		#endif*/

		//Linear Path
		linout1= lin_b0 * in_buffer[i] + lin_b1 * lin_x1 - 
				lin_a1 * lin_y1[1] - lin_a2 * lin_y1[0];		

		lin_x1=in_buffer[i];
		lin_y1[0]=lin_y1[1];
		lin_y1[1]=linout1;

		linout2= lin_b0 * linout1 + lin_b1 * lin_y1[0] - 
				lin_a1 * lin_y2[1] - lin_a2 * lin_y2[0];
		
		lin_y2[0]= lin_y2[1];
		lin_y2[1]= linout2;

		//non-linear path
		//stage 1
		nonlinout1a= nlin_b0 * in_buffer[i] + nlin_b1 * nlin_x1a - 
				nlin_a1 * nlin_y1a[1] - nlin_a2 * nlin_y1a[0];

		nlin_x1a=in_buffer[i];
		nlin_y1a[0]=nlin_y1a[1];
		nlin_y1a[1]=nonlinout1a;

		nonlinout2a= nlin_b0 * nonlinout1a + nlin_b1 * nlin_y1a[0] - 
				nlin_a1 * nlin_y2a[1] - nlin_a2 * nlin_y2a[0];
		nlin_y2a[0]= nlin_y2a[1];
		nlin_y2a[1]= nonlinout2a;

	/*	#ifdef PROFILE
		if(i==0)
		{
			  end_count_process = tc[T2_COUNT];
			  dtcm_profile_buffer[1+((seg_index-1)*3)]=start_count_process-end_count_process;
		}
		#endif	*/
		
		//MOC efferent effects
		nonlinout2a*=MOC;
		
		//stage 2
		abs_x= ABS(nonlinout2a);

		if(abs_x<dispThresh)
		{			
			compressedNonlin= a * nonlinout2a;
		}
		else if(abs_x>0.0)//compress
		{	
			
			compressedNonlin=SIGN(nonlinout2a) * ctBM * (REAL)expk(c * logk((accum)a*abs_x*recip_ctBM));
			//compressedNonlin= sign * ctBM * exp(c * log(a * abs_x * recip_ctBM));
		}	
		else
		{
			compressedNonlin=0.0;
		}			

		//stage 3 
		nonlinout1b= nlin_b0 * compressedNonlin + nlin_b1 * nlin_x1b -
				 nlin_a1 * nlin_y1b[1] - nlin_a2 * nlin_y1b[0];

		nlin_x1b=compressedNonlin;
		nlin_y1b[0]=nlin_y1b[1];
		nlin_y1b[1]=nonlinout1b;

		nonlinout2b= nlin_b0 * nonlinout1b + nlin_b1 * nlin_y1b[0] - 
				nlin_a1 * nlin_y2b[1] - nlin_a2 * nlin_y2b[0];

		nlin_y2b[0]= nlin_y2b[1];
		nlin_y2b[1]= nonlinout2b;
	
		//save to buffer
		out_buffer[i]=linout2*lin_gain + nonlinout2b;
		//out_buffer[i]=compressedNonlin;//nlin_b0;//nonlinout1a;//nonlinout2b;//linout1;//nonlinout1a;
	}
		
	return segment_offset;
}

void transfer_handler(uint tid, uint ttag)
{
	if (ttag==DMA_READ)
	{
#ifdef PROFILE
  end_count_read = tc[T2_COUNT];
  dtcm_profile_buffer[seg_index*3]=(REAL)(start_count_read-end_count_read);
#ifdef PRINT
  io_printf (IO_BUF, "read complete in %d ticks\n",start_count_read-end_count_read);
#endif
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
		
		#ifdef PROFILE
		  start_count_process = tc[T2_COUNT];
		#endif
		
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
			
		#ifdef PROFILE
			  end_count_process = tc[T2_COUNT];
			  dtcm_profile_buffer[1+((seg_index-1)*3)]=start_count_process-end_count_process;
		#ifdef PRINT
			io_printf (IO_BUF, "process complete in %d ticks (segment %d)\n",start_count_process-end_count_process,seg_index);
		#endif
		#endif			
		
		spin1_trigger_user_event(NULL,NULL);
	}
	else if (ttag==DMA_WRITE)
	{

        #ifdef PROFILE
          end_count_write = tc[T2_COUNT];
          dtcm_profile_buffer[2+((seg_index-1)*3)]=start_count_write-end_count_write;
        #ifdef PRINT
          io_printf (IO_BUF, "write complete in %d ticks\n",start_count_write-end_count_write);
        #endif
        #endif
		//flip write buffers
		write_switch=!write_switch;
		//send MC packet to connected IHC/AN models
        while (!spin1_send_mc_packet(key, 0, WITH_PAYLOAD))
        {
            spin1_delay_us(1);
        }

		//send acknowledgement MC packet to connected OME model
        /*while (!spin1_send_mc_packet(ome_key, 2, WITH_PAYLOAD))
        {
            spin1_delay_us(1);
        }*/
	}
	else
	{
		#ifdef PRINT
		io_printf(IO_BUF,"[core %d] invalid %d DMA tag!\n",coreID,ttag);
		#endif
	}

}

void app_done ()
{
  // report simulation time
  io_printf (IO_BUF, "[core %d] simulation lasted %d ticks\n", coreID,
             spin1_get_simulation_time());

  //copy profile data
#ifdef PROFILE
  //io_printf (IO_BUF, "[core %d] saving profile data...\n", coreID);
	for (uint i=0;i<3*TOTAL_TICKS;i++)
	{
	//	profile_buffer[i]  = dtcm_profile_buffer[i];
	}
#endif
  
  // say goodbye
  io_printf (IO_BUF, "[core %d] stopping simulation\n", coreID);
  io_printf (IO_BUF, "[core %d] -------------------\n", coreID);
}

void c_main()
{
  // Get core and chip IDs
  coreID = spin1_get_core_id ();
  chipID = spin1_get_chip_id ();

  //set timer tick
  spin1_set_timer_tick (TIMER_TICK_PERIOD);

  app_init();

  //setup callbacks
  //process channel once data input has been read to DTCM
  spin1_callback_on (DMA_TRANSFER_DONE,transfer_handler,0);
  //reads from DMA to DTCM every tick
  //spin1_callback_on (TIMER_TICK,data_read,-1);
  spin1_callback_on (MCPL_PACKET_RECEIVED,data_read,-1);
  spin1_callback_on (USER_EVENT,data_write,0);

  spin1_start (SYNC_WAIT);
  
  app_done ();

}

