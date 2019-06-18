// DO NOT EDIT! THIS FILE WAS GENERATED FROM src/DRNL_SpiNN_old.c

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
//#include "startupValues.h"
#include "DRNL_SpiNN.h"
#include "spin1_api.h"
#include "math.h"
#include "complex.h"
#include "random.h"
#include "stdfix-exp.h"
#include "log.h"
#define TIMER_TICK_PERIOD 23000//REALTIME (23ms to process 100 44100Hz samples
#define TOTAL_TICKS 240//173//197       
#define PROFILE
//#define LOOP_PROFILE
//#define PRINT

//=========GLOBAL VARIABLES============//
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

REAL cf,nlin_b0,nlin_b1,nlin_b2,nlin_a1,nlin_a2,nlBWp,nlBWq,linBWp,linBWq,linCFp,linCFq,	nlin_bw,nlin_phi,nlin_theta,nlin_cos_theta,nlin_sin_theta,nlin_alpha,

lin_b0,lin_b1,lin_b2,lin_a1,lin_a2,lin_bw,lin_phi,lin_theta,lin_cos_theta,lin_sin_theta,lin_alpha,
lin_gain,

a,ctBM,dispThresh,recip_ctBM,compressedNonlin,MOC;
//ctBM,dispThresh;

accum c;

REAL complex lin_z1,lin_z2,lin_z3,lin_tf,nlin_z1,nlin_z2,nlin_z3,nlin_tf;

REAL lin_x1[2],lin_x2[2];
REAL lin_y1[2],lin_y2[2];

REAL nlin_x1a[2],nlin_x2a[2];
REAL nlin_y1a[2],nlin_y2a[2];
REAL nlin_x1b[2],nlin_x2b[2];
REAL nlin_y1b[2],nlin_y2b[2];

//uint seed_selection[SEED_SEL_SIZE];//TODO:this needs to be moved to SDRAM

int start_count_process;
int end_count_process;
int start_count_read;
int end_count_read;
int start_count_write;
int end_count_write;


REAL *dtcm_buffer_a;
REAL *dtcm_buffer_b;
REAL *dtcm_buffer_x;
REAL *dtcm_buffer_y;
REAL *dtcm_profile_buffer;

REAL *sdramin_buffer;
REAL *sdramout_buffer;
REAL *profile_buffer;


//application initialisation
void app_init(void)
{
	Fs=SAMPLING_FREQUENCY;
	dt=(1.0/Fs);
	seg_index=0;
	read_switch=0;
	write_switch=0;
	
	/* say hello */
	
	io_printf (IO_BUF, "[core %d] -----------------------\n", coreID);
	io_printf (IO_BUF, "[core %d] starting simulation\n", coreID);
	
	// Allocate buffers somewhere in SDRAM
	
	//output results buffer
	sdramout_buffer = (REAL *) sark_xalloc (sv->sdram_heap,
					 MAX_SIGNAL_S*(uint)44100. * sizeof(REAL),
					 coreID,
					 ALLOC_LOCK);	

	sdramin_buffer = (REAL *) sark_xalloc (sv->sdram_heap,
					MAX_SIGNAL_S*(uint)44100. *sizeof(REAL),
					coreID|32,
					ALLOC_LOCK);
	
	profile_buffer = (REAL *) sark_xalloc (sv->sdram_heap,
					3 * ((uint)44100./SEGSIZE) *sizeof(REAL),
					coreID|64,
					ALLOC_LOCK);
	
	// and a buffer in DTCM
	
	dtcm_buffer_a = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_buffer_b = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_buffer_x = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_buffer_y = (REAL *) sark_alloc (SEGSIZE, sizeof(REAL));
	dtcm_profile_buffer = (REAL *) sark_alloc (3*TOTAL_TICKS, sizeof(REAL));
	
	if (dtcm_buffer_a == NULL ||dtcm_buffer_b == NULL ||dtcm_buffer_x == NULL ||dtcm_buffer_y == NULL 
			||  sdramout_buffer == NULL || sdramin_buffer == NULL || profile_buffer == NULL || dtcm_profile_buffer == NULL)
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
	
		for (uint i=0;i<MAX_SIGNAL_S* ((uint)44100.);i++)
		{
			sdramout_buffer[i]  = 0;
		}
		for (uint i=0;i<MAX_SIGNAL_S * (uint)44100.;i++)
		{
			sdramin_buffer[i]  = 0;
		}
		
		for (uint i=0;i<3 * TOTAL_TICKS;i++)
		{
			dtcm_profile_buffer[i]  = 0;
			profile_buffer[i]  = 0;
		}
		
		io_printf (IO_BUF, "[core %d] dtcm buffer a @ 0x%08x\n", coreID,
				   (uint) dtcm_buffer_a);
		io_printf (IO_BUF, "[core %d] sdram out buffer @ 0x%08x\n", coreID,
				   (uint) sdramout_buffer);
		io_printf (IO_BUF, "[core %d] sdram in buffer @ 0x%08x\n", coreID,
				   (uint) sdramin_buffer);	
		io_printf (IO_BUF, "[core %d] profile buffer @ 0x%08x\n", coreID,
				   (uint) profile_buffer);	
	}
	
	//============MODEL INITIALISATION================//

	//set center frequency TODO:change to a model input parameter
	cf=1000.0;

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
	lin_x1[0]=0.0;
	lin_x1[1]=0.0;
	lin_y1[0]=0.0;
	lin_y1[1]=0.0;
	
	lin_x2[0]=0.0;
	lin_x2[1]=0.0;
	lin_y2[0]=0.0;
	lin_y2[1]=0.0;	

	nlin_x1a[0]=0.0;
	nlin_x1a[1]=0.0;
	nlin_y1a[0]=0.0;
	nlin_y1a[1]=0.0;

	compressedNonlin=0.0;
	
	nlin_x2a[0]=0.0;
	nlin_x2a[1]=0.0;
	nlin_y2a[0]=0.0;
	nlin_y2a[1]=0.0;

	nlin_x1b[0]=0.0;
	nlin_x1b[1]=0.0;
	nlin_y1b[0]=0.0;
	nlin_y1b[1]=0.0;
	
	nlin_x2b[0]=0.0;
	nlin_x2b[1]=0.0;
	nlin_y2b[0]=0.0;
	nlin_y2b[1]=0.0;

	MOC=1.0; // TODO change this to be a model input	

	

#ifdef PROFILE
    // configure timer 2 for profiling
    // enabled, free running, interrupt disabled, no pre-scale, 32 bit, free-running mode
    tc[T2_CONTROL] = TIMER2_CONF;
#endif
    
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
		io_printf (IO_BUF, "[core %d] segment %d written to @ 0x%08x - 0x%08x\n", coreID,seg_index,
							  (uint) &sdramout_buffer[out_index],(uint) &sdramout_buffer[out_index+(NUMFIBRES-1)*SEGSIZE+SEGSIZE-1]);
#endif
	}
}

//DMA read
void data_read(uint ticks, uint null)
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
#ifdef PRINT
			io_printf (IO_BUF, "buff_a read\n");
#endif
		}
		else
		{
			dtcm_buffer_in=dtcm_buffer_b;
			read_switch=0;
#ifdef PRINT
			io_printf (IO_BUF, "buff_b read\n");
#endif
		}

#ifdef PRINT
		io_printf (IO_BUF, "[core %d] sdram DMA read @ 0x%08x (segment %d)\n", coreID,
					  (uint) &sdramin_buffer[(seg_index)*SEGSIZE],seg_index+1);
#endif
		
		spin1_dma_transfer(DMA_READ,&sdramin_buffer[seg_index*SEGSIZE], dtcm_buffer_in, DMA_READ,
			   SEGSIZE*sizeof(REAL));
	}
	
	// stop if desired number of ticks reached
	if (ticks > TOTAL_TICKS) 
	{
		io_printf (IO_BUF, "spinn_exit\n");
		spin1_exit (0); 
	}
	
}


uint process_chan(REAL *out_buffer,REAL *in_buffer) 
{  
	uint segment_offset=SEGSIZE*(seg_index-1);
	uint i,j,k;
		
	uint si=0;
	REAL linout1,linout2,nonlinout1a,nonlinout2a,nonlinout1b,nonlinout2b,abs_x,sign;
	accum accum_comp;
		
#ifdef PRINT
	io_printf (IO_BUF, "[core %d] segment %d (offset=%d) starting processing\n", coreID,seg_index,segment_offset);
#endif
	
	for(i=0;i<SEGSIZE;i++)
	{
	/*	#ifdef PROFILE
		if(i==0)
		{
		  start_count_process = tc[T2_COUNT];
		}
		#endif*/

		//Linear Path
		lin_x1[1]= in_buffer[i];	
		linout1= lin_b0 * lin_x1[1] + lin_b1 * lin_x1[0] - 
				lin_a1 * lin_y1[1] - lin_a2 * lin_y1[0];
		
		lin_x1[0]= lin_x1[1];
		lin_y1[0]=lin_y1[1];
		lin_y1[1]=linout1;

		//lin_x2[0]= lin_x2[1];
		//lin_x2[1]= linout1;		
		
		//linout2= lin_b0 * lin_x2[1] + lin_b1 * lin_x2[0] - 
		//		lin_a1 * lin_y2[1] - lin_a2 * lin_y2[0];
		linout2= lin_b0 * lin_y1[1] + lin_b1 * lin_y1[0] - 
				lin_a1 * lin_y2[1] - lin_a2 * lin_y2[0];
		
		lin_y2[0]= lin_y2[1];
		lin_y2[1]= linout2;

		//non-linear path
		//stage 1
		nlin_x1a[1]= in_buffer[i];
		nonlinout1a= nlin_b0 * nlin_x1a[1] + nlin_b1 * nlin_x1a[0] - 
				nlin_a1 * nlin_y1a[1] - nlin_a2 * nlin_y1a[0];

		nlin_x1a[0]= nlin_x1a[1];
		nlin_y1a[0]=nlin_y1a[1];
		nlin_y1a[1]=nonlinout1a;

		//nlin_x2a[0]= nlin_x2a[1];
		//nlin_x2a[1]= nonlinout1a;

		//nonlinout2a= nlin_b0 * nlin_x2a[1] + nlin_b1 * nlin_x2a[0] - 
		//		nlin_a1 * nlin_y2a[1] - nlin_a2 * nlin_y2a[0];
		nonlinout2a= nlin_b0 * nlin_y1a[1] + nlin_b1 * nlin_y1a[0] - 
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
			sign= SIGN(nonlinout2a);
			compressedNonlin= sign * ctBM * (REAL)expk(c * logk((accum)a*abs_x*recip_ctBM));
			//compressedNonlin= sign * ctBM * exp(c * log(a * abs_x * recip_ctBM));
		}	
		else
		{
			compressedNonlin=0.0;
		}			

		//stage 3 
		nlin_x1b[1]= compressedNonlin;//nonlinout2a;//
		nonlinout1b= nlin_b0 * nlin_x1b[1] + nlin_b1 * nlin_x1b[0] -
				 nlin_a1 * nlin_y1b[1] - nlin_a2 * nlin_y1b[0];

		nlin_x1b[0]= nlin_x1b[1];
		nlin_y1b[0]=nlin_y1b[1];
		nlin_y1b[1]=nonlinout1b;

		//nlin_x2b[0]= nlin_x2b[1];
		//nlin_x2b[1]= nonlinout1b;

		//nonlinout2b= nlin_b0 * nlin_x2b[1] + nlin_b1 * nlin_x2b[0] - 
		//		nlin_a1 * nlin_y2b[1] - nlin_a2 * nlin_y2b[0];

		nonlinout2b= nlin_b0 * nlin_y1b[1] + nlin_b1 * nlin_y1b[0] - 
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
		
		#ifdef PROFILE
		  start_count_process = tc[T2_COUNT];
		#endif
		
		//choose current buffers
		if(!read_switch && !write_switch)
		{
#ifdef PRINT 
io_printf (IO_BUF, "buff_b-->buff_x\n");
#endif
			index_x=process_chan(dtcm_buffer_x,dtcm_buffer_b);
		}
		else if(!read_switch && write_switch)
		{
#ifdef PRINT
io_printf (IO_BUF, "buff_b-->buff_y\n"); 
#endif
			index_y=process_chan(dtcm_buffer_y,dtcm_buffer_b);
		}
		else if(read_switch && !write_switch)
		{
#ifdef PRINT
io_printf (IO_BUF, "buff_a-->buff_x\n");
#endif	
			index_x=process_chan(dtcm_buffer_x,dtcm_buffer_a);

		}
		else
		{
#ifdef PRINT
io_printf (IO_BUF, "buff_a-->buff_y\n");
#endif
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
		profile_buffer[i]  = dtcm_profile_buffer[i];
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

  //setup callbacks
  //process channel once data input has been read to DTCM
  spin1_callback_on (DMA_TRANSFER_DONE,transfer_handler,0);
  //reads from DMA to DTCM every tick
  spin1_callback_on (TIMER_TICK,data_read,-1);
  spin1_callback_on (USER_EVENT,data_write,0);

  app_init();

  spin1_start (SYNC_WAIT);
  
  app_done ();

}

