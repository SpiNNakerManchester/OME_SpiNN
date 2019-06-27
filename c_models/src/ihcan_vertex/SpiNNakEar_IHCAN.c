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
#include "IHC_AN_softfloat.h"
#include "spin1_api.h"
#include "math.h"
#include "random.h"
#include "stdfix-exp.h"
#include "log.h"
#include <data_specification.h>
#include <recording.h>
#include <profiler.h>
#include <profile_tags.h>
#include <simulation.h>
#include <debug.h>

//#define PROFILE
#define BITFIELD //define this if using spike (bitfield) output

//=========GLOBAL VARIABLES============//
REAL Fs,dt,max_rate;
uint coreID;
uint chipID;
uint test_DMA;
uint seg_index;
uint cbuff_numseg;
uint shift_index;
uint spike_count=0;
uint data_read_count=0;
uint data_write_count=0;
uint mc_rx_count=0;

uint read_switch;
bool write_switch;
bool app_complete=false;
uint processing;
uint index_x;
uint index_y;
REAL r_max_recip;
REAL dt_spikes;
REAL dt_segment;
uint spike_seg_size;
uint seg_output;
uint seg_output_n_bytes;
SEED_TYPE local_seed;

int start_count_process;
int end_count_process;
int start_count_read;
int end_count_read;
int start_count_write;
int end_count_write;
uint refrac[NUMFIBRES];

double *dtcm_buffer_a;
double *dtcm_buffer_b;
uint32_t *dtcm_buffer_x;
uint32_t *dtcm_buffer_y;

double *sdramin_buffer;

IHC_ciliaParams Cilia;
IHC_preSynParams preSyn;
IHC_synParams Synapse;

startupVars startupValues;
//recurring values
double past_ciliaDisp;
REAL IHCVnow;
REAL mICaCurr;
REAL CaCurr[NUMFIBRES];
REAL ANCleft[NUMFIBRES];
REAL ANAvail[NUMFIBRES];
REAL ANRepro[NUMFIBRES];

uint32_t NUMLSR,NUMMSR,NUMHSR;

startupVars generateStartupVars(void)
{
	startupVars out;
	REAL Gu0,kt0,IHCV,gmax,u0,u1,s1,s0,ga,gk,et,ek,rpc,ekp,gamma,
	beta,mICaCurr,mICaCurr_pow,gmaxcalsr,gmaxcamsr,gmaxcahsr,eca,
	taucalsr,taucamsr,taucahsr,CaCurrLSR,CaCurrMSR,CaCurrHSR,
	kt0LSR,kt0MSR,kt0HSR,ANCleftLSR,ANCleftMSR,ANCleftHSR,ANAvailLSR,
	ANAvailMSR,ANAvailHSR,ANReproLSR,ANReproMSR,ANReproHSR,z,power,y,
	x,l,r,MLSR,MMSR,MHSR;

	//======Calculate Gu0===========//
	gmax=6e-9;
	u0=0.3e-9;
	u1=1e-9;
	s1=1e-9;
	s0=6e-9;
	ga=1e-10;
	Gu0=ga+gmax/(1+(exp(u0/s0)*(1+exp(u1/s1))));
	//======Calculate IHCVnow=========//
	gk=2.1e-8;
	et=0.1;
	ek=-0.08;
	rpc=0.04;
	ekp=ek+rpc*et;
	IHCV=(gk*ekp+Gu0*et)/(Gu0+gk);

	//=======Calculate mICaCurrent========//
	gamma=100.;
	beta=400.;
	mICaCurr=1./(1.+exp(-gamma*IHCV)*(1./beta));

	//=======Calculate CaCurrent========//
	gmaxcalsr=20e-9;
	gmaxcamsr=20e-9;
	gmaxcahsr=20e-9;
	eca=0.066;
	taucalsr=200e-6;
	taucamsr=350e-6;
	taucahsr=500e-6;
	mICaCurr_pow=mICaCurr;
	for (int i=0;i<2;i++)
	{
	    mICaCurr_pow*=mICaCurr;
	}
	CaCurrLSR=((gmaxcalsr*mICaCurr_pow)*(IHCV-eca))*taucalsr;
	CaCurrMSR=((gmaxcamsr*mICaCurr_pow)*(IHCV-eca))*taucamsr;
	CaCurrHSR=((gmaxcahsr*mICaCurr_pow)*(IHCV-eca))*taucahsr;

	//========Calculate AN==========//
	z=40e12;
	power=3;
	y=15;
	x=300;
	MLSR=4;
	MMSR=4;
	MHSR=4;
	l=150;
	r=300;

	kt0LSR=-z*pow(CaCurrLSR,power);
	kt0MSR=-z*pow(CaCurrMSR,power);
	kt0HSR=-z*pow(CaCurrHSR,power);
	ANCleftLSR=(kt0LSR*y*MLSR)/(y*(l+r)+kt0LSR*l);
	ANCleftMSR=(kt0MSR*y*MMSR)/(y*(l+r)+kt0MSR*l);
	ANCleftHSR=(kt0HSR*y*MHSR)/(y*(l+r)+kt0HSR*l);
	ANAvailLSR=round((ANCleftLSR*(l+r))/kt0LSR);
	ANAvailMSR=round((ANCleftMSR*(l+r))/kt0MSR);
	ANAvailHSR=round((ANCleftHSR*(l+r))/kt0HSR);
	ANReproLSR=(ANCleftLSR*r)/x;
	ANReproMSR=(ANCleftMSR*r)/x;
	ANReproHSR=(ANCleftHSR*r)/x;

	out.IHCVnow0=IHCV;
	out.Ekp=ekp;
	out.mICaCurr0=mICaCurr;
	out.CaCurrLSR0=CaCurrLSR;
	out.CaCurrMSR0=CaCurrMSR;
	out.CaCurrHSR0=CaCurrHSR;
	out.ANCleftLSR0=ANCleftLSR;
	out.ANCleftMSR0=ANCleftMSR;
	out.ANCleftHSR0=ANCleftHSR;
	out.ANAvailLSR0=ANAvailLSR;
	out.ANAvailMSR0=ANAvailMSR;
	out.ANAvailHSR0=ANAvailHSR;
	out.ANReproLSR0=ANReproLSR;
	out.ANReproMSR0=ANReproMSR;
	out.ANReproHSR0=ANReproHSR;

	return out;
}
//data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    RECORDING,
    PROFILER}regions;

// The parameters to be read from memory
enum params {
    DATA_SIZE = 0,
    DRNLCOREID,
    COREID,
    DRNLAPPID,
    DRNL_KEY,
    RESAMPLE,
    FS,
    AN_KEY,
    IS_RECORDING,
    N_LSR,
    N_MSR,
    N_HSR,
    SEED
};

uint data_size;
uint placement_coreID;
uint drnl_coreID;
uint drnl_appID;
uint drnl_key;
uint mask;
uint resamp_fac;
uint sampling_freq;
uint an_key;
uint32_t *seeds;
uint32_t is_recording;
uint32_t recording_flags;
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
	shift_index=0;
	cbuff_numseg=4;
	read_switch=0;
	write_switch=0;
	r_max_recip=REAL_CONST(1.)/(REAL)RDM_MAX;

	io_printf (IO_BUF, "[core %d] -----------------------\n", coreID);
	io_printf (IO_BUF, "[core %d] starting simulation\n", coreID);
	//obtain data spec
	address_t data_address = data_specification_get_data_address();
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

    is_recording = params[IS_RECORDING];
    log_info("is_recording=%d",is_recording);//not used so always will record
    if (!initialise_recording()) return false;

    // Get the size of the data in words
    data_size = params[DATA_SIZE];
    //get the core ID of the parent DRNL
    drnl_coreID = params[DRNLCOREID];
    placement_coreID = params[COREID];//not used
    drnl_appID = params[DRNLAPPID];//not used
    //parent DRNL data key
    drnl_key = params[DRNL_KEY];
    //comms protocol mask to obtain commands from received key
    mask = 3;
    //factor to perform vesicle model resampling by
    resamp_fac = params[RESAMPLE];
    sampling_freq = params[FS];
    an_key = params[AN_KEY];
    NUMLSR = params[N_LSR];
    NUMMSR = params[N_MSR];
    NUMHSR = params[N_HSR];

    //RNG seeds
    seeds = &params[SEED];
    log_info("parent DRNL key=%d",drnl_key);
    log_info("data_size=%d",data_size);
    log_info("mask=%d",mask);
    log_info("DRNL ID=%d",drnl_coreID);
    log_info("AN key=%d",an_key);
    log_info("n_lsr=%d",NUMLSR);
    log_info("n_msr=%d",NUMMSR);
    log_info("n_hsr=%d",NUMHSR);

    #ifdef PROFILE
    // configure timer 2 for profiling
    profiler_init(
    data_specification_get_region(PROFILER, data_address));
    #endif

    Fs=(REAL)sampling_freq;
	dt=(1.0/Fs);
	seg_output = NUMFIBRES * SEGSIZE;
	#ifdef BITFIELD
    seg_output /= 8;
    if (seg_output==0)seg_output=1;
    seg_output_n_bytes=seg_output * sizeof(uint32_t);
	#endif
	#ifndef BITFIELD
	seg_output_n_bytes=seg_output * sizeof(REAL);
	#endif
    log_info("seg output (in bytes)=%d\n",seg_output_n_bytes);
    spike_seg_size = seg_output/NUMFIBRES;

	max_rate=Fs/(REAL)resamp_fac;
	dt_spikes=(REAL)resamp_fac*dt;
	dt_segment = dt*SEGSIZE;

	//Alocate buffers in DTCM
	//input buffers
	dtcm_buffer_a = (double *) sark_alloc (SEGSIZE, sizeof(double));
	dtcm_buffer_b = (double *) sark_alloc (SEGSIZE, sizeof(double));
    //output buffers
	#ifdef BITFIELD
    dtcm_buffer_x = (uint32_t *) sark_alloc (seg_output, sizeof(uint32_t));
	dtcm_buffer_y = (uint32_t *) sark_alloc (seg_output, sizeof(uint32_t));
	#endif
	#ifndef BITFIELD
	dtcm_buffer_x = (REAL *) sark_alloc (seg_output, sizeof(REAL));
	dtcm_buffer_y = (REAL *) sark_alloc (seg_output, sizeof(REAL));
	#endif

	if (dtcm_buffer_a == NULL ||dtcm_buffer_b == NULL ||
	    dtcm_buffer_x == NULL ||dtcm_buffer_y == NULL)
	{
		test_DMA = FALSE;
		io_printf (IO_BUF, "[core %d] error - cannot allocate buffer\n"
		           ,coreID);
	}
	else
	{
		test_DMA = TRUE;
		// initialize sections of DTCM
		for (uint i = 0; i < SEGSIZE; i++)
		{
			dtcm_buffer_a[i]   = 0;
			dtcm_buffer_b[i]   = 0;
		}
		for (uint i = 0; i < seg_output; i++)
		{
			dtcm_buffer_x[i]   = 0;
			dtcm_buffer_y[i]   = 0;
		}

		io_printf (IO_BUF, "[core %d] dtcm buffer a @ 0x%08x\n", coreID,
				   (uint) dtcm_buffer_a);
	}

    while (sdramin_buffer==NULL)//if first time in this callback setup input buffer
    {
        //obtain pointer to shared SDRAM circular buffer with parent DRNL
        spin1_delay_us(1000);
        sdramin_buffer = (double *) sark_tag_ptr (drnl_coreID, 0);
        //TODO: implement a time out here
    }
    io_printf(IO_BUF,"[core %d] sdram in buffer @ 0x%08x\n", coreID,
                   (uint) sdramin_buffer);

	//============MODEL INITIALISATION================//
	//calculate startup values
	startupValues=generateStartupVars();
    //initialise random number generator
	io_printf (IO_BUF, "[core %d][chip %d] local_seeds=",coreID,chipID);
	for(uint i=0;i<4;i++)
	{
        local_seed[i] = seeds[i];
		io_printf (IO_BUF, "%u ",local_seed[i]);
	}
	io_printf (IO_BUF,"\n");

	//initialise random number gen
	validate_mars_kiss64_seed (local_seed);

	//initialise cilia
	Cilia.tc= 0.00012;
	Cilia.filter_b1= 1.0;
	Cilia.filter_b2= (double)dt/Cilia.tc - 1.0;
	Cilia.filter_a1= (double)dt/Cilia.tc;
	Cilia.C=REAL_CONST(0.3);
	Cilia.u0=REAL_CONST(0.3e-9);
	Cilia.recips0=REAL_CONST(1.)/REAL_CONST(6e-9);
	Cilia.u1=REAL_CONST(1e-9);
	Cilia.recips1=REAL_CONST(1.)/REAL_CONST(1e-9);
	Cilia.Gmax=REAL_CONST(6e-9);
	Cilia.Ga=REAL_CONST(0.1e-9);
	Cilia.dtCap=(dt/5e-12);
	Cilia.Et=REAL_CONST(0.1);
	Cilia.Gk=REAL_CONST(2.1e-8);
	Cilia.Ek=REAL_CONST(-0.08);
	Cilia.Rpc=REAL_CONST(0.04);

	//==========Recurring Values=================//
	past_ciliaDisp=0.0;
	IHCVnow=startupValues.IHCVnow0;
	mICaCurr=startupValues.mICaCurr0;;
	for(uint i=0;i<NUMLSR;i++)
	{
		CaCurr[i]=startupValues.CaCurrLSR0;
		ANCleft[i]=startupValues.ANCleftLSR0;
		ANAvail[i]=startupValues.ANAvailLSR0;
		ANRepro[i]=startupValues.ANReproLSR0;
		refrac[i]=0;
		preSyn.GmaxCa[i]=20e-9;
		preSyn.recTauCa[i]=1./200e-6;
		Synapse.M[i]=REAL_CONST(4.);
	}
	for(uint i=0;i<NUMMSR;i++)
	{
		CaCurr[i+NUMLSR]=startupValues.CaCurrMSR0;
		ANCleft[i+NUMLSR]=startupValues.ANCleftMSR0;
		ANAvail[i+NUMLSR]=startupValues.ANAvailMSR0;
		ANRepro[i+NUMLSR]=startupValues.ANReproMSR0;
		refrac[i+NUMLSR]=0;
		preSyn.GmaxCa[i+NUMLSR]=20e-9;
		preSyn.recTauCa[i+NUMLSR]=1./350e-6;
		Synapse.M[i+NUMLSR]=REAL_CONST(4.);
	}
	for(uint i=0;i<NUMHSR;i++)
	{
		CaCurr[i+NUMLSR+NUMMSR]=startupValues.CaCurrHSR0;
		ANCleft[i+NUMLSR+NUMMSR]=startupValues.ANCleftHSR0;
		ANAvail[i+NUMLSR+NUMMSR]=startupValues.ANAvailHSR0;
		ANRepro[i+NUMLSR+NUMMSR]=startupValues.ANReproHSR0;
		refrac[i+NUMLSR+NUMMSR]=0;
		preSyn.GmaxCa[i+NUMLSR+NUMMSR]=20e-9;
		preSyn.recTauCa[i+NUMLSR+NUMMSR]=1./500e-6;
		Synapse.M[i+NUMLSR+NUMMSR]=REAL_CONST(4.);
	}

	//=========initialise the pre synapse params========//
	preSyn.recipBeta=(1./400.0);
	preSyn.gamma=REAL_CONST(100.);
	preSyn.dtTauM=(dt/5e-5);
	preSyn.Eca=0.066;
	preSyn.power=REAL_CONST(3.);
	preSyn.z=40e12;

	//=======initialise the synapse params=======//
	Synapse.ldt=REAL_CONST(150.)*dt_spikes;
	Synapse.ydt=REAL_CONST(15.)*dt_spikes;
	if(Synapse.ydt>REAL_CONST(1.))
	{
		Synapse.ydt=REAL_CONST(1.);
	}
	Synapse.xdt=REAL_CONST(300.)*dt_spikes;
	Synapse.rdt=REAL_CONST(300.)*dt_spikes;
	Synapse.refrac_period=((7.5e-4)/dt_spikes);

    return true;
}


void app_done()
{
    // report simulation time
    io_printf (IO_BUF, "[core %d] simulation produced %d spikes\n",
                coreID,spike_count);
    //copy profile data
    #ifdef PROFILE
        profiler_finalise();
    #endif
    // say goodbye
    io_printf (IO_BUF, "[core %d] stopping simulation\n", coreID);
    io_printf (IO_BUF, "[core %d] -------------------\n", coreID);
}

void app_end(uint null_a,uint null_b)
{
    recording_finalise();
    log_info("total simulation ticks = %d",
        simulation_ticks);
    io_printf (IO_BUF, "spinn_exit seg_index:%d data_read:%d data_write:%d mc_rx_count:%d\n",seg_index,
                data_read_count,data_write_count,mc_rx_count);
    app_done();
    app_complete=true;
    simulation_ready_to_read();
}

recording_complete_callback_t record_finished(void)
{
    #ifdef PROFILE
      profiler_write_entry_disable_irq_fiq(PROFILER_EXIT |
                                            PROFILER_TIMER);
    #endif
    data_write_count++;
    //flip write buffers
    write_switch=!write_switch;
}

void data_write(uint null_a, uint null_b)
{
    #ifdef BITFIELD
//	uint8_t *dtcm_buffer_out;
	uint32_t *dtcm_buffer_out;
	#endif
	#ifndef BITFIELD
	REAL *dtcm_buffer_out;
	#endif
	uint out_index;
	if(test_DMA == TRUE)
	{
	    //Select available buffer
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

        recording_record_and_notify(0, dtcm_buffer_out,
                            seg_output_n_bytes,record_finished);

	}
}

//DMA read
void data_read(uint mc_key, uint payload)
{
    mc_rx_count++;
    if(mc_key==drnl_key)
    {
        //measure time between each call of this function (should approximate the global clock in OME)
        #ifdef PROFILE
	     profiler_write_entry_disable_irq_fiq(PROFILER_ENTER |
	                                            PROFILER_TIMER);
		#endif
        double *dtcm_buffer_in;
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
            spin1_dma_transfer(DMA_READ,&sdramin_buffer[(seg_index & (cbuff_numseg-1))*SEGSIZE],
                            dtcm_buffer_in, DMA_READ,SEGSIZE*sizeof(double));
            data_read_count++;

        }
	}
	else{
	    log_info("unexpected mc packet received 0x%x",mc_key);
        //rt_error(RTE_SWERR);
	}
}
////---------Main segment processing loop-----////
//select correct output buffer type
#ifdef BITFIELD
uint process_chan(uint32_t *out_buffer,double *in_buffer)
#endif
#ifndef BITFIELD
uint process_chan(REAL *out_buffer,double *in_buffer)
#endif
{
    #ifdef BITFIELD
	uint segment_offset=seg_output*(seg_index-1);
	#endif
	#ifndef BITFIELD
	uint segment_offset=NUMFIBRES*spike_seg_size*(seg_index-1);
	#endif
	uint i,j,k;

	REAL term_1;
	REAL term_2;
	REAL term_3;
	double cilia_disp;
	REAL utconv;
	REAL ex1;
	REAL ex2;
	REAL ex3;
	REAL Guconv;
	REAL mICaINF;
	REAL micapowconv;
	REAL ICa;
	REAL sub1;
	REAL sub2;
	REAL pos_CaCurr;
	REAL CaCurr_pow;
	REAL releaseProb_pow;
	REAL Repro_rate;
	REAL Synapse_ypow;
	REAL Synapse_xpow;
	REAL compare;
	REAL vrr;
	REAL releaseProb;
	REAL M_q;
	REAL Probability;
	REAL ejected;
	uint spikes;
	REAL reprocessed;
	REAL replenish;
	REAL reuptakeandlost;
	REAL reuptake;
	double filter_1;

	uint si=0;
	uint spike_index=0;
	uint spike_shift=0;

    #ifdef BITFIELD
    //clear output buffer values (needed for bitfield writing)
    for (uint i = 0; i < seg_output; i++)
    {
        out_buffer[i]   = 0;
    }
    #endif

	for(i=0;i<SEGSIZE;i++)
	{
	    //==========Cilia filter===============//
        filter_1 = Cilia.filter_b1 * in_buffer[i] + Cilia.filter_b2
                    * past_ciliaDisp;
        cilia_disp = filter_1 - Cilia.filter_a1 * past_ciliaDisp;
		past_ciliaDisp=cilia_disp * Cilia.C;
		//===========Apply Scaler============//
  	  	utconv=past_ciliaDisp;
		//=========Apical Conductance========//
		ex1=expk((accum)(-(utconv-Cilia.u1)*Cilia.recips1));
		ex2=expk((accum)(-(utconv-Cilia.u0)*Cilia.recips0));
  	  	Guconv=Cilia.Ga + (Cilia.Gmax/(REAL_CONST(1.)+
  	  	        (REAL)ex2*(REAL_CONST(1.)+(REAL)ex1)));
		//========Receptor Potential=========//
		IHCVnow+=((-Guconv*(IHCVnow-Cilia.Et))-
		         (Cilia.Gk*(IHCVnow-startupValues.Ekp)))*
		           Cilia.dtCap;
		//================mICa===============//
		ex3=expk((accum)-preSyn.gamma*IHCVnow);
		mICaINF=REAL_CONST(1.)/(REAL_CONST(1.)+(REAL)ex3*
		        preSyn.recipBeta);
		mICaCurr+=(mICaINF-mICaCurr)*preSyn.dtTauM;
		//================ICa================//
		micapowconv=mICaCurr;
		for(k=0;k<preSyn.power-1;k++)
		{
			micapowconv=micapowconv*mICaCurr;
		}
		//============Fibres=============//
		for (j=0;j<NUMFIBRES;j++)
		{
			//======Synaptic Ca========//
			ICa=preSyn.GmaxCa[j]*micapowconv*(IHCVnow-
			                                preSyn.Eca);
			sub1 = ICa*dt;
			sub2 = (CaCurr[j]*dt)*preSyn.recTauCa[j];
			CaCurr[j] += sub1 - sub2;
			//invert Ca
			pos_CaCurr=REAL_CONST(-1.)*CaCurr[j];
			if(i%resamp_fac==0)
			{
				//=====Vesicle Release Rate MAP_BS=====//
				CaCurr_pow=pos_CaCurr*preSyn.z;
				for(k=0;k<(uint)preSyn.power-1;k++)
				{
					CaCurr_pow=CaCurr_pow*pos_CaCurr*preSyn.z;
				}
				vrr = CaCurr_pow;
				//=====Release Probability=======//
				releaseProb=vrr*dt_spikes;
				M_q=Synapse.M[j]-ANAvail[j];
				if(M_q<REAL_CONST(0.))
				{
					M_q=REAL_CONST(0.);
				}
				//===========Ejected============//
				releaseProb_pow=REAL_CONST(1.);
				for(k=0;k<(uint)ANAvail[j];k++)
				{
					releaseProb_pow=releaseProb_pow*
					                (REAL_CONST(1.)-releaseProb);
				}
				Probability=REAL_CONST(1.)-releaseProb_pow;
				if (refrac[j]>0)
				{
					refrac[j]--;
				}
				if(Probability>((REAL)random_gen()*r_max_recip))
				{
					ejected=REAL_CONST(1.);
					if (refrac[j]<=0)
					{
						spikes = 1;
                        spin1_send_mc_packet(an_key|j,0,NO_PAYLOAD);

						refrac[j]=(uint)(Synapse.refrac_period +
						           (((REAL)random_gen()*r_max_recip)*
						            Synapse.refrac_period)+REAL_CONST(0.5));
					}
					else
					{
						spikes=0;
					}
				}
				else
				{
					ejected=REAL_CONST(0.);
					spikes=0;
				}

				//=========Reprocessed=========//
				Repro_rate=Synapse.xdt;
				Synapse_xpow=REAL_CONST(1.);
				Synapse_ypow=REAL_CONST(1.);
				for(k=0;k<(uint)M_q;k++)
				{
					Synapse_ypow=Synapse_ypow*(REAL_CONST(1.)-Synapse.ydt);
				}
				for(k=0;k<(uint)ANRepro[j];k++)
				{
					Synapse_xpow=Synapse_xpow*(REAL_CONST(1.)-Repro_rate);
				}
				Probability=REAL_CONST(1.)-Synapse_xpow;
				if(Probability>((REAL)random_gen()*r_max_recip))
				{
					reprocessed=REAL_CONST(1.);
				}
				else
				{
					reprocessed=REAL_CONST(0.);
				}
				//========Replenish==========//
				Probability=REAL_CONST(1.)-Synapse_ypow;
				if(Probability>((REAL)random_gen()*r_max_recip))
				{
					replenish=REAL_CONST(1.);
				}
				else
				{
					replenish=REAL_CONST(0.);
				}
				//==========Update Variables=========//
				ANAvail[j]=ANAvail[j]+replenish+reprocessed-ejected;
				reuptakeandlost=(Synapse.rdt+Synapse.ldt)*ANCleft[j];
				reuptake=Synapse.rdt*ANCleft[j];
				ANCleft[j]= ANCleft[j]+ejected-reuptakeandlost;
   		        ANRepro[j]=ANRepro[j]+ reuptake-reprocessed;
				//=======write output value to SDRAM========//
				#ifdef BITFIELD
				spike_shift = (SEGSIZE-1)-i;//only using SEGSIZE LSBs of 32bit word
                if(spikes)
                {
                    spike_count++;
                    out_buffer[j]|=(spikes<<spike_shift);
                }
                #endif
                #ifndef BITFIELD
                out_buffer[(j*spike_seg_size)+i] =vrr;
                #endif
			}
		}
	}
	return segment_offset;
}

void transfer_handler(uint tid, uint ttag)
{
    //increment segment index
    seg_index++;
    //choose current available buffers
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

    //triggers a write to recording region callback
    spin1_trigger_user_event(NULL,NULL);
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
    uint32_t timer_period;
    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    if(app_init(&timer_period))
    {
        // Set timer tick (in microseconds)
        log_info("setting timer tick callback for %d microseconds",
        timer_period);
        spin1_set_timer_tick(timer_period);
        //setup callbacks
        //process channel once data input has been read to DTCM
        simulation_dma_transfer_done_callback_on(DMA_READ,transfer_handler);
        //reads from DMA to DTCM every MC packet received
        spin1_callback_on (MC_PACKET_RECEIVED,data_read,-1);
        spin1_callback_on (USER_EVENT,data_write,0);
        spin1_callback_on (TIMER_TICK,count_ticks,0);

        simulation_run();
    }
}
