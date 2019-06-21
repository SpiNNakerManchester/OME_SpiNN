/*
 * startupValues.h
 *
 *  Created on: 25 Nov 2016
 *      Author: rjames
 */

#ifndef STARTUPVALUES_H_
#define STARTUPVALUES_H_

#include "IHC_AN_softfloat.h"

typedef struct{

	REAL mICaCurr0,CaCurrLSR0,CaCurrMSR0,CaCurrHSR0,ANCleftLSR0,ANCleftMSR0,ANCleftHSR0,
		ANAvailLSR0,ANAvailMSR0,ANAvailHSR0,ANReproLSR0,ANReproMSR0,ANReproHSR0,Ekp,IHCVnow0;

}startupVars;

startupVars generateStartupVars(void);

#endif /* STARTUPVALUES_H_ */
