/*
 * createCSV.h
 *
 *  Created on: 29 Nov 2016
 *      Author: rjames
 */

#ifndef CREATECSV_H_
#define CREATECSV_H_

#include "IHC_AN_softfloat.h"

void readCSV(REAL buff[],char *filename,int row, int col);

void createCSV(char *filename, REAL a[],int row, int col);

#endif /* CREATECSV_H_ */
