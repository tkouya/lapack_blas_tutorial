/********************************************************************************/
/* tkaux.h : Auxiliary Routines for textbook "LAPACK/BLAS Tutorial"             */
/* Copyright (C) 2015 Tomonori Kouya                                            */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
#include <stdio.h>
#include <math.h>
#include <complex.h>

#ifndef USE_IMKL
	#include "cblas.h"
#endif // USE_IMKL

#ifndef __TKAUX_H__

#define __TKAUX_H__

// macros
#define DMAX(a, b) (((a) > (b)) ? (a) : (b))
#define DMIN(a, b) (((a) < (b)) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif

// printf_dvector(format, dvec, dim, interval -> printf(format, index, dvec[index + interval])
void printf_dvector(const char *, double *, int, int);

// printf_dvecto2r -> printf(format, index, dvec1[index], dvec2[index])
void printf_dvector2(const char *, double *, double *, int, int);

// zero clear
void set0_dvector(double *, int, int);

// zero clear
void set0_dmatrix(double *, int, int);

// zero clear
void set0_zmatrix(double complex *, int, int);

// Column Major <- Row major
// colm_mat <- rowm_mat
void row2col_dmatrix(double *, int, int, double *);

// Row Major <- Column major
// colm_mat <- rowm_mat
void col2row_dmatrix(double *, int, int, double *);

// norm relative error (2-norm)
// || dvec_err - dvec_true ||_2 / || dvec_true ||_2
double dreldiff_dvector(double *, int, int, double *, int);

#ifdef __cplusplus
}
#endif

#endif // __TKAUX_H__
