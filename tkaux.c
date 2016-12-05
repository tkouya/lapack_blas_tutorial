/********************************************************************************/
/* tkaux.c : Auxiliary Routines for textbook "LAPACK/BLAS Tutorial"             */
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

#include "tkaux.h"

// printf_dvector -> printf(format, index, dvec[index])
void printf_dvector(const char *format, double *dvec, int dim_dvec, int inc_dvec)
{
	int index;

	for(index = 0; index < dim_dvec; index += inc_dvec)
		printf(format, index, dvec[index]);
}

// zero clear
void set0_dvector(double *dvec, int dim_dvec, int inc_dvec)
{
	int i;

	for(i = 0; i < dim_dvec; i += inc_dvec)
		dvec[i] = 0.0;
}

// zero clear
void set0_dmatrix(double *dmat, int row_dim, int col_dim)
{
	int index;

	for(index = 0; index < row_dim * col_dim; index++)
		dmat[index] = 0.0;
}

// zero clear
void set0_zmatrix(double complex *zmat, int row_dim, int col_dim)
{
	int index;

	for(index = 0; index < row_dim * col_dim; index++)
		zmat[index] = 0.0 + 0.0 * I;
}

// Column Major <- Row major
// colm_mat <- rowm_mat
void row2col_dmatrix(double *colm_mat, int colm_row_dim, int colm_col_dim, double *rowm_mat)
{
	int i, j;

	for(j = 0; j < colm_col_dim; j++)
	{
		for(i = 0; i < colm_row_dim; i++)
			colm_mat[i + j * colm_row_dim] = rowm_mat[i * colm_col_dim + j];
			//printf("%d <= %d\n", i + j * colm_row_dim, i * colm_col_dim + j);
	}
}

// Row Major <- Column major
// colm_mat <- rowm_mat
void col2row_dmatrix(double *rowm_mat, int rowm_row_dim, int rowm_col_dim, double *colm_mat)
{
	int i, j;

	for(i = 0; i < rowm_row_dim; i++)
	{
		for(j = 0; j < rowm_col_dim; j++)
			rowm_mat[i * rowm_col_dim + j] = colm_mat[i + j * rowm_row_dim];
		//	printf("%d <= %d\n", i * rowm_col_dim + j, i + rowm_row_dim * j);
	}
}

// norm elative error 
// || dvec_err - dvec_true ||_2 / || dvec_true ||_2
double dreldiff_dvector(double *dvec_err, int dim_dvec, int inc_dvec_err, double *dvec_true, int inc_dvec_true)
{
	double reldiff, norm_dvec_true, *diff_vec;

	diff_vec = (double *)calloc(dim_dvec, sizeof(double));

	// diff_vec := -dvec_err + dvec_true
	cblas_dcopy(dim_dvec, dvec_err, 1, diff_vec, 1);
	cblas_daxpy(dim_dvec, -1.0, dvec_true,  inc_dvec_true, diff_vec, 1);
	reldiff = cblas_dnrm2(dim_dvec, diff_vec, 1);
	norm_dvec_true = cblas_dnrm2(dim_dvec, dvec_true, 1);
	if(norm_dvec_true > 0.0)
		reldiff /= norm_dvec_true;

	free(diff_vec);

	return reldiff;
}
