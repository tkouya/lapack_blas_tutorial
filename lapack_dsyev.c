/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample Program of DSYEV                       */
/*                        with Intel Math Kernel */
/* Last Update: 2025-02-04 (Tue) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // 2025-02-04(Tue)
#include "cblas.h"
#include "lapacke.h"

#define MIN(i, j) ((i) < (j) ? (i) : (j))
#define MAX(i, j) ((i) > (j) ? (i) : (j))

int main()
{
	int i, j, dim; // dimension of vectors
	int info;
	double *ma;
	double *cmat, *ma_org, *diag, alpha, beta;
	int absmax_eig_index, absmin_eig_index;
	double *eig, absmax_eig, absmin_eig, abs_eig, ma_norm;

	// input dimension
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// Initialize
	eig = (double *)calloc(sizeof(double), dim);
	ma = (double *)calloc(sizeof(double), dim * dim);
	ma_org = (double *)calloc(sizeof(double), dim * dim);
	diag = (double *)calloc(sizeof(double), dim * dim);
	cmat = (double *)calloc(sizeof(double), dim * dim);

	// input ma, ma_org
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			// A = Hilbert matrix
			//ma[i * dim + j] = 1.0 / (double)(i + j + 1);
			// A = Frank matrix
			ma[i * dim + j] = (double)(dim - MAX(i, j));
			// A = Random matrix
			//ma[i * dim + j] = (double)rand() / (double)RAND_MAX;
		
			// ma_org = ma
			ma_org[i * dim + j] = ma[i * dim + j];
		}
	}

	ma_norm = cblas_dnrm2(dim * dim, ma, 1);

	// print
/*	printf("A = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		for(j = 0; j < dim; j++)
			printf("%10g ", ma[i * dim + j]);
		printf("\n");

	}
*/
	// solve A * V = \lambda * V
	// 'V' ... get eigenvectors
	info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', dim, ma, dim, eig);

	// error occurs if info > 0
	if(info > 0)
	{
		printf("QR decomposition failed! (%d) \n", info);
		return EXIT_FAILURE;
	}
	else if(info < 0)
	{
		printf("%d-th argument of DSYEV is illegal!\n", info);
		return EXIT_FAILURE;
	}


	printf("Envenvectors: \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		for(j = 0; j < dim; j++)
			printf("%5g ", ma[i * dim + j]);
		printf("\n");

	}

	// diag := diag(eig)
	absmax_eig_index = absmin_eig_index = 0;
	absmax_eig = absmin_eig = fabs(eig[0]);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			diag[i * dim + j] = 0.0;
		diag[i * dim + i] = eig[i];

		abs_eig = fabs(eig[i]);
		if(absmax_eig < abs_eig)
		{
			absmax_eig = abs_eig;
			absmax_eig_index = i;
		}
		if(absmin_eig > abs_eig)
		{
			absmin_eig = abs_eig;
			absmin_eig_index = i;
		}
	}

	// print
	printf("Eigenvalues = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		printf("%10g\n", eig[i]);
	}
	printf("\n");

	printf("absmax_eig = %25.17e\n", eig[absmax_eig_index]);
	printf("absmin_eig = %25.17e\n", eig[absmin_eig_index]);
	printf("cond2      = %25.17e\n", absmax_eig / absmin_eig);

	alpha = 1.0;
	beta = 0.0;

	// ev^T * A * ev - lambda * I ?
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dim, dim, dim, alpha, ma, dim, ma_org, dim, beta, cmat, dim);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, alpha, cmat, dim, ma, dim, beta, ma_org, dim);

	alpha = -1.0;
	cblas_daxpy(dim * dim, alpha, diag, 1, ma_org, 1);

	printf("||ev^T * A * ev - lambda||_F / ||A||_F  = %15.7e\n", cblas_dnrm2(dim * dim, ma_org, 1) / ma_norm);

	/*
	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		for(j = 0; j < dim; j++)
			printf("%5g ", ma_org[i * dim + j]);
		printf("\n");
	}
*/
	// free
	free(eig);
	free(cmat);
	free(ma);
	free(ma_org);
	free(diag);

	return EXIT_SUCCESS;
}
