/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample Program of DGEEV                       */
/*                        with Intel Math Kernel */
/* Last Update: 2013-01-16 (Wed) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "lapacke.h"

int main()
{
	int i, j, dim; // dimension of vectors
	int info;
	double *ma, *revec, *levec;
	double complex *right_ev, *left_ev, *cmat, *ma_org, alpha, beta, *ceig;
	double *re_eig, *im_eig, ma_norm;

	// input dimension
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// Initialize
	re_eig = (double *)calloc(dim, sizeof(double));
	im_eig = (double *)calloc(dim, sizeof(double));
	ma = (double *)calloc(dim * dim, sizeof(double));
	ma_org = (double complex *)calloc(dim * dim, sizeof(double complex));
	levec = (double *)calloc(dim * dim, sizeof(double));
	revec = (double *)calloc(dim * dim, sizeof(double));
	left_ev = (double complex *)calloc(dim * dim, sizeof(double complex));
	right_ev = (double complex *)calloc(dim * dim, sizeof(double complex));
	cmat = (double complex *)calloc(dim * dim, sizeof(double complex));
	ceig = (double complex *)calloc(dim, sizeof(double complex));

	// input va and vb
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			// A = Lotkin matrix
			//if(i == 0) ma[j] = 1.0;
			//else ma[i * dim + j] = 1.0 / (double)(i + j + 1);
			// A = Random matrix
			ma[i * dim + j] = (double)rand() / (double)RAND_MAX;
		
			// ma_org = ma
			ma_org[i * dim + j] = ma[i * dim + j] + 0.0 * I;
		}
	}

	ma_norm = cblas_dnrm2(dim * dim, ma, 1);

	// print
	/*
	printf("A = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		for(j = 0; j < dim; j++)
			printf("%10g ", ma[i * dim + j]);
		printf("\n");

	}
	*/

	// solve A * V = \lambda * V
	// 'V', 'V' ... get left and right eigenvectors
	// 'N', 'V' ... get right eigenvector
	// 'V', 'N' ... get left eigenvector
	// 'N', 'N' ... get no eigenvectors
	info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'V', 'V', dim, ma, dim, re_eig, im_eig, levec, dim, revec, dim);

	// error occurs if info > 0
	if(info > 0)
	{
		printf("QR decomposition failed! (%d) \n", info);
		return EXIT_FAILURE;
	}
	else if(info < 0)
	{
		printf("%d-th argument of DGEEV is illegal!\n", info);
		return EXIT_FAILURE;
	}

	// reconstruction of eigenvectors
	for(i = 0; i < dim; i++)
	{
		ceig[i] = re_eig[i] + im_eig[i] * I;

		// 固有値が実数の時
		if(im_eig[i] == 0.0)
		{
			for(j = 0; j < dim; j++)
			{
				right_ev[j * dim + i] = revec[j * dim + i] + 0.0 * I;
				left_ev[j * dim + i] = levec[j * dim + i] + 0.0 * I;
				//left_ev[j + i  * dim] = levec[j + i  * dim] + 0.0 * I;
			}
		}
		// 固有値が複素数になる場合
		else if(im_eig[i] > 0.0)
		{
			for(j = 0; j < dim; j++)
			{
				right_ev[j * dim + i] = revec[j * dim + i] + revec[j * dim + i + 1] * I;
				left_ev[j * dim + i] = levec[j * dim + i]  - levec[j * dim + i + 1] * I;
			}
		}
		else
		{
			for(j = 0; j < dim; j++)
			{
				right_ev[j * dim + i] = revec[j * dim + i - 1] - revec[j * dim + i] * I;
				left_ev[j * dim + i] = levec[j * dim + i - 1] + levec[j * dim + i] * I;
			}
		}
	}

	// print
	printf("Eigenvalues = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		printf("(%10g, %10g)\n", re_eig[i], im_eig[i]);

	}
	printf("\n");

	alpha = 1.0 + 0.0 * I;
	beta = 0.0 + 0.0 * I;
	set0_zmatrix(cmat, dim, dim);

	// A * right_ev - \lambda * right_ev == 0 ?
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, (void *)&alpha, ma_org, dim, right_ev, dim, (void *)&beta, cmat, dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			cmat[i + j * dim] -= ceig[i] * right_ev[i + j * dim];
	}
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			printf("(%5g, %5g) ", creal(right_ev[i * dim + j]), cimag(right_ev[i * dim + j]));
		printf("\n");
	}
	/* 
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			printf("(%5g, %5g) ", creal(cmat[i * dim + j]), cimag(cmat[i * dim + j]));
		printf("\n");
	}
	*/
	printf("||A * right_ev - lambda * right_ev||_F / ||A||_F = %25.17e\n", cblas_dznrm2(dim * dim, cmat, 1) / ma_norm);

	set0_zmatrix(cmat, dim, dim);

	// left_ev * A - \lambda * left_ev == 0 ?
	cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dim, dim, dim, (void *)&alpha, ma_org, dim, left_ev, dim, (void *)&beta, cmat, dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			cmat[i + j * dim] -= ceig[i] * left_ev[i + j * dim ];
	}
	/*for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			printf("(%5g, %5g) ", creal(cmat[i * dim + j]), cimag(cmat[i * dim + j]));
		printf("\n");
	}
	*/
	printf("||A^T * left_ev - lambda * left_ev||_F / ||A||_F = %25.17e\n", cblas_dznrm2(dim * dim, cmat, 1) / ma_norm);


	// free
	free(re_eig);
	free(im_eig);
	free(revec);
	free(levec);
	free(right_ev);
	free(left_ev);
	free(ma);
	free(ma_org);
	free(cmat);
	free(ceig);

	return EXIT_SUCCESS;
}
