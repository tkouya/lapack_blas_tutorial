/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample Program of DGEEV with MAGMA            */
/* Last Update: 2015-04-06 (Mon) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>

// LAPACKE and BLAS on CPU
#include "cblas.h"
#include "lapacke.h"

// CUDA & MAGMA
#include "cuda.h"
#include "magma.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

int main()
{
	int i, j, dim; // dimension of vectors
	int info;
	double *ma, *revec, *levec;
	double complex *right_ev, *left_ev, *cmat, *ma_org, alpha, beta, *ceig;
	double *re_eig, *im_eig, ma_norm;
	int lwork_num;
	double *h_work;

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

	lwork_num = MAX(dim * (5 + 2 * dim), dim * (2 + magma_get_dgehrd_nb(dim)));
	magma_malloc_pinned((void **)&h_work, sizeof(double) * lwork_num);

	// input va and vb
	for(i = 0; i < dim; i++)
	{
		// Column major
		for(j = 0; j < dim; j++)
		{
			// A = Lotkin matrix
			//if(i == 0) ma[j] = 1.0;
			//else ma[i + j * dim] = 1.0 / (double)(i + j + 1);
			// A = Random matrix
			ma[i + j * dim] = (double)rand() / (double)RAND_MAX;
		
			// ma_org = ma
			ma_org[i + j * dim] = ma[i + j * dim] + 0.0 * I;
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

	// MAGMA開始
	magma_init();

	// solve A * V = \lambda * V
	// MagmaVec, MagmaVec ... get left and right eigenvectors
	// MagmaNoVec, MagmaVec ... get right eigenvector
	// MagmaVec, MagmaNoVec ... get left eigenvector
	// MagmaNoVec, MagmaNoVec ... get no eigenvectors
	magma_dgeev(MagmaVec, MagmaVec, dim, ma, dim, re_eig, im_eig, levec, dim, revec, dim, h_work, (magma_int_t)lwork_num, &info);
	if(info != 0)
		printf("magma_dgeev: return error %d: %s\n", (int)info, magma_strerror(info));

	// MAGMA終了
	magma_finalize();

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
				right_ev[j + i * dim] = revec[j + i * dim] + 0.0 * I;
				left_ev[j + i * dim] = levec[j + i * dim] + 0.0 * I;
				//left_ev[j + i  * dim] = levec[j + i  * dim] + 0.0 * I;
			}
		}
		// 固有値が複素数になる場合
		else if(im_eig[i] > 0.0)
		{
			for(j = 0; j < dim; j++)
			{
				right_ev[j + i * dim] = revec[j + i * dim] + revec[j + (i + 1) * dim] * I;
				left_ev[j + i * dim] = levec[j + i * dim]  - levec[j + (i + 1) * dim] * I;
			}
		}
		else
		{
			for(j = 0; j < dim; j++)
			{
				right_ev[j + i * dim] = revec[j + (i - 1) * dim] - revec[j + i * dim] * I;
				left_ev[j + i * dim] = levec[j + (i - 1) * dim] + levec[j + i * dim] * I;
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
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, (void *)&alpha, ma_org, dim, right_ev, dim, (void *)&beta, cmat, dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			cmat[i * dim + j] -= ceig[i] * right_ev[i * dim + j];
			//cmat[i + j * dim] -= ceig[i] * right_ev[i + j * dim];
	}
	 
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			printf("(%5g, %5g) ", creal(right_ev[i + j * dim]), cimag(right_ev[i * dim + j]));
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
	cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, dim, dim, dim, (void *)&alpha, ma_org, dim, left_ev, dim, (void *)&beta, cmat, dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			cmat[i * dim + j] -= ceig[i] * left_ev[i * dim + j];
			//cmat[i + j * dim] -= ceig[i] * left_ev[i + j * dim ];
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
