/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Solver for Linear equation with MAGMA         */
/* Last Update: 2016-12-01 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// BLAS on CPU
#include "cblas.h"

// CUDA & MAGMA
#include "cuda.h"
#include "magma.h"

int main()
{
	int i, j, dim;
	int inc_vec_x, inc_vec_b;
	int *pivot, info;

	double *mat_a, *vec_b, *vec_x; // CPU
	double alpha, beta;

	// input dimension
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// initialize a matrix and vectors on CPU
	mat_a = (double *)calloc(dim * dim, sizeof(double));
	vec_x = (double *)calloc(dim, sizeof(double));
	vec_b = (double *)calloc(dim, sizeof(double));

	// input mat_a and vec_x on CPU
	for(j = 0; j < dim; j++)
	{
		// Column-major
		for(i = 0; i < dim; i++)
		{
			mat_a[i + j * dim] = (double)rand() / (double)RAND_MAX;
			if(rand() % 2 != 0)
				mat_a[i + j * dim] = -mat_a[i + j * dim];
		}
		vec_x[j] = 1.0 / (double)(j + 1);
	}

	// size(vec_x) == size(vec_b)
	inc_vec_x = inc_vec_b = 1;

	// vec_b := 1.0 * mat_a * vec_x + 0.0 * vec_b
	alpha = 1.0;
	beta = 0.0;
	cblas_dgemv(CblasColMajor, CblasNoTrans, dim, dim, alpha, mat_a, dim, vec_x, inc_vec_x, beta, vec_b, inc_vec_b);

	// print
/*	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3f ", mat_a[i + j * dim]);
		printf("]  %10.3f = %10.3f\n", vec_x[i], vec_b[i]);
	}
*/
	// start MAGMA
	magma_init();

	// initialize pivot area
	pivot = (int *)calloc(sizeof(int), dim);

	// solve A * X = C -> C := X
	magma_dgesv(dim, 1, mat_a, dim, pivot, vec_b, dim, &info);

	printf("info = %d\n", info);

	// print
/*	printf("calculated x = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d -> %3d: ", i, pivot[i]);
		printf("%25.17e ", vec_b[i]);
		printf("\n");
	}

	// diff
	printf("x - calculated x = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		printf("%10.2e ", fabs((vec_x[i] - vec_b[i]) / vec_x[i]));
		printf("\n");
	}
*/
	// norm relative error: ||vec_b - vec_x||_2 / ||vec_x||_2
	cblas_daxpy(dim, -1.0, vec_x, 1, vec_b, 1);
	printf("||x - calculated x||_2 = %10.7e\n", cblas_dnrm2(dim, vec_b, 1) / cblas_dnrm2(dim, vec_x, 1));

	// free
	free(mat_a);
	free(vec_x);
	free(vec_b);
	free(pivot);

	// finalize MAGMA
	magma_finalize();

	return EXIT_SUCCESS;
}
