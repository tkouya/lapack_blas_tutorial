/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Solver for Linear equation with DSGESV        */
/* Last Update: 2016-11-30 (Wed) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lapacke.h"
#include "cblas.h"

#define IJMAX(i, j) ( ((i) > (j)) ? (i) : (j) )
#define IJMIN(i, j) (((i) < (j)) ? (i) : (j))

int main()
{
	lapack_int i, j, k, dim, shift, index, ku, kl;
	lapack_int inc_vec_x, inc_vec_b;
	lapack_int *pivot, info;

	double *mat_a, *vec_b, *vec_x, *mat_a_full;
	double alpha, beta;
	double running_time;

	// input dimension of a linear equation to be solved
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// initialize a tridiagonal matrix(mat_a) and vectors
	kl = 1;
	ku = 1; // necessary for pivoting
	mat_a = (double *)calloc((kl * 2 + ku + 1) * dim, sizeof(double));
	vec_x = (double *)calloc(dim, sizeof(double));
	vec_b = (double *)calloc(dim, sizeof(double));

	// dim * dim
	mat_a_full = (double *)calloc(dim * dim, sizeof(double));

	for(i = 0; i < dim; i++)
		vec_b[i] = 0.0;

	// input mat_a and vec_x
	for(i = 0; i < dim; i++)
	{
		// mat_a_full := 0
		for(j = 0; j < dim; j++)
			mat_a_full[i * dim + j] = 0.0;
	}

	for(i = 0; i < dim; i++)
	{
		// upper subdiagonal element
		if((i + 1) < dim)
		{
			j = i + 1;
			index = i * dim + j;
			mat_a_full[index] = (double)(j + 1);

			if((i + j + 1) % 2 != 0)
				mat_a_full[index] = -mat_a_full[index];
		}

		// diagonal element
		j = i;
		index = i * dim + j;
		mat_a_full[index] = 1.0 / (double)(i + j + 1);
		if((i + j + 1) % 2 != 0)
			mat_a_full[index] = -mat_a_full[index];

		mat_a_full[index] += 2.0;

		// lower subdiagonal element
		if((i - 1) >= 0)
		{
			j = i - 1;
			index = i * dim + j;
			mat_a_full[index] = 1.0 / (double)(i + 1);

			if((i + j + 1) % 2 != 0)
				mat_a_full[index] = -mat_a_full[index];
		}

		//vec_x[i] = 1.0 / (double)(i + 1);
		vec_x[i] = 1.0 ;
	}

	// print
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			printf("%10.3e ", mat_a_full[i * dim + j]);
		printf("\n");
	}
	printf("\n");

	// convert
	for(j = 0; j < dim; j++)
	{
		k = ku - j;
		for(i = IJMAX(0, j - ku); i < IJMIN(dim, j + kl + 1); i++)
		{
			printf("(%d, %d) -> (%d, %d)\n", i, j, k + i, j);
			mat_a[(k + i) * dim + j] = mat_a_full[i * dim + j];
		}
	}

	// print
	for(i = 0; i < (ku + kl + 1); i++)
	{
		for(j = 0; j < dim; j++)
			printf("%10.3e ", mat_a[i * dim + j]);
		printf("\n");
	}
	printf("\n");

	// size(vec_x) == size(vec_b)
	inc_vec_x = inc_vec_b = 1;

	// vec_b := 1.0 * mat_a * vec_x + 0.0 * vec_b
	alpha = 1.0;
	beta = 0.0;
	//cblas_dgbmv(CblasRowMajor, CblasTrans, dim, dim, kl, ku, alpha, mat_a, dim, vec_x, inc_vec_x, beta, vec_b, inc_vec_b);
	cblas_dgbmv(CblasColMajor, CblasNoTrans, dim, dim, kl, ku, alpha, mat_a, dim, vec_x, inc_vec_x, beta, vec_b, inc_vec_b);

	// print
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
		{
			if(j == (i + 1))
				printf("%10.3e ", mat_a[shift + j]);
			else if(j == i)
				printf("%10.3e ", mat_a[shift + dim + j]);
			else if(j == (i - 1))
				printf("%10.3e ", mat_a[shift + 2 * dim + j]);
			else 
				printf("%10.3e ", 0.0);
		}

		printf("]  %10.3f = %10.3f\n", vec_x[i], vec_b[i]);
	}

	// initialize pivot
	pivot = (lapack_int *)calloc(dim, sizeof(lapack_int));

	// solve A * X = C -> C := X
	info = LAPACKE_dgbsv(LAPACK_COL_MAJOR, dim, kl, ku, 1, mat_a, kl * 2 + ku + 1, pivot, vec_b, dim);

	printf("info = %d\n", info);

	// print
	printf("calculated x = \n");
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
	
	// free
	free(mat_a);
	free(vec_x);
	free(vec_b);
	free(pivot);
	free(mat_a_full);

	return EXIT_SUCCESS;
}
