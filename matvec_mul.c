/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Mutiplication of matrix and vector            */
/* Last Update: 2016-12-01 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cblas.h"

int main()
{
	int i, j, dim;
	int inc_vec_x, inc_vec_b;

	double *mat_a, *vec_b, *vec_x;
	double alpha, beta;

	// ŸŒ³”“ü—Í
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// •Ï”‰Šú‰»
	mat_a = (double *)calloc(dim * dim, sizeof(double));
	vec_x = (double *)calloc(dim, sizeof(double));
	vec_b = (double *)calloc(dim, sizeof(double));

	// mat_a‚Ævec_x‚É’l“ü—Í
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			mat_a[i * dim + j] = (double)(i + j + 1);
			if((i + j + 1) % 2 != 0)
				mat_a[i * dim + j] *= -1.0;
		}
		vec_x[i] = 1.0 / (double)(i + 1);
	}

	// size(vec_x) == size(vec_b)
	inc_vec_x = inc_vec_b = 1;

	// vec_b := 1.0 * mat_a * vec_x + 0.0 * vec_b
	alpha = 1.0;
	beta = 0.0;
	cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, alpha, mat_a, dim, vec_x, inc_vec_x, beta, vec_b, inc_vec_b);

	// o—Í
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3lf ", mat_a[i * dim + j]);
		printf("]  %10.3lf = %10.3lf\n", vec_x[i], vec_b[i]);
	}

	// •Ï”Á‹
	free(mat_a);
	free(vec_x);
	free(vec_b);

	return EXIT_SUCCESS;
}
