/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Multiplication of C99 complex matrix          */
/*                                    and vector */
/* Last Update: 2016-12-01 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // C99 complex data type

#include "cblas.h"

int main()
{
	int i, j, dim;
	int inc_vec_x, inc_vec_b;

	double complex *mat_a, *vec_b, *vec_x;
	double complex alpha, beta;

	// input dimension
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// initialize a matrix and vectors
	mat_a = (double complex *)calloc(dim * dim, sizeof(double complex));
	vec_x = (double complex *)calloc(dim, sizeof(double complex));
	vec_b = (double complex *)calloc(dim, sizeof(double complex));

	// input mat_a and vec_x
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			mat_a[i * dim + j] = (double)(i + j + 1) - (double)(i + j + 1) * I;
			if((i + j + 1) % 2 != 0)
				mat_a[i * dim + j] *= -1.0;
		}
		vec_x[i] = 1.0 / (double)(i + 1) + 1.0 / (double)(i + 1) * I;
	}

	// size(vec_x) == size(vec_b)
	inc_vec_x = inc_vec_b = 1;

	// vec_b := 1.0 * mat_a * vec_x + 0.0 * vec_b
	alpha = 1.0;
	beta = 0.0;
	cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, (void *)&alpha, mat_a, dim, vec_x, inc_vec_x, (void *)&beta, vec_b, inc_vec_b);

	// print
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%6.3lf %+-6.3lf * I ", creal(mat_a[i * dim + j]), cimag(mat_a[i * dim + j]));
		printf("]  %6.3lf %+-6.3lf * I = %6.3lf %+-6.3lf * I\n", creal(vec_x[i]), cimag(vec_x[i]), creal(vec_b[i]), cimag(vec_b[i]));
	}

	// free
	free(mat_a);
	free(vec_x);
	free(vec_b);

	return EXIT_SUCCESS;
}
