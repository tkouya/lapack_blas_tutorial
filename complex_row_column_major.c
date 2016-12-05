/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample probram with C99 complex row-major     */
/*                     and column major matrices */
/* Last Update: 2016-12-01 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // C99 complex data type

#include "cblas.h"

int main()
{
	int i, j, row_dim, col_dim;

	double complex *mat_a;

	// input dimension
	printf("Row    Dim = "); scanf("%d", &row_dim);
	printf("Column Dim = "); scanf("%d", &col_dim);

	if((row_dim <= 0) || (col_dim <= 0))
	{
		printf("Illegal dimension! (row_dim = %d, col_dim = %d)\n", row_dim, col_dim);
		return EXIT_FAILURE;
	}

	// initialize matrix area
	mat_a = (double complex *)calloc(row_dim * col_dim, sizeof(double complex));

	printf("Row Major: %d x %d\n", row_dim, col_dim);

	// Row Major 
	// mat_a = A
	// A = [1+i 2+2i ....... n+ni]
	//     [n+i n+2i ...... 2n+ni]
	//     [.....................]
	//     [(m-1)n+1+i ..... mn+ni]
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			mat_a[i * col_dim + j] = (double)(i * col_dim + j + 1) + (double)(j + 1) * I;
	}

	// print (1)
	printf("1 dimension: \n");
	printf("[");
	for(i = 0; i < row_dim * col_dim; i++)
		printf(" %6.3lf %+-6.3lf * i ", creal(mat_a[i]), cimag(mat_a[i]));
	printf("]\n");

	// print(2)
	printf("2 dimension: \n");
	for(i = 0; i < row_dim; i++)
	{
		printf("[");
		for(j = 0; j < col_dim; j++)
			printf(" %6.3lf %+-6.3lf * i ", creal(mat_a[i * col_dim + j]), cimag(mat_a[i * col_dim + j]));
		printf("]\n");
	}

	printf("Column Major: %d x %d\n", row_dim, col_dim);

	// Column Major
	// mat_a = A
	// A = [1+i 2+2i ....... n+ni]
	//     [n+i n+2i ...... 2n+ni]
	//     [.....................]
	//     [(m-1)n+1+i ..... mn+ni]
	for(j = 0; j < col_dim; j++)
	{
		for(i = 0; i < row_dim; i++)
			mat_a[i + row_dim * j] = (double)(i * col_dim + j + 1) + (double)(j + 1) * I;
	}

	// print (1)
	printf("1 dimension: \n");
	printf("[");
	for(i = 0; i < row_dim * col_dim; i++)
		printf(" %6.3lf  %+-6.3lf * i ", creal(mat_a[i]), cimag(mat_a[i]));
	printf("]\n");

	// print (2)
	printf("2 dimension: \n");
	for(i = 0; i < row_dim; i++)
	{
		printf("[");
		for(j = 0; j < col_dim; j++)
			printf(" %6.3lf %+-6.3lf * i ", creal(mat_a[i + row_dim * j]), cimag(mat_a[i + row_dim * j]));
		printf("]\n");
	}

	// free
	free(mat_a);

	return EXIT_SUCCESS;
}
