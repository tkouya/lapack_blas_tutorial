/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample Program of BLAS Level3                 */
/*                        with Intel Math Kernel */
/* Last Update: 2011-06-10 (Fri) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cblas.h"

int main()
{
	int i, j, dim;
	double *ma, *mb, *mc;
	double alpha, beta;

	// input dimension
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// Initialize
	ma = (double *)calloc(dim * dim, sizeof(double));
	mb = (double *)calloc(dim * dim, sizeof(double));
	mc = (double *)calloc(dim * dim, sizeof(double));

	// input va and vb
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			ma[i * dim + j] = sqrt(2.0) * (i + j + 1);
			mb[i * dim + j] = sqrt(2.0) * (dim * 2 - (i + j + 1));
		}
	}

	// print
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3f ", ma[i * dim + j]);
		printf("] [");
		for(j = 0; j < dim; j++)
			printf("%10.3f ", mb[i * dim + j]);
		printf("]\n");
	}

	// mc := 1.0 * ma * mb + 0.0 * mc
	alpha = 1.0;
	beta = 0.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, alpha, ma, dim, mb, dim, beta, mc, dim);

	// print
	printf(" = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d: [", i);
		for(j = 0; j < dim; j++)
			printf("%10.3f ", mc[i * dim + j]);
		printf("]\n");
	}

	return EXIT_SUCCESS;
}
