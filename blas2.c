/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample Program of BLAS Level2                 */
/*                        with Intel Math Kernel */
/* Last Update: 2013-08-05 (Mon) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"

int main()
{
	int i, j, dim;
	int inc_vb, inc_vc;
	double *ma, *vb, *vc;
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
	vb = (double *)calloc(dim, sizeof(double));
	vc = (double *)calloc(dim, sizeof(double));

	// input ma and vb
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			ma[i * dim + j] = sqrt(2.0) * (double)(dim - (i + j + 1));
		vb[i] = sqrt(2.0) * (double)(i + 1);
	}

	//vc := vb
	inc_vb = inc_vc = 1;

	// vc := 1.0 * ma * vb
	alpha = 1.0;
	beta = 0.0;
	cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, alpha, ma, dim, vb, inc_vb, beta, vc, inc_vc);

	// print
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3f ", ma[i * dim + j]);
		printf("]  %10.3f = %10.3f\n", vb[i], vc[i]);
	}

	// free
	free(ma);
	free(vb);
	free(vc);

	return EXIT_SUCCESS;
}
