/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample Program of BLAS Level1                 */
/*                        with Intel Math Kernel */
/* Last Update: 2011-06-10 (Fri) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"

int main()
{
	int i, dim;
	int inc_vb, inc_vc, inc_va;
	double *va, *vb, *vc;
	double alpha;

	// input dimension
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// Initialize
	va = (double *)calloc(dim, sizeof(double));
	vb = (double *)calloc(dim, sizeof(double));
	vc = (double *)calloc(dim, sizeof(double));

	// input va and vb
	for(i = 0; i < dim; i++)
	{
		va[i] = i + 1;
		vb[i] = dim - (i + 1);
	}

	//vc := vb
	inc_vb = inc_vc = inc_va = 1;
	cblas_dcopy(dim, vb, inc_vb, vc, inc_vc);

	// vc := 1.0 * va + vb
	alpha = 1.0;
	cblas_daxpy(dim, alpha, va, inc_va, vc, inc_vc);

	// print
	for(i = 0; i < dim; i++)
		printf("%10.3f + %10.3f = %10.3f\n", *(va + i), *(vb + i), *(vc + i));

	// free
	free(va);
	free(vb);
	free(vc);

	return EXIT_SUCCESS;
}
