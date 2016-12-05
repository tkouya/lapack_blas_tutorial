/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Jacobi Iterative Refinement                   */
/* Last Update: 2015-02-20 (Fri) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"

// relative difference
double reldiff_dvector(double *vec1, double *vec2, int dim)
{
	double *tmp_vec;
	double ret, norm;

	tmp_vec = (double *)calloc(dim, sizeof(double));

	cblas_dcopy(dim, vec1, 1, tmp_vec, 1);
	cblas_daxpy(dim, -1.0, vec2, 1, tmp_vec, 1);
	ret = cblas_dnrm2(dim, tmp_vec, 1);
	norm = cblas_dnrm2(dim, vec1, 1);

	if(norm != 0.0)
		ret /= norm;

	return ret;
}

int main()
{
	int i, j, dim, itimes;
	double *ma, *md, *vy, *vx, *vx_true, *vx_new, *vb;
	double reps, aeps;

	// input dimension
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// Initialize
	ma = (double *)calloc(dim * dim, sizeof(double));
	md = (double *)calloc(dim, sizeof(double));
	vy = (double *)calloc(dim, sizeof(double));
	vx = (double *)calloc(dim, sizeof(double));
	vx_new = (double *)calloc(dim, sizeof(double));
	vx_true = (double *)calloc(dim, sizeof(double));
	vb = (double *)calloc(dim, sizeof(double));

	// input ma and vx_true
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			ma[i * dim + j] = sqrt(2.0) * (double)(i + j + 1);
		ma[i * dim + i] = sqrt(2.0) * dim * dim;
		vx_true[i] = sqrt(2.0) * (double)(i + 1);
	}

	// vb := 1.0 * ma * vx_true
	cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0, ma, dim, vx_true, 1, 0.0, vb, 1);

	reps = 1.0e-10;
	aeps = 0.0;

	// md := D^{-1} = diag[1/a11, 1/a22, ..., 1/ann]
	for(i = 0; i < dim; i++)
		md[i] = 1.0 / ma[i * dim + i];

	// vx := 0
	for(i = 0; i < dim; i++)
		vx[i] = 0.0;

	// Jacobi Iteration
	for(itimes = 0; itimes < dim * 10; itimes++)
	{
		// vy := vb
		cblas_dcopy(dim, vb, 1, vy, 1);

		// vx_new := vx
		cblas_dcopy(dim, vx, 1, vx_new, 1);

		// y := b - A * x
		cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, -1.0, ma, dim, vx, 1, 1.0, vy, 1);
		//for(i = 0; i < dim; i++)
		//	printf("%3d %15.7e %15.7e\n", i, vx[i], vy[i]);
		//printf("\n");

		// x_new := x + D^{-1} * y
		cblas_dsbmv(CblasRowMajor, CblasUpper, dim, 0, 1.0, md, 1, vy, 1, 1.0, vx_new, 1);
		//for(i = 0; i < dim; i++)
		//	vx_new[i] = vx[i] + md[i] * vy[i];

		//for(i = 0; i < dim; i++)
		//	printf("%3d %15.7e %15.7e: %15.7e\n", i, vy[i], vx_new[i], md[i]);
		//printf("\n");

		// || x_new - x || < reps || x_new || + aeps
		cblas_daxpy(dim, -1.0, vx_new, 1, vx, 1);
		if(cblas_dnrm2(dim, vx, 1) <= reps * cblas_dnrm2(dim, vx_new, 1) + aeps)
		{
			// vx := vx_new
			cblas_dcopy(dim, vx_new, 1, vx, 1);
			break;
		}

		printf("%3d %10.3e\n", itimes, cblas_dnrm2(dim, vx, 1));

		// vx := vx_new
		cblas_dcopy(dim, vx_new, 1, vx, 1);
	}

	// print
	printf("Iterative Times = %d\n", itimes);
	printf("Rel.Diff = %10.3e\n", reldiff_dvector(vx, vx_true, dim));

	for(i = 0; i < dim; i++)
	{
		printf("%3d %25.17e %25.17e\n", i, vx[i], vx_true[i]);
	}

	// free
	free(ma);
	free(md);
	free(vy);
	free(vx);
	free(vx_true);
	free(vx_new);
	free(vb);


	return EXIT_SUCCESS;
}
