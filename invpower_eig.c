/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Inverse Power Method for min eigenvalue       */
/*                                 & eigenvetor  */
/* Last Update: 2015-02-23 (Mon) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"
#include "lapacke.h"

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
	int i, j, dim, itimes, *pivot, info;
	double *ma, *ma_org, *vy, *vx, *vx_new, *vb, eig, eig_old, y_norm;
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
	ma_org = (double *)calloc(dim * dim, sizeof(double));
	vy = (double *)calloc(dim, sizeof(double));
	vx = (double *)calloc(dim, sizeof(double));

	// input ma
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			ma[i * dim + j] = sqrt(2.0) * (double)(i + j + 1);
		ma[i * dim + i] = sqrt(2.0) * dim * dim;
	}
	cblas_dcopy(dim * dim, ma, 1, ma_org, 1);

	reps = 1.0e-10;
	aeps = 0.0;


	// vx := [1 1 ... 1] / sqrt(dim)
	for(i = 0; i < dim; i++)
		vx[i] = 1.0 / sqrt((double)dim);

	eig_old = eig = 0.0;

	// Inverse Power Method
	// ピボット初期化
	pivot = (lapack_int *)calloc(dim, sizeof(lapack_int));

	// LU分解
	info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim, dim, ma, dim, pivot);
	//printf("DGETRF info = %d\n", info);

	// vy := vx
	cblas_dcopy(dim, vx, 1, vy, 1);

	for(itimes = 0; itimes < dim * 10; itimes++)
	{
		// y := A^(-1) * x
		info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', dim, 1, ma, dim, pivot, vy, 1);
		//printf("DGETRS info = %d\n", info);
		//for(i = 0; i < dim; i++)
		//	printf("%3d %15.7e %15.7e\n", i, vx[i], vy[i]);
		//printf("\n");

		// eig^{-1} = (A^(-1) * x, x) / (x, x) = (A^(-1) * x, x)
		eig = 1.0 / cblas_ddot(dim, vy, 1, vx, 1);

		// |eig_old - eig | <= reps * |eig| + aeps
		if(fabs(eig_old - eig) <= reps * fabs(eig) + aeps)
			break;

		printf("%3d %25.17e\n", itimes, eig);

		eig_old = eig;

		// x := y / ||y||
		y_norm = 1.0 / cblas_dnrm2(dim, vy, 1);
		cblas_dscal(dim, y_norm, vy, 1);
		cblas_dcopy(dim, vy, 1, vx, 1);
	}

	// eig * x == A * x?
	cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0, ma_org, dim, vx, 1, 0.0, vy, 1);
	cblas_dscal(dim, eig, vx, 1);

	// print
	printf("Iterative Times = %d\n", itimes);
	printf("Min. Eigenvalue = %25.17e\n", eig);
	printf("Rel.Diff = %10.3e\n", reldiff_dvector(vx, vy, dim));

	for(i = 0; i < dim; i++)
	{
		printf("%3d %25.17e %25.17e\n", i, eig * vx[i], vy[i]);
	}

	// free
	free(ma);
	free(ma_org);
	free(vy);
	free(vx);


	return EXIT_SUCCESS;
}
