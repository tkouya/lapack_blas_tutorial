/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample Program of DEGCON                      */
/*                        with Intel Math Kernel */
/* Last Update: 2016-11-30 (Wed) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#ifdef USE_IMKL
	#include "mkl.h" // for Intel Math Kernel Library
	#include "mkl_lapack.h" // for dlange
	#include "mkl_lapacke.h" // for dgecon
	#include "mkl_cblas.h" // for dcopy
#else // USE_IMKL
	#include "cblas.h"
	#include "lapacke.h"
#endif // USE_IMKL

int main()
{
	int i, j, dim; // dimension of vectors
	int info, *pivot;
	double *ma, *ma_work;
	double norm1, cond1; // ||A||_1 and cond_1(A)
	double normi, condi; // ||A||_inf and cond_inf(A)
	char str_mkl_version[1024];

#ifdef USE_IMKL
	// print MKL version
	MKL_Get_Version_String(str_mkl_version, 1024);
	printf("%s\n", str_mkl_version);
#endif // USE_IMKL

	// input dimension
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// Initialize
	ma = (double *)calloc(sizeof(double), dim * dim);
	ma_work = (double *)calloc(sizeof(double), dim * dim);
	pivot = (int *)calloc(dim, sizeof(int));

	// input va and vb
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			// A = hirbert matrix
			*(ma + i * dim + j) = 1.0 / (double)(i + j + 1);
			/* ma[i * dim + j] = (double)rand() / (double)(RAND_MAX);
			if(rand() % 2 != 0)
				ma[i * dim + j] = -ma[i * dim + j];
			*/
		}
	}

	// A := A
	cblas_dcopy(dim * dim, ma, 1, ma_work, 1);

	// print
/*	printf("A = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		for(j = 0; j < dim; j++)
			printf("%10f ", *(ma + i * dim + j));
		printf("\n");

	}
*/
	// Get ||A||_1
	norm1 = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', dim, dim, ma_work, dim);

	// LU decomposition
	info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim, dim, ma_work, dim, pivot);
	//printf("DGETRF info = %d\n", info);

	// Compute condition number of A
	info = LAPACKE_dgecon(LAPACK_ROW_MAJOR, '1', dim, ma_work, dim, norm1, &cond1);

	// error occurs if info > 0
	if(info < 0)
	{
		printf("The %d-th parameter is illegal!\n", -info);
		return EXIT_FAILURE;
	}

	// A := A
	cblas_dcopy(dim * dim, ma, 1, ma_work, 1);

	// Get ||A||_inf
	normi = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'I', dim, dim, ma_work, dim);

	// LU decomposition
	info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim, dim, ma_work, dim, pivot);
	//printf("DGETRF info = %d\n", info);

	// Compute condition number of A
	info = LAPACKE_dgecon(LAPACK_ROW_MAJOR, 'I', dim, ma_work, dim, normi, &condi);

	// error occurs if info > 0
	if(info < 0)
	{
		printf("The %d-th parameter is illegal!\n", -info);
		return EXIT_FAILURE;
	}

	// print norm and condition number of A
	printf("||A||_1   = %25.17e, cond_1(A)   = %25.17e\n", norm1, 1.0 / cond1);
	printf("||A||_inf = %25.17e, cond_inf(A) = %25.17e\n", normi, 1.0 / condi);

	// free
	free(ma);
	free(ma_work);

	return EXIT_SUCCESS;
}
