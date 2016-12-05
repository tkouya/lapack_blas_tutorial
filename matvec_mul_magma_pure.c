/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Mutiplication of matrix and vector            */
/*                                with magmablas */
/* Last Update: 2016-12-01 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// BLAS on CPU
#include "cblas.h"

// MAGMA
#include "magma.h"

int main()
{
	int i, j, dim;
	int inc_vec_x, inc_vec_b;

	double *mat_a, *vec_b, *vec_x, *vec_b_gpu; // CPU
	double *dev_mat_a, *dev_vec_b, *dev_vec_x; // GPU
	double alpha, beta;

	// input dimension
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// start MAGMA
	magma_init();

	// initialize a matrix and vectors on CPU
	magma_dmalloc_cpu(&mat_a, dim * dim);
	magma_dmalloc_cpu(&vec_b, dim);
	magma_dmalloc_cpu(&vec_x, dim);

	// input mat_a and vec_x
	for(j = 0; j < dim; j++)
	{
		for(i = 0; i < dim; i++)
		{
			// column-major
			mat_a[i + j * dim] = (double)rand() / (double)RAND_MAX;
			if(rand() % 2 != 0)
				mat_a[i + j * dim] *= -1.0;
		}
		vec_x[j] = 1.0 / (double)(j + 1);
	}

	// size(vec_x) == size(vec_b)
	inc_vec_x = inc_vec_b = 1;

	// vec_b := 1.0 * mat_a * vec_x + 0.0 * vec_b
	alpha = 1.0;
	beta = 0.0;
	cblas_dgemv(CblasColMajor, CblasNoTrans, dim, dim, alpha, mat_a, dim, vec_x, inc_vec_x, beta, vec_b, inc_vec_b);

	// print
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3lf ", mat_a[i + j * dim]);
		printf("]  %10.3lf = %10.3lf\n", vec_x[i], vec_b[i]);
	}

	// GPU

	// initialize a matrix and vectors on GPU (MAGMA)
	magma_dmalloc(&dev_mat_a, dim * dim);
	magma_dmalloc(&dev_vec_x, dim);
	magma_dmalloc(&dev_vec_b, dim);

	// dev_vec_x on GPU -> vec_x_gpu on CPU
	magma_dmalloc_cpu(&vec_b_gpu, dim);

	// CPU(Host) -> GPU(device)

	// mat_a -> dev_mat_a
	magma_dsetmatrix(dim, dim, mat_a, dim, dev_mat_a, dim);

	// size(vec_x) == size(vec_b)
	inc_vec_x = inc_vec_b = 1;

	// vec_x -> dev_vec_x
	magma_dsetvector(dim, vec_x, inc_vec_x, dev_vec_x, inc_vec_x);

	// vec_b := 1.0 * mat_a * vec_x + 0.0 * vec_b
	alpha = 1.0;
	beta = 0.0;
	magmablas_dgemv(MagmaNoTrans, dim, dim, alpha, dev_mat_a, dim, dev_vec_x, inc_vec_x, beta, dev_vec_b, inc_vec_b);

	// dev_vec_b -> vec_b_gpu
	magma_dgetvector(dim, dev_vec_b, inc_vec_b, vec_b_gpu, inc_vec_b);

	// print
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3lf ", mat_a[i + j * dim]);
		printf("]  %10.3lf = %10.3lf\n", vec_x[i], vec_b_gpu[i]);
	}

	// relative difference: ||vec_b_gpu - vec_b||_2 / ||vec_b||_2
	cblas_daxpy(dim, -1.0, vec_b, inc_vec_b, vec_b_gpu, inc_vec_b);
	printf("||vec_b_gpu - vec_b||_2 / ||vec_b||_2 = %15.7e\n", cblas_dnrm2(dim, vec_b_gpu, 1) / cblas_dnrm2(dim, vec_b, 1));

	// free
	magma_free_cpu(mat_a);
	magma_free_cpu(vec_x);
	magma_free_cpu(vec_b);
	magma_free_cpu(vec_b_gpu);

	magma_free(dev_mat_a);
	magma_free(dev_vec_b);
	magma_free(dev_vec_x);

	// finalize MAGMA
	magma_finalize();

	return EXIT_SUCCESS;
}
