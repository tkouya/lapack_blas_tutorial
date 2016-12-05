/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Multiplication of matrix and vector           */
/*                                   with OpenMP */
/* Last Update: 2016-11-30 (Wed) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// rowwise only
// my_matvec_mul: vec_b := mat_a * vec_x
#ifdef _OPENMP
void my_matvec_mul_omp(double *vec_b, double *mat_a, int row_dim, int col_dim, double *vec_x)
{
	int i, j, row_index;

	// parallelized main loop
	#pragma omp parallel for private(j)
	for(i = 0; i < row_dim; i++)
	{
		//printf("Thread No. %d:\n", omp_get_thread_num());
		vec_b[i] = 0.0;
		row_index = row_dim * i;

		for(j = 0; j < col_dim; j++)
			vec_b[i] += mat_a[row_index + j] * vec_x[j];
	}
}
#endif // _OPENMP

// rowwise only
// my_matvec_mul: vec_b := mat_a * vec_x
void my_matvec_mul(double *vec_b, double *mat_a, int row_dim, int col_dim, double *vec_x)
{
	int i, j, row_index;

	// serial main loop
	for(i = 0; i < row_dim; i++)
	{
		vec_b[i] = 0.0;
		row_index = row_dim * i;

		for(j = 0; j < col_dim; j++)
			vec_b[i] += mat_a[row_index + j] * vec_x[j];
	}
}

int main()
{
	int i, j, dim;

	double *mat_a, *vec_b, *vec_x;

	// input dimension of square matrix and vector
	printf("Dim = "); scanf("%d", &dim);

	// input maximum number of threads
#ifdef _OPENMP
	int num_threads;

	printf("Max.Num.threads = "); scanf("%d", &num_threads);
	omp_set_num_threads(num_threads);
	printf("#threads = %d\n", omp_get_max_threads());
#endif // _OPENMP

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// initialize a matrix and vectors
	mat_a = (double *)calloc(dim * dim, sizeof(double));
	vec_x = (double *)calloc(dim, sizeof(double));
	vec_b = (double *)calloc(dim, sizeof(double));

	// input mat_a and vec_x
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

	// vec_b := mat_a * vec_x
#ifdef _OPENMP
	my_matvec_mul_omp(vec_b, mat_a, dim, dim, vec_x);
#else
	my_matvec_mul(vec_b, mat_a, dim, dim, vec_x);
#endif // _OPENMP

	// print
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3lf ", mat_a[i * dim + j]);
		printf("]  %10.3lf = %10.3lf\n", vec_x[i], vec_b[i]);
	}

	// free
	free(mat_a);
	free(vec_x);
	free(vec_b);

	return EXIT_SUCCESS;
}
