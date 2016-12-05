/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Solver for Linear equation                    */
/*                        with original routines */
/* Last Update: 2016-11-30 (Wed) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// rowwise only
// my_matvec_mul: vec_b := mat_a * vec_x
void my_matvec_mul(double *vec_b, double *mat_a, int row_dim, int col_dim, double *vec_x)
{
	int i, j, row_index;

	// main loop
	for(i = 0; i < row_dim; i++)
	{
		vec_b[i] = 0.0;
		row_index = row_dim * i;

		for(j = 0; j < col_dim; j++)
			vec_b[i] += mat_a[row_index + j] * vec_x[j];
	}
}

// rowwise only
// my_linear_eq_solve: solve mat_a * x = vec_b in x -> vec_b := x
int my_linear_eq_solve(double *mat_a, int dim, int *pivot, double *vec_b)
{
	int i, j, k, row_index_j, row_index_i, max_j, tmp_index;
	double absmax_aji, abs_aji, pivot_aii, vec_x;

	// initialize pivot vector
	for(i = 0; i < dim; i++)
		pivot[i] = i;

	// forward 
	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		absmax_aji = fabs(mat_a[pivot[i] * dim + i]);
		max_j = i;
		for(j = i + 1; j < dim; j++)
		{
			abs_aji = mat_a[pivot[j] * dim + i];
			if(absmax_aji < abs_aji)
			{
				max_j = j;
				absmax_aji = abs_aji;
			}
		}
		if(max_j != i)
		{
			tmp_index = pivot[max_j];
			pivot[max_j] = pivot[i];
			pivot[i] = tmp_index;
		}

		// select pivoted column
		row_index_i = pivot[i] * dim;
		pivot_aii = mat_a[row_index_i + i];

		// error
		if(fabs(pivot_aii) <= 0.0)
			return -1;

		for(j = i + 1; j < dim; j++)
		{
			row_index_j = pivot[j] * dim;
			mat_a[row_index_j + i] /= pivot_aii;

			for(k = i + 1; k < dim; k++)
				mat_a[row_index_j + k] -= mat_a[row_index_j + i] * mat_a[row_index_i + k];
		}
	}
	
	// forward substitution
	for(j = 0; j < dim; j++)
	{
		vec_x = vec_b[pivot[j]];
		for(i = j + 1; i < dim; i++)
			vec_b[pivot[i]] -= mat_a[pivot[i] * dim + j] * vec_x;
	}

	// backward substitution
	for(i = dim - 1; i >= 0; i--)
	{
		vec_x = vec_b[pivot[i]];
		row_index_i = pivot[i] * dim;
		for(j = i + 1; j < dim; j++)
			vec_x -= mat_a[row_index_i + j] * vec_b[pivot[j]];

		vec_b[pivot[i]] = vec_x / mat_a[row_index_i + i];
	}

	// reordering
	for(i = 0; i < dim; i++)
	{
		if(pivot[i] != i)
		{
			for(j = i + 1; j < dim; j++)
			{
				if(pivot[j] == i)
				{
					vec_x = vec_b[pivot[i]];
					vec_b[pivot[i]] = vec_b[i];
					vec_b[i] = vec_x;
					pivot[j] = pivot[i];
					pivot[i] = i;
				}
			}
		}
	}

	return 0;
}

int main()
{
	int i, j, dim;
	int *pivot, info;

	double *mat_a, *vec_b, *vec_x;
	double alpha, beta;
	double running_time;

	// input dimension of linear equation to be solved
	printf("Dim = "); scanf("%d", &dim);

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
			mat_a[i * dim + j] = 1.0 / (double)(i + j + 1);
			if((i + j + 1) % 2 != 0)
				mat_a[i * dim + j] = -mat_a[i * dim + j];
		}
		mat_a[i * dim + i] += 2.0;
		vec_x[i] = 1.0 / (double)(i + 1);
	}

	// vec_b := mat_a * vec_x
	my_matvec_mul(vec_b, mat_a, dim, dim, vec_x);

	// print
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3f ", mat_a[i * dim + j]);
		printf("]  %10.3f = %10.3f\n", vec_x[i], vec_b[i]);
	}

	// initialize pivot vector
	pivot = (int *)calloc(sizeof(int), dim);

	// solve A * x = b -> b := x
	info = my_linear_eq_solve(mat_a, dim, pivot, vec_b);

	printf("info = %d\n", info);

	// print
	printf("calculated x = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d -> %3d: ", i, pivot[i]);
		printf("%25.17e ", vec_b[pivot[i]]);
		printf("\n");
	}

	// diff
	printf("x - calculated x = \n");
	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		printf("%10.2e ", fabs((vec_x[i] - vec_b[i]) / vec_x[i]));
		printf("\n");
	}
	
	// free
	free(mat_a);
	free(vec_x);
	free(vec_b);
	free(pivot);

	return EXIT_SUCCESS;
}
