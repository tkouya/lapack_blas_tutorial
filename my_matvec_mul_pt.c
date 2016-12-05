/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Multiplication of matrix and vector           */
/*                                  with Pthread */
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

	// メインループ
	for(i = 0; i < row_dim; i++)
	{
		vec_b[i] = 0.0;
		row_index = row_dim * i;

		for(j = 0; j < col_dim; j++)
			vec_b[i] += mat_a[row_index + j] * vec_x[j];
	}
}

/* Struct for Thread */
typedef struct {
	double *vec_b;
	double *mat_a;
	int row_dim;
	int col_dim;
	double *vec_x;

	int i; // i th row
	int num_threads, thread_index;
} packed_my_matvec_mul_t; 

/* parallelized computation */
void thread_my_matvec_mul(void *arg_org)
{
	packed_my_matvec_mul_t *arg;
	int i, j, row_index, row_dim, col_dim;
	double *vec_b, *mat_a, *vec_x;

	arg = (packed_my_matvec_mul_t *)arg_org;

	vec_b = arg->vec_b;
	mat_a = arg->mat_a;
	row_dim = arg->row_dim;
	col_dim = arg->col_dim;
	vec_x = arg->vec_x;
	i = arg->i;

#ifdef DEBUG
	printf("Start thread No.%d/%d\n", arg->thread_index, arg->num_threads);
#endif

	vec_b[i] = 0.0;
	row_index = row_dim * i;

	for(j = 0; j < col_dim; j++)
		vec_b[i] += mat_a[row_index + j] * vec_x[j];

#ifdef DEBUG
	printf("End thread No.%d/%d\n", arg->thread_index, arg->num_threads);
#endif
}

// Parallelized LU
int _pthread_my_matvec_mul(double *vec_b, double *mat_a, int row_dim, int col_dim, double *vec_x, long int num_threads)
{
	int thread_i, i;
	pthread_t thread[128];
	packed_my_matvec_mul_t *th_arg[128];

	/* not necessary to be parallelized */
	if(num_threads <= 1)
	{
		my_matvec_mul(vec_b, mat_a, row_dim, col_dim, vec_x);
		return 0;
	}

	// メインループ
	for(i = 0; i < row_dim; i += num_threads)
	{
		// Initialize argument for pthread
		for(thread_i = 0; thread_i < num_threads; thread_i++)
		{
			th_arg[thread_i] = (packed_my_matvec_mul_t *)malloc(sizeof(packed_my_matvec_mul_t));
			th_arg[thread_i]->vec_b = vec_b;
			th_arg[thread_i]->mat_a = mat_a;
			th_arg[thread_i]->row_dim = row_dim;
			th_arg[thread_i]->col_dim = col_dim;
			th_arg[thread_i]->vec_x = vec_x;
			th_arg[thread_i]->i = i + thread_i;
			th_arg[thread_i]->num_threads = num_threads;
			th_arg[thread_i]->thread_index = thread_i;
		}

		// Creat threads
		for(thread_i = 0; thread_i < num_threads; thread_i++)
			pthread_create(&thread[thread_i], NULL, thread_my_matvec_mul, (void *)th_arg[thread_i]);

		// Join threads
		for(thread_i = 0; thread_i < num_threads; thread_i++)
			pthread_join(thread[thread_i], NULL);
	}

	return 0;
}


int main()
{
	int i, j, dim;

	double *mat_a, *vec_b, *vec_x;

	// 次元数入力
	printf("Dim = "); scanf("%d", &dim);

	// 最大スレッド数入力
	int num_threads;

	printf("Max.Num.threads = "); scanf("%d", &num_threads);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// 変数初期化
	mat_a = (double *)calloc(dim * dim, sizeof(double));
	vec_x = (double *)calloc(dim, sizeof(double));
	vec_b = (double *)calloc(dim, sizeof(double));

	// mat_aとvec_xに値入力
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
	_pthread_my_matvec_mul(vec_b, mat_a, dim, dim, vec_x, num_threads);

//	my_matvec_mul(vec_b, mat_a, dim, dim, vec_x);

	// 出力
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3lf ", mat_a[i * dim + j]);
		printf("]  %10.3lf = %10.3lf\n", vec_x[i], vec_b[i]);
	}

	// 変数消去
	free(mat_a);
	free(vec_x);
	free(vec_b);

	return EXIT_SUCCESS;
}
