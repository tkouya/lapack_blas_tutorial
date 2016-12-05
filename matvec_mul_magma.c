/*************************************************/
/* matvec_mul_magma.c : 実行列×実ベクトル(magmablas)
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// BLAS on CPU
#include "cblas.h"

// mycuda_calloc, mycuda_free関数
#include "mycuda.h" // CUDA
#include "cublas_v2.h" // cuBLAS

// MAGMA
#include "magma.h"

int main()
{
	int i, j, dim;
	int inc_vec_x, inc_vec_b;

	double *mat_a, *vec_b, *vec_x, *vec_b_gpu; // CPU
	double *dev_mat_a, *dev_vec_b, *dev_vec_x; // GPU
	double alpha, beta;

	// cuBLAS用
	cublasStatus_t status;
	cublasHandle_t handle;

	// 次元数入力
	printf("Dim = "); scanf("%d", &dim);

	if(dim <= 0)
	{
		printf("Illegal dimension! (dim = %d)\n", dim);
		return EXIT_FAILURE;
	}

	// 変数初期化 on CPU
	mat_a = (double *)calloc(dim * dim, sizeof(double));
	vec_x = (double *)calloc(dim, sizeof(double));
	vec_b = (double *)calloc(dim, sizeof(double));

	// mat_aとvec_xに値入力
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

	// 出力
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3lf ", mat_a[i + j * dim]);
		printf("]  %10.3lf = %10.3lf\n", vec_x[i], vec_b[i]);
	}

	// GPU

	// MAGMA初期化
	magma_init();

	// 変数初期化 on GPU (MAGMA)
	dev_mat_a = (double *)mycuda_calloc(dim * dim, sizeof(double));
	dev_vec_x = (double *)mycuda_calloc(dim, sizeof(double));
	dev_vec_b = (double *)mycuda_calloc(dim, sizeof(double));

	// dev_vec_x on GPU -> vec_x_gpu on CPU
	vec_b_gpu = (double *)calloc(dim, sizeof(double));

	// CPU(Host) -> GPU(device)

	// cuBLAS初期化
	status = cublasCreate(&handle);

	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("Fail to initialize cuBLAS!\n");

		mycuda_free(dev_mat_a);
		mycuda_free(dev_vec_b);
		mycuda_free(dev_vec_x);
		cublasDestroy(handle);

		return 0;
	}

	// mat_a -> dev_mat_a
	status = cublasSetMatrix(dim, dim, sizeof(double), mat_a, dim, dev_mat_a, dim);
	if(status != CUBLAS_STATUS_SUCCESS)
		printf("mat_a -> dev_mat_a: cublasSetMatrix failed.\n");

	// size(vec_x) == size(vec_b)
	inc_vec_x = inc_vec_b = 1;

	// vec_x -> dev_vec_x
	status = cublasSetVector(dim, sizeof(double), vec_x, inc_vec_x, dev_vec_x, inc_vec_x);
	if(status != CUBLAS_STATUS_SUCCESS)
		printf("vec_x -> dev_vec_x: cublasSetVector failed.\n");

	// vec_b := 1.0 * mat_a * vec_x + 0.0 * vec_b
	alpha = 1.0;
	beta = 0.0;
	magmablas_dgemv(MagmaNoTrans, dim, dim, alpha, dev_mat_a, dim, dev_vec_x, inc_vec_x, beta, dev_vec_b, inc_vec_b);

	if(status != CUBLAS_STATUS_SUCCESS)
		printf("cublasDgemv failed.\n");

	// 同期
	cudaDeviceSynchronize(handle);

	// dev_vec_b -> vec_b_gpu
	status = cublasGetVector(dim, sizeof(double), dev_vec_b, inc_vec_b, vec_b_gpu, inc_vec_b);
	if(status != CUBLAS_STATUS_SUCCESS)
		printf("dev_vec_b -> vec_b_gpu: cublasGetVector failed.\n");

	// 出力
	for(i = 0; i < dim; i++)
	{
		printf("[");
		for(j = 0; j < dim; j++)
			printf("%10.3lf ", mat_a[i + j * dim]);
		printf("]  %10.3lf = %10.3lf\n", vec_x[i], vec_b_gpu[i]);
	}

	// 相対差異:: ||vec_b_gpu - vec_b||_2 / ||vec_b||_2
	cblas_daxpy(dim, -1.0, vec_b, inc_vec_b, vec_b_gpu, inc_vec_b);
	printf("||vec_b_gpu - vec_b||_2 / ||vec_b||_2 = %15.7e\n", cblas_dnrm2(dim, vec_b_gpu, 1) / cblas_dnrm2(dim, vec_b, 1));

	// 変数消去
	free(mat_a);
	free(vec_x);
	free(vec_b);
	free(vec_b_gpu);

	mycuda_free(dev_mat_a);
	mycuda_free(dev_vec_b);
	mycuda_free(dev_vec_x);

	// MAGMA終了
	magma_finalize();

	return EXIT_SUCCESS;
}
