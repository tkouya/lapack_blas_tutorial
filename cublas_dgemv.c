/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample Program of BLAS Level3                 */
/*                        with Intel Math Kernel */
/* Last Update: 2011-06-10 (Fri) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "get_secv.h"
#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#else
	#include "cblas.h"
#endif

#ifdef USE_CUDA
	#include "cuda.h"
	#include "cublas_v2.h"
#ifdef USE_MAGMA
	#include "magma.h"
#endif
#endif

// normal 3 loops version
void handmade_dmatmul(double *c, double *a, double *b, int dim)
{
	int i, j, k;

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			*(c + i * dim + j) = 0.0;
			for(k = 0; k < dim; k++)
				*(c + i * dim + j) += *(a + i * dim + k) * *(b + k * dim + j);
		}
	}
}

// flag = 0 ... use DGEMM in BLAS
// flag = 1 ... use handmade matmul 
double get_gflops_dgemm(int flag, int dim)
{
#ifdef USE_ATLAS
	int i, j;
#else
	MKL_INT i, j; // dimension of vectors
#endif
	double *ma, *mb, *mc;
	double alpha, beta;
	double running_time;
	double gflops;
	int itimes, max_itimes = 10000;
	double tmp;

	// Initialize
	ma = (double *)calloc(dim * dim, sizeof(double));
	mb = (double *)calloc(dim * dim, sizeof(double));
	mc = (double *)calloc(dim * dim, sizeof(double));

	// column major
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			*(ma + i + j * dim) = sqrt(2.0) * (i + j + 1);
			*(mb + i + j * dim) = sqrt(2.0) * (dim * 2 - (i + j + 1));
		}
	}

	if(flag == 0)
	{
		// mc := 1.0 * ma * mb + 0.0 * mc
		alpha = 1.0;
		beta = 0.0;
		tmp = 0.0;
		for(itimes = 0; itimes < max_itimes; itimes++)
		{
			running_time = get_real_secv();
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, alpha, ma, dim, mb, dim, beta, mc, dim);
			tmp += get_real_secv() - running_time; // end
			if(tmp > 0.0)
			{
				running_time = tmp / (itimes + 1);
				break;
			}	
		}
	}
	else if(flag == 1)
	{
		tmp = 0.0;
		for(itimes = 0; itimes < max_itimes; itimes++)
		{
			running_time = get_real_secv();
			handmade_dmatmul(mc, ma, mb, dim);
			tmp += get_real_secv() - running_time; // end
			if(tmp > 0.0)
			{
				running_time = tmp / (itimes + 1);
				break;
			}
		}
	
	}

	// print
/*	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		for(j = 0; j < dim; j++)
			printf("%10f ", *(mc + i * dim + j));
		printf("\n");
	}
*/
//	printf("dim = %d, Running Time(sec) = %f\n", dim, running_time);
	gflops = 2.0 * (double)dim * (double)dim * (double)dim / running_time / 1024.0 / 1024.0 / 1024.0;
//	printf("%f Gflops\n", gflops);

	free(ma);
	free(mb);
	free(mc);

	return gflops;
}

// GPGPU
#ifdef USE_CUDA

// GPU上に行列格納領域を確保
void *mycuda_calloc(int num_elements, size_t size_element)
{
	cudaError_t cuda_error;
	void *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, num_elements * size_element);

	if(cuda_error != cudaSuccess)
	{
		printf("device memory allocation failed!(num_elements = %d, size = %d)\n", num_elements, size_element);
		return NULL;
	}

	return ret;
}

// GPU上のメモリ領域を解放
void mycuda_free(void *mem)
{
	cudaFree(mem);
}

// flag = 0 ... use DGEMM in cuBLAS
// flag = 1 ... use magma_dgemm in MAGMA
double get_gflops_cuda_dgemm(int flag, int dim, double *total_running_time)
{
	int i, j; // dimension of vectors
	double *ma, *mb, *mc, *mc_host; // on CPU
	double *dev_ma, *dev_mb, *dev_mc; // on GPU
	double alpha, beta;
	double running_time, tmp_time, tmp_total_time;
	double gflops;
	int itimes, max_itimes = 10000;
	cublasStatus_t status;
	cublasHandle_t handle;

	// Initialize on CPU
	ma = (double *)calloc(dim * dim, sizeof(double));
	mb = (double *)calloc(dim * dim, sizeof(double));
	mc = (double *)calloc(dim * dim, sizeof(double));
	mc_host = (double *)calloc(dim * dim, sizeof(double));

	// Initialize on GPU
	dev_ma = (double *)mycuda_calloc(dim * dim, sizeof(double));
	dev_mb = (double *)mycuda_calloc(dim * dim, sizeof(double));
	dev_mc = (double *)mycuda_calloc(dim * dim, sizeof(double));

	// input ma and mb
	for(j = 0; j < dim; j++)
	{
		for(i = 0; i < dim; i++)
		{
			// column major
			ma[i + j * dim] = sqrt(2.0) * (i + j + 1);
			mb[i + j * dim] = sqrt(2.0) * (dim * 2 - (i + j + 1));
			mc[i + j * dim] = 0.0;
			mc_host[i + j * dim] = 0.0;
		}
	}

	// set matrix
	status = cublasCreate(&handle);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cuBLASの初期化に失敗しました。\n");

		mycuda_free(dev_ma);
		mycuda_free(dev_mb);
		mycuda_free(dev_mc);
		cublasDestroy(handle);

		return 0;
	}

/*	status = cublasSetMatrix(dim, dim, sizeof(double), ma, dim, dev_ma, dim);
	if(status != CUBLAS_STATUS_SUCCESS)
		printf("ma -> dev_ma: cublasSetMatrix失敗しました。\n");
	status = cublasSetMatrix(dim, dim, sizeof(double), mb, dim, dev_mb, dim);
	if(status != CUBLAS_STATUS_SUCCESS)
		printf("mb -> dev_mb: cublasSetMatrix失敗しました。\n");
*/

	// cublasDgemm
	if(flag == 0)
	{
		// mc := 1.0 * ma * mb + 0.0 * mc
		alpha = 1.0; // on CPU
		beta = 0.0; // on CPU
		tmp_time = 0.0;
		tmp_total_time = 0.0;
		for(itimes = 0; itimes < max_itimes; itimes++)
		{
			tmp_total_time = get_real_secv();

			status = cublasSetMatrix(dim, dim, sizeof(double), ma, dim, dev_ma, dim);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("ma -> dev_ma: cublasSetMatrix失敗しました。\n");
			status = cublasSetMatrix(dim, dim, sizeof(double), mb, dim, dev_mb, dim);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("mb -> dev_mb: cublasSetMatrix失敗しました。\n");

			tmp_time = get_real_secv();
			status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, dev_ma, dim, dev_mb, dim, &beta, dev_mc, dim);
			//magma_dgemm(MagmaNoTrans, MagmaNoTrans, dim, dim, dim, alpha, dev_ma, dim, dev_mb, dim, beta, dev_mc, dim);
			cudaDeviceSynchronize(); // <--CPUタイマーを使う際には必須！
			running_time += get_real_secv() - tmp_time; // end
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("cublasDgemm失敗しました。\n");

			//printf("%f \n", get_real_secv() - running_time);

			status = cublasGetMatrix(dim, dim, sizeof(double), dev_mc, dim, mc, dim);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("dev_mc -> mc: cublasGetMatrix失敗しました。\n");

			*total_running_time += get_real_secv() - tmp_total_time;

			if(*total_running_time > 1.0)
				break;
		}
	}
#ifdef USE_MAGMA
	// magmablas_dgemm
	else if(flag == 1)
	{
		// mc := 1.0 * ma * mb + 0.0 * mc
		alpha = 1.0; // on CPU
		beta = 0.0; // on CPU
		tmp_time = 0.0;
		tmp_total_time = 0.0;
		for(itimes = 0; itimes < max_itimes; itimes++)
		{
			tmp_total_time = get_real_secv();

			status = cublasSetMatrix(dim, dim, sizeof(double), ma, dim, dev_ma, dim);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("ma -> dev_ma: cublasSetMatrix失敗しました。\n");
			status = cublasSetMatrix(dim, dim, sizeof(double), mb, dim, dev_mb, dim);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("mb -> dev_mb: cublasSetMatrix失敗しました。\n");

			tmp_time = get_real_secv();
			magma_dgemm(MagmaNoTrans, MagmaNoTrans, dim, dim, dim, alpha, dev_ma, dim, dev_mb, dim, beta, dev_mc, dim);
			cudaDeviceSynchronize(); // <--CPUタイマーを使う際には必須！
			running_time += get_real_secv() - tmp_time; // end
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("cublasDgemm失敗しました。\n");

			//printf("%f \n", get_real_secv() - running_time);

			status = cublasGetMatrix(dim, dim, sizeof(double), dev_mc, dim, mc, dim);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("dev_mc -> mc: cublasGetMatrix失敗しました。\n");

			*total_running_time += get_real_secv() - tmp_total_time;

			if(*total_running_time > 1.0)
				break;
		}
	}
#endif // USE_MAGMA

	running_time /= (itimes + 1);
	*total_running_time /= (itimes + 1);
	//printf("itimes = %d, total_time, running_time = %f, %f \n", itimes, *total_running_time, running_time);

	// on CPU
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, alpha, ma, dim, mb, dim, beta, mc_host, dim);
	// ||mc_host - mc||_F
	cblas_daxpy(dim * dim, -1.0, mc, 1, mc_host, 1);
	//printf("||mc - mc_host||_F = %25.17e\n", cblas_dnrm2(dim * dim, mc_host, 1) / cblas_dnrm2(dim * dim, mc, 1));

	// print
/*	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		for(j = 0; j < dim; j++)
			printf("%10f ", *(mc + i * dim + j));
		printf("\n");
	}
*/
//	printf("dim = %d, Running Time(sec) = %lf (%lf ms)\n", dim, running_time, running_time * 1000);
	gflops = 2.0 * (double)dim * (double)dim * (double)dim / running_time / 1024.0 / 1024.0 / 1024.0;
	//printf("%f Gflops\n", gflops);

	// free on CPU
	free(ma);
	free(mb);
	free(mc);

	// free on GPU
	mycuda_free(dev_ma);
	mycuda_free(dev_mb);
	mycuda_free(dev_mc);

	cublasDestroy(handle);

	return gflops;
}
#endif

int main()
{
	int start_dim, end_dim, step_dim, dim;
	char str_mkl_version[1024];
	int max_num_threads;
	double total_time[2] = {0.0, 0.0};

	// print MKL version
#ifndef USE_IMKL
	MKL_Get_Version_String(str_mkl_version, 1024);
	printf("%s\n", str_mkl_version);

	max_num_threads = mkl_get_max_threads();
	printf("Max Number of Threads: %d\n", max_num_threads);
	mkl_set_num_threads(max_num_threads);
#endif

	// input dimension
	printf("Start Dim = "); scanf("%d", &start_dim);
	printf("End   Dim = "); scanf("%d", &end_dim);
	printf("Step  Dim = "); scanf("%d", &step_dim);

	printf("  DIM  DGEMM(GFlops)");
#ifdef USE_CUDA
	printf(" cuBLAS(GFlops)");
#endif
#ifdef USE_MAGMA
	printf(" magmablas(GFlops)");
	printf(" Total time(sec)");
#endif
	printf("\n");

	for(dim = start_dim; dim <= end_dim; dim += step_dim)
	{
		printf("%5d %10.3f", dim, get_gflops_dgemm(0, dim));
#ifdef USE_CUDA
		printf(" %10.3f", get_gflops_cuda_dgemm(0, dim, &total_time[0]));
#endif
#ifdef USE_MAGMA
		printf(" %10.3f", get_gflops_cuda_dgemm(1, dim, &total_time[1]));
#endif
		if(total_time[0] > 0.0)
			printf(" %10.3f", total_time[0]);
		if(total_time[1] > 0.0)
			printf(" %10.3f", total_time[1]);
		printf("\n");
	}

	return EXIT_SUCCESS;
}
