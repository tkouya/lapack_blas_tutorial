/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample Program of BLAS Level 2                */
/*                       with Intel Math Kernel, */
/*                              cuBLAS and MAGMA */
/* Last Update: 2015-03-27 (Fri) T.Kouya         */
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
	// mycuda_calloc, mycuda_free関数
	#include "mycuda.h"
	#include "cublas_v2.h"
#ifdef USE_MAGMA
	#include "magma.h"
#endif
#endif

// column major only
// my_matvec_mul: vec_b := mat_a * vec_x
void my_matvec_mul_col(double *vec_b, double *mat_a, int row_dim, int col_dim, double *vec_x)
{
	int i, j;

	// メインループ
	for(i = 0; i < row_dim; i++)
	{
		vec_b[i] = 0.0;
		for(j = 0; j < col_dim; j++)
			vec_b[i] += mat_a[i + j * row_dim] * vec_x[j];
	}
}

// flag = 0 ... use DGEMV in BLAS
// flag = 1 ... use handmade mavec_mul
double get_gflops_dgemv(int flag, int dim, double *total_running_time, int *iterative_times)
{
#ifdef USE_ATLAS
	int i, j;
#else
	MKL_INT i, j; // dimension of vectors
#endif
	double *ma, *vb, *vc;
	double alpha, beta;
	double running_time;
	double gflops;
	int inc_vb, inc_vc;
	int itimes, max_itimes = 200000;
	double tmp_time;

	// Initialize
	ma = (double *)calloc(dim * dim, sizeof(double));
	vb = (double *)calloc(dim, sizeof(double));
	vc = (double *)calloc(dim, sizeof(double));

	// row major
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			ma[i + j * dim] = sqrtf(2.0) * (i + j + 1);

		vb[i] = sqrtf(2.0) * (dim * 2 - i);
	}

	if(flag == 0)
	{
		// vc := 1.0 * ma * vb + 0.0 * vc
		alpha = 1.0;
		beta = 0.0;
		tmp_time = 0.0;
		inc_vb = 1;
		inc_vc = 1;
		for(itimes = 0; itimes < max_itimes; itimes++)
		{
			running_time = get_real_secv();
			cblas_dgemv(CblasColMajor, CblasNoTrans, dim, dim, alpha, ma, dim, vb, inc_vb, beta, vc, inc_vc);
			tmp_time += get_real_secv() - running_time; // end
			if((tmp_time > 1.0) && (itimes > 5))
			{
				running_time = tmp_time / (itimes + 1);
				break;
			}	
		}
		*total_running_time = tmp_time;
		*iterative_times = itimes;
	}
	else if(flag == 1)
	{
		tmp_time = 0.0;
		for(itimes = 0; itimes < max_itimes; itimes++)
		{
			running_time = get_real_secv();
			my_matvec_mul_col(vc, ma, dim, dim, vb);
			tmp_time += get_real_secv() - running_time; // end
			if((tmp_time > 1.0) && (itimes > 5))
			{
				running_time = tmp_time / (itimes + 1);
				break;
			}
		}
		*total_running_time = tmp_time;
		*iterative_times = itimes;
	}

	// print
/*	for(i = 0; i < dim; i++)
	{
		printf("%3d: ", i);
		printf("%10f ", vc[i]);
		printf("\n");
	}
*/
//	printf("dim = %d, Running Time(sec) = %f\n", dim, running_time);
	gflops = (2.0 * (double)dim * (double)dim - (double)dim) / running_time / 1024.0 / 1024.0 / 1024.0;
//	printf("%f Gflops\n", gflops);

	free(ma);
	free(vb);
	free(vc);

	return gflops;
}

// GPGPU
#ifdef USE_CUDA

// flag = 0 ... use DGEMV in cuBLAS
// flag = 1 ... use magma_dgemv in MAGMA
double get_gflops_cuda_dgemv(int flag, int dim, double *total_running_time, int *iterative_times)
{
	int i, j; // dimension of vectors
	double *ma, *vb, *vc, *vc_host; // on CPU
	double *dev_ma, *dev_vb, *dev_vc; // on GPU
	double alpha, beta;
	double running_time, tmp_time, tmp_total_time;
	double gflops;
	int inc_vb, inc_vc;
	int itimes, max_itimes = 10000;
	cublasStatus_t status;
	cublasHandle_t handle;

	// Initialize on CPU
	ma = (double *)calloc(dim * dim, sizeof(double));
	vb = (double *)calloc(dim, sizeof(double));
	vc = (double *)calloc(dim, sizeof(double));
	vc_host = (double *)calloc(dim, sizeof(double));

	// Initialize on GPU
	dev_ma = (double *)mycuda_calloc(dim * dim, sizeof(double));
	dev_vb = (double *)mycuda_calloc(dim, sizeof(double));
	dev_vc = (double *)mycuda_calloc(dim, sizeof(double));

	// input ma and mb
	for(j = 0; j < dim; j++)
	{
		for(i = 0; i < dim; i++)
		{
			// column major
			ma[i + j * dim] = sqrt(2.0) * (i + j + 1);
		}
		vb[i] = sqrt(2.0) * (dim * 2 - i);
		vc[i] = 0.0;
		vc_host[i] = 0.0;
	}

	// set matrix
	status = cublasCreate(&handle);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("cuBLASの初期化に失敗しました。\n");

		mycuda_free(dev_ma);
		mycuda_free(dev_vb);
		mycuda_free(dev_vc);
		cublasDestroy(handle);

		return 0;
	}

	inc_vb = 1;
	inc_vc = 1;

	// cublasDgemm
	if(flag == 0)
	{
		// vc := 1.0 * ma * vb + 0.0 * vc
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
			status = cublasSetVector(dim, sizeof(double), (void *)vb, inc_vb, (void *)dev_vb, inc_vb);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("vb -> dev_vb: cublasSetVector失敗しました。\n");

			tmp_time = get_real_secv();
			status = cublasDgemv(handle, CUBLAS_OP_N, dim, dim, &alpha, dev_ma, dim, dev_vb, inc_vb, &beta, dev_vc, inc_vc);
			//magma_dgemv(MagmaNoTrans, dim, dim, alpha, dev_ma, dim, dev_vb, inv_vb, beta, dev_vc, inc_vc);
			cudaDeviceSynchronize(); // <--CPUタイマーを使う際には必須！
			running_time += get_real_secv() - tmp_time; // end
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("cublasDgemv失敗しました。\n");

			//printf("%f \n", get_real_secv() - running_time);

			status = cublasGetVector(dim, sizeof(double), dev_vc, inc_vc, vc, inc_vc);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("dev_vc -> vc: cublasGetMatrix失敗しました。\n");

			*total_running_time += get_real_secv() - tmp_total_time;

			if((*total_running_time > 1.0) && (itimes > 5))
				break;
		}

		*iterative_times = itimes;
	}
#ifdef USE_MAGMA
	// magmablas_dgemm
	else if(flag == 1)
	{
		// initialize
		magma_init();
		
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
			status = cublasSetVector(dim, sizeof(double), vb, inc_vb, dev_vb, inc_vb);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("vb -> dev_vb: cublasSetVector失敗しました。\n");

			tmp_time = get_real_secv();
			magma_dgemv(MagmaNoTrans, dim, dim, alpha, dev_ma, dim, dev_vb, inc_vb, beta, dev_vc, inc_vc);
			cudaDeviceSynchronize(); // <--CPUタイマーを使う際には必須！
			running_time += get_real_secv() - tmp_time; // end

			//printf("%f \n", get_real_secv() - running_time);

			status = cublasGetVector(dim, sizeof(double), dev_vc, inc_vc, vc, inc_vc);
			if(status != CUBLAS_STATUS_SUCCESS)
				printf("dev_vc -> vc: cublasGetVector失敗しました。\n");

			*total_running_time += get_real_secv() - tmp_total_time;

			if((*total_running_time > 1.0) && (itimes > 5))
				break;
		}

		// finalize
		magma_finalize();

		*iterative_times = itimes;
	}
#endif // USE_MAGMA

	running_time /= (itimes + 1);
	*total_running_time /= (itimes + 1);
	//printf("itimes = %d, total_time, running_time = %f, %f \n", itimes, *total_running_time, running_time);

	// on CPU
	cblas_dgemv(CblasColMajor, CblasNoTrans, dim, dim, alpha, ma, dim, vb, inc_vb, beta, vc_host, inc_vc);
	// ||vc_host - vc||_F
	cblas_daxpy(dim, -1.0, vc, 1, vc_host, 1);
	//printf("||vc - vc_host||_F = %25.17e\n", cblas_snrm2(dim, vc_host, 1) / cblas_snrm2(dim, vc, 1));

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
	gflops = (2.0 * (double)dim * (double)dim - (double)dim) / running_time / 1024.0 / 1024.0 / 1024.0;
	//printf("%f Gflops\n", gflops);

	// free on CPU
	free(ma);
	free(vb);
	free(vc);

	// free on GPU
	mycuda_free(dev_ma);
	mycuda_free(dev_vb);
	mycuda_free(dev_vc);

	cublasDestroy(handle);

	return gflops;
}
#endif

#ifdef USE_ATLAS
	#define ALT_NTHREADS 4
#endif

int main()
{
	int start_dim, end_dim, step_dim, dim;
	char str_mkl_version[1024];
	int max_num_threads;
	double total_time[3] = {0.0, 0.0, 0.0};
	int iterative_times[3] = {0, 0, 0};

	// print MKL version
#ifdef USE_IMKL
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

	printf("  DIM  DGEMV(GFlops)");
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
		printf("%5d %10.3lg", dim, get_gflops_dgemv(0, dim, &total_time[0], &iterative_times[0]));
#ifdef USE_CUDA
		printf(" %10.3lg", get_gflops_cuda_dgemv(0, dim, &total_time[1], &iterative_times[1]));
#endif
#ifdef USE_MAGMA
		printf(" %10.3lg", get_gflops_cuda_dgemv(1, dim, &total_time[2], &iterative_times[2]));
#endif
		if(total_time[0] > 0.0)
			printf(" %10.3lg(%6d)", total_time[0], iterative_times[0]);
		if(total_time[1] > 0.0)
			printf(" %10.3lg(%6d)", total_time[1], iterative_times[1]);
		if(total_time[2] > 0.0)
			printf(" %10.3lg(%6d)", total_time[2], iterative_times[2]);
		printf("\n");
	}

	return EXIT_SUCCESS;
}
