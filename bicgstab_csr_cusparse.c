/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Jacobi Iterative Refinement with Sparse BLAS  */
/* Last Update: 2015-05-01 (Fri) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mycuda.h"
#include "tkaux.h"
//#include "cblas.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"

#include "mm/matrix_market_io.h" 	// Matrix Market(MTX) I/O

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

// BiCGSTAB
int bicgstab_csr_cusparse(double *dev_answer, cusparseHandle_t cusparse_handle, cusparseMatDescr_t mat_descripter, double *dev_mat_val, int mat_num_nonzero, int *dev_mat_row_csr, int *dev_mat_col_index, cublasHandle_t cublas_handle, double *dev_vec_b, double reps, double aeps, double history_norm2[], int dim, int maxtimes)
{
	long int i, j, times, return_val;
	double one = 1.0, zero = 0.0;
	double alpha, alpha_num   , alpha_den;
	double beta , beta_num    ;
	double rho  , old_rho     ;
	double omega, omega_den   ;
	double dtmp , init_resnorm;
	double *dev_vec[9], *zero_vec; /* Temporary Vectors */

/* Set initial value */
	zero_vec = (double *)calloc(dim, sizeof(double));
	set0_dvector(zero_vec, dim, 1);
	for(i = 0; i < 9; i++)
	{
		dev_vec[i] = (double *)mycuda_calloc(dim, sizeof(double));
		cublasSetVector(dim, sizeof(double), zero_vec, 1, (void *)dev_vec[i], 1);
	}

	/* dev_vec[0] ... approximation of solution */
	/* dev_vec[1] ... residual : b - a * vec[0]  == r */
	/* dev_vec[2] ... (b - a * vec[0])^T  == r^t */
	/* dev_vec[3] ... p */
	/* dev_vec[4] ... p^T */
	/* dev_vec[5] ... v */
	/* dev_vec[6] ... s */
	/* dev_vec[7] ... s^T */
	/* dev_vec[8] ... t */

	cublasDcopy(cublas_handle, dim, dev_vec_b, 1, dev_vec[1], 1); 
	cublasDcopy(cublas_handle, dim, dev_vec_b, 1, dev_vec[2], 1);

	cublasDdot(cublas_handle, dim, dev_vec[1], 1, dev_vec[1], 1, &beta_num);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		cublasDdot(cublas_handle, dim, dev_vec[2], 1, dev_vec[1], 1, &rho);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(bicgstab, %ld)\n", times);
			return_val= -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := r */
			cublasDcopy(cublas_handle, dim, dev_vec[1], 1, dev_vec[3], 1);
		}
		else
		{
			beta = (rho / old_rho) * (alpha / omega);

			/* p := r + beta * (p - omega * v) */
			//add_cmul_dvector(vec[4], vec[3], -omega, vec[5]);
			//add_cmul_dvector(vec[3], vec[1], beta, vec[4]);

			cublasDcopy(cublas_handle, dim, dev_vec[3], 1, dev_vec[4], 1);
			dtmp = -omega;
			cublasDaxpy(cublas_handle, dim, &dtmp, dev_vec[5], 1, dev_vec[4], 1);
			cublasDcopy(cublas_handle, dim, dev_vec[1], 1, dev_vec[3], 1);
			cublasDaxpy(cublas_handle, dim, &beta, dev_vec[4], 1, dev_vec[3], 1);
		}
		/* precondition */
		
		/* v := Apt */
		//mul_drsmatrix_dvec(vec[5], a, vec[3]);
		//mkl_cspblas_dcsrgemv("N", &dim, mat_a_val, mat_ia, mat_ja, vec[3], vec[5]);
		cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim, dim, mat_num_nonzero, &one, mat_descripter, dev_mat_val, dev_mat_row_csr, dev_mat_col_index, dev_vec[3], &zero, dev_vec[5]);
	//	cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, dim, dim, mat_num_nonzero, &one, mat_descripter, dev_mat_val, dev_mat_row_csr, dev_mat_col_index, dev_vec[3], &zero, dev_vec[5]);

		//alpha_den = cblas_ddot(dim, vec[2], 1, vec[5], 1);
		cublasDdot(cublas_handle, dim, dev_vec[2], 1, dev_vec[5], 1, &alpha_den);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(bicgstab, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;

		/* s = r - alpha v */
		//add_cmul_dvector(vec[6], vec[1], -alpha, vec[5]);
		cublasDcopy(cublas_handle, dim, dev_vec[1], 1, dev_vec[6], 1);
		dtmp = -alpha;
		cublasDaxpy(cublas_handle, dim, &dtmp, dev_vec[5], 1, dev_vec[6], 1);

		/* Stopping Criteria */
		cublasDnrm2(cublas_handle, dim, dev_vec[6], 1, &dtmp);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			//add_cmul_dvector(vec[0], vec[0], alpha, vec[3]);
			cublasDaxpy(cublas_handle, dim, &alpha, dev_vec[3], 1, dev_vec[0], 1);

			cublasDcopy(cublas_handle, dim, dev_vec[0], 1, dev_answer, 1);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* precondition */
		//mul_drsmatrix_dvec(vec[8], a, vec[6]);
		//mkl_cspblas_dcsrgemv("N", &dim, mat_a_val, mat_ia, mat_ja, vec[6], vec[8]);
		cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim, dim, mat_num_nonzero, &one, mat_descripter, dev_mat_val, dev_mat_row_csr, dev_mat_col_index, dev_vec[6], &zero, dev_vec[8]);
	//	cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, dim, dim, mat_num_nonzero, &one, mat_descripter, dev_mat_val, dev_mat_row_csr, dev_mat_col_index, dev_vec[6], &zero, dev_vec[8]);

		/* omega = (t, s) / (t, t) */
		//omega_den = cblas_ddot(dim, vec[8], 1, vec[8], 1);
		cublasDdot(cublas_handle, dim, dev_vec[8], 1, dev_vec[8], 1, &omega_den);
		if(omega_den == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(bicgstab, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}
		cublasDdot(cublas_handle, dim, dev_vec[8], 1, dev_vec[6], 1, &omega);
		if(omega == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(bicgstab, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		omega = omega / omega_den;

		/* x = x + alpha pt + omega st */
		//add_cmul_dvector(vec[4], vec[0], alpha, vec[3]);
		//add_cmul_dvector(vec[0], vec[4], omega, vec[6]);
		cublasDaxpy(cublas_handle, dim, &alpha, dev_vec[3], 1, dev_vec[0], 1);
		cublasDaxpy(cublas_handle, dim, &omega, dev_vec[6], 1, dev_vec[0], 1);

		/* residual */
		//add_cmul_dvector(vec[1], vec[6], -omega, vec[8]);
		cublasDcopy(cublas_handle, dim, dev_vec[6], 1, dev_vec[1], 1);
		dtmp = -omega;
		cublasDaxpy(cublas_handle, dim, &dtmp, dev_vec[8], 1, dev_vec[1], 1);

		//beta_num = cblas_ddot(dim, vec[1], 1, vec[1], 1);
		cublasDdot(cublas_handle, dim, dev_vec[1], 1, dev_vec[1], 1, &beta_num);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);

		if(history_norm2 != NULL)
			history_norm2[times] = dtmp / init_resnorm;

		if(dtmp <= aeps + reps * init_resnorm)
		{
			cublasDcopy(cublas_handle, dim, dev_vec[0], 1, dev_answer, 1);
			return_val = times; // Fix!
			break; // Fix!
		}

		old_rho = rho;
	}

	/* Not converge */
	cublasDcopy(cublas_handle, dim, dev_vec[0], 1, dev_answer, 1);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		mycuda_free(dev_vec[i]);

	free(zero_vec);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(bicgstab, %ld)\n", times);
		return_val = -5;
	}

	return return_val; // Fix!

}

int main()
{
	int i, j, itimes;
	int dim, maxtimes, row_dim, col_dim;
	double *md, *vy, *vx, *vx_true, *vx_new, *vb, *his_norm2_res;
	double *dev_md, *dev_vy, *dev_vx, *dev_vx_true, *dev_vx_new, *dev_vb, *dev_his_norm2_res;
	double alpha, beta;
	double reps, aeps;

	// A: sparse matrix
	unsigned char *mtx_fname;
	// 疎行列 COOrdinate format
	double *ma_val;
	int *ma_row_index, *ma_col_index;
	// 疎行列 CSR format
	double *ma_val_csr;
	// cooソート
	int *dev_ma_coo_perm, ma_coo_buff_size;
	void *dev_ma_coo_buff;
	// 共通
	int ma_num_nonzero, diag_flag, info;
	MM_typecode matcode;

	// cublas
	cublasHandle_t cublas_handle;

	// cuSPARSE
	cudaError_t cuda_error;
	cusparseStatus_t status;
	cusparseHandle_t handle;
	cusparseMatDescr_t mat_descripter;
	double *dev_ma_val, *dev_ma_val_sorted;
	int *dev_ma_row_index, *dev_ma_col_index;
	int *dev_ma_row_csr;

	// input mtx file
	//mtx_fname = "mm/b1_ss/b1_ss.mtx"; // (1,1) == (0, 0) element lack!
	//mtx_fname = "mm/sptest.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/diagtest.mtx"; // Diagonal matrix
	//mtx_fname = "mm/cage4/cage4.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/cavity10/cavity10.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/circuit_2/circuit_2.mtx";
	mtx_fname = "mm/t2d_q4/t2d_q4.mtx";
	//mtx_fname = "mm/circuit5M/circuit5M.mtx"; // exceed 2GB!

	// ヘッダ情報を表示(ファイルが読めるかどうか一度チェックする)
	if(mm_print_header_mtx_crd(mtx_fname, 100) != MM_SUCCESS)
		return EXIT_FAILURE;

	// read mtx file as coordinate format
	mm_read_mtx_crd(mtx_fname, &row_dim, &col_dim, &ma_num_nonzero, &ma_row_index, &ma_col_index, &ma_val, &matcode);

	// square matrix?
	if(row_dim != col_dim)
	{
		fprintf(stderr, "ERROR: ma is not square matrix(row_dim, col_dim = %d, %d)\n", row_dim, col_dim);
		return EXIT_FAILURE;
	}

	dim = row_dim;

	// 1-based index -> 0-based index
	for(i = 0; i < ma_num_nonzero; i++)
	{
		--ma_row_index[i];
		--ma_col_index[i];
	}

	// cuSPARSE初期化
	status = cusparseCreate(&handle);
	if(status != CUSPARSE_STATUS_SUCCESS)
	{
		fprintf(stderr, "ERROR: failed to initialize cuSPARSE!\n");
		return EXIT_FAILURE;
	}

	// cudaMalloc & cudaMemcpy
	dev_ma_val = (double *)mycuda_calloc(ma_num_nonzero, sizeof(double));
	dev_ma_val_sorted = (double *)mycuda_calloc(ma_num_nonzero, sizeof(double));
	dev_ma_row_index = (int *)mycuda_calloc(ma_num_nonzero, sizeof(int));
	dev_ma_col_index = (int *)mycuda_calloc(ma_num_nonzero, sizeof(int));
	dev_ma_row_csr = (int *)mycuda_calloc(row_dim + 1, sizeof(int));

	cuda_error = cudaMemcpy(dev_ma_val, ma_val, ma_num_nonzero * sizeof(double), cudaMemcpyHostToDevice);
	if(cuda_error != cudaSuccess)
		fprintf(stderr, "ERROR: cudaMemcpy error!(ma_val -> dev_ma_val)\n");
	cuda_error = cudaMemcpy(dev_ma_row_index, ma_row_index, ma_num_nonzero * sizeof(int), cudaMemcpyHostToDevice);
	if(cuda_error != cudaSuccess)
		fprintf(stderr, "ERROR: cudaMemcpy error!(ma_row_index -> dev_ma_row_index)\n");
	cuda_error = cudaMemcpy(dev_ma_col_index, ma_col_index, ma_num_nonzero * sizeof(int), cudaMemcpyHostToDevice);
	if(cuda_error != cudaSuccess)
		fprintf(stderr, "ERROR: cudaMemcpy error!(ma_col_index -> dev_ma_col_index)\n");

	// バッファサイズ設定とバッファ確保，順序ベクトル確保
	cusparseXcoosort_bufferSizeExt(handle, row_dim, col_dim, ma_num_nonzero, dev_ma_row_index, dev_ma_col_index, (size_t *)&ma_coo_buff_size);
	cudaMalloc(&dev_ma_coo_buff, sizeof(char) * ma_coo_buff_size);
	cudaMalloc((void **)&dev_ma_coo_perm, sizeof(int) * ma_num_nonzero);
	cusparseCreateIdentityPermutation(handle, ma_num_nonzero, dev_ma_coo_perm);

	// COOソート
	cusparseXcoosortByRow(handle, row_dim, col_dim, ma_num_nonzero, dev_ma_row_index, dev_ma_col_index, dev_ma_coo_perm, dev_ma_coo_buff);

	// 値もソート : dev_ma_val -> dev_ma_val_sorted
	cusparseDgthr(handle, ma_num_nonzero, dev_ma_val, dev_ma_val_sorted, dev_ma_coo_perm, CUSPARSE_INDEX_BASE_ZERO);

	// クリア
	cudaFree(dev_ma_coo_buff);
	cudaFree(dev_ma_coo_perm);


	// COO to CSR
	//status = cusparseXcoo2csr(handle, dev_ma_row_index, ma_num_nonzero, row_dim, dev_ma_row_csr, CUSPARSE_INDEX_BASE_ONE); // 変換しない場合
	status = cusparseXcoo2csr(handle, dev_ma_row_index, ma_num_nonzero, row_dim, dev_ma_row_csr, CUSPARSE_INDEX_BASE_ZERO);
	if(status != CUSPARSE_STATUS_SUCCESS)
	{
		fprintf(stderr, "ERROR: convert coo to csr index!\n");
		
		cusparseDestroy(handle);
		return EXIT_FAILURE;
	}

	// 行列descripter初期化
	status = cusparseCreateMatDescr(&mat_descripter);
	if(status != CUSPARSE_STATUS_SUCCESS)
	{
		fprintf(stderr, "ERROR: failed to initialize a matrix descripter!\n");

		cusparseDestroy(handle);
		return EXIT_FAILURE;
	}

	// 疎行列セット
	cusparseSetMatType(mat_descripter, CUSPARSE_MATRIX_TYPE_GENERAL);
//	cusparseSetMatIndexBase(mat_descripter, CUSPARSE_INDEX_BASE_ONE); // 変換しない場合
	cusparseSetMatIndexBase(mat_descripter, CUSPARSE_INDEX_BASE_ZERO);

	// Initialize on CPU
	vy = (double *)calloc(dim, sizeof(double));
	vx = (double *)calloc(dim, sizeof(double));
	vx_new = (double *)calloc(dim, sizeof(double));
	vx_true = (double *)calloc(dim, sizeof(double));
	vb = (double *)calloc(dim, sizeof(double));

	// input ma and vx_true
	for(i = 0; i < dim; i++)
		vx_true[i] = (double)(i + 1);

	// Initialize on GPU
	dev_vy = (double *)mycuda_calloc(dim, sizeof(double));
	dev_vx = (double *)mycuda_calloc(dim, sizeof(double));
	dev_vx_new = (double *)mycuda_calloc(dim, sizeof(double));
	dev_vx_true = (double *)mycuda_calloc(dim, sizeof(double));
	dev_vb = (double *)mycuda_calloc(dim, sizeof(double));

	// CPU -> GPU
	cuda_error = cudaMemcpy(dev_vx, vx_true, dim * sizeof(double), cudaMemcpyHostToDevice);
	if(cuda_error != cudaSuccess)
		fprintf(stderr, "ERROR: cudaMemcpy error!(vx_true -> dev_vx)\n");

	// 行列・ベクトル積(SpMV)
	// vb := 1.0 * ma * vx_true + 0.0 * vb
	alpha = 1.0;
	beta = 0.0;
//	status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, dim, dim, ma_num_nonzero, &alpha, mat_descripter, dev_ma_val, dev_ma_row_csr, dev_ma_col_index, dev_vx, &beta, dev_vb);
//	status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim, dim, ma_num_nonzero, &alpha, mat_descripter, dev_ma_val, dev_ma_row_csr, dev_ma_col_index, dev_vx, &beta, dev_vb);
	status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim, dim, ma_num_nonzero, &alpha, mat_descripter, dev_ma_val_sorted, dev_ma_row_csr, dev_ma_col_index, dev_vx, &beta, dev_vb);

	// GPU -> CPU
	cuda_error = cudaMemcpy(vb, dev_vb, dim * sizeof(double), cudaMemcpyDeviceToHost);

	// print vb
/*	printf("b := A * x\n");
	for(i = 0; i < dim; i++)
		printf("%d %25.17e %25.17e\n", i, vb[i], vx_true[i]);
*/
	reps = 1.0e-10;
	aeps = 0.0;

	maxtimes = dim * 10;
	his_norm2_res = (double *)calloc(maxtimes, sizeof(double));
	dev_his_norm2_res = (double *)mycuda_calloc(maxtimes, sizeof(double));

	// BiCGSTAB
	cublasCreate(&cublas_handle);
//	itimes = bicgstab_csr_cusparse(dev_vx, handle, mat_descripter, dev_ma_val, ma_num_nonzero, dev_ma_row_csr, dev_ma_col_index, cublas_handle, dev_vb, reps, aeps, his_norm2_res, dim, maxtimes);
	itimes = bicgstab_csr_cusparse(dev_vx, handle, mat_descripter, dev_ma_val_sorted, ma_num_nonzero, dev_ma_row_csr, dev_ma_col_index, cublas_handle, dev_vb, reps, aeps, his_norm2_res, dim, maxtimes);
	cublasDestroy(cublas_handle);

	if(itimes < 0)
		itimes = maxtimes;

//	cudaMemcpy(his_norm2_res, dev_his_norm2_res, maxtimes * sizeof(double), cudaMemcpyDeviceToHost);
	for(i = 0; i < itimes; i++)
		printf("%3d %10.3e\n", i, his_norm2_res[i]);

	free(his_norm2_res);

	// print
	cudaMemcpy(vx, dev_vx, dim * sizeof(double), cudaMemcpyDeviceToHost);

	printf("Iterative Times = %d\n", itimes);
	printf("Rel.Diff = %10.3e\n", reldiff_dvector(vx, vx_true, dim));

/*	for(i = 0; i < dim; i++)
	{
		printf("%3d %25.17e %25.17e\n", i, vx[i], vx_true[i]);
	}
*/
	// free
	mycuda_free(dev_vy);
	mycuda_free(dev_vx);
	mycuda_free(dev_vx_new);
	mycuda_free(dev_vx_true);
	mycuda_free(dev_vb);

	free(ma_val);
	free(ma_row_index);
	free(ma_col_index);

	free(vy);
	free(vx);
	free(vx_true);
	free(vx_new);
	free(vb);

	// finalize cuSPARSE
	cusparseDestroyMatDescr(mat_descripter);
	cusparseDestroy(handle);

	return EXIT_SUCCESS;

	

}
