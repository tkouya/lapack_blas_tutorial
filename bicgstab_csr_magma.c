/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Jacobi Iterative Refinement with Sparse BLAS  */
/* Last Update: 2015-04-16 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tkaux.h"
//#include "cblas.h"

//#include "cublas_v2.h"
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#include "magma.h"
#include "magmasparse.h"		// Sparse BLAS in MAGMA
#include "magma_lapack.h"

//#include "mm/matrix_market_io.h" 	// Matrix Market(MTX) I/O

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
	int i, j, itimes;
	int dim, maxtimes, dim1;
	double *md, *vy, *vx, *vx_true, *vx_new, *vb, *his_norm2_res;
	double reps, aeps;

	// MAGMA
	magma_dopts doptions; // Options for sparse solver
	magma_queue_t queue;

	// A: sparse matrix
	char *mtx_fname;
	// 疎行列 CSR format
	magma_d_sparse_matrix ma_csr, dev_ma_csr;
	magma_d_vector dev_vx, dev_vb;
	magma_int_t row_dim, col_dim;
//	MM_typecode matcode;

	// magma initialize
	magma_init();
//	magma_print_environment();
	
	magma_queue_create(&queue);

	// input mtx file
	//mtx_fname = "mm/b1_ss/b1_ss.mtx"; // (1,1) == (0, 0) element lack!
	//mtx_fname = "mm/sptest.mtx"; // all diagonal elements are found!
	//mtx_fname = "/home/tkouya/na/lapack/mm/sptest.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/diagtest.mtx"; // Diagonal matrix
	//mtx_fname = "mm/cage4/cage4.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/cavity10/cavity10.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/circuit_2/circuit_2.mtx";
	//mtx_fname = "mm/t2d_q4/t2d_q4.mtx";
	mtx_fname = "mm/circuit5M/circuit5M.mtx"; // exceed 2GB!

	// ヘッダ情報を表示(ファイルが読めるかどうか一度チェックする)
//	if(mm_print_header_mtx_crd(mtx_fname, 100) != MM_SUCCESS)
//		return EXIT_FAILURE;

	// read mtx file as coordinate format and convert CSR format
	magma_d_csr_mtx(&ma_csr, mtx_fname, queue);
	//magma_dm_5stencil(5, &ma_csr, queue);

	row_dim = (int)ma_csr.num_rows;
	col_dim = (int)ma_csr.num_cols;

	printf("row_dim, col_dim, num_nonzeros = %d, %d, %d\n", row_dim, col_dim, (int)ma_csr.nnz);

	// square matrix?
	if(row_dim != col_dim)
	{
		fprintf(stderr, "ERROR: ma is not square matrix(row_dim, col_dim = %d, %d)\n", row_dim, col_dim);
		return EXIT_FAILURE;
	}

	dim = row_dim;

	// Initialize
	vy = (double *)calloc(dim, sizeof(double));
	vx = (double *)calloc(dim, sizeof(double));
	vx_new = (double *)calloc(dim, sizeof(double));
	vx_true = (double *)calloc(dim, sizeof(double));
	vb = (double *)calloc(dim, sizeof(double));

	// input ma and vx_true
	for(i = 0; i < dim; i++)
		vx_true[i] = (double)(i + 1);

	magma_dvinit(&dev_vb, Magma_DEV, dim, dim, 0.0, queue);
	magma_dvinit(&dev_vx, Magma_DEV, dim, dim, 0.0, queue);
	magma_dvset(dim, 1, vx_true, &dev_vx, queue);

	// CPU -> GPU
	magma_d_mtransfer(ma_csr, &dev_ma_csr, Magma_CPU, Magma_DEV, queue);

	// 行列・ベクトル積(SpMV)
	// vb := 1.0 * ma * vx_true + 0.0 * vb
	magma_d_spmv(1.0, dev_ma_csr, dev_vx, 0.0, dev_vb, queue);

	// GPU -> CPU
	dim1 = 1;
	magma_dvget(dev_vb, &dim, &dim1, &vb, queue);

	// print vb
	printf("b := A * x\n");
	for(i = 0; i < dim; i++)
		printf("%d %25.17e\n", i, vb[i]);

	return 0;

#if 0


	reps = 1.0e-10;
	aeps = 0.0;

	maxtimes = dim * 10;
	his_norm2_res = (double *)calloc(maxtimes, sizeof(double));

	// BiCGSTAB
	itimes = bicgstab_csr(vx, ma_val_csr, ma_ia_csr, ma_ja_csr, vb, reps, aeps, his_norm2_res, dim, maxtimes);

	if(itimes < 0)
		itimes = maxtimes;

	for(i = 0; i < itimes; i++)
		printf("%3d %10.3e\n", i, his_norm2_res[i]);

	free(his_norm2_res);

	// print
	printf("Iterative Times = %d\n", itimes);
	printf("Rel.Diff = %10.3e\n", reldiff_dvector(vx, vx_true, dim));

	for(i = 0; i < dim; i++)
	{
		printf("%3d %25.17e %25.17e\n", i, vx[i], vx_true[i]);
	}

	// free
	free(ma_val);
	free(ma_row_index);
	free(ma_col_index);

	free(ma_val_csr);
	free(ma_ia_csr);
	free(ma_ja_csr);

	free(vy);
	free(vx);
	free(vx_true);
	free(vx_new);
	free(vb);
#endif // 0

	magma_queue_destroy(queue);
	magma_finalize();

	return EXIT_SUCCESS;

	

}
