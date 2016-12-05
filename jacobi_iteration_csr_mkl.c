/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Jacobi Iterative Refinement with Sparse BLAS  */
/* Last Update: 2015-04-22 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tkaux.h"

#include "mkl.h"		// Sparse BLAS in Intel Math Kernel
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

// Jacobi Iteration Method using COOrdinate format
int jacobi_iteration_csr(double *answer, double *mat_a_val, int *mat_ia, int *mat_ja, double *diagonal, double *vec_b, double reps, double aeps, double history_norm2[], int dim, int maxtimes)
{
	int i, itimes;
	double *vx, *vx_new, *vy, *vtmp;
	double init_resnorm2;

	// Initialize
	vx = (double *)calloc(dim, sizeof(double));
	vx_new = (double *)calloc(dim, sizeof(double));
	vy = (double *)calloc(dim, sizeof(double));
	vtmp = (double *)calloc(dim, sizeof(double));

	// vx := 0
	set0_dvector(vx, dim, 1);

	// vy := b
	init_resnorm2 = cblas_dnrm2(dim, vec_b, 1);

	// Jacobi Iteration
	for(itimes = 0; itimes < maxtimes; itimes++)
	{
		// vy := vb
		cblas_dcopy(dim, vec_b, 1, vy, 1);

		// vx_new := vx
		cblas_dcopy(dim, vx, 1, vx_new, 1);

		// y := b - A * x
		mkl_cspblas_dcsrgemv("N", &dim, mat_a_val, mat_ia, mat_ja, vx, vtmp);
		cblas_daxpy(dim, -1.0, vtmp, 1, vy, 1);

		// 
		if(history_norm2 != NULL)
			history_norm2[itimes] = cblas_dnrm2(dim, vy, 1) / init_resnorm2;

		// x_new := x + D^{-1} * y
		//cblas_dsbmv(CblasRowMajor, CblasUpper, dim, 0, 1.0, md, 1, vy, 1, 1.0, vx_new, 1);
		for(i = 0; i < dim; i++)
			vx_new[i] = vx[i] + diagonal[i] * vy[i];

		//for(i = 0; i < dim; i++)
		//	printf("%3d %15.7e %15.7e: %15.7e\n", i, vy[i], vx_new[i], diagonal[i]);
		//printf("\n");

		// || x_new - x || < reps || x_new || + aeps
		cblas_daxpy(dim, -1.0, vx_new, 1, vx, 1);
		if(cblas_dnrm2(dim, vx, 1) <= reps * cblas_dnrm2(dim, vx_new, 1) + aeps)
			break;

		//printf("%3d %10.3e\n", itimes, cblas_dnrm2(dim, vx, 1));

		// vx := vx_new
		cblas_dcopy(dim, vx_new, 1, vx, 1);
	}

	// vx := vx_new
	cblas_dcopy(dim, vx_new, 1, answer, 1);

	// free
	free(vx);
	free(vx_new);
	free(vy);
	free(vtmp);

	return itimes;

}


int main()
{
	int i, j, itimes, maxtimes;
	int row_dim, col_dim, dim;
	double *md, *vy, *vx, *vx_true, *vx_new, *vb, *his_norm2_res;
	double reps, aeps;

	// A: sparse matrix
	unsigned char *mtx_fname;
	// 疎行列 COOrdinate format
	double *ma_val;
	int *ma_row_index, *ma_col_index;
	// 疎行列 CSR format
	double *ma_val_csr;
	int *ma_ia_csr, *ma_ja_csr;
	// 共通
	int ma_num_nonzero, diag_flag, job[6], info;
	MM_typecode matcode;

	// input mtx file
	//mtx_fname = "mm/b1_ss/b1_ss.mtx"; // (1,1) == (0, 0) element lack!
	//mtx_fname = "mm/sptest.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/diagtest.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/cage4/cage4.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/cavity10/cavity10.mtx"; // all diagonal elements are found!
	mtx_fname = "mm/t2d_q4/t2d_q4.mtx";

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

	// Initialize
	md = (double *)calloc(dim, sizeof(double));
	vy = (double *)calloc(dim, sizeof(double));
	vx = (double *)calloc(dim, sizeof(double));
	vx_new = (double *)calloc(dim, sizeof(double));
	vx_true = (double *)calloc(dim, sizeof(double));
	vb = (double *)calloc(dim, sizeof(double));

	// Initialize for CSR format
	ma_val_csr = (double *)calloc(ma_num_nonzero, sizeof(double));
	ma_ia_csr = (int *)calloc(ma_num_nonzero, sizeof(int));
	ma_ja_csr = (int *)calloc(ma_num_nonzero, sizeof(int));

	// input ma and vx_true
	for(i = 0; i < dim; i++)
		vx_true[i] = (double)(i + 1);

	// md := D^{-1} = diag[1/a11, 1/a22, ..., 1/ann]
	diag_flag = 0;

	// 1-based index -> 0-based index & 対角成分探索
	for(i = 0; i < ma_num_nonzero; i++)
	{
		--ma_row_index[i];
		--ma_col_index[i];

		// 対角要素探索
		if(ma_row_index[i] == ma_col_index[i])
		{
			md[ma_row_index[i]] = 1.0 / ma_val[i];
			diag_flag++; // 見つかった！
			//printf("%d: %d - %d\n", i, ma_row_index[i], ma_col_index[i]);
		}
	}
	if(diag_flag < dim)
	{
		fprintf(stderr, "ERROR: a part of diagonal elements lack! (dim = %d, #diag. = %d)\n", dim, diag_flag);
		return EXIT_FAILURE;
	}

	// Convert COO -> CSR

	// job[0] = 0: CSR to COO
	//          1: COO to CSR
	//          2: COO to CSR and sorting
	job[0] = 1;
	// job[1] = 0: 0-based index in CSR
	//          1: 1-based index in CSR
	job[1] = 0;
	// job[2] = 0: 0-based index in COO
	//          1: 1-based index in COO
	job[2] = 0;
	// job[3] is not used
	job[3] = 0;
	// job[4] = nzmax (CSR job[0] = 0) or nnz (COO job[0] = 1, 2)
	job[4] = ma_num_nonzero;
	// job[5] : job indicator
	// CSR -> COO: job[5] = 3 (all allrays are filled), 1(row_index only), 2(row_, col_index only)
	// COO -> CSR: job[5] = 0 (all arrays are filled), 1 (ia only), 2
	job[5] = 0;

	mkl_dcsrcoo(job, &dim, ma_val_csr, ma_ja_csr, ma_ia_csr, &ma_num_nonzero, ma_val, ma_row_index, ma_col_index, &info);

	// 行列・ベクトル積(SpMV)
	// vb := ma * vx_true
	mkl_cspblas_dcsrgemv("N", &dim, ma_val_csr, ma_ia_csr, ma_ja_csr, vx_true, vb);

	// print vb
/*	printf("b := A * x, a(i,i)\n");
	for(i = 0; i < dim; i++)
		printf("%d %25.17e %25.17e\n", i, vb[i], md[i]);
*/
	reps = 1.0e-10;
	aeps = 0.0;

	maxtimes = dim * 10;
	his_norm2_res = (double *)calloc(maxtimes, sizeof(double));

	// Jacobi iteration
	itimes = jacobi_iteration_csr(vx, ma_val_csr, ma_ia_csr, ma_ja_csr, md, vb, reps, aeps, his_norm2_res, dim, maxtimes);

	if(itimes < 0)
		itimes = maxtimes;

	for(i = 0; i < itimes; i++)
		printf("%3d %10.3e\n", i, his_norm2_res[i]);

	free(his_norm2_res);
	// print
	printf("Iterative Times = %d\n", itimes);
	printf("Rel.Diff = %10.3e\n", reldiff_dvector(vx, vx_true, dim));

/*	for(i = 0; i < dim; i++)
	{
		printf("%3d %25.17e %25.17e\n", i, vx[i], vx_true[i]);
	}
*/
	// free
	free(ma_val);
	free(ma_row_index);
	free(ma_col_index);
	
	free(ma_val_csr);
	free(ma_ia_csr);
	free(ma_ja_csr);

	free(md);
	free(vy);
	free(vx);
	free(vx_true);
	free(vx_new);
	free(vb);

	return EXIT_SUCCESS;
}
