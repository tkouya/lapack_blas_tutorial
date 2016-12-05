/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Jacobi Iterative Refinement with Sparse BLAS  */
/* Last Update: 2015-04-16 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tkaux.h"

#include "cblas.h"				// BLAS
#include "blas_sparse.h"		// Sparse BLAS
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

// Jacobi Iteration Method
int jacobi_iteration(double *answer, blas_sparse_matrix mat_a, double *diagonal, double *vec_b, double reps, double aeps, double history_norm2[], int dim, int maxtimes)
{
	int i, itimes;
	double *vx, *vx_new, *vy;
	double init_resnorm2;

	// Initialize
	vx = (double *)calloc(dim, sizeof(double));
	vx_new = (double *)calloc(dim, sizeof(double));
	vy = (double *)calloc(dim, sizeof(double));

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
		BLAS_dusmv(blas_no_trans, -1.0, mat_a, vx, 1, vy, 1);

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
	blas_sparse_matrix ma; // 疎行列
	int ma_num_nonzero, diag_flag;
	double *ma_val;
	int *ma_row_index, *ma_col_index;
	MM_typecode matcode;

	// input mtx file
	//mtx_fname = "mm/b1_ss/b1_ss.mtx"; // (1,1) == (0, 0) element lack!
	//mtx_fname = "mm/sptest.mtx"; // all diagonal elements are found!
	//mtx_fname = "mm/diagtest.mtx"; // all diagonal elements are found!
	mtx_fname = "mm/cage4/cage4.mtx"; // all diagonal elements are found!

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

	// input ma and vx_true
	for(i = 0; i < dim; i++)
		vx_true[i] = (double)(i + 1);

	// Sparse BLASハンドル生成
	ma = BLAS_duscr_begin(dim, dim);

	// md := D^{-1} = diag[1/a11, 1/a22, ..., 1/ann]
	diag_flag = 0;

	// 1-based index -> 0-based index & 要素を挿入 & 対角成分探索
	for(i = 0; i < ma_num_nonzero; i++)
	{
		BLAS_duscr_insert_entry(ma, ma_val[i], --ma_row_index[i], --ma_col_index[i]);

		// 対角要素探索
		if(ma_row_index[i] == ma_col_index[i])
		{
			md[ma_row_index[i]] = 1.0 / ma_val[i];
			diag_flag++; // 見つかった！
		}
	}
	if(diag_flag < dim)
	{
		fprintf(stderr, "ERROR: a part of diagonal elements lack! (dim = %d, #diag. = %d)\n", dim, diag_flag);
		return EXIT_FAILURE;
	}

	// 行列生成完了
	BLAS_duscr_end(ma);

	// 行列・ベクトル積(SpMV)
	// vb := 1.0 * ma * vx_true
	set0_dvector(vb, dim, 1);
	BLAS_dusmv(blas_no_trans, 1.0, ma, vx_true, 1, vb, 1);

	// print vb
	printf("b := A * x, a(i,i)\n");
	for(i = 0; i < dim; i++)
		printf("%d %25.17e %25.17e\n", i, vb[i], md[i]);

	reps = 1.0e-10;
	aeps = 0.0;

	maxtimes = dim * 10;
	his_norm2_res = (double *)calloc(maxtimes, sizeof(double));

	// Jacobi iteration
	itimes = jacobi_iteration(vx, ma, md, vb, reps, aeps, his_norm2_res, dim, maxtimes);

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

	free(md);
	free(vy);
	free(vx);
	free(vx_true);
	free(vx_new);
	free(vb);


	return EXIT_SUCCESS;
}
