/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Complex row-major and column major matrices   */
/* Last Update: 2016-11-30 (Wed) T.Kouya         */
/*************************************************/
#include <iostream>
#include <cstdio>

#include "lapacke.h"
#include "cblas.h"

using namespace std;

int main()
{
	int i, j, row_dim, col_dim;

	lapack_complex_double *mat_a;

	// 次元数入力
	printf("Row    Dim = "); scanf("%d", &row_dim);
	printf("Column Dim = "); scanf("%d", &col_dim);

	if((row_dim <= 0) || (col_dim <= 0))
	{
		printf("Illegal dimension! (row_dim = %d, col_dim = %d)\n", row_dim, col_dim);
		return EXIT_FAILURE;
	}

	// 変数初期化
	mat_a = (lapack_complex_double *)calloc(row_dim * col_dim, sizeof(lapack_complex_double));

	printf("Row Major: %d x %d\n", row_dim, col_dim);

	// Row Major (行方向格納形式)
	// mat_a = Aに値入力
	// A = [1+i 2+2i ....... n+ni]
	//     [n+i n+2i ...... 2n+ni]
	//     [.....................]
	//     [(m-1)n+1+i ..... mn+ni]
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			mat_a[i * col_dim + j] = lapack_make_complex_double((double)(i * col_dim + j + 1), (double)(j + 1));
			//mat_a[i * col_dim + j] = lapack_make_complex_double((double)(i * col_dim + j + 1), (double)(j + 1));
	}

	// 出力(1)
	printf("1 dimension: \n");
	printf("[");
	for(i = 0; i < row_dim * col_dim; i++)
		printf(" %6.3lf %+-6.3lf * i ", lapack_complex_double_real(mat_a[i]), lapack_complex_double_imag(mat_a[i]));
	printf("]\n");

	// 出力(2)
	printf("2 dimension: \n");
	for(i = 0; i < row_dim; i++)
	{
		printf("[");
		for(j = 0; j < col_dim; j++)
			printf(" %6.3lf %+-6.3lf * i ", lapack_complex_double_real(mat_a[i * col_dim + j]), lapack_complex_double_imag(mat_a[i * col_dim + j]));
		printf("]\n");
	}

	printf("Column Major: %d x %d\n", row_dim, col_dim);

	// Column Major (列方向格納形式)
	// mat_a = Aに値入力
	// A = [1   2   ....... n]
	//     [n+1 n+2 ...... 2n]
	//     [.................]
	//     [(m-1)n+1 ..... mn]
	for(j = 0; j < col_dim; j++)
	{
		for(i = 0; i < row_dim; i++)
			mat_a[i + row_dim * j] = lapack_make_complex_double((double)(i * col_dim + j + 1), (double)(j + 1));
	}

	// 出力(1)
	printf("1 dimension: \n");
	printf("[");
	for(i = 0; i < row_dim * col_dim; i++)
		printf(" %6.3lf  %+-6.3lf * i ", lapack_complex_double_real(mat_a[i]), lapack_complex_double_imag(mat_a[i]));
	printf("]\n");

	// 出力(2)
	printf("2 dimension: \n");
	for(i = 0; i < row_dim; i++)
	{
		printf("[");
		for(j = 0; j < col_dim; j++)
			printf(" %6.3lf %+-6.3lf * i ", lapack_complex_double_real(mat_a[i + row_dim * j]), lapack_complex_double_real(mat_a[i + row_dim * j]));
		printf("]\n");
	}

	// 変数消去
	free(mat_a);

	return EXIT_SUCCESS;
}
