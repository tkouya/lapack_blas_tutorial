/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Integral equation by using derivative free    */
/*                                        solver */
/* Last Update: 2016-12-02 (Fri) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef USE_IMKL
	#include "mkl.h" // for Intel Math Kernel Library
	#include "mkl_cblas.h"
	#include "mkl_lapacke.h"
#else // USE_IMKL
	#include "cblas.h"
	#include "lapacke.h"
#endif // USE_IMKL

#ifdef _OPENMP
	#include "omp.h"
#endif // _OPENMP

#include "tkaux.h"
#include "get_secv.h"
#include "gauss_integral.h"

// Definition of the integral equation to be solved
//#include "ex1.c" // Example 1
#include "prob1.c" // Exersise 8.1 (1)
//#include "prob2.c" // Exersise 8.1 (2)

// difference: [u, v; F]
// vfunc(ret_vec, vec): ret_vec = vfunc(vec)
void difference_dmat_parallel(double *ret_mat, double *u, double *v, double (* vfunc_index)(int, double *, int), int dim)
{
	int i, j, k, thread_index, num_threads;
	double *arg_vec[256];
	double den[128], num[128]; // Max. #thread = 128

#ifdef _OPENMP
	num_threads = omp_get_max_threads();
#else // _OPENMP
	num_threads = 1;
#endif // _OPENMP

	// initialize
	for(i = 0; i < num_threads * 2; i++)
		arg_vec[i] = (double *)calloc(dim, sizeof(double));

	// main loop
	for(i = 0; i < dim; i++)
	{
#ifdef _OPENMP
		#pragma omp parallel for private(k, thread_index) 
#endif // _OPENMP
		for(j = 0; j < dim; j++)
		{
#ifdef _OPENMP
			thread_index = omp_get_thread_num();
#else // _OPENMP
			thread_index = 0;
#endif // _OPENMP

			// arg_vec[0] := [.........., u_j, v_j+1, ..., v_dim-1]
			for(k = 0; k <= j; k++)
				arg_vec[thread_index * 2][k] = u[k];
			for(k = j + 1; k < dim; k++)
				arg_vec[thread_index * 2][k] = v[k];
			
			// arg_vec[1] := [..., u_j-1, v_j, .........., v_dim-1]
			for(k = 0; k < j; k++)
				arg_vec[thread_index * 2 + 1][k] = u[k];
			for(k = j; k < dim; k++)
				arg_vec[thread_index * 2 + 1][k] = v[k];

			// (F_i(arg_vec[0]) - F_i(arg_vec[1])) / (u_j - v_j)
			num[thread_index] = vfunc_index(i, arg_vec[thread_index * 2], dim) - vfunc_index(i, arg_vec[thread_index * 2 + 1], dim);
			den[thread_index] = u[j] - v[j];
			
			if(den[thread_index] == 0.0)
			{
				fprintf(stderr, "WARNING: dmat_difference (%d, %d) is divided by zero!\n", i, j);
			}
			else
				num[thread_index] /= den[thread_index];
			
			ret_mat[i * dim + j] = num[thread_index];
		}
	}

	// free
	for(i = 0; i < num_threads * 2; i++)
		free(arg_vec[i]);
}


// difference: [u, v; F]
// vfunc(ret_vec, vec): ret_vec = vfunc(vec)
void difference_dmat_serial(double *ret_mat, double *u, double *v, double (* vfunc_index)(int, double *, int), int dim)
{
	int i, j, k;
	double *arg_vec[2];
	double den, num;

	// initialize
	arg_vec[0] = (double *)calloc(dim, sizeof(double));
	arg_vec[1] = (double *)calloc(dim, sizeof(double));

	// main loop
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{

			// arg_vec[0] := [.........., u_j, v_j+1, ..., v_dim-1]
			for(k = 0; k <= j; k++)
				arg_vec[0][k] = u[k];
			for(k = j + 1; k < dim; k++)
				arg_vec[0][k] = v[k];
			
			// arg_vec[1] := [..., u_j-1, v_j, .........., v_dim-1]
			for(k = 0; k < j; k++)
				arg_vec[1][k] = u[k];
			for(k = j; k < dim; k++)
				arg_vec[1][k] = v[k];

			// (F_i(arg_vec[0]) - F_i(arg_vec[1])) / (u_j - v_j)
			num = vfunc_index(i, arg_vec[0], dim) - vfunc_index(i, arg_vec[1], dim);
			den = u[j] - v[j];
			
			if(den == 0.0)
			{
				fprintf(stderr, "WARNING: dmat_difference (%d, %d) is divided by zero!\n", i, j);
			}
			else
				num /= den;
			
			ret_mat[i * dim + j] = num;
		}
	}

	// free
	free(arg_vec[0]);
	free(arg_vec[1]);
}

// New Secant Method
// work_vec[0] = delta + zeta
void new_secant_1step_dvector(double *next_x, double *xm1, double *x0, double (* vfunc_index)(int, double *, int), void (* vfunc)(double *, double *, int), double *work_vec[4], double *diff_mat, int dim)
{
	int i;
	int *pivot;

	pivot = (int *)calloc(dim, sizeof(int));

	// Solve [x[-1], x[0]; F] delta = -F(x[0])
#ifdef USE_PARALLEL_DIFFMAT
	difference_dmat_parallel(diff_mat, xm1, x0, vfunc_index, dim);
#else // USE_PARALLEL_DIFFMAT
	difference_dmat_serial(diff_mat, xm1, x0, vfunc_index, dim);
#endif // USE_PARALLEL_DIFFMAT

	//print_dmatrix(diff_mat);

	vfunc(work_vec[0], x0, dim);
	cblas_dscal(dim, -1.0, work_vec[0], 1);

	LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim, dim, diff_mat, dim, pivot);

	LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', dim, 1, diff_mat, dim, pivot, work_vec[0], 1);

	// y = x[0] + delta
	cblas_dcopy(dim, work_vec[0], 1, work_vec[2], 1);
	cblas_daxpy(dim, 1.0, x0, 1, work_vec[2], 1);

	// Solve [x[-1], x[0]; F] zeta = -F(y)
	vfunc(work_vec[3], work_vec[2], dim);
	cblas_dscal(dim, -1.0, work_vec[3], 1);
	
	LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', dim, 1, diff_mat, dim, pivot, work_vec[3], 1);
	
	// x[1] := x[0] + delta + zeta
	cblas_daxpy(dim, 1.0, work_vec[3], 1, work_vec[0], 1);
	cblas_daxpy(dim, 1.0, work_vec[2], 1, work_vec[3], 1);

	cblas_dcopy(dim, work_vec[3], 1, next_x, 1);

	free(pivot);
}

// Secant Method
// work_vec[0] = delta
void secant_1step_dvector(double *next_x, double *xm1, double *x0, double (* vfunc_index)(int, double *, int), void (* vfunc)(double *, double *, int), double *work_vec[4], double *diff_mat, int dim)
{
	int i;
	int *pivot;

	pivot = (int *)calloc(dim, sizeof(int));

	// Solve [x[-1], x[0]; F] delta = -F(x[0])
#ifdef USE_PARALLEL_DIFFMAT
	difference_dmat_parallel(diff_mat, xm1, x0, vfunc_index, dim);
#else // USE_PARALLEL_DIFFMAT
	difference_dmat_serial(diff_mat, xm1, x0, vfunc_index, dim);
#endif // USE_PARALLEL_DIFFMAT

	//printf_dvector("%3d %25.17e\n", diff_mat, dim * dim, 1);

	vfunc(work_vec[0], x0, dim);
	cblas_dscal(dim, -1.0, work_vec[0], 1);

	LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim, dim, diff_mat, dim, pivot);

	LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', dim, 1, diff_mat, dim, pivot, work_vec[0], 1);

	// x[1] := x[0] + delta
	cblas_dcopy(dim, x0, 1, next_x, 1);
	cblas_daxpy(dim, 1.0, work_vec[0], 1, next_x, 1);

	free(pivot);
}

// New secant method
int derivative_free_iteration_dvector(double *ret_vec, double *xm1, double *x0, double (* vfunc_index)(int, double *, int), void (* vfunc)(double *, double *, int), int dim, double rel_tol, double abs_tol, int maxtimes)
{
	int iter_times, i;
	double *work_vec[4], *old_old_vec, *old_vec, *tmp_vec;
	double *work_mat;
	double start_time, end_time;

	// initialize
	init_derivative_free_iteration_dvector(dim);

	for(i = 0; i < 4; i++)
		work_vec[i] = (double *)calloc(dim, sizeof(double));
	work_mat = (double *)calloc(dim * dim, sizeof(double));

	old_old_vec = (double *)calloc(dim, sizeof(double));
	cblas_dcopy(dim, xm1, 1, old_old_vec, 1);

	old_vec = (double *)calloc(dim, sizeof(double));
	cblas_dcopy(dim, x0, 1, old_vec, 1);

	//printf("initial value:\n"); printf_dvector("%5d, %25.17e\n", xm1, dim, 1); printf_dvector("%5d, %25.17e\n", x0, dim, 1);

	// main loop
	start_time = get_real_secv();
	for(iter_times = 0; iter_times < maxtimes; iter_times++)
	{

	#ifdef USE_NEW_SECANT
		// ret_vec := old_vec + delta + zeta
		new_secant_1step_dvector(ret_vec, old_old_vec, old_vec, vfunc_index, vfunc, work_vec, work_mat, dim);
	#else // USE_NEW_SECANT
		// ret_vec := old_vec + delta
		secant_1step_dvector(ret_vec, old_old_vec, old_vec, vfunc_index, vfunc, work_vec, work_mat, dim);
	#endif // USE_NEW_SECANT

		//printf("%ld:\n", iter_times); printf_dvector("%5d, %25.17e\n", ret_vec, dim, 1);
		printf("%5d, %25.17e\n", iter_times, cblas_dnrm2(dim, work_vec[0], 1));

		// ||tmp_vec|| < rel_tol * ||old_vec|| + abs_tol ?
		if(cblas_dnrm2(dim, work_vec[0], 1) < rel_tol * cblas_dnrm2(dim, old_vec, 1) + abs_tol)
			break;

		// old_old_vec := old_vec
		// old_vec := ret_vec
		cblas_dcopy(dim, old_vec, 1, old_old_vec, 1);
		cblas_dcopy(dim, ret_vec, 1, old_vec, 1);
	}
	end_time = get_real_secv() - start_time;

	printf("elapsed time(s) of derivative_free_iteration_dvector: %f\n", end_time);

	if(iter_times >= maxtimes)
		fprintf(stderr, "Warning: derivative_free_iteration is not convergent!(%d iter_times)\n", iter_times);

	// free
	for(i = 0; i < 4; i++)
		free(work_vec[i]);
	free(work_mat);

	free(old_old_vec);
	free(old_vec);

	free_derivative_free_iteration_dvector();

	return iter_times;
}

int main(int argc, char *argv[])
{
	int i, dim, itimes;
	double *dvec_u, *dvec_v, *dvec_ans;
	double *dmat;
	double start_time, end_time;

//	dim = 4;
//	dim = 8;
//	dim = 16;
//	dim = 32;
//	dim = 64;
//	dim = 512;
//	dim = 1024;

	if(argc <= 1)
	{
		printf("Usage: %s [dimension] \n", argv[0]);
		return EXIT_SUCCESS;
	}

	dim = atoi(argv[1]);

	if(dim <= 0)
	{
		fprintf(stderr, "ERROR: dimension( = %d) is illegal!\n", dim);
		return EXIT_FAILURE;
	}

	dvec_u = (double *)calloc(dim, sizeof(double));
	dvec_v = (double *)calloc(dim, sizeof(double));
	dvec_ans = (double *)calloc(dim, sizeof(double));
	dmat = (double *)calloc(dim * dim, sizeof(double));

#ifdef USE_IMKL
	char str_mkl_version[1024];
	int max_num_threads;

	MKL_Get_Version_String(str_mkl_version, 1024);
	printf("%s\n", str_mkl_version);

	max_num_threads = mkl_get_max_threads();
	printf("Max Number of Threads: %d\n", max_num_threads);
	mkl_set_num_threads(max_num_threads);
#endif

	srand(10);

#ifdef _OPENMP
	#pragma omp parallel for
#endif // _OPENMP
	for(i = 0; i < dim; i++)
	{
		dvec_u[i] = (double)rand() / (double)RAND_MAX;
		dvec_v[i] = (double)rand() / (double)RAND_MAX;
	}

//	difference_dmat(dmat, dvec_u, dvec_v, vf_index, dim);
//	difference_dmat(dmat, dvec_u, dvec_v, vf1_index, dim);

//	printf_dvector("%3d %25.17e\n", dmat, dim * dim, 1);

	// secant or new secant method
	start_time = get_real_secv();
	itimes = derivative_free_iteration_dvector(dvec_ans, dvec_u, dvec_v, vf_index, vf, dim, 1.0e-10, 1.0e-50, dim * 2);
	end_time = get_real_secv() - start_time;

	//printf("dvec_ans: \n");	printf_dvector("%5d, %25.17e\n", dvec_ans, dim, 1);
	printf("elapsed time(s): %f\n", end_time);

	free(dvec_u);
	free(dvec_v);
	free(dvec_ans);
	free(dmat);

	return EXIT_SUCCESS;
}
