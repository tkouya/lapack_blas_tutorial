/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Hammerstein Integral equation                 */
/* Last Update: 2016-12-02 (Fri) T.Kouya         */
/*************************************************/
//
// Example 1: x(s) = 1 + 1/2 * int[0, 1] K(s, t) (|x(t)| + (x(t))^2) dt
//            where K(s, t) = (1 - s) * t (t \le s) or (1 - t) * s (s \le t)
//
double get_bij(int i, int j, double *abscissa, double *weight)
{
	double ret;

	// abscissa[i] = s_i
	// abscissa[j] = t_j
	if(j <= i)
		ret = weight[j] * abscissa[j] * (1.0 - abscissa[i]);
	else
		ret = weight[j] * abscissa[i] * (1.0 - abscissa[j]);

	return ret;
}

void get_bmatrix(double *B, int row_dim, int col_dim, double *abscissa, double *weight)
{
	int i, j;

#ifdef _OPENMP
	#pragma omp parallel for private(j)
#endif // _OPENMP
	for(i = 0; i < row_dim; i++)
		for(j = 0; j < col_dim; j++)
			B[i * col_dim + j] = get_bij(i, j, abscissa, weight);

}

double *dmat_b;
double *dvec_abscissa, *dvec_weight, *dvec_xhat, *dvec_xsqr;

void init_derivative_free_iteration_dvector(int dim)
{
	dvec_abscissa = (double *)calloc(dim, sizeof(double));
	dvec_weight = (double *)calloc(dim, sizeof(double));
	dvec_xhat = (double *)calloc(dim, sizeof(double));
	dvec_xsqr = (double *)calloc(dim, sizeof(double));

	gauss_integral_eig_d(dvec_abscissa, dvec_weight, dim, GAUSS_LEGENDRE);
	dshifted_gauss_legendre(0.0, 1.0, dvec_abscissa, dvec_weight, dim);

	dmat_b = (double *)calloc(dim * dim, sizeof(double));

	get_bmatrix(dmat_b, dim, dim, dvec_abscissa, dvec_weight);

}

void free_derivative_free_iteration_dvector(void)
{
	free(dvec_abscissa);
	free(dvec_weight);
	free(dvec_xhat);
	free(dvec_xsqr);

	free(dmat_b);
}

// ret = |x|
void abs_dvector(double *ret, double *x, int dim)
{
	int i;

#ifdef _OPENMP
	#pragma omp parallel for
#endif // _OPENMP
	for(i = 0; i < dim; i++)
		ret[i] = fabs(x[i]);
}

// ret = x^2
void sqr_dvector(double *ret, double *x, int dim)
{
	int i;

#ifdef _OPENMP
	#pragma omp parallel for
#endif // _OPENMP
	for(i = 0; i < dim; i++)
		ret[i] = x[i] * x[i];
}

// ret = 1
void allone_dvector(double *ret, int dim)
{
	int i;

#ifdef _OPENMP
	#pragma omp parallel for
#endif // _OPENMP
	for(i = 0; i < dim; i++)
		ret[i] = 1.0;
}

// double vfunc_index(int, DVector)
double vf_index(int index, double *x, int dim)
{
	double ret, tmp;
	int j;

	// 0.5 * B * (x_hat + x_sqr)
	tmp = 0.0;
#ifdef _OPENMP
	#pragma omp parallel for reduction(+:tmp)
#endif // _OPENMP
	for(j = 0; j < dim; j++)
		tmp += dmat_b[index * dim + j] * (fabs(x[j]) + x[j] * x[j]);

	ret = x[index] - 1.0 - 0.5 * tmp;

	return ret;
}


// void vfunc(DVector, DVector)
void vf(double *ret_vec, double *x, int dim)
{
	double ret;
	int i;
	double *tmp_vec;

	tmp_vec = (double *)calloc(dim, sizeof(double));

	// 0.5 * B * (x_hat + x_sqr)
	abs_dvector(dvec_xhat, x, dim);
	sqr_dvector(dvec_xsqr, x, dim);
	cblas_daxpy(dim, 1.0, dvec_xhat, 1, dvec_xsqr, 1);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0, dmat_b, dim, dvec_xsqr, 1, 0.0, ret_vec, 1);
	cblas_dscal(dim, 0.5, ret_vec, 1);

	// x - 1 - 0.5 * B * (x_hat + x_sqr)
	allone_dvector(tmp_vec, dim);
	cblas_daxpy(dim, 1.0, ret_vec, 1, tmp_vec, 1);
	cblas_dcopy(dim, x, 1, ret_vec, 1);
	cblas_daxpy(dim, -1.0, tmp_vec, 1, ret_vec, 1);

	free(tmp_vec);
}
