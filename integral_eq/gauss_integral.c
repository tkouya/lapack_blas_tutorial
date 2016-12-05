/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Gaussian integration formulas                 */
/*                              with LAPACK/BLAS */
/* Last Update: 2016-12-01 (Thu) T.Kouya         */
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

#include "gauss_integral.h"

/* trimat:= [vec[1][0] vec[0][0  ]             0 ... 0     ] */
/*          [vec[2][0] vec[1][1  ] vec[0][1  ] 0 ... 0     ] */
/*          [        ..............................        ] */
/*          [0 ... 0   vec[2][n-3] vec[1][n-2] vec[0][n-2] ] */
/*          [0 ... 0   0           vec[2][n-2] vec[1][n-1] ] */
/*                                                           */
int init_dmatrix_tri(double *trimat[3], int dim)
{
	if((trimat[0] = (double *)calloc(sizeof(double), dim - 1)) == NULL)
		return -1;

	if((trimat[1] = (double *)calloc(sizeof(double), dim)) == NULL)
	{
		free(trimat[0]);
		return -2;
	}
	if((trimat[2] = (double *)calloc(sizeof(double), dim - 1)) == NULL)
	{
		free(trimat[0]);
		free(trimat[1]);
		return -3;
	}

	return 0;
}

void free_dmatrix_tri(double *trimat[3])
{
	free(trimat[0]);
	free(trimat[1]);
	free(trimat[2]);
}

void print_dmatrix_tri(double *trimat[3], int dim)
{
	long int i;

	printf("%3d:                          %25.17e %25.17e\n", 0, trimat[1][0], trimat[0][0]);
	for(i = 1; i < dim - 1; i++)
		printf("%3ld:%25.17e %25.17e %25.17e\n", i, trimat[2][i - 1], trimat[1][i], trimat[0][i]);
	printf("%3ld:%25.17e %25.17e\n", i, trimat[2][dim - 2], trimat[1][dim - 1]);
}

/* coef[2] x^2 + coef[1] x + coef[0] = 0 */
/* Output: ans_re[0] + sqrt(-1)*ans_im[0], ans_re[1] + sqrt(-1) * ans_im[1] */
int dquadratic_eq(double ans_re[2], double ans_im[2], double coef[3])
{
	double d, den, tmp;

//	printf("coef: %10.3e x^2 + %10.3e x + %10.3e = 0\n", coef[2], coef[1], coef[0]);

	ans_re[0] = 0.0;
	ans_re[1] = 0.0;
	ans_im[0] = 0.0;
	ans_im[1] = 0.0;

	if(coef[2] == 0.0)
	{
		if(coef[1] == 0.0)
		{
			fprintf(stderr, "ERROR: No answers (dquadraric_eq)\n");
			return -1;
		}
		ans_re[0] = -coef[0] / coef[1];
		return 1; // number of answers
	}

	den = 2.0 * coef[2];
	d = coef[1] * coef[1] - 4.0 * coef[2] * coef[0];
	ans_re[0] = -coef[1] / den;

	/* anss are real numbers */
	if(d == 0.0)
	{
		ans_re[1] = ans_re[0];
	}
	else if(d > 0)
	{
		tmp = sqrt(d);
		if(coef[1] > 0.0)
			tmp = -tmp / den;
		else
			tmp = tmp / den;
		ans_re[0] -= tmp;
		ans_re[1] = coef[0] / (coef[2] * ans_re[0]);
	}
	/* complex numbers */
	else
	{
		tmp = sqrt(-d) / den;
		ans_re[1] = ans_re[0];
		ans_im[0] = tmp;
		ans_im[1] = -tmp;
	}
	return 2;
}


/* Eigen polynomial of Unsymmetric Real Tridiagonal Matrix   */
/*                                                           */
/* trimat:= [vec[1][0] vec[0][0  ]             0 ... 0     ] */
/*          [vec[2][0] vec[1][1  ] vec[0][1  ] 0 ... 0     ] */
/*          [        ..............................        ] */
/*          [0 ... 0   vec[2][n-3] vec[1][n-2] vec[0][n-2] ] */
/*          [0 ... 0   0           vec[2][n-2] vec[1][n-1] ] */
/*                                                           */
void get_dtrimat_legendre(double *trimat[3], double *diagmat, int dim)
{
	long int i, j, k;
	double tmp, a1, a2, a3, a4, a, b, c;
	double di, dip1;

	/* Frank Matrix */
	for(i = 0; i < dim; i++)
	{
		// Legendre
		a1 = (double)(i + 1);
		a2 = 0.0;
		a3 = (double)(2 * i + 1);
		a4 = (double)i;

		a = a3 / a1;
		b = a2 / a1;
		c = a4 / a1;
		for(j = 0; j < dim; j++)
		{
			if(j == (i - 1))
			{
				tmp = c / a;
				trimat[2][j] = tmp;
			}
			else if(j == i)
			{
				tmp = -b / a;
				trimat[1][j] = tmp;
			}
			else if(j == (i + 1))
			{
				tmp = 1.0 / a;
				trimat[0][j - 1] = tmp;
			}
		}
	}

	/* to be Symmetric */
	di = 1.0;
	diagmat[0] = di;
	for(i = 0; i < dim - 1; i++)
	{
		dip1 = sqrt(trimat[0][i] / trimat[2][i]) * di;
		trimat[0][i] = di / dip1 * trimat[0][i];
		trimat[2][i] = dip1 / di * trimat[2][i];
		di = dip1;
		diagmat[i + 1] = dip1;
	}
}


/* Eigen polynomial of Unsymmetric Real Tridiagonal Matrix   */
/*                                                           */
/* trimat:= [vec[1][0] vec[0][0  ]             0 ... 0     ] */
/*          [vec[2][0] vec[1][1  ] vec[0][1  ] 0 ... 0     ] */
/*          [        ..............................        ] */
/*          [0 ... 0   vec[2][n-3] vec[1][n-2] vec[0][n-2] ] */
/*          [0 ... 0   0           vec[2][n-2] vec[1][n-1] ] */
/*                                                           */
void get_dtrimat_leguerre(double *trimat[3], double *diagmat, int dim)
{
	long int i, j, k;
	double tmp, a1, a2, a3, a4, a, b, c;
	double di, dip1;

	/* Frank Matrix */
	for(i = 0; i < dim; i++)
	{
		// Leguerre: alpha = 0.0
		a1 = (double)(i + 1);
		a2 = (double)(2 * i + 1);
		a3 = (double)(-1);
		a4 = (double)i;

		a = a3 / a1;
		b = a2 / a1;
		c = a4 / a1;
		for(j = 0; j < dim; j++)
		{
			if(j == (i - 1))
			{
				trimat[2][j] = c / a;
			}
			else if(j == i)
			{
				trimat[1][j] = -b / a;
			}
			else if(j == (i + 1))
			{
				trimat[0][j - 1] = 1.0 / a;
			}
		}
	}

	/* to be Symmetric */
	di = 1.0;
	diagmat[0] = di;

	for(i = 0; i < dim - 1; i++)
	{
		dip1 = sqrt(trimat[0][i] / trimat[2][i]) * di;
		trimat[0][i] = di / dip1 * trimat[0][i];

		trimat[2][i] = dip1 / di * trimat[2][i];
		di = dip1;
		diagmat[i + 1] = dip1;
	}

}

/* Eigen polynomial of Unsymmetric Real Tridiagonal Matrix   */
/*                                                           */
/* trimat:= [vec[1][0] vec[0][0  ]             0 ... 0     ] */
/*          [vec[2][0] vec[1][1  ] vec[0][1  ] 0 ... 0     ] */
/*          [        ..............................        ] */
/*          [0 ... 0   vec[2][n-3] vec[1][n-2] vec[0][n-2] ] */
/*          [0 ... 0   0           vec[2][n-2] vec[1][n-1] ] */
/*                                                           */
void get_dtrimat_hermite(double *trimat[3], double *diagmat, int dim)
{
	long int i, j, k;
	double tmp, a1, a2, a3, a4, a, b, c, next_a, next_c;
	double di, dip1;

	/* Frank Matrix */
	for(i = 0; i < dim; i++)
	{
		// Hermite
		a1 = (double)(1);
		a2 = (double)(0);
		a3 = (double)(2);
		a4 = (double)(2 * i);

		a = a3 / a1;
		b = a2 / a1;
		c = a4 / a1;
		for(j = 0; j < dim; j++)
		{
			if(j == (i - 1))
			{
				trimat[2][j] = c / a;
			}
			else if(j == i)
			{
				trimat[1][j] = -b / a;
			}
			else if(j == (i + 1))
			{
				trimat[0][j - 1] = 1.0 / a;
			}
		}
	}

	/* to be Symmetric */
	for(i = 0; i < dim; i++)
	{
		// Hermite
		a1 = (double)(1);
		a2 = (double)(0);
		a3 = (double)(2);
		a4 = (double)(2 * i);

		a = a3 / a1;
		b = a2 / a1;
		c = a4 / a1;
		next_a = a;                          // ONLY for Hermite!!
		next_c = (double)(2 * (i + 1)) / a1; // ONLY for Hermite!!
		for(j = 0; j < dim; j++)
		{
			if(j == (i - 1))
			{
				trimat[2][j] = trimat[0][j];
			}
			else if(j == i)
			{
				trimat[1][j] = -b / a;
			}
			else if(j == (i + 1))
			{
				trimat[0][j - 1] = sqrt(next_c / (a * next_a));
			}
		}
		
	}

}

//
void gauss_integral_eig_d(double *abscissa, double *weight, int deg, int gauss_int_coef)
{
	double *dtrimat[3], *dtrimat_org[3];
	double *dinit_vec, *dweight_vecs;
	double dmu0, dweight, daeps, dreps;
	int i, j;

/* Double */
	/* initialize */
	init_dmatrix_tri(dtrimat, deg);
	init_dmatrix_tri(dtrimat_org, deg);
	dinit_vec = (double *)calloc(sizeof(double), deg);
	dweight_vecs = (double *)calloc(sizeof(double), deg * deg);

	/* get problem */
	/* abscissas & weight */
	switch(gauss_int_coef)
	{
		case GAUSS_LEGUERRE:
			get_dtrimat_leguerre(dtrimat, dinit_vec, deg);
			get_dtrimat_leguerre(dtrimat_org, dinit_vec, deg);
			dmu0 = 1.0;
			break;
		case GAUSS_HERMITE:
			get_dtrimat_hermite(dtrimat, dinit_vec, deg);
			get_dtrimat_hermite(dtrimat_org, dinit_vec, deg);
			dmu0 = sqrt(M_PI);
			break;
		default: // gauss_legendre
		case GAUSS_LEGENDRE:
			get_dtrimat_legendre(dtrimat, dinit_vec, deg);
			get_dtrimat_legendre(dtrimat_org, dinit_vec, deg);
			dmu0 = 2.0;
			break;
	}

	// dweight_vecs := I
	for(i = 0; i < deg; i++)
	{
		for(j = 0; j < deg; j++)
			dweight_vecs[i * deg + j] = 0.0;
		dweight_vecs[i * deg + i] = 1.0;
	}

	// eigenvalues and eigenvectors of symmetric  tridiagonal matrix
	LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'V', deg, dtrimat[1], dtrimat[2], dweight_vecs, deg);

/*	printf("Eigenvalues: \n");
	for(i = 0; i < deg; i++)
	{
		printf("%3d: %10g\n", i, dtrimat[1][i]);
		for(j = 0; j < deg; j++)
			printf("\t %10g\n", dweight_vecs[i * deg]);
	}
*/
//	print_dmatrix_tri(dtrimat);
//	print_dvector(dinit_vec);

	/* abscissa */
	cblas_dcopy(deg, dtrimat[1], 1, abscissa, 1);

	for(i = 0; i < deg; i++)
	{
		dweight = dweight_vecs[i];
		dweight = dweight * dweight * dmu0;
		//printf("%5d, %25.17e, %25.17e\n", i, dtrimat[1][i], dweight);

		/* weight */
		weight[i] = dweight;
	}

	/* free */
	free_dmatrix_tri(dtrimat);
	free_dmatrix_tri(dtrimat_org);
	free(dinit_vec);
	free(dweight_vecs);
}

//shifted_gauss_legendre(dvec_abscissa, dvec_weight, dim);
void dshifted_gauss_legendre(double min_val, double max_val, double *abscissa, double *weight, int deg)
{
	int i, div;
	double x, trans1, trans2;

	div = deg;

	/* transform of variables */
	trans1 = (max_val - min_val) / 2.0;
	trans2 = (max_val + min_val) / 2.0;

	/* Main loop */
	/* sum^N w_i * func(t) */
	for(i = 0; i < div; i++)
	{
		/* x = (b - a) / 2 * t + (b + a) / 2 */
		abscissa[i] = trans1 * abscissa[i] + trans2;

		//printf("%25.17e, %25.17e\n", abscissa[i], weight[i]);
	}
}

/* int[max_val, min_val] func(x) dx */
double dgauss_legendre_integral(double min_val, double max_val, double (* func)(double), double *abscissa, double *weight, int deg)
{
	int i, div;
	double x, trans1, trans2, func_val, ret;

	div = deg;

	/* transform of variables */
	trans1 = (max_val - min_val) / 2.0;
	trans2 = (max_val + min_val) / 2.0;

	ret = 0.0;

	/* Main loop */
	/* sum^N w_i * func(t) */
	for(i = 0; i < div; i++)
	{
		/* x = (b - a) / 2 * t + (b + a) / 2 */
		x = trans1 * abscissa[i] + trans2;
		ret += func(x) * weight[i];
	}
	ret *= trans1;

	return ret;
}

#ifdef DEBUG

// usage
void usage(const char *progname)
{
	printf("$ %s [kind of Gaussian Integration Scheme] [deg]\n", progname);
	printf("     0: Gauss-Legendre, 1: Gauss-Leguerre, 2: Gauss-Hermite\n");
}

// ex1: cos(x)
double ex1_func(double x)
{
	return cos(x);
}

// ex2: x^2
double ex2_func(double x)
{
	return x * x;
}

int main(int argc, char *argv[])
{
	int deg = 3, kind_of_scheme, i;
	double *abscissa, *weight;

	if(argc <= 1)
	{
		usage(argv[0]);
		return EXIT_SUCCESS;
	}

	kind_of_scheme = atoi(argv[1]);

	if(kind_of_scheme == 1)
	{
		kind_of_scheme = GAUSS_LEGUERRE;
		printf("Gauss-Leguerre ");
	}
	else if(kind_of_scheme == 2)
	{
		kind_of_scheme = GAUSS_HERMITE;
		printf("Gauss-Hermite ");
	}
	else
	{
		kind_of_scheme = GAUSS_LEGENDRE;
		printf("Gauss-Legendre ");
	}

	if(argc >= 3)
	{
		deg = atoi(argv[2]);
		if(deg <= 1)
			deg = 2;
	}

	printf(" %d points Formura:\n", deg);

	abscissa = (double *)calloc(deg, sizeof(double));
	weight = (double *)calloc(deg, sizeof(double));

	gauss_integral_eig_d(abscissa, weight, deg, kind_of_scheme);

	for(i = 0; i < deg; i++)
		printf("%5d: %25.17e %25.17e\n", i, abscissa[i], weight[i]);

	if(kind_of_scheme == GAUSS_LEGENDRE)
	{
		// Check values of constant integrals with the given Gauss-Legedre formula
		printf("ex1: int[0, PI/2] cos x dx = 1 approx %25.17e\n", dgauss_legendre_integral(0.0, M_PI / 2, ex1_func, abscissa, weight, deg));
		printf("ex2: int[0, 1   ] x^2 dx = 1/3 approx %25.17e\n", dgauss_legendre_integral(0.0, 1.0, ex2_func, abscissa, weight, deg));
	}

	free(abscissa);
	free(weight);

	return EXIT_SUCCESS;
}
#endif // DEBUG