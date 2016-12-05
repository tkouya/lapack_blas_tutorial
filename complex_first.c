/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample program with C99 complex type          */
/* Last Update: 2016-12-01 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <math.h>
#include <complex.h> // C99 Complex data type

int main()
{
	float  complex cc = 0.0, ca = -2.0 + 2.0 * I, cb = 3.0 - 3.0 * I;
	double complex zc = 0.0, za = -2.0 + 2.0 * I, zb = 3.0 - 3.0 * I;
	double relerr;

	// basic arithmetic: float complex
	printf("--- float data type(single precsion floating-point number) ---\n");
	cc = ca + cb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) + (%25.17e %+-25.17e * I)\n", crealf(cc), cimagf(cc), crealf(ca), cimagf(ca), crealf(cb), cimagf(cb));
	cc = ca - cb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) - (%25.17e %+-25.17e * I)\n", crealf(cc), cimagf(cc), crealf(ca), cimagf(ca), crealf(cb), cimagf(cb));
	cc = ca * cb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) * (%25.17e %+-25.17e * I)\n", crealf(cc), cimagf(cc), crealf(ca), cimagf(ca), crealf(cb), cimagf(cb));
	cc = ca / cb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) / (%25.17e %+-25.17e * I)\n", crealf(cc), cimagf(cc), crealf(ca), cimag(ca), creal(cb), cimagf(cb));

	// absolute value and square root: float
	cc = cabsf(ca);
	printf("%25.17e %+-25.17e * I := |%25.17e %+-25.17e * I|\n", crealf(cc), cimagf(cc), crealf(ca), cimagf(ca));
	cc = csqrtf(cb);
	printf("%25.17e %+-25.17e * I:= sqrt(%25.17e %+-25.17e * I)\n", crealf(cc), cimagf(cc), crealf(cb), cimagf(cb));

	// basic arithmetic: double
	printf("--- double data type(double precsion floating-point number) ---\n");
	zc = za + zb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) + (%25.17e %+-25.17e * I)\n", creal(zc), cimag(zc), creal(za), cimag(za), creal(zb), cimag(zb));
	zc = za - zb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) - (%25.17e %+-25.17e * I)\n", creal(zc), cimag(zc), creal(za), cimag(za), creal(zb), cimag(zb));
	zc = za * zb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) * (%25.17e %+-25.17e * I)\n", creal(zc), cimag(zc), creal(za), cimag(za), creal(zb), cimag(zb));
	zc = za / zb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) / (%25.17e %+-25.17e * I)\n", creal(zc), cimag(zc), creal(za), cimag(za), creal(zb), cimag(zb));

	// absolute value and square root: double
	zc = cabs(za);
	printf("%25.17e %+-25.17e * I := |%25.17e %+-25.17e * I|\n", creal(zc), cimag(zc), creal(za), cimag(za));
	zc = csqrt(zb);
	printf("%25.17e %+-25.17e * I:= sqrt(%25.17e %+-25.17e * I)\n", creal(zc), cimag(zc), creal(zb), cimag(zb));

	// relative error of float square root
	relerr = cabs(cc - zc);
	if(cabs(zc) > 0.0)
		relerr /= cabs(zc);

	printf("Single Prec.  : %25.17e + %25.17e * I\n", creal(cc), cimag(cc));
	printf("Double Prec.  : %25.17e + %25.17e * I\n", creal(zc), cimag(zc));
	printf("Relative Error: %10.3e\n", relerr);

	// real part
	relerr = fabs(creal(cc) - creal(zc));
	if(fabs(creal(zc)) > 0.0)
		relerr /= fabs(creal(zc));

	printf("Relative Error(real): %10.3e\n", relerr);

	// imaginary part
	relerr = fabs(cimag(cc) - cimag(zc));
	if(fabs(cimag(zc)) > 0.0)
		relerr /= fabs(cimag(zc));

	printf("Relative Error(imag): %10.3e\n", relerr);

	return 0;
}
