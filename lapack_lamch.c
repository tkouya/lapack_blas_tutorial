/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample program to get properties              */
/*                        of floating-point type */
/*                        with original routines */
/* Last Update: 2016-12-01 (Thu) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <math.h>

#include "lapacke.h"

int main()
{
	float seps, sufth, softh;
	double deps, dufth, dofth;

	// float
	seps = LAPACK_slamch("E");
	sufth = LAPACK_slamch("U");
	softh = LAPACK_slamch("O");
	printf("eps : Machine Epsilon(float)     : %15.7e\n", seps);
	printf("ufth: Underflow Threshold(float) : %15.7e\n", sufth);
	printf("ofth: Overflow Threshold(float)  : %15.7e\n", softh);

	// double
	deps = LAPACK_dlamch("E");
	dufth = LAPACK_dlamch("U");
	dofth = LAPACK_dlamch("O");
	printf("Machine Epsilon(double)     : %25.17e\n", deps);
	printf("Underflow Threshold(double) : %25.17e\n", dufth);
	printf("Overflow Threshold(double)  : %25.17e\n", dofth);

	return EXIT_SUCCESS;
}
