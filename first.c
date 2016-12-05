/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* First programming with float & double types   */
/* Last Update: 2016-11-30 (Wed) T.Kouya         */
/*************************************************/
#include <stdio.h>
#include <math.h>

int main()
{
	float sc = 0.0, sa = -2.0, sb = 3.0;
	double dc = 0.0, da = -2.0, db = 3.0;
	double relerr;

	// basic arithmetic: float
	printf("--- float data type(single precsion floating-point number) ---\n");
	sc = sa + sb;
	printf("%25.17e := %25.17e + %25.17e\n", sc, sa, sb);
	sc = sa - sb;
	printf("%25.17e := %25.17e - %25.17e\n", sc, sa, sb);
	sc = sa * sb;
	printf("%25.17e := %25.17e * %25.17e\n", sc, sa, sb);
	sc = sa / sb;
	printf("%25.17e := %25.17e / %25.17e\n", sc, sa, sb);

	// absolute value and square root: float
	sc = fabsf(sa);
	printf("%25.17e := |%25.17e|\n", sc, sa);
	sc = sqrtf(sb);
	printf("%25.17e := sqrt(%25.17e)\n", sc, sb);

	// basic arithmetic: double
	printf("--- double data type(double precsion floating-point number) ---\n");
	dc = da + db;
	printf("%25.17e := %25.17e + %25.17e\n", dc, da, db);
	dc = da - db;
	printf("%25.17e := %25.17e - %25.17e\n", dc, da, db);
	dc = da * db;
	printf("%25.17e := %25.17e * %25.17e\n", dc, da, db);
	dc = da / db;
	printf("%25.17e := %25.17e / %25.17e\n", dc, da, db);

	// absolute value and square root: double
	dc = fabs(da);
	printf("%25.17e := |%25.17e|\n", dc, da);
	dc = sqrt(db);
	printf("%25.17e := sqrt(%25.17e)\n", dc, db);

	// relative error of single precision square root
	relerr = fabs(sc - dc);
	if(fabs(dc) > 0.0)
		relerr /= fabs(dc);

	printf("Single Prec.  : %25.17e\n", sc);
	printf("Double Prec.  : %25.17e\n", dc);
	printf("Relative Error: %10.3e\n", relerr);

	return 0;
}
