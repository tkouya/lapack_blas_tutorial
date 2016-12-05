/*************************************************/
/* LAPACK/BLAS Tutorial                          */
/* Sample program with C++ complex type          */
/* Last Update: 2016-12-01 (Thu) T.Kouya         */
/*************************************************/
#include <iostream>
#include <complex>
#include <cstdio>

using namespace std;

int main()
{
	complex<float> cc = 0.0, ca = complex<float>(-2.0, 2.0), cb = complex<float>(3.0, -3.0);
	complex<double> zc = 0.0, za = complex<double>(-2.0, 2.0), zb = complex<double>(3.0,  -3.0);
	double relerr;

	// basic arithmetic: float complex
	cout << "--- float data type(single precsion floating-point number) ---" << endl;
	cc = ca + cb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) + (%25.17e %+-25.17e * I)\n", cc.real(), cc.imag(), ca.real(), ca.imag(), cb.real(), cb.imag());
	cc = ca - cb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) - (%25.17e %+-25.17e * I)\n", cc.real(), cc.imag(), ca.real(), ca.imag(), cb.real(), cb.imag());
	cc = ca * cb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) * (%25.17e %+-25.17e * I)\n", cc.real(), cc.imag(), ca.real(), ca.imag(), cb.real(), cb.imag());
	cc = ca / cb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) / (%25.17e %+-25.17e * I)\n", cc.real(), cc.imag(), ca.real(), ca.imag(), cb.real(), cb.imag());

	// absolute value and square root: float
	cc = abs(ca);
	printf("%25.17e %+-25.17e * I := |%25.17e %+-25.17e * I|\n", cc.real(), cc.imag(), ca.real(), ca.imag());
	cc = sqrt(cb);
	printf("%25.17e %+-25.17e * I:= sqrt(%25.17e %+-25.17e * I)\n", cc.real(), cc.imag(), cb.real(), cb.imag());

	// basic arithmetic: double
	printf("--- double data type(double precsion floating-point number) ---\n");
	zc = za + zb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) + (%25.17e %+-25.17e * I)\n", zc.real(), zc.imag(), za.real(), za.imag(), zb.real(), zb.imag());
	zc = za - zb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) - (%25.17e %+-25.17e * I)\n", zc.real(), zc.imag(), za.real(), za.imag(), zb.real(), zb.imag());
	zc = za * zb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) * (%25.17e %+-25.17e * I)\n", zc.real(), zc.imag(), za.real(), za.imag(), zb.real(), zb.imag());
	zc = za / zb;
	printf("%25.17e %+-25.17e * I := (%25.17e %+-25.17e * I) / (%25.17e %+-25.17e * I)\n", zc.real(), zc.imag(), za.real(), za.imag(), zb.real(), zb.imag());

	// absolute value and square root: double
	zc = abs(za);
	printf("%25.17e %+-25.17e * I := |%25.17e %+-25.17e * I|\n", zc.real(), zc.imag(), za.real(), za.imag());
	zc = sqrt(zb);
	printf("%25.17e %+-25.17e * I:= sqrt(%25.17e %+-25.17e * I)\n", zc.real(), zc.imag(), zb.real(), zb.imag());

	// relative error of float square root
	relerr = abs((complex<double>)cc - zc);
	if(abs(zc) > 0.0)
		relerr /= abs(zc);

	printf("Single Prec.  : %25.17e %+-25.17e * I\n", cc.real(), cc.imag());
	printf("Double Prec.  : %25.17e %+-25.17e * I\n", zc.real(), zc.imag());
	printf("Relative Error: %10.3e\n", relerr);

	// real part
	relerr = abs(cc.real() - zc.real());
	if(abs(zc.real()) > 0.0)
		relerr /= abs(zc.real());

	printf("Relative Error(real): %10.3e\n", relerr);

	// imaginary part
	relerr = abs(cc.imag() - zc.imag());
	if(abs(zc.imag()) > 0.0)
		relerr /= abs(zc.imag());

	printf("Relative Error(imag): %10.3e\n", relerr);

	return 0;
}
