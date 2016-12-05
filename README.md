============================================================
LAPACK/BLAS Tutorial: List of sample programs

 2016-12-02 (Fri) Tomonori Kouya
============================================================

The following programs are published for all readers of "LAPACK/BLAS Tutorial" written in JAPANESE.

*Common  
	lapack_gcc.inc  
	lapack_icc.inc  
	lapack_win_intel.inc  
	Makefile.unix  
	Makefile.win_intel  
	windows\  

*Chapter 1  
	first.c  
	complex_first.c  
	complex_first_cpp.c  

*Chapter 2  
	my_matvec_mul.c  
	matvec_mul.c  
	complex_matvec_mul.c  
	my_linear_eq.c  
	linear_eq.c  
	row_column_major.c  
	complex_row_column_major.c  
	lapack_complex_row_column_major.c  
	lapack_complex_row_column_major.cc  

*Chapter 3  
	blas1.c  
	blas2.c  
	blas3.c  
	jacobi_iteration.c  
	power_eig.c  

*Chapter 4  
	linear_eq_dgetrf.c  
	linear_eq_dsgesv.c  
	linear_eq_dsposv.c  
	lapack_dgecon.c  
	lapack_lamch.c  
	invpower_eig.c  
	lapack_dgeev.c  
	lapack_dsyev.c  
	lapack_ssyev.c  

*Chapter 5  
	my_matvec_mul_pt.c  
	my_matvec_mul_omp.c  
	my_linear_eq_omp.c  

*Chapter 6  
	jacobi_iteration_mkl.c  
	jacobi_iteration_csr_mkl.c  
	bicgstab_mkl.c  
	bicgstab_csr_mkl.c  
	mm/matrix_market_io.h  
	mm/matrix_market_io.c  

*Chapter 7 (Caution: These programs cannot be compiled on Windows!)  
	mycuda_daxpy.cu  
	matvec_mul_cublas.c  
	matvec_mul_magma.c  
	matvec_mul_magma_pure.c  
	linear_eq_magma.c  
	lapack_dgeev_magma.c  
	bicgstab_csr_cusparse.c  

*Chapter 8  
	integral_eq/Makefile.unix  
	integral_eq/Makefile.win_intel  
	integral_eq/gauss_integral.h  
	integral_eq/gauss_integral.c  
	integral_eq/iteration.c  

