#*************************************************#
# LAPACK/BLAS Tutorial                            #
# Configuration file for Intel C compiler         #
# Last Update: 2016-12-02 (Fri) T.Kouya           #
#*************************************************#
CC=icc
FC=ifort
CPP=icpc

INC = -I/usr/local/include
LIB = -L/usr/local/lib -L/usr/lib64 -L/opt/intel/lib/intel64 -lifcore

CBLAS_INC = $(INC)
CBLAS_LIB = $(LIB) -lcblas -lrefblas -lm

LAPACKE_INC = -I/usr/local/include/lapacke $(CBLAS_INC)
LAPACKE_LIB = -L/usr/local/lib -llapacke -llapack $(CBLAS_LIB) -L/opt/intel/lib/intel64 -lifcore

IMKL_INC=-I/opt/intel/include -I/opt/intel/mkl/include 
#IMKL_LIB=-L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -L/opt/intel/lib/intel64 -lifcore
IMKL_LIB=-L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L/opt/intel/lib/intel64 -lifcore -liomp5

OPENMP = -openmp -openmp-report2
OPENMP_LIB = 
