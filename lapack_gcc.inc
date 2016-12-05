#*************************************************#
# LAPACK/BLAS Tutorial                            #
# Configuration file for GNU compiler collection  #
# Last Update: 2016-12-02 (Fri) T.Kouya           #
#*************************************************#
CC=gcc
FC=gfortran
CPP=g++

INC = -I/usr/local/include
LIB = -L/usr/local/lib/gcc  -lgfortran -lm

CBLAS_INC = $(INC)
CBLAS_LIB = $(LIB) -lcblas -lrefblas -lgfortran

LAPACKE_INC = -I/usr/local/include/lapacke $(CBLAS_INC)
LAPACKE_LIB = -L/usr/local/lib/gcc -llapacke -llapack $(CBLAS_LIB)

IMKL_INC=-I/opt/intel/mkl/include 
#IMKL_LIB=-L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -L/opt/intel/lib/intel64 -lifcore
#IMKL_LIB=-L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L/opt/intel/lib/intel64 -lifcore -liomp5

OPENMP = -fopenmp
OPENMP_LIB = -lgomp
