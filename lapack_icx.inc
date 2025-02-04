#*************************************************#
# LAPACK/BLAS Tutorial                            #
# Configuration file for Intel C compiler         #
# Last Update: 2026-02-04 (Tue) T.Kouya           #
#*************************************************#
CC=icx-cc
FC=ifx
CPP=icx
INTEL_ONEAPI=/opt/intel/oneapi

INC = -I/usr/local/include
LIB = -L/usr/local/lib -L/usr/lib64 -L/opt/intel/lib/intel64 -lifcore

CBLAS_INC = $(INC)
CBLAS_LIB = $(LIB) -lcblas -lrefblas -lm

LAPACKE_INC = -I/usr/local/include/lapacke $(CBLAS_INC)
LAPACKE_LIB = -L/usr/local/lib -llapacke -llapack $(CBLAS_LIB) -L$(INTEL_ONEAPI)/compiler/latest/lib -lifcore

IMKL_INC=-I$(INTEL_ONEAPI)/mkl/latest/include
IMKL_LIB=-L$(INTEL_ONEAPI)/mkl/latest/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L$(INTEL_ONEAPI)/compiler/latest/lib -lifcore -liomp5

OPENMP = -openmp -qopt-report=max
OPENMP_LIB = 
