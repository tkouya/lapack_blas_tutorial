#*************************************************#
# LAPACK/BLAS Tutorial                            #
# Configuration file for Intel C compiler         #
#                                      on Windows #
# Last Update: 2016-12-02 (Fri) T.Kouya           #
#*************************************************#
CC=icl /TC /Qstd=c99 /D:WIN32
FC=ifort
CPP=icl /TP

VCL=cl

INTEL_ROOT = "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows"
INTEL_LIB_ROOT = $(INTEL_ROOT)\compiler\lib\intel64
INTEL_MKL_ROOT = $(INTEL_ROOT)\mkl\lib\intel64

CBLAS_INC = $(INC)
CBLAS_LIB = /link $(INTEL_MKL_ROOT)\mkl_intel_lp64.lib /link $(INTEL_MKL_ROOT)\mkl_core.lib /link $(INTEL_MKL_ROOT)\mkl_sequential.lib

IMKL_INC = $(INC)
IMKL_LIB = /link $(INTEL_MKL_ROOT)\mkl_intel_lp64.lib /link $(INTEL_MKL_ROOT)\mkl_core.lib /link $(INTEL_MKL_ROOT)\mkl_sequential.lib

LAPACKE_INC = $(INC)
LAPACKE_LIB = /link $(INTEL_MKL_ROOT)\mkl_intel_lp64.lib /link $(INTEL_MKL_ROOT)\mkl_core.lib /link $(INTEL_MKL_ROOT)\mkl_sequential.lib

OPENMP = -openmp -openmp-report2
OPENMP_LIB = 
