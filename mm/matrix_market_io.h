/*************************************************/
/* I/O functions for MatrixMarcket format        */
/*                                               */
/* Version 0.1: 2015-04-15(Wed)                  */
/*                                               */
/* Original code:                                */
/* http://math.nist.gov/MatrixMarket/mmio-c.html */
/* *************** Public Domain *************** */
/*************************************************/

#ifndef __MATRIX_MARKET_IO_H__
#define __MATRIX_MARKET_IO_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Max. length of string per one line
#define MM_MAX_LINE_LEN 4096

// Max. length of token
#define MM_MAX_TOKEN_LEN 128

// Bunner of Matrix Market format
#define MM_BANNER "%%MatrixMarket"

// return codes
#define MM_ERROR  (-1)
#define MM_SUCCESS (0)

// boolean values
#define MM_TRUE  (1)
#define MM_FALSE (0)

// The number of kinds of Matrix Market typecode
#define MM_NUM_TYPECODE 4

// Matrix Type code
typedef unsigned char MM_typecode[MM_NUM_TYPECODE];

/******************************************************/
/* Matrix Market Internal definitions of Matrix types */
/* type[4] ... 4-character sequence                   */
/*                                                    */
/* Object    Sparse                  Storage          */
/*           or Dense  Data Type     Scheme           */
/*                                                    */
/* type[0]   type[1]    type[2]      type[3]          */
/*                                                    */
/* M(atrix)  C(oord)    R(real)      G(eneral)        */
/*           A(rray)    C(omplex)    H(ermitian)      */
/*                      P(attern)    S(ymmetric)      */
/*                      I(nteger) (s)K(ew-symmetric)  */
/******************************************************/

#define MM_MATRIX_STR		"matrix"
#define mm_is_matrix(typecode)		((typecode)[0] == 'M')
#define mm_set_matrix(typecode)		((*typecode)[0] = 'M')

#define MM_SPARSE_STR		"coordinate"
#define mm_is_sparse(typecode)		((typecode)[1] == 'C')
#define mm_set_sparse(typecode)		((*typecode)[1] = 'C')

#define MM_COODINATE_STR	"coordinate"
#define mm_is_coodinate(typecode)	((typecode)[1] == 'C')
#define mm_set_coodinate(typecode)	((*typecode)[1] = 'C')

#define MM_DENSE_STR		"array"
#define mm_is_dense(typecode)		((typecode)[1] == 'A')
#define mm_set_dense(typecode)		((*typecode)[1] = 'A')

#define MM_ARRAY_STR		"array"
#define mm_is_array(typecode)		((typecode)[1] == 'A')
#define mm_set_array(typecode)		((*typecode)[1] = 'A')

#define MM_REAL_STR			"real"
#define mm_is_real(typecode)		((typecode)[2] == 'R')
#define mm_set_real(typecode)		((*typecode)[2] = 'R')

#define MM_COMPLEX_STR		"complex"
#define mm_is_complex(typecode)		((typecode)[2] == 'C')
#define mm_set_complex(typecode)	((*typecode)[2] = 'C')

#define MM_PATTERN_STR		"pattern"
#define mm_is_pattern(typecode)		((typecode)[2] == 'P')
#define mm_set_pattern(typecode)	((*typecode)[2] = 'P')

#define MM_INTEGER_STR		"integer"
#define mm_is_integer(typecode)		((typecode)[2] == 'I')
#define mm_set_integer(typecode)	((*typecode)[2] = 'I')

#define MM_GENERAL_STR		"general"
#define mm_is_general(typecode)		((typecode)[3] == 'G')
#define mm_set_general(typecode)	((*typecode)[3] = 'G')

#define MM_HERMITIAN_STR	"hermitian"
#define mm_is_hermitian(typecode)	((typecode)[3] == 'H')
#define mm_set_hermitian(typecode)	((*typecode)[3] = 'H')

#define MM_SYMMETRIC_STR	"symmetric"
#define mm_is_symmetric(typecode)	((typecode)[3] == 'S')
#define mm_set_symmetric(typecode)	((*typecode)[3] = 'S')

#define MM_SKEW_STR			"skew-symmetric"
#define mm_is_skew(typecode)		((typecode)[3] == 'K')
#define mm_set_skew(typecode)		((*typecode)[3] = 'K')

/* clear type codes */
#define mm_clear_typecode(typecode)	((*typecode)[0] = (*typecode)[1] = (*typecode)[2] = ' ', (*typecode)[3] = 'G')
#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)

/* error codes */
#define MM_COULD_NOT_READ_FILE	11
#define MM_COULD_NOT_READ_FILE_STR	"MM_ERROR: cannot read the file!"

#define MM_COULD_NOT_WRITE_FILE	12
#define MM_COULD_NOT_WRITE_FILE_STR	"MM_ERROR: cannot write the file!"

#define MM_NOT_MATRIX			13
#define MM_NOT_MATRIX_STR			"MM_ERROR: not matrix!"

#define MM_NO_HEADER			14
#define MM_NO_HEADER_STR			"MM_ERROR: cannot find MM header!"

#define MM_UNSUPPORTED_TYPE		15
#define MM_UNSUPPORTED_TYPE_STR		"MM_ERROR: is unsupported type!"

#define MM_LINE_TOO_LONG		16
#define MM_LINE_TOO_LONG_STR		"MM_ERROR: length of line exceeded!"

#define MM_PREMATURE_EOF		17
#define MM_PREMATURE_EOF_STR		"MM_ERROR: premature EOF!"

#define MM_UNDEFINED_ERROR		18
#define MM_UNDEFINED_ERROR_STR		"MM_ERROR: cannot specify the king of errors!"

/******************************************/
/* Functions defined in mmio              */
/*                                        */
/* mm_is_valid                            */
/* mm_typecode_to_str                     */
/* mm_read_banner                         */
/* mm_read_mtx_crd_size                   */
/* mm_write_crd_size                      */
/* mm_read_unsymmetric_sparse (obsolete?) */
/* mm_read_mtx_array_size                 */
/* mm_write_mtx_array_size                */
/* mm_read_mtx_crd_data                   */
/* mm_read_mtx_crd_entry                  */
/* mm_read_mtx_crd                        */
/* mm_write_mtx_crd                       */
/******************************************/
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// NULL or print str
unsigned char *mm_put_str(unsigned char *str);

/* print error messages */
int mm_print_error(unsigned char *str, int error_code);

// convert type codes to strings
unsigned char *mm_typecode_to_str(MM_typecode matcode);

// check matcode
int mm_is_valid(MM_typecode matcode);

// read banner
int mm_read_banner(FILE *fp, MM_typecode *matcode);

// write banner
int mm_write_banner(FILE *fp, MM_typecode matcode);

// read coodinate matrix size
int mm_read_mtx_crd_size(FILE *fp, int *row_dim, int *col_dim, int *num_nonzeros);

// write coodinate matrix size
int mm_write_crd_size(FILE *fp, int row_dim, int col_dim, int num_nonzeros);

// read unsymmetric double precision sparse matrix
// I cannot understand thre reason why this function is nessesary. For practical training ?
int mm_read_unsymmetric_sparse(const char *fname, int *row_dim, int *col_dim, int *num_nonzeros, double **val, int **row_index, int **col_index);

// read mtx array size
int mm_read_mtx_array_size(FILE *fp, int *row_dim, int *col_dim);

// write mtx array size
int mm_write_mtx_array_size(FILE *fp, int row_dim, int col_dim);

// read coodinate matrix when row_index, col_index, val are already allocated
int mm_read_mtx_crd_data(FILE *fp, int row_dim, int col_dim, int num_nonzeros, int row_index[], int col_index[], double val[], MM_typecode matcode);

// read one line(entry) of coodinate matrix 
int mm_read_mtx_crd_entry(FILE *fp, int *row_index, int *col_index, double *real, double *imag, MM_typecode matcode);

// read any kinds of coodinate matrix
// After finishing to read the whole data, mm_read_mtx_crd fills
// row_dim, col_dim, num_nonzeros, row_index, col_index and array of values,
// and return typecode such as "MCRS" which means "Matrix, Coordinate, Real, Symmetric".
int mm_read_mtx_crd(const char *fname, int *row_dim, int *col_dim, int *num_nonzeros, int **row_index, int **col_index, double **val, MM_typecode *matcode);

// write mtx format file
int mm_write_mtx_crd(unsigned char *fname, int row_dim, int col_dim, int num_nonzeros, int row_index[], int col_index[], double val[], MM_typecode matcode);

/******************************************/
/* Functions newly defined                */
/*                                        */
/* mm_read_mtx_array_data                 */
/* mm_read_mtx_array_entry                */
/* mm_read_mtx_array                      */
/* mm_write_mtx_array                     */
/* mm_print_header_mtx_crd                */
/* mm_print_header_mtx_array              */
/******************************************/

// read array type matrix when val is already allocated
int mm_read_mtx_array_data(FILE *fp, int row_dim, int col_dim, double val[], MM_typecode matcode);

// read one line(entry) of coodinate matrix 
int mm_read_mtx_array_entry(FILE *fp, double *real, double *imag, MM_typecode matcode);

// read general real or complex array matrix
// After finishing to read the whole data, mm_read_mtx_crd fills
// row_dim, col_dim and array of values,
// and return typecode such as "MARG" which means "Matrix, Array, Real, General".
int mm_read_mtx_array(const char *fname, int *row_dim, int *col_dim, double **val, MM_typecode *matcode);

// write mtx format file
int mm_write_mtx_array(unsigned char *fname, int row_dim, int col_dim, double val[], MM_typecode matcode);

// read any kinds of coodinate matrix within specified number of lines
int mm_print_header_mtx_crd(const char *fname, int num_lines_print);

// read any kinds of coodinate matrix within specified number of lines
int mm_print_header_mtx_array(const char *fname, int num_lines_print);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // __MATRIX_MARKET_IO_H__
