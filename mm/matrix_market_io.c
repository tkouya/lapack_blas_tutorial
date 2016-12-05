/*************************************************/
/* I/O functions for MatrixMarcket format        */
/*                                               */
/* Version 0.2: 2016-12-01(Thu)                  */
/* Version 0.1: 2015-04-15(Wed)                  */
/*                                               */
/* Original code:                                */
/* http://math.nist.gov/MatrixMarket/mmio-c.html */
/* *************** Public Domain *************** */
/*************************************************/

#include <stdio.h>
#include "matrix_market_io.h" // compatible to original mmoi.h, mmio.c

/***************************************/
/* functions                           */
/***************************************/

// NULL or print str
unsigned char *mm_put_str(unsigned char *str)
{
	if(str == NULL)
		return "";
	else
		return str;
}

/* print error messages */
int mm_print_error(unsigned char *str, int error_code)
{
	switch(error_code)
	{
		case MM_COULD_NOT_READ_FILE:
			fprintf(stderr, "%s %s\n", mm_put_str(str), MM_COULD_NOT_READ_FILE_STR);
			break;
		case MM_COULD_NOT_WRITE_FILE:
			fprintf(stderr, "%s %s\n", mm_put_str(str), MM_COULD_NOT_WRITE_FILE_STR);
			break;
		case MM_NOT_MATRIX:
			fprintf(stderr, "%s %s\n", mm_put_str(str), MM_NOT_MATRIX_STR);
			break;
		case MM_NO_HEADER:
			fprintf(stderr, "%s %s\n", mm_put_str(str), MM_NO_HEADER_STR);
			break;
		case MM_UNSUPPORTED_TYPE:
			fprintf(stderr, "%s %s\n", mm_put_str(str), MM_UNSUPPORTED_TYPE_STR);
			break;
		case MM_LINE_TOO_LONG:
			fprintf(stderr, "%s %s\n", mm_put_str(str), MM_LINE_TOO_LONG_STR);
			break;
		case MM_PREMATURE_EOF:
			fprintf(stderr, "%s %s\n", mm_put_str(str), MM_PREMATURE_EOF_STR);
			break;
		default:
			fprintf(stderr, "%s %s\n", mm_put_str(str), MM_UNDEFINED_ERROR_STR);
			break;
	}

	return error_code;
}

// convert type codes to strings
unsigned char *mm_typecode_to_str(MM_typecode matcode)
{
	unsigned char buf[MM_MAX_LINE_LEN];
	unsigned char *ret_str;
	unsigned char *type[MM_NUM_TYPECODE];

	/* Check Matrix type */
	if(mm_is_matrix(matcode))
		type[0] = MM_MATRIX_STR;
	else
	{
		mm_print_error(NULL, MM_NOT_MATRIX);
		type[0] = MM_NOT_MATRIX_STR;
	}

	/* Check Coodinate or Array types */
	if(mm_is_sparse(matcode))
		type[1] = MM_SPARSE_STR;
	else
	{
		if(mm_is_dense(matcode))
			type[1] = MM_DENSE_STR;
		else
			return NULL;
	}

	/* Check element data type */
	if(mm_is_real(matcode))
		type[2] = MM_REAL_STR;
	else
	{
		if(mm_is_complex(matcode))
			type[2] = MM_COMPLEX_STR;
		else
		{
			if(mm_is_pattern(matcode))
				type[2] = MM_PATTERN_STR;
			else
			{
				if(mm_is_integer(matcode))
					type[2] = MM_INTEGER_STR;
				else
					return NULL;
			}
		}
	}

	/* Check symmetry type */
	if(mm_is_general(matcode))
		type[3] = MM_GENERAL_STR;
	else
	{
		if(mm_is_symmetric(matcode))
			type[3] = MM_SYMMETRIC_STR;
		else
		{
			if(mm_is_hermitian(matcode))
				type[3] = MM_HERMITIAN_STR;
			else
			{
				if(mm_is_skew(matcode))
					type[3] = MM_SKEW_STR;
				else
					return NULL;
			}
		}
	}

	/* print all types */
	sprintf(buf, "%s %s %s %s", type[0], type[1], type[2], type[3]);

	ret_str = (unsigned char *)calloc((int)strlen(buf) + 1, sizeof(unsigned char));

	strcpy(ret_str, buf);

	return ret_str;
}

// check matcode
int mm_is_valid(MM_typecode matcode)
{
	if(!mm_is_matrix(matcode))
		return MM_FALSE;

	if(mm_is_dense(matcode) && mm_is_pattern(matcode))
		return MM_FALSE;

	if(mm_is_real(matcode) && mm_is_hermitian(matcode))
		return MM_FALSE;

	if(mm_is_pattern(matcode) && (mm_is_hermitian(matcode) || mm_is_skew(matcode)))
		return MM_FALSE;

	// all property is OK!
	return MM_TRUE;
}

// read banner
int mm_read_banner(FILE *fp, MM_typecode *matcode)
{
	unsigned char line[MM_MAX_LINE_LEN];
	unsigned char banner[MM_MAX_TOKEN_LEN];
	unsigned char mtx[MM_MAX_TOKEN_LEN];
	unsigned char crd[MM_MAX_TOKEN_LEN];
	unsigned char data_type[MM_MAX_TOKEN_LEN];
	unsigned char storage_scheme[MM_MAX_TOKEN_LEN];
	unsigned char *pt;

	// clear type code
	mm_clear_typecode(matcode);

	if(fgets(line, MM_MAX_LINE_LEN, fp) == NULL)
		return mm_print_error(NULL, MM_PREMATURE_EOF);

	if(sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) != 5)
		return mm_print_error(NULL, MM_PREMATURE_EOF);

	// convert to lower case
	for(pt = mtx; *pt != '\0'; *pt = tolower(*pt), pt++); 
	for(pt = crd; *pt != '\0'; *pt = tolower(*pt), pt++); 
	for(pt = data_type; *pt != '\0'; *pt = tolower(*pt), pt++); 
	for(pt = storage_scheme; *pt != '\0'; *pt = tolower(*pt), pt++);

	// (1) check banner
	if(strncmp(banner, MM_BANNER, strlen(MM_BANNER)) != 0)
		return mm_print_error(banner, MM_NO_HEADER);

	// (2) first field shoud be "mtx"
	if(strcmp(mtx, MM_MATRIX_STR) != 0)
		return mm_print_error(mtx, MM_UNSUPPORTED_TYPE);

	mm_set_matrix(matcode);

	// (3) sparse or dense
	if(strcmp(crd, MM_SPARSE_STR) == 0)
		mm_set_sparse(matcode);
	else
	{
		if(strcmp(crd, MM_DENSE_STR) == 0)
			mm_set_dense(matcode);
		else
			return mm_print_error(crd, MM_UNSUPPORTED_TYPE);
	}

#ifdef DEBUG
//	printf("banner        : -%s-\n", banner);
//	printf("mtx           : -%s-\n", mtx);
//	printf("crd           : -%s-\n", crd);
//	printf("data_type     : -%s-\n", data_type);
//	printf("storage_scheme: -%s-\n", storage_scheme);
#endif // DEBUG

	// (4) data type
	if(strcmp(data_type, MM_REAL_STR) == 0)
		mm_set_real(matcode);
	else
	{
		if(strcmp(data_type, MM_COMPLEX_STR) == 0)
			mm_set_complex(matcode);
		else
		{
			if(strcmp(data_type, MM_PATTERN_STR) == 0)
				mm_set_pattern(matcode);
			else
			{
				if(strcmp(data_type, MM_INTEGER_STR) == 0)
					mm_set_integer(matcode);
				else
					return mm_print_error(data_type, MM_UNSUPPORTED_TYPE);
			}
		}
	}

	// (6) storage scheme
	if(strcmp(storage_scheme, MM_GENERAL_STR) == 0)
		mm_set_general(matcode);
	else
	{
		if(strcmp(storage_scheme, MM_SYMMETRIC_STR) == 0)
			mm_set_symmetric(matcode);
		else
		{
			if(strcmp(storage_scheme, MM_HERMITIAN_STR) == 0)
				mm_set_hermitian(matcode);
			else
			{
				if(strcmp(storage_scheme, MM_SKEW_STR) == 0)
					mm_set_skew(matcode);
				else
					return mm_print_error(storage_scheme, MM_UNSUPPORTED_TYPE);
			}
		}
	}

#ifdef DEBUG
	printf("%s\n", mm_typecode_to_str(*matcode));
#endif // DEBUG

	return MM_SUCCESS;
}

// write banner
int mm_write_banner(FILE *fp, MM_typecode matcode)
{
	unsigned char *str = mm_typecode_to_str(matcode);
	int ret_code;

	ret_code = fprintf(fp, "%s %s\n", MM_BANNER, str);

	free(str);

	if(ret_code != 2)
		return mm_print_error(NULL, MM_COULD_NOT_WRITE_FILE);
	else
		return MM_SUCCESS;
}

// read coodinate matrix size
int mm_read_mtx_crd_size(FILE *fp, int *row_dim, int *col_dim, int *num_nonzeros)
{
	unsigned char line[MM_MAX_LINE_LEN];
	int num_items_read;

	*row_dim = 0;
	*col_dim = 0;
	*num_nonzeros = 0;

	// skip comments
	do
	{
		if(fgets(line, MM_MAX_LINE_LEN, fp) == NULL)
			return mm_print_error(NULL, MM_PREMATURE_EOF);
	} while(line[0] == '%');

	// line[] : brank or row_dim, col_dim, num_nonzeros
	if(sscanf(line, "%d %d %d", row_dim, col_dim, num_nonzeros) == 3)
	{
#ifdef DEBUG
//		printf("(1) line = %s\n", line);
//		printf("(1) row_dim, col_dim, num_nonzeros = %d, %d, %d\n", *row_dim, *col_dim, *num_nonzeros);
#endif // DEBUG
		return MM_SUCCESS;
	}
	else
	{
		do
		{
			num_items_read = fscanf(fp, "%d %d %d", row_dim, col_dim, num_nonzeros);

			if(num_items_read == EOF)
				return mm_print_error(NULL, MM_PREMATURE_EOF);

		} while(num_items_read != 3);
#ifdef DEBUG
//		printf("(2) row_dim, col_dim, num_nonzeros = %d, %d, %d\n", *row_dim, *col_dim, *num_nonzeros);
#endif // DEBUG
	}
	
	return MM_SUCCESS;
}

// write coodinate matrix size
int mm_write_crd_size(FILE *fp, int row_dim, int col_dim, int num_nonzeros)
{
	if(fprintf(fp, "%d %d %d\n", row_dim, col_dim, num_nonzeros) != 3)
		return mm_print_error(NULL, MM_COULD_NOT_WRITE_FILE);
	else
		return MM_SUCCESS;
}

// read unsymmetric double precision sparse matrix
// I cannot understand thre reason for being this function. For practical training ?
int mm_read_unsymmetric_sparse(const char *fname, int *row_dim, int *col_dim, int *num_nonzeros, double **val, int **row_index, int **col_index)
{
	FILE *fp;
	MM_typecode matcode;
	int in_row_dim, in_col_dim, in_num_nonzeros, i;
	int *in_row_index, *in_col_index;
	double *in_val;

	// open file
	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "mm_read_unsymmetric_sparse: cannot open %s!\n", fname);
		return MM_ERROR;
	}

	if(mm_read_banner(fp, &matcode) != MM_SUCCESS)
	{
		fprintf(stderr, "mm_read_unsymmetric_sparse: could not process Matrix Market Banner in file [%s]\n", fname);
		return MM_ERROR;
	}
	
	if(!(mm_is_real(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)))
	{
		fprintf(stderr, "Sorry, This application dose not support Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
		return MM_ERROR;
	}
	
	// set the size of sparse matrix
	if(mm_read_mtx_crd_size(fp, &in_row_dim, &in_col_dim, &in_num_nonzeros) != 0)
	{
		fprintf(stderr, "read_unsymmetric_sparse: could not get sparse matrix size\n");
		return MM_ERROR;
	}

	*row_dim = in_row_dim;
	*col_dim = in_col_dim;
	*num_nonzeros = in_num_nonzeros;
	
	// reserve memory for matrix elements
	in_row_index = (int *)calloc(in_num_nonzeros, sizeof(int));
	in_col_index = (int *)calloc(in_num_nonzeros, sizeof(int));
	in_val = (double *)calloc(in_num_nonzeros, sizeof(double));

	*val = in_val;
	*row_index = in_row_index;
	*col_index = in_col_index;

	// read values of matrix
	for(i = 0; i < in_num_nonzeros; i++)
	{
		fscanf(fp, "%d %d %lg\n", &in_row_index[i], &in_col_index[i], &in_val[i]);

		// 1-base index to 0-base index
		in_row_index[i]--;
		in_col_index[i]--;
	}

	return MM_SUCCESS;
}

// read mtx array size
int mm_read_mtx_array_size(FILE *fp, int *row_dim, int *col_dim)
{
	unsigned char line[MM_MAX_LINE_LEN];
	int num_items_read;

	*row_dim = 0;
	*col_dim = 0;

	// unread comments
	do
	{
		if(fgets(line, MM_MAX_LINE_LEN, fp) == NULL)
			return mm_print_error(NULL, MM_PREMATURE_EOF);
	} while(line[0] == '%');

	// blank or row_dim, col_dim
	if(sscanf(line, "%d %d", row_dim, col_dim) == 2)
	{
#ifdef DEBUG
//		printf("(1) line = %s\n", line);
//		printf("(1) row_dim, col_dim = %d, %d\n", *row_dim, *col_dim);
#endif // DEBUG
		return MM_SUCCESS;
	}
	else
	{
		do
		{
			num_items_read = fscanf(fp, "%d %d", row_dim, col_dim);
			if(num_items_read == EOF)
				return mm_print_error(NULL, MM_PREMATURE_EOF);
		} while(num_items_read != 2);
	}
#ifdef DEBUG
//	printf("row_dim, col_dim = %d, %d\n", *row_dim, *col_dim);
#endif // DEBUG

	return MM_SUCCESS;
}

// write mtx array size
int mm_write_mtx_array_size(FILE *fp, int row_dim, int col_dim)
{
	if(fprintf(fp, "%d %d\n", row_dim, col_dim) != 2)
		return mm_print_error(NULL, MM_COULD_NOT_WRITE_FILE);
	else
		return MM_SUCCESS;
}

// read coodinate matrix when row_index, col_index, val are already allocated
int mm_read_mtx_crd_data(FILE *fp, int row_dim, int col_dim, int num_nonzeros, int row_index[], int col_index[], double val[], MM_typecode matcode)
{
	int i;

	if(mm_is_complex(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
		{
			if(fscanf(fp, "%d %d %lg %lg", &row_index[i], &col_index[i], &val[2 * i], &val[2 * i + 1]) != 4)
				return mm_print_error(NULL, MM_PREMATURE_EOF);
		}
	}
	else if(mm_is_real(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
		{
			if(fscanf(fp, "%d %d %lg", &row_index[i], &col_index[i], &val[i]) != 3)
				return mm_print_error(NULL, MM_PREMATURE_EOF);
		}
	}
	else if(mm_is_pattern(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
		{
			if(fscanf(fp, "%d %d", &row_index[i], &col_index[i]) != 2)
				return mm_print_error(NULL, MM_PREMATURE_EOF);
		}
	}
	else
		return mm_print_error(NULL, MM_UNSUPPORTED_TYPE);

	return MM_SUCCESS;
}

// read one line(entry) of coodinate matrix 
int mm_read_mtx_crd_entry(FILE *fp, int *row_index, int *col_index, double *real, double *imag, MM_typecode matcode)
{
	if(mm_is_complex(matcode))
	{
		if(fscanf(fp, "%d %d %lg %lg", row_index, col_index, real, imag) != 4)
			return mm_print_error(NULL, MM_PREMATURE_EOF);
	}
	else if(mm_is_real(matcode))
	{
		if(fscanf(fp, "%d %d %lg", row_index, col_index, real) != 3)
			return mm_print_error(NULL, MM_PREMATURE_EOF);
	}
	else if(mm_is_pattern(matcode))
	{
		if(fscanf(fp, "%d %d", row_index, col_index) != 2)
			return mm_print_error(NULL, MM_PREMATURE_EOF);
	}
	else
		return mm_print_error(NULL, MM_UNSUPPORTED_TYPE);

	return MM_SUCCESS;
}

// read any kinds of coodinate matrix
// After finishing to read the whole data, mm_read_mtx_crd fills
// row_dim, col_dim, num_nonzeros, row_index, col_index and array of values,
// and return typecode such as "MCRS" which means "Matrix, Coordinate, Real, Symmetric".
int mm_read_mtx_crd(const char *fname, int *row_dim, int *col_dim, int *num_nonzeros, int **row_index, int **col_index, double **val, MM_typecode *matcode)
{
	FILE *fp;
	int ret_code;

	// support standard input (stdin)
	if(strcmp(fname, "stdin") == 0)
		fp = stdin;
	else
	{
		// open file
		fp = fopen(fname, "r");
		if(fp == NULL)
		{
			fprintf(stderr, "mm_read_mtx_crd: cannot open %s!\n", fname);
			return MM_COULD_NOT_READ_FILE;
		}
	}

	ret_code = mm_read_banner(fp, matcode);
	if(ret_code != MM_SUCCESS)
	{
		fprintf(stderr, "mm_read_mtx_crd: could not process Matrix Market Banner in file [%s]\n", fname);
		return ret_code;
	}
	
	if(!(mm_is_valid(*matcode) && mm_is_matrix(*matcode) && mm_is_sparse(*matcode)))
	{
		fprintf(stderr, "mm_read_mtx_crd: Sorry, This application dose not support Matrix Market type: [%s]\n", mm_typecode_to_str(*matcode));
		return MM_UNSUPPORTED_TYPE;
	}

	// set the size of sparse matrix
	ret_code = mm_read_mtx_crd_size(fp, row_dim, col_dim, num_nonzeros);
	if(ret_code != 0)
	{
		fprintf(stderr, "mm_read_mtx_crd: could not get sparse matrix size\n");
		return ret_code;
	}

#ifdef DEBUG
//	printf("row_dim, col_dim, num_nonzeros = %d, %d, %d\n", *row_dim, *col_dim, *num_nonzeros);
#endif // DEBUG

	// reserve memory for matrix elements
	*row_index = (int *)calloc(*num_nonzeros, sizeof(int));
	*col_index = (int *)calloc(*num_nonzeros, sizeof(int));

	if(mm_is_complex(*matcode))
	{
		*val = (double *)calloc((*num_nonzeros) * 2, sizeof(double));
		if(val == NULL)
		{
			fprintf(stderr, "cannot allocate matrix elements!(complex: %d x %d)\n", *row_dim, *col_dim);
			return MM_ERROR;
		}

		ret_code = mm_read_mtx_crd_data(fp, *row_dim, *col_dim, *num_nonzeros, *row_index, *col_index, *val, *matcode);
		if(ret_code != MM_SUCCESS)
			return ret_code;
	}
	else if(mm_is_real(*matcode))
	{
		*val = (double *)calloc((*num_nonzeros), sizeof(double));
		if(val == NULL)
		{
			fprintf(stderr, "cannot allocate matrix elements!(real: %d x %d)\n", *row_dim, *col_dim);
			return MM_ERROR;
		}

		ret_code = mm_read_mtx_crd_data(fp, *row_dim, *col_dim, *num_nonzeros, *row_index, *col_index, *val, *matcode);
		if(ret_code != MM_SUCCESS)
			return ret_code;
	}
	else if(mm_is_pattern(*matcode))
	{
		ret_code = mm_read_mtx_crd_data(fp, *row_dim, *col_dim, *num_nonzeros, *row_index, *col_index, *val, *matcode);
		if(ret_code != MM_SUCCESS)
			return ret_code;
	}

	if(fp != stdin)
		fclose(fp);

	return MM_SUCCESS;
}

// write mtx format file
int mm_write_mtx_crd(unsigned char *fname, int row_dim, int col_dim, int num_nonzeros, int row_index[], int col_index[], double val[], MM_typecode matcode)
{
	FILE *fp;
	int i;

	// support standard output (stdout)
	if(strcmp(fname, "stdout") == 0)
		fp = stdout;
	else
	{
		fp = fopen(fname, "w");
		if(fp == NULL)
			return mm_print_error(fname, MM_COULD_NOT_WRITE_FILE);
	}

	// print header of MTX format
	fprintf(fp, "%s ", MM_BANNER);
	fprintf(fp, "%s\n", mm_typecode_to_str(matcode));
	fprintf(fp, "%d %d %d\n", row_dim, col_dim, num_nonzeros);

	// print values of matrix
	if(mm_is_pattern(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
			fprintf(fp, "%d %d\n", row_index[i], col_index[i]);
	}
	else if(mm_is_real(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
			fprintf(fp, "%d %d %20.16g\n", row_index[i], col_index[i], val[i]);
	}
	else if(mm_is_complex(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
			fprintf(fp, "%d %d %20.16g %20.16g\n", row_index[i], col_index[i], val[2 * i], val[2 * i + 1]);
	}
	else
	{
		if(fp != stdout)
			fclose(fp);
		return mm_print_error(fname, MM_UNSUPPORTED_TYPE);
	}

	if(fp != stdout)
		fclose(fp);

	return MM_SUCCESS;
}

// read array type matrix when val is already allocated
int mm_read_mtx_array_data(FILE *fp, int row_dim, int col_dim, double val[], MM_typecode matcode)
{
	int i, num_nonzeros = row_dim * col_dim;

	if(mm_is_complex(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
		{
			if(fscanf(fp, "%lg %lg", &val[2 * i], &val[2 * i + 1]) != 2)
				return mm_print_error(NULL, MM_PREMATURE_EOF);
		}
	}
	else if(mm_is_real(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
		{
			if(fscanf(fp, "%lg", &val[i]) != 1)
				return mm_print_error(NULL, MM_PREMATURE_EOF);
		}
	}
	else
		return mm_print_error(NULL, MM_UNSUPPORTED_TYPE);

	return MM_SUCCESS;
}

// read one line(entry) of coodinate matrix 
int mm_read_mtx_array_entry(FILE *fp, double *real, double *imag, MM_typecode matcode)
{
	int i;

	if(mm_is_complex(matcode))
	{
		if(fscanf(fp, "%lg %lg", real, imag) != 2)
			return mm_print_error(NULL, MM_PREMATURE_EOF);
	}
	else if(mm_is_real(matcode))
	{
		if(fscanf(fp, "%lg", real) != 1)
			return mm_print_error(NULL, MM_PREMATURE_EOF);
	}
	else
		return mm_print_error(NULL, MM_UNSUPPORTED_TYPE);

	return MM_SUCCESS;
}

// read general real or complex array matrix
// After finishing to read the whole data, mm_read_mtx_crd fills
// row_dim, col_dim and array of values,
// and return typecode such as "MARG" which means "Matrix, Array, Real, General".
int mm_read_mtx_array(const char *fname, int *row_dim, int *col_dim, double **val, MM_typecode *matcode)
{
	FILE *fp;
	int ret_code, num_nonzeros;

	// support standard input (stdin)
	if(strcmp(fname, "stdin") == 0)
		fp = stdin;
	else
	{
		// open file
		fp = fopen(fname, "r");
		if(fp == NULL)
		{
			fprintf(stderr, "mm_read_mtx_array: cannot open %s!\n", fname);
			return MM_COULD_NOT_READ_FILE;
		}
	}

	ret_code = mm_read_banner(fp, matcode);
	if(ret_code != MM_SUCCESS)
	{
		fprintf(stderr, "mm_read_mtx_array: could not process Matrix Market Banner in file [%s]\n", fname);
		return ret_code;
	}
	
	if(!(mm_is_valid(*matcode) && mm_is_matrix(*matcode) && mm_is_array(*matcode) || mm_is_symmetric(*matcode) || mm_is_skew(*matcode)))
	{
		fprintf(stderr, "mm_read_mtx_array: Sorry, This application dose not support Matrix Market type: [%s]\n", mm_typecode_to_str(*matcode));
		return MM_UNSUPPORTED_TYPE;
	}

	// set the size of dense matrix
	ret_code = mm_read_mtx_array_size(fp, row_dim, col_dim);
	if(ret_code != 0)
	{
		fprintf(stderr, "mm_read_mtx_array: could not get dense matrix size\n");
		return ret_code;
	}

#ifdef DEBUG
//	printf("row_dim, col_dim = %d, %d\n", *row_dim, *col_dim);
#endif // DEBUG

	num_nonzeros = (*row_dim) * (*col_dim);

	if(mm_is_complex(*matcode))
	{
		*val = (double *)calloc(num_nonzeros * 2, sizeof(double));
		if(val == NULL)
		{
			fprintf(stderr, "cannot allocate matrix elements!(complex: %d x %d)\n", *row_dim, *col_dim);
			return MM_ERROR;
		}

		ret_code = mm_read_mtx_array_data(fp, *row_dim, *col_dim, *val, *matcode);
		if(ret_code != MM_SUCCESS)
			return ret_code;
	}
	else if(mm_is_real(*matcode))
	{
		*val = (double *)calloc(num_nonzeros, sizeof(double));
		if(val == NULL)
		{
			fprintf(stderr, "cannot allocate matrix elements!(real: %d x %d)\n", *row_dim, *col_dim);
			return MM_ERROR;
		}

		ret_code = mm_read_mtx_array_data(fp, *row_dim, *col_dim, *val, *matcode);
		if(ret_code != MM_SUCCESS)
			return ret_code;
	}

	if(fp != stdin)
		fclose(fp);

	return MM_SUCCESS;
}

// write mtx format file
int mm_write_mtx_array(unsigned char *fname, int row_dim, int col_dim, double val[], MM_typecode matcode)
{
	FILE *fp;
	int i, num_nonzeros = row_dim * col_dim;

	// support standard output (stdout)
	if(strcmp(fname, "stdout") == 0)
		fp = stdout;
	else
	{
		fp = fopen(fname, "w");
		if(fp == NULL)
			return mm_print_error(fname, MM_COULD_NOT_WRITE_FILE);
	}

	// print header of MTX format
	fprintf(fp, "%s ", MM_BANNER);
	fprintf(fp, "%s\n", mm_typecode_to_str(matcode));
	fprintf(fp, "%d %d\n", row_dim, col_dim);

	// print values of matrix
	if(mm_is_real(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
			fprintf(fp, "%20.16g\n", val[i]);
	}
	else if(mm_is_complex(matcode))
	{
		for(i = 0; i < num_nonzeros; i++)
			fprintf(fp, "%20.16g %20.16g\n", val[2 * i], val[2 * i + 1]);
	}
	else
	{
		if(fp != stdout)
			fclose(fp);
		return mm_print_error(fname, MM_UNSUPPORTED_TYPE);
	}

	if(fp != stdout)
		fclose(fp);

	return MM_SUCCESS;
}

// read any kinds of coodinate matrix within specified number of lines
int mm_print_header_mtx_crd(const char *fname, int num_lines_print)
{
	FILE *fp;
	int ret_code;
	MM_typecode matcode;
	int row_dim, col_dim, num_nonzeros, i;
	int *row_index, *col_index;
	double *val;

	// support standard input (stdin)
	if(strcmp(fname, "stdin") == 0)
		fp = stdin;
	else
	{
		// open file
		fp = fopen(fname, "r");
		if(fp == NULL)
		{
			fprintf(stderr, "mm_print_header_mtx_crd: cannot open %s!\n", fname);
			return MM_COULD_NOT_READ_FILE;
		}
	}

	ret_code = mm_read_banner(fp, &matcode);
	if(ret_code != MM_SUCCESS)
	{
		fprintf(stderr, "mm_print_header_mtx_crd: could not process Matrix Market Banner in file [%s]\n", fname);
		return ret_code;
	}
	
	if(!(mm_is_valid(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)))
	{
		fprintf(stderr, "mm_print_header_mtx_crd: Sorry, This application dose not support Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
		return MM_UNSUPPORTED_TYPE;
	}

	// set the size of sparse matrix
	ret_code = mm_read_mtx_crd_size(fp, &row_dim, &col_dim, &num_nonzeros);
	if(ret_code != 0)
	{
		fprintf(stderr, "mm_print_header_mtx_crd: could not get sparse matrix size\n");
		return ret_code;
	}

#ifdef DEBUG
//	printf("row_dim, col_dim, num_nonzeros = %d, %d, %d\n", *row_dim, *col_dim, *num_nonzeros);
#endif // DEBUG

	printf("Matrix type : %s\n", mm_typecode_to_str(matcode));
	printf("row_dim     : %d\n", row_dim);
	printf("col_dim     : %d\n", col_dim);
	printf("num_nonzeros: %d\n", num_nonzeros);

	num_nonzeros = (num_nonzeros < num_lines_print) ? num_nonzeros : num_lines_print;

	// reserve memory for matrix elements
	row_index = (int *)calloc(num_nonzeros, sizeof(int));
	col_index = (int *)calloc(num_nonzeros, sizeof(int));

	if(mm_is_complex(matcode))
	{
		val = (double *)calloc(num_nonzeros * 2, sizeof(double));
		if(val == NULL)
		{
			fprintf(stderr, "cannot allocate matrix elements!(complex: %d x %d)\n", row_dim, col_dim);
			return MM_ERROR;
		}

		ret_code = mm_read_mtx_crd_data(fp, row_dim, col_dim, num_nonzeros, row_index, col_index, val, matcode);
		if(ret_code != MM_SUCCESS)
		{
			free(row_index);
			free(col_index);
			free(val);

			return ret_code;
		}

		for(i = 0; i < num_nonzeros; i++)
			printf("%d %d %20.16g %20.16g\n", row_index[i], col_index[i], val[2 * i], val[2 * i + 1]);

		free(row_index);
		free(col_index);
		free(val);
	}
	else if(mm_is_real(matcode))
	{
		val = (double *)calloc(num_nonzeros, sizeof(double));
		if(val == NULL)
		{
			fprintf(stderr, "cannot allocate matrix elements!(real: %d x %d)\n", row_dim, col_dim);
			return MM_ERROR;
		}

		ret_code = mm_read_mtx_crd_data(fp, row_dim, col_dim, num_nonzeros, row_index, col_index, val, matcode);
		if(ret_code != MM_SUCCESS)
		{
			free(row_index);
			free(col_index);
			free(val);

			return ret_code;
		}

		for(i = 0; i < num_nonzeros; i++)
			printf("%d %d %20.16g\n", row_index[i], col_index[i], val[i]);

		free(row_index);
		free(col_index);
		free(val);
	}
	else if(mm_is_pattern(matcode))
	{
		ret_code = mm_read_mtx_crd_data(fp, row_dim, col_dim, num_nonzeros, row_index, col_index, val, matcode);
		if(ret_code != MM_SUCCESS)
		{
			free(row_index);
			free(col_index);

			return ret_code;
		}

		for(i = 0; i < num_nonzeros; i++)
			printf("%d %d\n", row_index[i], col_index[i]);

		free(row_index);
		free(col_index);
	}

	if(fp != stdin)
		fclose(fp);

	return MM_SUCCESS;
}

// read any kinds of coodinate matrix within specified number of lines
int mm_print_header_mtx_array(const char *fname, int num_lines_print)
{
	FILE *fp;
	int ret_code;
	MM_typecode matcode;
	int row_dim, col_dim, num_nonzeros, i;
	double real, imag;

	// support standard input (stdin)
	if(strcmp(fname, "stdin") == 0)
		fp = stdin;
	else
	{
		// open file
		fp = fopen(fname, "r");
		if(fp == NULL)
		{
			fprintf(stderr, "mm_print_header_mtx_array: cannot open %s!\n", fname);
			return MM_COULD_NOT_READ_FILE;
		}
	}

	ret_code = mm_read_banner(fp, &matcode);
	if(ret_code != MM_SUCCESS)
	{
		fprintf(stderr, "mm_print_header_mtx_array: could not process Matrix Market Banner in file [%s]\n", fname);
		return ret_code;
	}
	
	if(!(mm_is_valid(matcode) && mm_is_matrix(matcode) && mm_is_dense(matcode)))
	{
		fprintf(stderr, "mm_print_header_mtx_array: Sorry, This application dose not support Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
		return MM_UNSUPPORTED_TYPE;
	}

	// set the size of sparse matrix
	ret_code = mm_read_mtx_array_size(fp, &row_dim, &col_dim);
	if(ret_code != 0)
	{
		fprintf(stderr, "mm_print_header_mtx_array: could not get dense matrix size\n");
		return ret_code;
	}

	num_nonzeros = row_dim * col_dim;

#ifdef DEBUG
	printf("row_dim, col_dim, num_nonzeros = %d, %d, %d\n", row_dim, col_dim, num_nonzeros);
#endif // DEBUG

	printf("Matrix type : %s\n", mm_typecode_to_str(matcode));
	printf("row_dim     : %d\n", row_dim);
	printf("col_dim     : %d\n", col_dim);
	num_nonzeros = (num_nonzeros < num_lines_print) ? num_nonzeros : num_lines_print;

	for(i = 0; i < num_nonzeros; i++)
	{
		mm_read_mtx_array_entry(fp, &real, &imag, matcode);
		if(mm_is_complex(matcode))
			printf("%20.16g %20.16g\n", real, imag);
		else if(mm_is_real(matcode))
			printf("%20.16g\n", real);
	}

	if(fp != stdin)
		fclose(fp);

	return MM_SUCCESS;
}


#ifdef DEBUG

int main(int argc, char *argv[])
{
	int i, j, row_dim, col_dim, num_nonzeros, ret;
	int *row_index, *col_index;
	double *element;
	MM_typecode matcode;

	// print usage
	if(argc <= 2)
	{
		printf("[usage] %s [\"crd\" or \"array\"] [MatrixMarket file name] [fname to write]\n", argv[0]);
		return 0;
	}

	// coordinate format
	if(strncmp(argv[1], "crd", 6) == 0)
	{
		// print info and 10 lines of elements
		mm_print_header_mtx_crd(argv[2], 10);

		printf("Open %s...\n", argv[2]);
		//ret = mm_read_unsymmetric_sparse(argv[2], &row_dim, &col_dim, &num_nonzeros, &element, &row_index, &col_index);
		ret = mm_read_mtx_crd(argv[2], &row_dim, &col_dim, &num_nonzeros, &row_index, &col_index, &element, &matcode);
		if(ret == MM_ERROR)
		{
			printf("MM_ERROR!\n");
			return MM_ERROR;
		}

		printf("matcode       : %c %c %c %c\n", matcode[0], matcode[1], matcode[2], matcode[3]);
		printf("matcode string: %s\n", mm_typecode_to_str(matcode));
		printf("row_dim, col_dim, num_nonzeros = %d, %d, %d\n", row_dim, col_dim, num_nonzeros);

		// print elements
/*		for(i = 0; i < num_nonzeros; i++)
			printf("mat(%d, %d) = %25.17e\n", row_index[i], col_index[i], element[i]);
*/
		// write test
		if(argc >= 4)
		{
			printf("Write %s...\n", argv[3]);
			mm_write_mtx_crd(argv[3], row_dim, col_dim, num_nonzeros, row_index, col_index, element, matcode);
		}
	}
	// coordinate format
	if(strncmp(argv[1], "array", 6) == 0)
	{
		// print info and 10 lines of elements
		mm_print_header_mtx_array(argv[2], 10);

		printf("Open %s...\n", argv[2]);
		//ret = mm_read_unsymmetric_sparse(argv[2], &row_dim, &col_dim, &num_nonzeros, &element, &row_index, &col_index);
		ret = mm_read_mtx_array(argv[2], &row_dim, &col_dim, &element, &matcode);
		if(ret == MM_ERROR)
		{
			printf("MM_ERROR!\n");
			return MM_ERROR;
		}

		printf("matcode       : %c %c %c %c\n", matcode[0], matcode[1], matcode[2], matcode[3]);
		printf("matcode string: %s\n", mm_typecode_to_str(matcode));
		printf("row_dim, col_dim = %d, %d\n", row_dim, col_dim);

		// print elements
/*		for(i = 0; i < row_dim; i++)
			for(j = 0; j < col_dim; j++)
				printf("mat(%d, %d) = %25.17e\n", i, j, element[i * col_dim + j]);
*/
		// write test
		if(argc >= 4)
		{
			printf("Write %s...\n", argv[3]);
			mm_write_mtx_array(argv[3], row_dim, col_dim, element, matcode);
		}
	}

	return 0;
}
#endif // DEBUG
