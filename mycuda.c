//
// mycuda.c ... CUDA用基礎ルーチン
// Copyright (c) 2015 T.Kouya
//

#include "mycuda.h"

// GPU上に行列格納領域を確保
void *mycuda_calloc(int num_elements, size_t size_element)
{
	cudaError_t cuda_error;
	void *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, num_elements * size_element);

	if(cuda_error != cudaSuccess)
	{
		printf("device memory allocation failed!(num_elements = %d, size = %d)\n", num_elements, (int)size_element);
		return NULL;
	}

	return ret;
}

// GPU上のメモリ領域を解放
void mycuda_free(void *mem)
{
	cudaFree(mem);
}

