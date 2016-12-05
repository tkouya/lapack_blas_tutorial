/********************************************************************************/
/* mycuda_daxpy : Original DAXPY based on CUDA                                  */
/* Copyright (C) 2015 Tomonori Kouya                                            */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
#include <stdio.h>

#include "cblas.h"
#include "mycuda.h"
#include "tkaux.h"

// y := alpha * x + y
__global__ void  mycuda_daxpy_kernel (int dim, double *ptr_alpha, double x[], int x_step_dim, double y[], int y_step_dim)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	int x_index = k * x_step_dim;
	int y_index = k * y_step_dim;

	if ((x_index < dim) && (y_index < dim))
	{
		y[y_index] = *ptr_alpha * x[x_index] + y[y_index];
	}
}

// Maximum threads per a block
#define MAX_NUM_THREADS_CUDA 8

// c := a * b
void  mycuda_daxpy(int dim, double *dev_alpha, double dev_x[], int x_step_dim, double dev_y[], int y_step_dim)
{
	dim3 threads(MAX_NUM_THREADS_CUDA);
	dim3 blocks(1);

	threads.x = (dim > MAX_NUM_THREADS_CUDA) ? MAX_NUM_THREADS_CUDA : dim;

	blocks.x = (dim > MAX_NUM_THREADS_CUDA) ? (dim / MAX_NUM_THREADS_CUDA) + 1 : 1;

	printf("Threads (x): %d\n", threads.x);
	printf("Blocks  (x): %d\n", blocks.x);

	mycuda_daxpy_kernel<<<blocks, threads>>>(dim, dev_alpha, dev_x, x_step_dim, dev_y, y_step_dim);
}

int main()
{
	int i, dim;
	double host_alpha, *host_x, *host_y; // on CPU
	double *dev_alpha, *dev_x, *dev_y;   // on GPU

	printf("dim = "); scanf("%d", &dim);

	host_x = (double *)calloc(dim, sizeof(double));
	host_y = (double *)calloc(dim, sizeof(double));

	// alpha = sqrt(2)
	// x[i] = sqrt(2) * i
	// y[i] = sqrt(3) * (dim - i)

	host_alpha = sqrt(2.0);
	for(i = 0; i < dim; i++)
	{
		host_x[i] = sqrt(2.0) * (double)(i + 1);
		host_y[i] = sqrt(3.0) * (double)(dim - i);
	}

	// host to device
	dev_alpha = (double *)mycuda_calloc(1, sizeof(double));
	dev_x = (double *)mycuda_calloc(dim, sizeof(double));
	dev_y = (double *)mycuda_calloc(dim, sizeof(double));

	// host_x -> dev_x
	// host_y -> dev_y
	cudaMemcpy((void *)dev_alpha, (void *)&host_alpha, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev_x, (void *)host_x, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev_y, (void *)host_y, dim * sizeof(double), cudaMemcpyHostToDevice);

	// y := alpha * x + y on CPU
	cblas_daxpy(dim, host_alpha, host_x, 1, host_y, 1);

	printf_dvector("%d %25.17e\n", host_y, dim, 1);

	// y := alpha * x + y on GPU
	mycuda_daxpy(dim, dev_alpha, dev_x, 1, dev_y, 1);

	// dev_y -> host_x
	cudaMemcpy((void *)host_x, (void *)dev_y, dim * sizeof(double), cudaMemcpyDeviceToHost);

	printf_dvector("%d %25.17e\n", host_x, dim, 1);

	mycuda_free(dev_x);
	mycuda_free(dev_y);

	free(host_x);
	free(host_y);

	return 0;
}
