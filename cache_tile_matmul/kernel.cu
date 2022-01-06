
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <cassert>

#include "gputimer.h"

using std::vector;
const int THREADS = 16;
const int SHMEM_SIZE = THREADS * THREADS * sizeof(int);

__global__ void matMul(int *c, int *b, int *a, int N_a, int N_b, int N)
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	//boundary check
	if (row < N_a && col < N_b) {
		int sum = 0;

		for (int k = 0; k < N; k++) {
			sum += a[N*row + k] * b[N_b*k + col];
		}

		c[N_b*row + col] = sum;
	}
}

__global__ void transposeMatTiled(int *b, int *a, int a_x, int a_y) {
	__shared__ int A[SHMEM_SIZE];

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	int out_corner_y = blockIdx.x * blockDim.x;
	int out_corner_x = blockIdx.y*blockDim.y;

	if (row < a_y && col < a_x) {
		A[blockDim.x*threadIdx.x + threadIdx.y] = a[row * a_x + col]; //transpose tile when copying to shared mem

		__syncthreads();

		b[a_y * (out_corner_y + threadIdx.y) + (out_corner_x + threadIdx.x)] = A[blockDim.x*threadIdx.y + threadIdx.x];
	}
}

__global__ void transposeMat(int *b, int *a, int a_x, int a_y) {
	__shared__ int A[SHMEM_SIZE];

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row < a_y && col < a_x) {
		b[a_y * col + row] = a[row * a_x + col];
	}
}


__global__ void matMulTiled(int *c, int *b, int *a, int N) {
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int row = by * blockDim.x + ty;
	int col = bx * blockDim.y + tx;

	int temp = 0;

	for (int i = 0; i < N; i+= blockDim.x) {
		A[(ty * blockDim.x) + tx] = a[row * N + i + tx];
		B[(ty * blockDim.x) + tx] = b[i * N + ty * N + col];

		__syncthreads();

		for (int j = 0; j < blockDim.x; j++) {
			temp += (A[(ty * blockDim.x) + j] * B[(j * blockDim.x) + tx]);
		}

		__syncthreads();
	}

	c[(row * N) + col] = temp;
	
}

void testTranspose(int ARRAY_SIZE_A_x, int ARRAY_SIZE_A_y) {
	GpuTimer timer;

	int *h_a;
	int *h_b;

	int bytes_a = sizeof(int) * ARRAY_SIZE_A_x * ARRAY_SIZE_A_y;

	h_a = (int*)malloc(bytes_a);
	h_b = (int*)malloc(bytes_a);

	for (int i = 0; i < ARRAY_SIZE_A_y; i++) {
		for (int j = 0; j < ARRAY_SIZE_A_x; j++) {
			h_a[ARRAY_SIZE_A_x * i + j] = rand() % 100;
		}
	}

	int *d_a, *d_b;
	cudaMalloc(&d_a, bytes_a);
	cudaMalloc(&d_b, bytes_a);

	cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_b, h_b, bytes_a, cudaMemcpyHostToDevice);

	//threads per dim
	//int THREADS = 16;
	int BLOCKS_x = (int)(ARRAY_SIZE_A_x - 1) / THREADS + 1;
	int BLOCKS_y = (int)(ARRAY_SIZE_A_y - 1) / THREADS + 1;

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS_x, BLOCKS_y);

	timer.Start();
	transposeMat<<<blocks, threads>>>(d_b, d_a, ARRAY_SIZE_A_x, ARRAY_SIZE_A_y);
	timer.Stop();

	std::cout << "transpose_parallel_per_row: %g ms.\nVerifying transpose...%s\n" << timer.Elapsed() << "\n";

	cudaMemcpy(h_b, d_b, bytes_a, cudaMemcpyDeviceToHost);
	/*
	for (int i = 0; i < ARRAY_SIZE_A_y; i++) {
		for (int j = 0; j < ARRAY_SIZE_A_x; j++) {
			std::cout << h_a[ARRAY_SIZE_A_x*i + j] << " ";
		}
		std::cout << "\n";
	}

	for (int j = 0; j < ARRAY_SIZE_A_x; j++) {
		for (int i = 0; i < ARRAY_SIZE_A_y; i++) {
			std::cout << h_b[ARRAY_SIZE_A_y*j + i] << " ";
		}
		std::cout << "\n";
	}
	*/
	for (int j = 0; j < ARRAY_SIZE_A_x; j++) {
		for (int i = 0; i < ARRAY_SIZE_A_y; i++) {
			assert(h_b[ARRAY_SIZE_A_y*j + i] == h_a[ARRAY_SIZE_A_x*i + j]);
		}
	}


	
}

int main()
{
	int const ARRAY_SIZE_A_x = 1024;
	int const ARRAY_SIZE_B_y = 8;
	int const ARRAY_SIZE_A_y = 1024;
	int const ARRAY_SIZE_B_x = 8;
	
	testTranspose(ARRAY_SIZE_A_x, ARRAY_SIZE_A_y); //square matrix
	testTranspose(ARRAY_SIZE_A_x/2, ARRAY_SIZE_A_y); //tall matrix
	testTranspose(ARRAY_SIZE_A_x, ARRAY_SIZE_A_y/2); //fat matrix
	/*
	assert(ARRAY_SIZE_A_x == ARRAY_SIZE_B_y);

	int bytes_a = sizeof(int) * ARRAY_SIZE_A_x * ARRAY_SIZE_A_y;
	int bytes_b = sizeof(int) * ARRAY_SIZE_B_x * ARRAY_SIZE_B_y;
	int bytes_c = sizeof(int) * ARRAY_SIZE_A_y * ARRAY_SIZE_B_x;

	int *h_a;
	int *h_b;
	
	int *h_c;

	h_a = (int*)malloc(bytes_a);
	h_b = (int*)malloc(bytes_b);
	h_c = (int*)malloc(bytes_c);

	//initialize matrices
	for (int i = 0; i < ARRAY_SIZE_A_y; i++) {
		for (int j = 0; j < ARRAY_SIZE_A_x; j++) {
			h_a[ARRAY_SIZE_A_x * i + j] = rand() % 100;
		}
	}

	for (int i = 0; i < ARRAY_SIZE_B_y; i++) {
		for (int j = 0; j < ARRAY_SIZE_B_x; j++) {
			h_b[ARRAY_SIZE_B_x * i + j] = rand() % 100;
		}
	}
	
	//device pointers
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes_a);
	cudaMalloc(&d_b, bytes_b);
	cudaMalloc(&d_c, bytes_c);

	cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

	//threads per dim
	int THREADS = 16;
	int BLOCKS_y = (int)(ARRAY_SIZE_B_x - 1) / THREADS + 1;
	int BLOCKS_x = (int)(ARRAY_SIZE_A_y - 1) / THREADS + 1;

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS_y, BLOCKS_x);

	matMul<<<blocks, threads>>>(d_c, d_b, d_a, ARRAY_SIZE_A_y, ARRAY_SIZE_B_x, ARRAY_SIZE_A_x);

	cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	
	// For every row...
	for (int i = 0; i < ARRAY_SIZE_A_y; i++) {
		// For every column...
		for (int j = 0; j < ARRAY_SIZE_B_x; j++) {
			// For every element in the row-column pair
			int tmp = 0;
			for (int k = 0; k < ARRAY_SIZE_A_x; k++) {
				// Accumulate the partial results
				tmp += (h_a[ARRAY_SIZE_A_x * i + k] * h_b[ARRAY_SIZE_B_x * k + j]);
			}

			// Check against the CPU result
			assert(tmp == h_c[ARRAY_SIZE_B_x * i + j]);
		}
	}
	*/
	std::cout << "Completed Successfully!";

	return 0;
}

