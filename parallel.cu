#include <bits/stdc++.h>
#include <cuda.h>
#include <stdlib.h>

#define WARP_SIZE 16

// CUDA MULTIPLY
// do matrix-matrix multiplication
__global__ void cuda_mat_multiply(const double* A, const double* B, double * C,
                                    int rowsa, int colsa,
                                    int rowsb, int colsb,
                                    int rowsc, int colsc){
    __shared__ double sA[32][32];   // Tile size of 32x32
    __shared__ double sB[32][32];
    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    double Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;
    for (int k = 0; k < (((colsa - 1)/ 32) + 1); k++){
        if ( (Row < rowsa) && (threadIdx.x + (k*32)) < colsa){
            sA[threadIdx.y][threadIdx.x] = A[(Row*colsa) + threadIdx.x + (k*32)];
        }
        else{
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();
        if ( Col < colsb && (threadIdx.y + k*32) < rowsb){
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*32)*colsb + Col];
        }
        else{
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < 32; ++j){
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (Row < rowsc && Col < colsc){
        C[Row*colsc + Col] = Cvalue;
    }
}

// CUDA MATRIX MATRIX ADDITION
// c = a + b, n is size of a
__global__ void cu_addition(const double *A, const double *B, double *C, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		C[tid] = __fadd_rd(A[tid], B[tid]);
		tid += stride;
	}
}


// CUDA TRANSPOSE
// do matrix transpose
__global__ void cuda_mat_transpose(const double* src, double* dst, int colssrc, int colsdst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int cdst = tid % colsdst;
		int rdst = tid / colsdst;
		int rsrc = cdst;
		int csrc = rdst;
		dst[tid] = src[rsrc * colssrc + csrc];
		tid += stride;
	}
}

// CUDA SIGMOID
// sigmoid non-linearity
__global__ void cu_sigmoid(double* src, double* dst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		double tmp = __fmul_rd(src[tid], -1.0);
		tmp = __expf(tmp);
		tmp = __fadd_rd(tmp, 1.0);
		dst[tid] = __fdividef(1.0, tmp);
		tid += stride;
	}
}

// CUDA MATRIX SCALAR ADDITION
// a += b, n is size of a
__global__ void cu_mat_scalar_addition(double *A, const double b, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fadd_rd(A[tid], b);
		tid += stride;
	}
}

// CUDA MATRIX SCALAR MULTIPLY
// a(i) *= b, n is size of a
__global__ void cu_mat_scalar_multiply(double *A, double B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fmul_rd(A[tid], B);
		tid += stride;
	}
}

// CUDA MATRIX SCALAR DIVIDE
// a(i) /= b, n is size of a
__global__ void cu_mat_scalar_divide(double *A, double B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fdiv_rd(A[tid], B);
		tid += stride;
	}
}

// CUDA ELEMENT WISE MULTIPLY
// a(i) *= b(i), n is size of a
__global__ void cu_elementWiseMultiply(double *A, const double *B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fmul_rd(A[tid], B[tid]);
		tid += stride;
	}
}


// CUDA DSIGMOID A
// derivative of sigmoid non-linearity using cache of forward passing matrix
__global__ void cu_dsigmoid_a(double* src, double* dst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		float tmp = __fsub_rd(1.0, src[tid]);
		dst[tid] = __fmul_rd(tmp, src[tid]);
		tid += stride;
	}
}


// CUDA DSIGMOID
// derivative of sigmoid non-linearity
__global__ void cu_dsigmoid(double* src, double* dst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		float tmp = __expf(src[tid]);
		float tmp2 = __fadd_rd(tmp, 1.0);
		tmp2 = __fmul_rd(tmp2, tmp2);
		dst[tid] = fdividef(tmp, tmp2);
		tid += stride;
	}
}
