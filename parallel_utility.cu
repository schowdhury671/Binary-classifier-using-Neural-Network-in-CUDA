#include "parallel.cu"
#define threadsPerBlock 32
using namespace std;


void init_2D_mat(double **(&arr), int row, int col) {
    arr = (double **)malloc(row * sizeof(double *));
    for (int i = 0; i < row; i++)
        arr[i] = (double *)malloc(col * sizeof(double));
}

double *serialize_2D_mat(double **mat, int r, int c) {
    double *res = new double[r*c];
    int k = 0;
    for(int i = 0; i < r; i++)
        for(int j = 0; j < c; j++)
            res[k++] = mat[i][j];

    return res;
}

double **deserialize_2D_mat(double *arr, int r, int c) {
    double **res;
    int k = 0;
    init_2D_mat(res, r, c);
    for(int i = 0; i < r; i++)
        for(int j = 0; j < c; j++)
            res[i][j] = arr[k++];

    return res;
}

// MULTIPLY
// returns a * b
double **cuda_mat_multiply_helper(double **hostA, double **hostB, int numARows, int numAColumns, int numBRows, int numBColumns){
    double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
    double *hostB_serial = serialize_2D_mat(hostB, numBRows, numBColumns);
    double * hostC; // The output C matrix
    double * deviceA;
    double * deviceB;
    double * deviceC;
    // Setting numCRows and numCColumns
    int numCRows = numARows;
    int numCColumns = numBColumns;
    
    hostC = (double *) malloc(sizeof(double)*numCRows*numCColumns);    

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*numARows*numAColumns);
    cudaMalloc((void **)&deviceB, sizeof(double)*numBRows*numBColumns);
    cudaMalloc((void **)&deviceC, sizeof(double)*numCRows*numCColumns);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB_serial, sizeof(double)*numBRows*numBColumns, cudaMemcpyHostToDevice);

    // Initialize the grid and block dimensions 
    dim3 dimBlock(32, 32, 1);    
    dim3 dimGrid((numCColumns/32) + 1, (numCRows/32) + 1, 1);

    //@@ Launch the GPU Kernel here
    cuda_mat_multiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);    


    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostC, deviceC, sizeof(double)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);        
    cudaFree(deviceC);    

    double **hostC_deserialised = deserialize_2D_mat(hostC, numCRows, numCColumns);
   
    return hostC_deserialised;
}

// ADD
// returns a + b
double **cu_addition_helper(double **hostA, double **hostB, int numARows, int numAColumns){

    double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
    double *hostB_serial = serialize_2D_mat(hostB, numARows, numAColumns);
    double * hostC; // The output C matrix
    double * deviceA;
    double * deviceB;
    double * deviceC;

    int numCRows = numARows;
    int numCColumns = numAColumns;

    hostC = (double *) malloc(sizeof(double)*numCRows*numCColumns);    

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*numARows*numAColumns);
    cudaMalloc((void **)&deviceB, sizeof(double)*numARows*numAColumns);
    cudaMalloc((void **)&deviceC, sizeof(double)*numCRows*numCColumns);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);


	int len = numARows * numAColumns;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    
    cu_addition<<<num_blocks, block_size>>>(deviceA, deviceB, deviceC, len);
    
    
    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostC, deviceC, sizeof(double)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);        
    cudaFree(deviceC);    

    double **hostC_deserialised = deserialize_2D_mat(hostC, numCRows, numCColumns);
    
    return hostC_deserialised;
}


// TRANSPOSE
// return matrix transpose
double **cuda_mat_transpose_helper(double **hostA, int numARows, int numAColumns){

    double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
    double * hostC; // The output C matrix
    double * deviceA, * deviceC;

    int numCRows = numAColumns;
    int numCColumns = numARows;

    hostC = (double *) malloc(sizeof(double)*numCRows*numCColumns);    

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*numARows*numAColumns);
    cudaMalloc((void **)&deviceC, sizeof(double)*numCRows*numCColumns);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);

	int len = numARows * numAColumns;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

    cuda_mat_transpose<<<num_blocks, block_size>>>(deviceA, deviceC, numAColumns, numARows, len);
   
    
    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostC, deviceC, sizeof(double)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);       
    cudaFree(deviceC);    

    double **hostC_deserialised = deserialize_2D_mat(hostC, numCRows, numCColumns);
    
    return hostC_deserialised;
}

// MULTIPLY ELEMENT WISE
// returns src(i) * a
double** cu_mat_scalar_multiply_helper(double **hostA, double scalar, int numARows, int numAColumns){

    double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
    double * deviceA; 

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*numARows*numAColumns);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);

	int len = numARows * numAColumns;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

	cu_mat_scalar_multiply<<<num_blocks, block_size>>>(deviceA, scalar, len);

    
    
    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostA_serial, deviceA, sizeof(double)*numARows*numAColumns, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);         

    double **hostA_deserialised = deserialize_2D_mat(hostA_serial, numARows, numAColumns);
    
    return hostA_deserialised;

}


// MULTIPLY ELEMENT WISE
// return a(i) * b(i)
double** cu_mat_elementwise_multiply_helper(double **hostA, double **hostB, int numARows, int numAColumns) {

    double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
    double *hostB_serial = serialize_2D_mat(hostB, numARows, numAColumns);
    double * deviceA, * deviceB;

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*numARows*numAColumns);
    cudaMalloc((void **)&deviceB, sizeof(double)*numARows*numAColumns);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);

	int len = numARows * numAColumns;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

	cu_elementWiseMultiply<<<num_blocks, block_size>>>(deviceA, deviceB, len);

    
    
    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostA_serial, deviceA, sizeof(double)*numARows*numAColumns, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);       
    cudaFree(deviceB);    

    double **hostC_deserialised = deserialize_2D_mat(hostA_serial, numARows, numAColumns);
    
    return hostC_deserialised;

}

// SIGMOID
// sigmoid non-linearity
double **cu_sigmoid_helper(double **hostA, int numARows, int numAColumns){
    
    double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
    double * hostC; // The output C matrix
    double * deviceA, * deviceC;

    int numCRows = numAColumns;
    int numCColumns = numARows;

    hostC = (double *) malloc(sizeof(double)*numCRows*numCColumns);    

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*numARows*numAColumns);
    cudaMalloc((void **)&deviceC, sizeof(double)*numCRows*numCColumns);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);

	int len = numARows * numAColumns;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    
    cu_sigmoid<<<num_blocks, block_size>>>(deviceA, deviceC, len);
   
    
    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostC, deviceC, sizeof(double)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);       
    cudaFree(deviceC);    
    
    double **hostC_deserialised = deserialize_2D_mat(hostC, numCRows, numCColumns);
    
    return hostC_deserialised;
}


// DERIVATIVE OF SIGMOID
// sigmoid derivative required for back propagation
double **cu_dsigmoid_helper(double **hostA, int numARows, int numAColumns){
	double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
    double * hostC; // The output C matrix
    double * deviceA, * deviceC;

    int numCRows = numAColumns;
    int numCColumns = numARows;

    hostC = (double *) malloc(sizeof(double)*numCRows*numCColumns);    

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*numARows*numAColumns);
    cudaMalloc((void **)&deviceC, sizeof(double)*numCRows*numCColumns);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);

	int len = numARows * numAColumns;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

    cu_dsigmoid<<<num_blocks, block_size>>>(deviceA, deviceC, len);
   
    
    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostC, deviceC, sizeof(double)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);       
    cudaFree(deviceC);    

    double **hostC_deserialised = deserialize_2D_mat(hostC, numCRows, numCColumns);
    
    return hostC_deserialised;
}



// ADD 2D AND 1D MATRIX
// returns a(i)(j) + b(j)
double **cu_2D_1D_addition_helper(double **hostA, double *hostB, int numARows, int numAColumns){

    double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
    double *hostB_converted = (double*)malloc(numARows * numAColumns * sizeof(double));
    int k = 0;
    for(int i = 0; i < numAColumns; i++) {
        for(int j = 0; j < numARows; j++) {
            hostB_converted[k++] = hostB[i];
        }
    }
    double * hostC; // The output C matrix
    double * deviceA;
    double * deviceB;
    double * deviceC;

    int numCRows = numARows;
    int numCColumns = numAColumns;

    hostC = (double *) malloc(sizeof(double)*numCRows*numCColumns);    

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*numARows*numAColumns);
    cudaMalloc((void **)&deviceB, sizeof(double)*numARows*numAColumns);
    cudaMalloc((void **)&deviceC, sizeof(double)*numCRows*numCColumns);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA_serial, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB_converted, sizeof(double)*numARows*numAColumns, cudaMemcpyHostToDevice);


	int len = numARows * numAColumns;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    
    cu_addition<<<num_blocks, block_size>>>(deviceA, hostB_converted, deviceC, len);
    
    //cudaError_t err1 = cudaPeekAtLastError();
    //cudaDeviceSynchronize();
    //printf( "Got CUDA error ... %s \n", cudaGetErrorString(err1));
    
    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostC, deviceC, sizeof(double)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);        
    cudaFree(deviceC);    

    double **hostC_deserialised = deserialize_2D_mat(hostC, numCRows, numCColumns);
    
    return hostC_deserialised;
}


// ADD 2 VECTORS
// returns a + b
double *cu_vec_addition_helper(double *hostA, double *hostB, int n){

    double * hostC; // The output C matrix
    double * deviceA;
    double * deviceB;
    double * deviceC;

    hostC = (double *) malloc(sizeof(double)*n);    

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*n);
    cudaMalloc((void **)&deviceB, sizeof(double)*n);
    cudaMalloc((void **)&deviceC, sizeof(double)*n);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(double)*n, cudaMemcpyHostToDevice);


	int len = n;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    
    cu_addition<<<num_blocks, block_size>>>(deviceA, deviceB, deviceC, len);

    
    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostC, deviceC, sizeof(double)*n, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);        
    cudaFree(deviceC);    
    
    return hostC;
}

// MULTIPLY VECTOR ELEMENT WISE
// returns src(i) * a
double* cu_vec_scalar_multiply_helper(double *hostA, double scalar, int n){

    double * deviceA; 

    // Allocating GPU memory
    cudaMalloc((void **)&deviceA, sizeof(double)*n);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA, sizeof(double)*n, cudaMemcpyHostToDevice);

	int len = n;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

	cu_mat_scalar_multiply<<<num_blocks, block_size>>>(deviceA, scalar, len);

    
    // Copy the results in GPU memory back to the CPU    
    cudaMemcpy(hostA, deviceA, sizeof(double)*n, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);         
    
    return hostA;

}
