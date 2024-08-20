#include <cstdlib>
#include <math.h>
#include <iostream>
#include <algorithm>

#define MATRIX_TYPE float 
#define MAX_RANDOM_VALUE 100
#define TILE_SIZE 32
#define ITERATIONS 100

using namespace std;

// <--- CUDA KERNELS --->

/**
 * Kernel that copies a matrix into another, used as baseline
 *
 * @param matrix - The matrix to read data from
 * @param transposedMatrix - The matrix to store data to
 * @param matrixSize - The size of both matrices
 */
__global__ void matrixCopy(const MATRIX_TYPE* matrix, MATRIX_TYPE* outputMatrix, int matrixSize) {

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = matrixSize * row + column;

    if ( index < matrixSize * matrixSize ) {

        outputMatrix[index] = matrix[index];
    }
}

/**
 * Kernel that computes the out-of-place naive matrix transposition, used as a baseline
 *
 * @param matrix - The matrix to transpose
 * @param transposedMatrix - The matrix to store the transposed values
 * @param matrixSize - The size of both matrices
 */
__global__ void naiveTranspose(const MATRIX_TYPE* matrix, MATRIX_TYPE* transposedMatrix, int matrixSize) {

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = matrixSize * row + column;
    int transposedIndex = matrixSize * column + row;

    if ( index < matrixSize * matrixSize && transposedIndex < matrixSize * matrixSize ) {

        transposedMatrix[transposedIndex] = matrix[index];
    }
}

/**
  * Kernel that computes the out-of-place matrix transposition, using memory coalescing to
  * enhance performance
  *
  * @param matrix - The matrix to transpose
  * @param transposedMatrix - The matrix to store the transposed values
  * @param matrixSize - The size of both matrices
  */
__global__ void coalescedTiledTranspose(const MATRIX_TYPE* matrix, MATRIX_TYPE* transposedMatrix, int matrixSize) {
	
    __shared__ MATRIX_TYPE tile[TILE_SIZE * TILE_SIZE];

    int index = matrixSize * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x); 
    int transposedIndex =  matrixSize * (blockIdx.x * blockDim.x + threadIdx.y) + (blockIdx.y * blockDim.y + threadIdx.x); 

    // Copying values from global memory to the tile in the shared memory
    if ( index < matrixSize * matrixSize ) {
		tile[threadIdx.y * TILE_SIZE + threadIdx.x] = matrix[index];	
    }

    __syncthreads();

    if ( transposedIndex < matrixSize * matrixSize) {
        transposedMatrix[transposedIndex] = tile[threadIdx.x * TILE_SIZE + threadIdx.y];
    }
}


/**
  * Kernel that computes the out-of-place matrix transposition, using memory coalescing and
  * padding the shared memory array to prevent bank conflict and enhance performance
  *
  * @param matrix - The matrix to transpose
  * @param transposedMatrix - The matrix to store the transposed values
  * @param matrixSize - The size of both matrices
  */
__global__ void coalescedPaddedTiledTranspose(const MATRIX_TYPE* matrix, MATRIX_TYPE* transposedMatrix, int matrixSize) {
	

    // Padding the array to prevent bank conflict
    __shared__ MATRIX_TYPE paddedTile[(TILE_SIZE + 1) * TILE_SIZE];

    int index = matrixSize * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x); 
    int transposedIndex =  matrixSize * (blockIdx.x * blockDim.x + threadIdx.y) + (blockIdx.y * blockDim.y + threadIdx.x); 

    // Copying values from global memory to the tile in the shared memory
    if ( index < matrixSize * matrixSize ) {
		paddedTile[threadIdx.x + (threadIdx.y * (TILE_SIZE + 1))] = matrix[index];	
    }

    __syncthreads();

    if ( transposedIndex < matrixSize * matrixSize) {
        transposedMatrix[transposedIndex] = paddedTile[threadIdx.y + (threadIdx.x * (TILE_SIZE + 1))];
    }
}

/**
  * Simple kernel used to awake the GPU before the computations
  */
__global__ void awakeKernel() { }

// <--- END CUDA KERNELS --->

/**
 * Initialize the matrix with random values
 * @param matrix - The matrix to initialize
 * @param matrixSize - The size of the matrix
 */
void initMatrix(MATRIX_TYPE* matrix, int matrixSize) {
    srand(time(NULL));
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        matrix[i] = (MATRIX_TYPE) rand();
    }
}

/**
  * Print a matrix
  * @param matrix - The matrix to print
  * @param matrixSize - The size of the matrix
  */
void printMatrix(const MATRIX_TYPE* matrix, int matrixSize) {
    
	cout << "Matrix of " << matrixSize << " X " << matrixSize << " elements" << endl;
	for (int i = 0; i < matrixSize * matrixSize; i++) {
		
		if (i % matrixSize == 0) { cout << endl;} 
		cout << matrix[i] << " ";
	}

	cout << endl;
}

/**
  * Process the execution time of each kernel and return the effective bandwidth
  * @param execTimes - An array of execution times
  * @param matrixSize - The size of the matrix
  * @return effectiveBandwidth - The effective bandwidth, calculated excluding the 5 highest and lowest times
  */
float processExecTimes(float* execTimes, int matrixSize) {

    // Sort the array
    sort(execTimes, execTimes + ITERATIONS);

    // Exclude the 5 highest and 5 lowest values from the average
    float average = 0.0f;
    
    for (int i = 5; i < ITERATIONS - 5; i++) {
        average += execTimes[i];
    }
    average = average / (ITERATIONS - 10);
    
    float effectiveBandwidth = (2 * matrixSize * matrixSize * sizeof(MATRIX_TYPE) / 1024) / (average * 1000);
    return effectiveBandwidth;
}

int main(int argc, char** argv) {

    // Check the size of the matrix is passed as an argument
    if (argc < 2) {
        cout
        << "You must specify the size of the matrix as an argument"
        << endl;

        return -1;
    }
    
    // Calculate the size of the matrix and initialize it
    int MATRIX_SIZE = 1 << atoi(argv[1]);

    // The matrix will be divided in GRID_SIZE X GRID_SIZE tiles
    int GRID_SIZE = MATRIX_SIZE / TILE_SIZE;
    if ( GRID_SIZE < 1 ) { GRID_SIZE = 1; } 

    // Dimension of the tile matrix, each tile is a block
    dim3 GRID_DIMENSION(GRID_SIZE, GRID_SIZE);

    // Dimension of a single tile or block
    dim3 BLOCK_DIMENSION(TILE_SIZE, TILE_SIZE);

    awakeKernel<<<GRID_DIMENSION, BLOCK_DIMENSION>>>();


    // Create and initialize matrices
    MATRIX_TYPE *matrixA, *matrixB;

    cudaMallocManaged(&matrixA, MATRIX_SIZE * MATRIX_SIZE * sizeof(MATRIX_TYPE));
    cudaMallocManaged(&matrixB, MATRIX_SIZE * MATRIX_SIZE * sizeof(MATRIX_TYPE));
    
    initMatrix(matrixA, MATRIX_SIZE);

    // <--- MATRIX COPY --->

    cout << "Computing copy of matrix of size: "  
	    << MATRIX_SIZE << " X " << MATRIX_SIZE << " and a grid of size " 
	    << GRID_SIZE << " X " << GRID_SIZE << endl;
  
    // Crate cuda event to register execution time
    cudaEvent_t startCopy, stopCopy;
    
    cudaEventCreate(&startCopy);
    cudaEventCreate(&stopCopy);
    
    // Array to store execution times
    float copyExecTimes[100];


    for(int i = 0; i < ITERATIONS; i++) {
    
        float elapsedTime = 0.0f;

        cudaEventRecord(startCopy);
       
        matrixCopy<<<GRID_DIMENSION, BLOCK_DIMENSION>>>(matrixA, matrixB, MATRIX_SIZE);
        
        cudaEventRecord(stopCopy);
        cudaEventSynchronize(stopCopy);

        cudaEventElapsedTime(&elapsedTime, startCopy, stopCopy);
        copyExecTimes[i] = elapsedTime;
    } 

    cudaDeviceSynchronize();
    
    cout << "MATRIX COPY EFFECTIVE BANDWITH (GB/s): " << processExecTimes(copyExecTimes, MATRIX_SIZE) << endl; 

    // Free resources
    cudaEventDestroy(startCopy);
    cudaEventDestroy(stopCopy);

    // <--- END MATRIX COPY --->


    // <--- NAIVE MATRIX TRANPOSITION --->

    cout << "Computing naive matrix transposition of a matrix of size "  
	    << MATRIX_SIZE << " X " << MATRIX_SIZE << " and a grid of size " 
	    << GRID_SIZE << " X " << GRID_SIZE << endl;
  
    // Crate cuda event to register execution time
    cudaEvent_t startNaive, stopNaive;
    
    cudaEventCreate(&startNaive);
    cudaEventCreate(&stopNaive);

    // Array to store execution times
    float naiveExecTimes[100];

    for(int i = 0; i < ITERATIONS; i++) {
    
        float elapsedTime = 0.0f;

        cudaEventRecord(startNaive);
       
        naiveTranspose<<<GRID_DIMENSION, BLOCK_DIMENSION>>>(matrixA, matrixB, MATRIX_SIZE);
        
        cudaEventRecord(stopNaive);
        cudaEventSynchronize(stopNaive);

        cudaEventElapsedTime(&elapsedTime, startNaive, stopNaive);
        naiveExecTimes[i] = elapsedTime;
    } 

    cudaDeviceSynchronize();
    
    cout << "NAIVE MATRIX TRANSPOSITION EFFECTIVE BANDWIDTH (GB/s): " << processExecTimes(naiveExecTimes, MATRIX_SIZE) << endl; 

    // Free resources
    cudaEventDestroy(startNaive);
    cudaEventDestroy(stopNaive);

    // <--- END NAIVE MATRIX TRANSPOSITION --->


    // <--- COALESCED MATRIX TRANPOSITION --->

    cout << "Computing the coalesce matrix transposition of a matrix of size "  
	    << MATRIX_SIZE << " X " << MATRIX_SIZE << " and a grid of size " 
	    << GRID_SIZE << " X " << GRID_SIZE << endl;
  
    // Crate cuda event to register execution time
    cudaEvent_t startCoalesced, stopCoalesced;
    
    cudaEventCreate(&startCoalesced);
    cudaEventCreate(&stopCoalesced);
    
    // Array to store execution times
    float coalescedExecTimes[100];

    for(int i = 0; i < ITERATIONS; i++) {
    
        float elapsedTime = 0.0f;

        cudaEventRecord(startCoalesced);
       
        coalescedTiledTranspose<<<GRID_DIMENSION, BLOCK_DIMENSION>>>(matrixA, matrixB, MATRIX_SIZE);
        
        cudaEventRecord(stopCoalesced);
        cudaEventSynchronize(stopCoalesced);
	
	cudaEventElapsedTime(&elapsedTime, startCoalesced, stopCoalesced);
        coalescedExecTimes[i] = elapsedTime;
    } 
 
    cudaDeviceSynchronize();
 
    cout << "COALESCED MATRIX TRANSPOSITION EFFECTIVE BANDWIDTH (GB/s): " << processExecTimes(coalescedExecTimes, MATRIX_SIZE) << endl; 

    // Free resources
    cudaEventDestroy(startCoalesced);
    cudaEventDestroy(stopCoalesced);

    // <--- END COALESCED MATRIX TRANSPOSITION --->

    // <--- COALESCED PADDED MATRIX TRANPOSITION --->
    cout << "Computing the padded coalesce matrix transposition of a matrix of size "  
	    << MATRIX_SIZE << " X " << MATRIX_SIZE << " and a grid of size " 
	    << GRID_SIZE << " X " << GRID_SIZE << endl;
  
    // Crate cuda event to register execution time
    cudaEvent_t startCoalescedPadded, stopCoalescedPadded;
    
    cudaEventCreate(&startCoalescedPadded);
    cudaEventCreate(&stopCoalescedPadded);
    
    // Array to store execution times
    float coalescedPaddedExecTimes[100];

    for(int i = 0; i < ITERATIONS; i++) {
    
        float elapsedTime = 0.0f;

        cudaEventRecord(startCoalescedPadded);
       
        coalescedPaddedTiledTranspose<<<GRID_DIMENSION, BLOCK_DIMENSION>>>(matrixA, matrixB, MATRIX_SIZE);
        
        cudaEventRecord(stopCoalescedPadded);
        cudaEventSynchronize(stopCoalescedPadded);

        cudaEventElapsedTime(&elapsedTime, startCoalescedPadded, stopCoalescedPadded);
        coalescedPaddedExecTimes[i] = elapsedTime;
    } 

    cudaDeviceSynchronize();
    
    cout << "PADDED COALESCED MATRIX TRANSPOSITION EFFECTIVE BANDWIDTH (GB/s): " << processExecTimes(coalescedPaddedExecTimes, MATRIX_SIZE) << endl; 
    // Free resources
    cudaEventDestroy(startCoalescedPadded);
    cudaEventDestroy(stopCoalescedPadded);

    // <--- END COALESCED PADDED MATRIX TRANSPOSITION --->
    
    // Free arrays memory
    cudaFree(matrixA);
    cudaFree(matrixB);

    return 0;
}

