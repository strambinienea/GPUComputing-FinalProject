#include <cstdlib>
#include <math.h>
#include <iostream>
#include <mmio.h>
#include <algorithm>

#define ITERATIONS 100

using namespace std;

// <== CUDA KERNELS ==>

/**
  * Simple kernel used to awake the GPU before the computations
  */
__global__ void awakeKernel() { }

/* <=== MATRIX CONVERSION FUNCTIONS ===>*/

/**
 * Utility function used to import a matrix from a Matrix Market file
 * This function requires the matrix to be highly sparse (more than 75% of zero-elements), with real values and squares.
 *
 * @param filename - Path to the Matrix Market file
 * @param matrixSize - Variable to store the size of the matrix, the matrix is square
 * @param nonZero - Variable to store the number of non-zero elements
 * @param rowIdx - Pointer to the array that will store the row indexes of the non-zero elements
 * @param colIdx - Pointer to the array that will store the column indexes of the non-zero elements
 * @param values - Pointer to the array that will store the values of the non-zero elements
 *
 * @return 0 if the operation was successful, 1 otherwise
 */
int readMatrixMarketFile(
        const char *filename,
        int *matrixSize,
        int *nonZero,
        int **rowIdx,
        int **colIdx,
        double **values
) {

    FILE *fd;
    MM_typecode matrixCode;
    int N;

    // Check if the file exists and is a valid Matrix Market file
    if ((fd = fopen(filename, "r")) == NULL || mm_read_banner(fd, &matrixCode) != 0) {
        cout << "File: " << filename << " could not be opened, or is not a valid Matrix Market file" << endl;
        return 1;
    }

    // Check the matrix is sparse and composed of real numbers
    if (mm_is_sparse(matrixCode) && mm_is_real(matrixCode)) {

        // If sparse read the size of the matrix and number of non-zero elements
        if (mm_read_mtx_crd_size(fd, matrixSize, &N, nonZero) != 0) {
            cout << "Could not read matrix size" << endl;
            return 1;
        }
    } else {
        cout << "The application only supports highly sparse matrices with values type real" << endl;
        return 1;
    }

    // Check the matrix is a square matrix
    if (*matrixSize != N) {
        cout << "The application only supports square matrices" << endl;
        return 1;
    }

    // Allocate memory for the row and column indexes
    *rowIdx = (int *) malloc(*nonZero * sizeof(int));
    *colIdx = (int *) malloc(*nonZero * sizeof(int));
    *values = (double *) malloc(*nonZero * sizeof(double));

    // Read the row and column indexes
    for (int i = 0; i < *nonZero; i++) {
        fscanf(fd, "%d %d %lg\n", &(*rowIdx)[i], &(*colIdx)[i], &(*values)[i]);
        // Adjust from 1-based to 0-based
        (*rowIdx)[i]--;
        (*colIdx)[i]--;
    }

    fclose(fd);

    cout << "Matrix Market file read successfully" << endl;
    return 0;
}

/**
 * Utility function used to convert a matrix from the COO format to the CSR format
 *
 * @param _rowIdx - Pointer to the array that stores the row indexes of the non-zero elements (COO format)
 * @param _colIdx - Pointer to the array that stores the column indexes of the non-zero elements (COO format)
 * @param _values - Pointer to the array that stores the values of the non-zero elements (COO format)
 * @param matrixSize - Size of the matrix, the matrix is square
 * @param nonZero - Number of non-zero elements
 * @param rowPtrs - Pointer to the array that will store the offset value of each row for the colIdx array
 * @param colIdx - Pointer to the array that will store the column indexes of the non-zero elements
 * @param values - Pointer to the array that will store the values of the non-zero elements
 */
void convertMatrixToCSR(
        const int *_rowIdx,
        const int *_colIdx,
        const double *_values,
        const int matrixSize,
        const int nonZero,
        int **rowPtrs,
        int **colIdx,
        double **values
) {

    // Allocate memory for the row pointers and column index, rowPtrs is of size matrix size
    *rowPtrs = (int *) malloc((matrixSize) * sizeof(int));
    *colIdx = (int *) malloc(nonZero * sizeof(int));
    *values = (double *) malloc(nonZero * sizeof(double));

    // The offset for the first row is always zero
    (*rowPtrs)[0] = 0;

    // Cycle every row pointer
    for ( int rowPtr = 0; rowPtr < matrixSize; rowPtr++) {

        // This offset is the value stored in the rowPtrs array
        int offset = (*rowPtrs)[rowPtr];

        // Cycle every non-zero element
        for ( int i = 0; i < nonZero; i++) {

            // If the row matches the rowPtr then it is the same row
            // so increment the offset and save the column index in the new column index array
            if (_rowIdx[i] == rowPtr) {
                (*colIdx)[offset] = _colIdx[i];
                (*values)[offset] = _values[i];
                offset++;
            }
        }

        // Save the offset in the next element of rowPtrs, first element offset is always zero
        (*rowPtrs)[rowPtr + 1] = offset;
    }
}


/**
 * Utility function used to convert a matrix from the CSR format to the padded CSR format
 * This format create a virtual matrix of matrixSize rows and padding columns,
 * where padding is the highest number of non-zero elements in a single row of the matrix
 * The colIdx contains the indexes of the column that store non-zero elements, with -1 used as a padding value,
 * similarly the values array uses 0 as a padding value.
 *
 * @param _rowPtrs - Pointer to the array that will store the offset value of each row for the colIdx array
 * @param _colIdx -  Pointer to the array that stores the column indexes of the non-zero elements
 * @param _values - Pointer to the array that stores the values of the non-zero elements
 * @param matrixSize - Size of the matrix, the matrix is square
 * @param padding - Variable to store the padding value, the highest number of elements in a row
 * @param colIdx - Pointer to the array that will store the column indexes of the non-zero elements
 * @param values - Pointer to the array that will store the values of the non-zero elements
 */
void convertCSRToPaddedCSR(
        const int *_rowPtrs,
        const int *_colIdx,
        const double *_values,
        const int matrixSize,
        int *padding,
        int **colIdx,
        double **values
) {

    // First find the highest number of elements in a row and use it to calculate the padding
    *padding = 0;
    for (int i = 1; i < matrixSize; i++) {
        if (_rowPtrs[i] - _rowPtrs[i - 1] > (*padding)) {
            (*padding) = _rowPtrs[i] - _rowPtrs[i - 1];
        }
    }

    // Allocate the memory for the new column index array
    *colIdx = (int *) malloc(matrixSize * (*padding) * sizeof(int));
    *values = (double *) malloc(matrixSize * (*padding) * sizeof(double));

    // Cycle every row of the matrix
    for (int i = 0; i < matrixSize; i++) {
        // Cycle every column of the matrix
        for ( int j = 0; j < (*padding); j ++ ) {

            // Calculate the current element by offsetting the base pointer with the index of the current column
            int elementPtr = _rowPtrs[i] + j;

            // If the element pointer is larger than the next offset in _rowPtrs,
            // it means there are no more elements in the current row, so start filling with -1 values
            if (elementPtr < _rowPtrs[i + 1]) {
                (*colIdx)[i * (*padding) + j] = _colIdx[elementPtr];
                (*values)[i * (*padding) + j] = _values[elementPtr];
            } else {
                (*colIdx)[i * (*padding) + j] = -1;
                (*values)[i * (*padding) + j] = 0;
            }
        }
    }
}

// <== UTILITY FUNCTIONS ==>

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

    // Check that the path to the Matrix Market file was provided as a command line argument
    if (argc != 2) {
        cout << "You must specify the path to the Matrix Market file" << endl;
        return 1;
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

