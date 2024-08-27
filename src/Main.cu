/**
 * @Author Enea Strambini
 */

#include <cstdlib>
#include <math.h>
#include <iostream>
#include <mmio.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cusparse.h>

#define ITERATIONS 100

using namespace std;

// <== CUDA KERNELS ==>

/**
 * Simple kernel used to tranpose a matrix in COO form
 * The kernel simply swaps the row and column arrays, whilst copying the values array
 *
 * @param rowIdx - Pointer to the array that stores the row indexes of the non-zero elements
 * @param colIdx - Pointer to the array that stores the column indexes of the non-zero elements
 * @param values - Pointer to the array that stores the values of the non-zero elements
 * @param nonZero - Number of non-zero elements
 * @param T_rowIdx - Pointer to the array that will store the row indexes of the transposed matrix
 * @param T_colIdx - Pointer to the array that will store the column indexes of the transposed matrix
 * @param T_values - Pointer to the array that will store
 */
__global__ void cooMatrixTranspose(
        const int* rowIdx,
        const int* colIdx,
        const double* values,
        const int nonZero,
        int* T_rowIdx,
        int* T_colIdx,
        double* T_values,
) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ( index < nonZero ) {
        T_rowIdx[index] = colIdx[index];
        T_colIdx[index] = rowIdx[index];
        T_values[index] = values[index];
    }
}

/**
 * Simple kernel used to tranpose a matrix in CSR form
 * To transpose a matrix in CSR form, the kernel will convert it in CSC form
 *
 * @param rowPtrs - Pointer to the array that will store the offset value of each row for the colIdx array
 * @param colIdx - Pointer to the array that stores the column indexes of the non-zero elements
 * @param values - Pointer to the array that stores the values of the non-zero elements
 * @param matrixSize - Size of the matrix, the matrix is square
 * @param nonZero - Number of non-zero elements
 * @param T_colPtrs - Pointer to the array that will store the offset value of each row for the rowIdx array
 * @param T_rowIdx - Pointer to the array that will store the row indexes of the transposed matrix
 * @param T_values - Pointer to the array that will store
 */
__global__ void csrMatrixTranspose(
        const int* rowPtrs,
        const int* colIdx,
        const double* values,
        const int matrixSize,
        const int nonZero,
        int* T_colPtrs,
        int* T_rowIdx,
        double* T_values
) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrixSize) {
        T_colPtrs[row] = rowPtrs[row];

        for (int i = rowPtrs[row]; i < rowPtrs[row + 1]; i++) {
            T_rowIdx[i] = colIdx[i];
            T_values[i] = values[i];
        }
    }
}


//__global__ void paddedCSRMatrixTranspose(
//        const int* rowPtrs,
//        const int* colIdx,
//        const double* values,
//        const int matrixSize,
//        const int nonZero,
//        const int padding,
//        int* T_colPtrs,
//        int* T_rowIdx,
//        double* T_values
//) {
//
//}

/**
  * Simple kernel used to awake the GPU before the computations
  */
__global__ void awakeKernel() { return; }

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

    // Allocate memory for the row and column indexes and values
    cudaMallocManaged(rowIdx, *nonZero * sizeof(int));
    cudaMallocManaged(colIdx, *nonZero * sizeof(int));
    cudaMallocManaged(values, *nonZero * sizeof(double));

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

    // Allocate memory for the row pointers, column index and values.
    // rowPtrs is of size matrix size
    cudaMallocManaged(rowPtrs, (matrixSize + 1) * sizeof(int));
    cudaMallocManaged(colIdx, nonZero * sizeof(int));
    cudaMallocManaged(values, nonZero * sizeof(double));

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
    (*rowPtrs)[matrixSize] = nonZero;
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

    // Allocate the memory for the new column index array and values array
    cudaMallocManaged(colIdx, matrixSize * (*padding) * sizeof(int));
    cudaMallocManaged(values, matrixSize * (*padding) * sizeof(double));

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

enum OPERATION {
    COO,
    CSR,
    PADDED_CSR,
    CUSPARSE
};

/**
  * Process the execution time of each kernel and return the effective bandwidth
  *
  * @param execTimes - An array of execution times
  * @param operation - The operation that was performed
  * @param nonZero - The number of non-zero elements in the matrix
  * @param matrixSize - The size of the matrix
  * @param padding - The padding used in the padded CSR format
  *
  * @return effectiveBandwidth - The effective bandwidth, calculated excluding the 5 highest and lowest times
  */
float processExecTimes(float* execTimes, OPERATION operation, int nonZero, int matrixSize = 0, int padding = 0) {

    // Sort the array
    sort(execTimes, execTimes + ITERATIONS);

    // Exclude the 5 highest and 5 lowest values from the average
    float average = 0.0f;

    for (int i = 5; i < ITERATIONS - 5; i++) {
        average += execTimes[i];
    }
    average = average / (ITERATIONS - 10);

    float effectiveBandwidth = 0.0f;

    // Calculate the effective bandwidth in GB/s
    switch (operation) {
        case COO: {
            // The effective bandwidth is found by multiplying the size of an array (nonZero times sizeof(double))
            // times 3 because it has to modify columns, rows and values arrays.
            // times 2 because it reads and writes
            //                        | column and row arrays  |   |     values array      |
            effectiveBandwidth = (2 * (2 * nonZero * sizeof(int) + nonZero * sizeof(double)) / 1024) / (average * 1000);
            break;
        }
        case CSR:
        case CUSPARSE: {
            // RowPtrs is of composed by matrixSize + 1 elements
            int rowPtrSize = (matrixSize + 1) * sizeof(int);

            // Both colIdx and values are composed by nonZero elements, but they differ in type
            int colIdxSize = nonZero * sizeof(int);
            int valuesSize = nonZero * sizeof(double);

            effectiveBandwidth = (2 * (rowPtrSize + colIdxSize + valuesSize) / 1024) / (average * 1000);
            break;
        }
        case PADDED_CSR: {
            // RowPtrs is of composed by matrixSize + 1 elements
            int rowPtrSize = (matrixSize + 1) * sizeof(int);

            // Both colIdx and values can be seen as matrices of matrixSize rows and padding columns
            int colIdxSize = matrixSize * padding * sizeof(int);
            int valuesSize = matrixSize * padding * sizeof(double);

            effectiveBandwidth = (2 * (rowPtrSize + colIdxSize + valuesSize) / 1024) / (average * 1000);
            break;
        }
        default: {
            cout << "Invalid operation" << endl;
            return -1;
        }
    }

    return effectiveBandwidth;
}

int main(int argc, char** argv) {

    // Check that the path to the Matrix Market file was provided as a command line argument
    if (argc != 2) {
        cout << "You must specify the path to the Matrix Market file" << endl;
        return 1;
    }

    // <== SETUP CODE ==>
    // In this part of the code the matrix will be read from the .mtx file and converted in the COO, CSR and padded
    // CSR formats. The matrix will later be used to perform the transposition using the GPU.

    // Read the matrix from the Matrix Market file
    int matrixSize, nonZero;
    int *rowIdx, *colIdx;
    double *values;

    if (readMatrixMarketFile(argv[1], &matrixSize, &nonZero, &rowIdx, &colIdx, &values) != 0) {
        cout << "Something went wrong while reading the Matrix Market file" << endl;
        return 1;
    }

    cout << "Matrix of size: " << matrixSize << " with " << nonZero << " non-zero elements" << endl;

#ifdef DEBUG
    cout << "Matrix in COO format" << endl;
    for (int i = 0; i < nonZero; i++) {
        cout << "Row: " << rowIdx[i] << " Col: " << colIdx[i] << " Vals: " << values[i] << endl;
    }
#endif

    // Convert the matrix from COO to CSR format
    int *CSRrowPtrs, *CSRcolIdx;
    double *CSRvalues;
    convertMatrixToCSR(rowIdx, colIdx, values, matrixSize, nonZero, &CSRrowPtrs, &CSRcolIdx, &CSRvalues);

#ifdef DEBUG
    cout << "Matrix in CSR format" << endl;
    for (int i = 0; i < matrixSize; i++) {
        for ( int j = CSRrowPtrs[i]; j < CSRrowPtrs[i + 1]; j++) {
            cout << "Row: " << i << " Col: " << CSRcolIdx[j] << " Vals: " << CSRvalues[j] << endl;
        }
    }
#endif

    // Convert the matrix from CSR to padded CSR format
    int *paddedColIdx;
    double *paddedValues;
    int padding;
    convertCSRToPaddedCSR(CSRrowPtrs, CSRcolIdx, CSRvalues, matrixSize, &padding, &paddedColIdx, &paddedValues);

#ifdef DEBUG
    cout << "Matrix in padded CSR format" << endl;
    for ( int i = 0; i < matrixSize; i++ ) {
        for ( int j = CSRrowPtrs[i]; j < CSRrowPtrs[i + 1]; j++) {
            cout << "Row: " << i << " Col: " << CSRcolIdx[j] << " Vals: " << CSRvalues[j] << endl;
        }
    }
#endif

    // <== KERNEL EXECUTION ==>

    // Awake the GPU before the computations
    int N_BLOCKS = 1;
    int N_THREADS = 1;

    awakeKernel<<<N_BLOCKS, N_THREADS>>>();

    // Create pointer for tranpose matrix arrays;
    int *T_rowIdx, *T_colIdx, *T_colPtr;
    double *T_values;

    // <== cuSPARSE TRANSPOSE ==>

    // Create cuda event to register execution time
    cudaEvent_t cusparseStart, cusparseStop;

    cudaEventCreate(&cusparseStart);
    cudaEventCreate(&cusparseStop);

    // Create the cuSPARSE handle used by the library functions
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);

    // Allocate memory for the transposed matrix, that is the same as the matrix in CSC format
    cudaMallocManaged(&T_colPtr, (matrixSize + 1) * sizeof(int));
    cudaMallocManaged(&T_rowIdx, nonZero * sizeof(int));
    cudaMallocManaged(&T_values, nonZero * sizeof(double));

    // Calculate the buffer size used by cuSPARSE method to perform the transposition
    // Only done once, since the operation is the same repeated multiple times
    size_t buffSize;
    cusparseStatus_t st1 = cusparseCsr2cscEx2_bufferSize(
        cusparseHandle,
        matrixSize,
        matrixSize,
        nonZero,
        CSRvalues,
        CSRrowPtrs,
        CSRcolIdx,
        T_values,
        T_colPtr,
        T_rowIdx,
        CUDA_R_64F,	
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT,
        &buffSize
    );

    cudaDeviceSynchronize();

    // Allocating the buffer
    void *buffer;
    cudaMallocManaged(&buffer, buffSize);

    float cusparseExecTimes[ITERATIONS];

    for (int i = 0; i < ITERATIONS; i++) {

        float elapsedTime = 0.0f;

        // Start the timer
        cudaEventRecord(cusparseStart);

        // Perform the transposition
        cusparseStatus_t status = cusparseCsr2cscEx2(
                cusparseHandle,
                matrixSize,
                matrixSize,
                nonZero,
                CSRvalues,
                CSRrowPtrs,
                CSRcolIdx,
                T_values,
                T_colPtr,
                T_rowIdx,
                CUDA_R_64F,
                CUSPARSE_ACTION_NUMERIC,
                CUSPARSE_INDEX_BASE_ZERO,
                CUSPARSE_CSR2CSC_ALG_DEFAULT,
                buffer
        );

        // Stop the timer and calculate the elapsed time
        cudaEventRecord(cusparseStop);
        cudaEventSynchronize(cusparseStop);

        cudaEventElapsedTime(&elapsedTime, cusparseStart, cusparseStop);
        cusparseExecTimes[i] = elapsedTime;
    }

    cudaDeviceSynchronize();

    cout << "CuSPARSE effective bandwidth: " << processExecTimes(cusparseExecTimes, CUSPARSE, nonZero, matrixSize) << " GB/s" << endl;
	
    // Destroy the cuSPARSE handle
    cusparseDestroy(cusparseHandle);

    // Freeing the memory
    cudaFree(T_colPtr);
    cudaFree(T_rowIdx);
    cudaFree(T_values);
    cudaFree(buffer);

    // Destroy the cuda events
    cudaEventDestroy(cusparseStart);
    cudaEventDestroy(cusparseStop);


    // <== COORDINATE LIST FORMAT (COO) ==>

    // Create cuda event to register execution time
    cudaEvent_t cooStart, cooStop;

    cudaEventCreate(&cooStart);
    cudaEventCreate(&cooStop);

    // Calculate the number of blocks and threads to use
    // Since the GPU allows a maximum of 1024 threads per block
    N_BLOCKS = (nonZero / 1024) + 1;
    N_THREADS = nonZero < 1024 ? nonZero : 1024;

    // Allocate memory for the transposed matrix
    cudaMallocManaged(&T_rowIdx, nonZero * sizeof(int));
    cudaMallocManaged(&T_colIdx, nonZero * sizeof(int));
    cudaMallocManaged(&T_values, nonZero * sizeof(double));

    cooMatrixTranspose<<<N_BLOCKS, N_THREADS>>>(rowIdx, colIdx, values, nonZero, T_rowIdx, T_colIdx, T_values);

    cudaDeviceSynchronize();

    cout << "Matrix in COO format" << endl;
    for (int i = 0; i < nonZero; i++) {
        cout << "Row: " << rowIdx[i] << " Col: " << colIdx[i] << " Vals: " << values[i] << endl;
    }

    cout << "Transposed matrix in COO format" << endl;
    for (int i = 0; i < nonZero; i++) {
        cout << "Row: " << T_rowIdx[i] << " Col: " << T_colIdx[i] << " Vals: " << T_values[i] << endl;
    }

    float cooExecTimes[ITERATIONS];

//    for (int i = 0; i < ITERATIONS; i++) {
//
//        float elapsedTime = 0.0f;
//
//        // Start the timer
//        cudaEventRecord(cooStart);
//
//        // Perform the transposition
//        cooMatrixTranspose<<<N_BLOCKS, N_THREADS>>>(rowIdx, colIdx, values, nonZero, T_rowIdx, T_colIdx, T_values);
//
//        // Stop the timer and calculate the elapsed time
//        cudaEventRecord(cooStop);
//        cudaEventSynchronize(cooStop);
//
//        cudaEventElapsedTime(&elapsedTime, cooStart, cooStop);
//        cooExecTimes[i] = elapsedTime;
//    }

    cudaDeviceSynchronize();

    cout << "COO effective bandwidth: " << processExecTimes(cooExecTimes, COO, nonZero) << " GB/s" << endl;

    cudaFree(T_rowIdx);
    cudaFree(T_colIdx);
    cudaFree(T_values);

    // Destroy the cuda events
    cudaEventDestroy(cooStart);
    cudaEventDestroy(cooStop);


    // <== COMPRESSED SPARSE ROW FORMAT (CSR) ==>

    // Create cuda event to register execution time
    cudaEvent_t csrStart, csrStop;

    cudaEventCreate(&csrStart);
    cudaEventCreate(&csrStop);

    // Calculate the number of blocks and threads to use
    // Since the GPU allows a maximum of 1024 threads per block
    N_BLOCKS = (matrixSize / 1024) + 1;
    N_THREADS = matrixSize < 1024 ? matrixSize : 1024;

    // Allocate memory for the transposed matrix
    cudaMallocManaged(&T_colPtr, nonZero * sizeof(int));
    cudaMallocManaged(&T_rowIdx, nonZero * sizeof(int));
    cudaMallocManaged(&T_values, nonZero * sizeof(double));

    float csrExecTimes[ITERATIONS];

    csrMatrixTranspose<<<N_BLOCKS, N_THREADS>>>(CSRrowPtrs, CSRcolIdx, CSRvalues, matrixSize, nonZero, T_colPtr, T_rowIdx, T_values);

    cudaDeviceSynchronize();

    cout << "Matrix in CSR format" << endl;
    for (int i = 0; i < matrixSize; i++) {
        for ( int j = CSRrowPtrs[i]; j < CSRrowPtrs[i + 1]; j++) {
            cout << "Row: " << i << " Col: " << CSRcolIdx[j] << " Vals: " << CSRvalues[j] << endl;
        }
    }

    cout << "Transposed matrix in CSR format" << endl;
    for (int i = 0; i < matrixSize; i++) {
        for ( int j = T_colPtr[i]; j < T_colPtr[i + 1]; j++) {
            cout << "Col: " << i << " Row: " << T_rowIdx[j] << " Vals: " << T_values[j] << endl;
        }
    }

//    for (int i = 0; i < ITERATIONS; i++) {
//
//        float elapsedTime = 0.0f;
//
//        // Start the timer
//        cudaEventRecord(csrStart);
//
//        // Perform the transposition
//        csrMatrixTranspose<<<N_BLOCKS, N_THREADS>>>(CSRrowPtrs, CSRcolIdx, CSRvalues, matrixSize, nonZero, T_colPtr, T_rowIdx, T_values);
//
//        // Stop the timer and calculate the elapsed time
//        cudaEventRecord(csrStop);
//        cudaEventSynchronize(csrStop);
//
//        cudaEventElapsedTime(&elapsedTime, csrStart, csrStop);
//        csrExecTimes[i] = elapsedTime;
//    }

    cudaDeviceSynchronize();

    cout << "CSR effective bandwidth: " << processExecTimes(csrExecTimes, CSR, nonZero, matrixSize) << " GB/s" << endl;

    // Free
    cudaFree(T_colPtr);
    cudaFree(T_rowIdx);
    cudaFree(T_values);

    // Destroy the cuda events
    cudaEventDestroy(csrStart);
    cudaEventDestroy(csrStop);


    // <== PADDED CSR FORMAT ==>

    // Create cuda event to register execution time
    cudaEvent_t paddedCSRStart, paddedCSRStop;

    cudaEventCreate(&paddedCSRStart);
    cudaEventCreate(&paddedCSRStop);


    // Destroy the cuda events
    cudaEventDestroy(paddedCSRStart);
    cudaEventDestroy(paddedCSRStop);

    // <== TEARDOWN ==>

    // Free the memory used by the COO matrix
    cudaFree(rowIdx);
    cudaFree(colIdx);
    cudaFree(values);

    // Free the memory used by the CSR matrix
    cudaFree(CSRrowPtrs);
    cudaFree(CSRcolIdx);
    cudaFree(CSRvalues);

    // Free the memory used by the padded CSR matrix
    cudaFree(paddedColIdx);
    cudaFree(paddedValues);


    return 0;
}

