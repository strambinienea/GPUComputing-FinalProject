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

/**
  * Process the execution time of each kernel and return the effective bandwidth
  * @param execTimes - An array of execution times
  * @param matrixSize - The size of the matrix
  * @return effectiveBandwidth - The effective bandwidth, calculated excluding the 5 highest and lowest times
  */
//float processExecTimes(float* execTimes, int matrixSize) {
//
//    // Sort the array
//    sort(execTimes, execTimes + ITERATIONS);
//
//    // Exclude the 5 highest and 5 lowest values from the average
//    float average = 0.0f;
//
//    for (int i = 5; i < ITERATIONS - 5; i++) {
//        average += execTimes[i];
//    }
//    average = average / (ITERATIONS - 10);
//
//    float effectiveBandwidth = (2 * matrixSize * matrixSize * sizeof(MATRIX_TYPE) / 1024) / (average * 1000);
//    return effectiveBandwidth;
//}

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
    awakeKernel<<<1, 1>>>();

    // <== cuSPARSE TRANSPOSE ==>

    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    
    // Create cuda event to register execution time
    cudaEvent_t cuSparse_Start, cuSparse_Stop;

    cudaEventCreate(&cuSparse_Start);
    cudaEventCreate(&cuSparse_Stop);

    cout << "Matrix in CSR format" << endl;
    for (int i = 0; i < matrixSize; i++) {
        for ( int j = CSRrowPtrs[i]; j < CSRrowPtrs[i + 1]; j++) {
            cout << "Row: " << i << " Col: " << CSRcolIdx[j] << " Vals: " << CSRvalues[j] << endl;
        }
    }

    int *cscColPtr, *cscRowIdx;
    double *cscValues;

    cudaMallocManaged(&cscColPtr, (matrixSize + 1) * sizeof(int));
    cudaMallocManaged(&cscRowIdx, nonZero * sizeof(int));
    cudaMallocManaged(&cscValues, nonZero * sizeof(double));

    size_t buffSize;

    cusparseStatus_t st1 = cusparseCsr2cscEx2_bufferSize(
        cusparseHandle,
        matrixSize,
        matrixSize,
        nonZero,
        CSRvalues,
        CSRrowPtrs,
        CSRcolIdx,
        cscValues,
        cscColPtr,
        cscRowIdx,
        CUDA_R_64F,	
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT,
        &buffSize
    );
    
    cudaDeviceSynchronize();

    void *buffer;
    cudaMallocManaged(&buffer, buffSize);
     
    cusparseStatus_t status = cusparseCsr2cscEx2(
        cusparseHandle,
        matrixSize,
        matrixSize,
        nonZero,
        CSRvalues,
        CSRrowPtrs,
        CSRcolIdx,
        cscValues,
        cscColPtr,
        cscRowIdx,
        CUDA_R_64F,	
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT,
        buffer
    );

    cudaDeviceSynchronize();

    cout << "Matrix in CSC format" << endl;
    for (int i = 0; i < matrixSize; i++) {
        for ( int j = cscColPtr[i]; j < cscColPtr[i + 1]; j++) {
            cout << "Col: " << i << " Row: " << cscRowIdx[j] << " Vals: " << cscValues[j] << endl;
        }
    }
    cusparseDestroy(cusparseHandle);

    cudaFree(cscColPtr);
    cudaFree(cscRowIdx);
    cudaFree(cscValues);
    cudaFree(buffer);

    // Destroy the cuda events
    cudaEventDestroy(cuSparse_Start);
    cudaEventDestroy(cuSparse_Stop);


    // <== COORDINATE LIST FORMAT (COO) ==>

    // Create cuda event to register execution time
    cudaEvent_t coo_Start, coo_Stop;

    cudaEventCreate(&coo_Start);
    cudaEventCreate(&coo_Stop);


    // Destroy the cuda events
    cudaEventDestroy(coo_Start);
    cudaEventDestroy(coo_Stop);


    // <== COMPRESSED SPARSE ROW FORMAT (CSR) ==>

    // Create cuda event to register execution time
    cudaEvent_t csr_Start, csr_Stop;

    cudaEventCreate(&csr_Start);
    cudaEventCreate(&csr_Stop);


    // Destroy the cuda events
    cudaEventDestroy(csr_Start);
    cudaEventDestroy(csr_Stop);


    // <== PADDED CSR FORMAT ==>

    // Create cuda event to register execution time
    cudaEvent_t paddedCSR_Start, paddedCSR_Stop;

    cudaEventCreate(&paddedCSR_Start);
    cudaEventCreate(&paddedCSR_Stop);


    // Destroy the cuda events
    cudaEventDestroy(paddedCSR_Start);
    cudaEventDestroy(paddedCSR_Stop);

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

