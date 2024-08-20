//
// Created by Enea Strambini on 20/08/24.
//
#include <cstdlib>
#include <iostream>
#include <mmio.h>

using namespace std;

int readMatrixMarketFile(const char *filename, int *M, int *N, int *nz, int **I, int **J) {

    FILE *f;
    MM_typecode matcode;

    // Check if the file exists and is a valid Matrix Market file
    if ((f = fopen(filename, "r")) == NULL || mm_read_banner(f, &matcode) != 0) {
        cout << "File: " << filename << " could not be opened, or is not a valid Matrix Market file" << endl;
        return 1;
    }

    // Check the matrix is no using complex numbers and is a sparse matrix
    if (mm_is_complex(matcode)) {
        cout << "The application does not support complex numbers" << endl;
        return 1;
    }

    if (mm_is_sparse(matcode)) {
        if (mm_read_mtx_crd_size(f, M, N, nz) != 0) {
            cout << "Could not read matrix size" << endl;
            return 1;
        }
    } else {
        cout << "The application only supports highly sparse matrices" << endl;
        return 1;
    }

    *I = (int *) malloc(*nz * sizeof(int));
    *J = (int *) malloc(*nz * sizeof(int));

    for (int i = 0; i < *nz; i++) {
        fscanf(f, "%d %d\n", &(*I)[i], &(*J)[i]);
        // Adjust from 1-based to 0-based
        (*I)[i]--;
        (*J)[i]--;
    }

    cout << "Matrix Market file read successfully" << endl;

    if (f != stdin) {
        fclose(f);
    }

    return 0;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <matrix-market-filename>" << endl;
        return 1;
    }

    int M, N, nz;
    int *I, *J;

    readMatrixMarketFile(argv[1], &M, &N, &nz, &I, &J);

    cout << "M: " << M << endl;
    cout << "N: " << N << endl;
    cout << "nz: " << nz << endl;


    free(I);
    free(J);

    return 0;
}