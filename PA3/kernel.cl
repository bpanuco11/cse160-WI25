__kernel void matrixMultiply(
    __global const int *A, __global const int *B, __global int *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Compute C = A^T B

  int row = get_global_id(0);
  int col = get_global_id(1);
  // Ensure we are within bounds
    if (row < numCRows && col < numCColumns) {
        int sum = 0;
        for (int k = 0; k < numARows; k++) {
            // Corrected indexing for A^T
            sum += A[row + k * numAColumns] * B[k * numBColumns + col];
        }
        C[row * numCColumns + col] = sum;
    }

}

// NORMAL MULTIPLY  

/*
// OpenCL kernel for matrix multiplication
__kernel void matrixMultiply(
    __global const int *A,  // Input matrix A (stored in global memory)
    __global const int *B,  // Input matrix B (stored in global memory)
    __global int *C,        // Output matrix C (stored in global memory)

    const unsigned int numARows,    // Number of rows in matrix A
    const unsigned int numAColumns, // Number of columns in matrix A
    const unsigned int numBRows,    // Number of rows in matrix B
    const unsigned int numBColumns, // Number of columns in matrix B
    const unsigned int numCRows,    // Number of rows in matrix C (same as numARows)
    const unsigned int numCColumns  // Number of columns in matrix C (same as numBColumns)
) {
    // Each thread computes a single element of matrix C
    int row = get_global_id(0);  // Get the row index assigned to this thread
    int col = get_global_id(1);  // Get the column index assigned to this thread

    // Ensure the thread is within valid matrix boundaries
    if (row < numCRows && col < numCColumns) {
        int sum = 0;  // Initialize sum for the (row, col) element of C

        // Perform the dot product of the row of A with the column of B
        for (int k = 0; k < numAColumns; k++) {
            // A[row, k] * B[k, col] contributes to C[row, col]
            sum += A[row * numAColumns + k] * B[k * numBColumns + col];
        }

        // Store the computed value in matrix C
        C[row * numCColumns + col] = sum;
    }
}


*/