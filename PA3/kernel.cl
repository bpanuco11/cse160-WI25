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
__kernel void matrixMultiply(
    __global const int *A, __global const int *B, __global int *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Compute C = A * B

  int row = get_global_id(0);
  int col = get_global_id(1);

  // Ensure we are within bounds
  if (row < numCRows && col < numCColumns) {
      int sum = 0;
      for (int k = 0; k < numAColumns; k++) {
          sum += A[row * numAColumns + k] * B[k * numBColumns + col];
      }
      C[row * numCColumns + col] = sum;
  }
}

*/