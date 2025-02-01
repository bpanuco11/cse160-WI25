__kernel void matrixMultiply(
    __global const int *A, __global const int *B, __global int *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns) {
    
    // block size for each tile.
    const int TILE_SIZE = 16;
    
    // Local memory for tiles of matrices A and B
    __local int A_Tile[TILE_SIZE][TILE_SIZE];  
    __local int B_Tile[TILE_SIZE][TILE_SIZE];  
    
    int row = get_global_id(0);  // Global row index for matrix C
    int col = get_global_id(1);  // Global column index for matrix C
    
    int localRow = get_local_id(0);  // Local row index within tile
    int localCol = get_local_id(1);  // Local column index within tile
    
    int sum = 0;  // Variable sum to accumulate result for C[row][col]
    
    // Iterate over tiles to perform matrix multiplication
    for (int t = 0; t < (numAColumns + TILE_SIZE - 1) / TILE_SIZE; t++) {
        
        // Load data into local memory (tiles of A and B)
        // only load data if within bounds!
        if (row < numARows && t * TILE_SIZE + localCol < numAColumns)
            A_Tile[localRow][localCol] = A[row * numAColumns + t * TILE_SIZE + localCol];
        else
            A_Tile[localRow][localCol] = 0;  // Pad with zero if out of bounds

        if (col < numBColumns && t * TILE_SIZE + localRow < numBRows)
            B_Tile[localRow][localCol] = B[(t * TILE_SIZE + localRow) * numBColumns + col];
        else
            B_Tile[localRow][localCol] = 0;  // Pad with zero if out of bounds
        
        // Synchronize threads to make sure all data is loaded into local memory
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum for this tile multiplication
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_Tile[localRow][k] * B_Tile[k][localCol];
        }
        
        /* Synchronize threads again to ensure all threads finish computing
          their partial sums before proceeding to the next tile load*/
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store the result into matrix C
    if (row < numARows && col < numBColumns) {
        C[row * numBColumns + col] = sum;
    }
}