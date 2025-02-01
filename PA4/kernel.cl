__kernel void matrixMultiply(
    __global const int *A, __global const int *B, __global int *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns) {
    
    // Define tile size
    const int TILE_SIZE = 16;
    
    // Local memory for tiles
    __local int Asub[TILE_SIZE][TILE_SIZE];
    __local int Bsub[TILE_SIZE][TILE_SIZE];
    
    // Global row and column indices
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    // Local row and column indices within the block
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);
    
    int sum = 0;
    
    // Iterate over tiles
    for (int t = 0; t < (numAColumns + TILE_SIZE - 1) / TILE_SIZE; t++) {
        
        // Load tiles into shared memory
        if (row < numARows && t * TILE_SIZE + localCol < numAColumns)
            Asub[localRow][localCol] = A[row * numAColumns + t * TILE_SIZE + localCol];
        else
            Asub[localRow][localCol] = 0;

        if (col < numBColumns && t * TILE_SIZE + localRow < numBRows)
            Bsub[localRow][localCol] = B[(t * TILE_SIZE + localRow) * numBColumns + col];
        else
            Bsub[localRow][localCol] = 0;
        
        // Synchronize to ensure tiles are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum for the tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }
        
        // Synchronize before next tile load
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store the result
    if (row < numARows && col < numBColumns) {
        C[row * numBColumns + col] = sum;
    }
}
