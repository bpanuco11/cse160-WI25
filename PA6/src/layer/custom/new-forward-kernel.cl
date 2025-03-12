#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void do_not_remove_this_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void prefn_marker_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void conv_forward_kernel(__global float *y, __global float *x, __constant float *k,
                                  const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Macros for indexing into the 4D tensors
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    // Compute the output dimensions
    const int H_out = H - K + 1;  // Output height
    const int W_out = W - K + 1;  // Output width
    
    // Number of grid tiles along the width direction
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    
    // Get the batch index (which input image this thread is working on)
    int b = get_global_id(2);
    
    // Get the output feature map index (which filter/kernel is being applied)
    int m = get_group_id(0);
    
    // Compute the (h, w) location of the output pixel this thread computes
    int h = (get_group_id(1) / W_grid) * TILE_WIDTH + get_local_id(1);
    int w = (get_group_id(1) % W_grid) * TILE_WIDTH + get_local_id(0);
    
    // Shared memory tile for storing a patch of the input tensor
    __local float tile_x[TILE_WIDTH + KERNEL_SZ - 1][TILE_WIDTH + KERNEL_SZ - 1];
    
    // Accumulator for the output value at (b, m, h, w)
    float accum = 0.0f;
    
    // Iterate over the input channels
    for (int c = 0; c < C; c++) {
        // Compute the starting indices of the input tile in the input tensor
        int in_h_start = (get_group_id(1) / W_grid) * TILE_WIDTH;
        int in_w_start = (get_group_id(1) % W_grid) * TILE_WIDTH;
        
        // Get local thread indices within the tile
        int local_h = get_local_id(1);
        int local_w = get_local_id(0);
        
        // Load the input patch into shared memory (each thread loads multiple values)
        for (int i = local_h; i < TILE_WIDTH + K - 1; i += TILE_WIDTH) {
            for (int j = local_w; j < TILE_WIDTH + K - 1; j += TILE_WIDTH) {
                int in_h = in_h_start + i;
                int in_w = in_w_start + j;
                
                // Ensure we do not read out of bounds; otherwise, use zero-padding
                if (in_h < H && in_w < W) {
                    tile_x[i][j] = x4d(b, c, in_h, in_w);
                } else {
                    tile_x[i][j] = 0.0f;
                }
            }
        }
        
        // Synchronize all threads to ensure shared memory is fully loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Perform convolution if within bounds
        if (h < H_out && w < W_out) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    accum += tile_x[get_local_id(1) + p][get_local_id(0) + q] * k4d(m, c, p, q);
                }
            }
        }
        
        // Synchronize before loading the next channel
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store the computed value in the output tensor if within bounds
    if (h < H_out && w < W_out) {
        y4d(b, m, h, w) = accum;
    }
    
    // Undefine the macros to avoid redefinition issues
    #undef y4d
    #undef x4d
    #undef k4d
}