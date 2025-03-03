#define TILE_WIDTH 16
#define KERNEL_SZ 7

// Maximum kernel size, assuming K = 7 is the max expected kernel size
#define MAX_KERNEL_SIZE 7
#define MAX_TILE_WIDTH 16


__kernel void do_not_remove_this_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void prefn_marker_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void conv_forward_kernel(
    __global float* y,       // Output tensor
    __global float* x,       // Input tensor
    __constant float* k,     // Kernel weights (constant memory for efficiency)
    int B,                   // Batch size
    int M,                   // Number of output feature maps
    int C,                   // Number of input channels
    int H,                   // Input height
    int W,                   // Input width
    int K                    // Kernel size (assumed square: KxK)
) {
    // Compute output dimensions
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    
    // Get global indices for the output tensor
    int col_out = get_global_id(0);  // Output pixel X
    int row_out = get_global_id(1);  // Output pixel Y
    int m = get_global_id(2);  // Feature map index
    
    // Ensure the thread is within valid bounds
    if (col_out >= W_out || row_out >= H_out || m >= M) return;
    
    // Iterate over batch size
    for (int b = 0; b < B; ++b) {
        float accum = 0.0f;

        for (int c = 0; c < C; ++c) {  // Loop over input channels
            for (int p = 0; p < K; ++p) {  // Loop over kernel rows
                for (int q = 0; q < K; ++q) {  // Loop over kernel columns
                    // Compute input image index
                    int img_idx = (b * C * H * W) + (c * H * W) + ((row_out + p) * W) + (col_out + q);
                    // Compute kernel index
                    int k_idx = (m * C * K * K) + (c * K * K) + (p * K) + q;
                    // Perform element-wise multiplication and accumulation
                    accum += x[img_idx] * k[k_idx];
                }
            }
        }
        // Compute output index
        int out_idx = (b * M * H_out * W_out) + (m * H_out * W_out) + (row_out * W_out) + col_out;
        // Store result in output tensor
        y[out_idx] = accum;
    }
}
