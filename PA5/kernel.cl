// OpenCL kernel function for performing 2D convolution with a given mask (kernel).
// This kernel assumes the input image has multiple channels (e.g., RGB) and applies
// the convolution operation independently to each channel.
__kernel void convolution2D(
    __global int *inputData,   // Pointer to the input image data (flattened array)
    __global int *outputData,  // Pointer to the output image data (flattened array)
    __constant int *maskData,  // Pointer to the convolution mask (kernel), stored in constant memory
    int width,                 // Width of the input image
    int height,                // Height of the input image
    int maskWidth,             // Width (and height) of the square convolution mask
    int imageChannels,         // Number of color channels in the image (e.g., 3 for RGB)
    int stride                // Stride: step size for sliding the mask over the image
) {
    // Get global indices for the output data
    int x = get_global_id(0); // X-coordinate of the output pixel
    int y = get_global_id(1); // Y-coordinate of the output pixel
    int c = get_global_id(2); // Channel index (R, G, or B)

    // Compute the mask's radius (assumes square mask)
    int maskRadius = maskWidth / 2;

    // Compute the width and height of the output image after applying the convolution
    int output_width = (width - maskWidth) / stride + 1;
    int output_height = (height - maskWidth) / stride + 1;

    // Ensure thread is within valid bounds
    if (x >= output_width || y >= output_height || c >= imageChannels) {
        return;  // If the thread is out of bounds, exit early
    }

    // Accumulator variable to store the convolution result for this pixel and channel
    int accum = 0;

    // Perform the convolution operation by iterating over the mask elements
    for (int j = 0; j < maskWidth; j++) {  // Loop over the mask rows
        for (int i = 0; i < maskWidth; i++) {  // Loop over the mask columns
            // Compute the corresponding input image coordinates
            int xOffset = x * stride + i;
            int yOffset = y * stride + j;

            // Ensure that the computed input coordinates are within image bounds
            if (xOffset < width && yOffset < height) {
                // Compute the index in the flattened input array
                int imageIndex = (yOffset * width + xOffset) * imageChannels + c;
                // Compute the index in the flattened mask array
                int maskIndex = j * maskWidth + i;

                // Perform element-wise multiplication and accumulate the result
                accum += inputData[imageIndex] * maskData[maskIndex];
            }
        }
    }

    // Compute the index for storing the result in the output array
    int outputIndex = (y * output_width + x) * imageChannels + c;
    outputData[outputIndex] = accum; // Store the computed convolution value in the output array
}
