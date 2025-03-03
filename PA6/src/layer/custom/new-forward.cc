
#include <cmath>
#include <iostream>

#include "kernel.h"
#include "device.h"

#include "opencl-new-forward.h"

#define TILE_WIDTH 16

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }


/*
void OpenCLInterface::conv_forward_opencl_prolog(const float *host_y, const float *host_x, const float *host_k, cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //@@ Allocate OpenCL memory here
    // Create memory buffers for input and output vectors
    // 
    // Do not create your own device/context/queue. 
    // Use this->opencl->[program, kernel, queue, context]
    // OpenCL (common for entire NN)
    //      class is defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl.h
    //      methods defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6%2Fsrc%2Flayer%2Fcustom%opencl.cc
    //      created and passed into the network here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/m2.cc
    //      it's pointer is kept in OpenCLInterface (THIS) class here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl-new-forward.h

    //@@ Copy memory to the OpenCL here
    // Copy input vectors to memory buffers
}


void OpenCLInterface::conv_forward_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //__global float *y, __constant float *x, __constant float *k,
    // const int B, const int M, const int C, const int H, const int W, const int K)
    // Set the arguments to our compute kernel
    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]

    //@@ Set the kernel dimensions and call the kernel

    //@@ Launch the OpenCL Kernel here
    // Execute the OpenCL kernel on the array
}


void OpenCLInterface::conv_forward_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //@@ Copy the output back to host

    // Read the memory buffer output_mem_obj to the local variable result
    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]

    //@@ Free the OpenCL memory here
    // Release OpenCL resources
}
*/


void OpenCLInterface::conv_forward_opencl_prolog(const float *host_y, const float *host_x, const float *host_k, 
    cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K) 
{
    // Allocate memory buffers on the device
    *device_y = clCreateBuffer(this->opencl->context, CL_MEM_READ_WRITE, B * M * (H - K + 1) * (W - K + 1) * sizeof(float), NULL, NULL);
    *device_x = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY, B * C * H * W * sizeof(float), NULL, NULL);
    *device_k = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY, M * C * K * K * sizeof(float), NULL, NULL);

    // Copy data from host to device
    clEnqueueWriteBuffer(this->opencl->queue, *device_x, CL_TRUE, 0, B * C * H * W * sizeof(float), host_x, 0, NULL, NULL);
    clEnqueueWriteBuffer(this->opencl->queue, *device_k, CL_TRUE, 0, M * C * K * K * sizeof(float), host_k, 0, NULL, NULL);
}


void OpenCLInterface::conv_forward_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K) 
{
    // Set kernel arguments
    clSetKernelArg(this->opencl->kernel, 0, sizeof(cl_mem), &device_y);
    clSetKernelArg(this->opencl->kernel, 1, sizeof(cl_mem), &device_x);
    clSetKernelArg(this->opencl->kernel, 2, sizeof(cl_mem), &device_k);
    clSetKernelArg(this->opencl->kernel, 3, sizeof(int), &B);
    clSetKernelArg(this->opencl->kernel, 4, sizeof(int), &M);
    clSetKernelArg(this->opencl->kernel, 5, sizeof(int), &C);
    clSetKernelArg(this->opencl->kernel, 6, sizeof(int), &H);
    clSetKernelArg(this->opencl->kernel, 7, sizeof(int), &W);
    clSetKernelArg(this->opencl->kernel, 8, sizeof(int), &K);

     // Compute output feature map dimensions
    //int H_out = H - K + 1;
    //int W_out = W - K + 1;
   
    // Define optimal work sizes
    //size_t local_work_size[3] = {TILE_WIDTH, TILE_WIDTH, 1}; // Tunable values for optimal execution

    size_t local_work_size[3] = {TILE_WIDTH, TILE_WIDTH, 1};

    size_t global_work_size[3] = { 
        (((W-K+1) + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH, 
        (( (H-K+1) + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH,
        M};


    // Execute the kernel
    clEnqueueNDRangeKernel(this->opencl->queue, this->opencl->kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL);
}

void OpenCLInterface::conv_forward_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, 
    cl_mem device_k, const int B, const int M, const int C, 
    const int H, const int W, const int K)
{
    cl_int err;

    // Copy the output data from the device to the host
    err = clEnqueueReadBuffer(this->opencl->queue, device_y, CL_TRUE, 0, 
    sizeof(float) * B * M * (H - K + 1) * (W - K + 1), host_y, 0, nullptr, nullptr);
    CHECK_ERR(err, "clEnqueueReadBuffer for output y");

    // Release OpenCL memory
    err = clReleaseMemObject(device_y);
    CHECK_ERR(err, "clReleaseMemObject for device_y");
    err = clReleaseMemObject(device_x);
    CHECK_ERR(err, "clReleaseMemObject for device_x");
    err = clReleaseMemObject(device_k);
    CHECK_ERR(err, "clReleaseMemObject for device_k");
}