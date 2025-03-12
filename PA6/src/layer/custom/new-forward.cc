
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

void OpenCLInterface::conv_forward_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;

    int output_height = H - K + 1;
    int output_width = W - K + 1;
    size_t num_tile_groups_width = (output_width + TILE_WIDTH - 1) / TILE_WIDTH;
    size_t num_tile_groups_height = (output_height + TILE_WIDTH - 1) / TILE_WIDTH;
    size_t global_item_size[3] = {(size_t)M * TILE_WIDTH, num_tile_groups_width * num_tile_groups_height * TILE_WIDTH, (size_t)B};
    size_t local_item_size[3] = {TILE_WIDTH, TILE_WIDTH, 1};

    // Set kernel arguments
    err = clSetKernelArg(this->opencl->kernel, 0, sizeof(cl_mem), &device_y);
    CHECK_ERR(err, "clSetKernelArg device_y");

    err = clSetKernelArg(this->opencl->kernel, 1, sizeof(cl_mem), &device_x);
    CHECK_ERR(err, "clSetKernelArg device_x");

    err = clSetKernelArg(this->opencl->kernel, 2, sizeof(cl_mem), &device_k);
    CHECK_ERR(err, "clSetKernelArg device_k");

    err = clSetKernelArg(this->opencl->kernel, 3, sizeof(int), &B);
    CHECK_ERR(err, "clSetKernelArg B");

    err = clSetKernelArg(this->opencl->kernel, 4, sizeof(int), &M);
    CHECK_ERR(err, "clSetKernelArg M");

    err = clSetKernelArg(this->opencl->kernel, 5, sizeof(int), &C);
    CHECK_ERR(err, "clSetKernelArg C");

    err = clSetKernelArg(this->opencl->kernel, 6, sizeof(int), &H);
    CHECK_ERR(err, "clSetKernelArg H");

    err = clSetKernelArg(this->opencl->kernel, 7, sizeof(int), &W);
    CHECK_ERR(err, "clSetKernelArg W");

    err = clSetKernelArg(this->opencl->kernel, 8, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg K");

    // Launch the OpenCL Kernel
    err = clEnqueueNDRangeKernel(this->opencl->queue, this->opencl->kernel, 3, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
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