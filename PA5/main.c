#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <string.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"
#include "img.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "kernel.cl"

#define COMPUTE_OUTPUT_DIM(input_dim, kernel_size, stride) \
    ((input_dim - kernel_size) / stride + 1)


void OpenCLConvolution2D(Image *input0, Matrix *input1, Image *result, int stride)
{
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    // Device input and output buffers
    cl_mem device_a, device_b, device_c;

    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get the ID for the specified kind of device type.
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "convolution2D", &err);
    CHECK_ERR(err, "clCreateKernel");


    // Query device for maximum workgroup size
    //size_t max_work_group_size;
    //err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    //CHECK_ERR(err, "clGetKernelWorkGroupInfo");

    //printf("Max work group size: %zu\n", max_work_group_size);

    // Allocate GPU memory
    device_a = clCreateBuffer(context,
        CL_MEM_READ_ONLY,
        input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS * sizeof(int),
        NULL,
        &err);
    CHECK_ERR(err, "clCreateBuffer input0");

    device_b = clCreateBuffer(context,
        CL_MEM_READ_ONLY,
        input1->shape[0] * input1->shape[1] * sizeof(int),
        NULL,
        &err);
    CHECK_ERR(err, "clCreateBuffer input1");

    device_c = clCreateBuffer(context,
        CL_MEM_WRITE_ONLY,
        result->shape[0] * result->shape[1] * IMAGE_CHANNELS * sizeof(int),
        NULL,
        &err);
    CHECK_ERR(err, "clCreateBuffer result");

    // Copy memory to the GPU
    err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS * sizeof(int), input0->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer input0");

    err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, input1->shape[0] * input1->shape[1] * sizeof(int), input1->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer input1");

    // Set the arguments to the compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_c);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_b);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &input0->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &input0->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 4");
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &input1->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 5");

    int imageChannels = IMAGE_CHANNELS;
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &imageChannels);
    CHECK_ERR(err, "clSetKernelArg 6");
    err |= clSetKernelArg(kernel, 7, sizeof(unsigned int), &stride);
    CHECK_ERR(err, "clSetKernelArg 7");

    // Calculate output dimensions
    int output_width = (input0->shape[1] - input1->shape[1]) / stride + 1;
    int output_height = (input0->shape[0] - input1->shape[0]) / stride + 1;

    // Adjust global work size based on device limits
    size_t global_work_size[3] = { (size_t)output_width, (size_t)output_height, (size_t)imageChannels };
    size_t local_work_size[3] = { 16, 16, 1 }; // Tunable based on device

    // Ensure global work size is divisible by local work size
    global_work_size[0] = (global_work_size[0] + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0];
    global_work_size[1] = (global_work_size[1] + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1];


    // Launch the GPU kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");

    // Copy memory back from GPU to CPU
    err = clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, result->shape[0] * result->shape[1] * IMAGE_CHANNELS * sizeof(int), result->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer");

    // Free GPU memory
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}




int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // get the dir from the input file
    int stride;
    char dir[256];
    strcpy(dir, dirname(strdup(input_file_a))); 

    // Host input and output vectors and sizes
    Image host_a, host_c, answer;
    Matrix host_b;
    
    cl_int err;

    err = LoadImgRaw(input_file_a, &host_a);
    CHECK_ERR(err, "LoadImg");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    // err = LoadImgTmp(input_file_c, &answer);
    err = LoadImgRaw(input_file_c, &answer);
    CHECK_ERR(err, "LoadImg");

    // Load stride
    err = LoadStride(dir, &stride);
    CHECK_ERR(err, "LoadStride");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer image
    // Calculate output dimensions
    // Get the dimensions of the input image (assuming they are stored in host_a)
  


     // Compute output dimensions using the formula
    rows = COMPUTE_OUTPUT_DIM(host_a.shape[0], host_b.shape[0], stride);
    cols = COMPUTE_OUTPUT_DIM(host_a.shape[1], host_b.shape[1], stride);
    //printf("host_b shape: [%d, %d]\n", host_b.shape[0], host_b.shape[1]);

    // Print the computed rows and cols
    //printf("Computed rows: %d ", rows);
    //printf("Computed cols: %d\n", cols);

    

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1] * IMAGE_CHANNELS);

    OpenCLConvolution2D(&host_a, &host_b, &host_c, stride);

    // Save the image
    SaveImg(input_file_d, &host_c);

    // Check the result of the convolution
    err = CheckImg(&answer, &host_c);
    CHECK_ERR(err, "CheckImg");

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}