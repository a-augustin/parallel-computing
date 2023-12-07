/* FIXME: Edit this file to complete the functionality of 2D separable 
 * convolution on the GPU. You may add additional kernel functions 
 * as necessary. 
 */
__constant__ float kernel_c[2 * HALF_WIDTH + 1];

__global__ void convolve_rows_kernel_naive(float* result, float* input, float* kernel, int num_cols, int num_rows)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= num_rows)
        return;

    for (int x = 0; x < num_cols; x++)
    {
        int j1 = x - HALF_WIDTH;
        int j2 = x + HALF_WIDTH;

        if (j1 < 0)
            j1 = 0;
        if (j2 >= num_cols)
            j2 = num_cols - 1;

        int i1 = j1 - x;

        j1 = j1 - x + HALF_WIDTH;
        j2 = j2 - x + HALF_WIDTH;

        result[thread_id * num_cols + x] = 0.0f;
        for (int i = i1, j = j1; j <= j2; j++, i++)
            result[thread_id * num_cols + x] += kernel[j] * input[thread_id * num_cols + x + i];
    }
}

__global__ void convolve_columns_kernel_naive(float* result, float* input, float* kernel, int num_cols, int num_rows)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= num_rows)
        return;

    for (int x = 0; x < num_cols; x++)
    {
        int j1 = thread_id - HALF_WIDTH;
        int j2 = thread_id + HALF_WIDTH;

        if (j1 < 0)
            j1 = 0;
        if (j2 >= num_rows)
            j2 = num_rows - 1;

        int i1 = j1 - thread_id;

        j1 = j1 - thread_id + HALF_WIDTH;
        j2 = j2 - thread_id + HALF_WIDTH;

        result[thread_id * num_cols + x] = 0.0f;
        for (int i = i1, j = j1; j <= j2; j++, i++)
            result[thread_id * num_cols + x] += kernel[j] * input[thread_id * num_cols + x + (i * num_cols)];
    }
}

__global__ void convolve_rows_kernel_optimized(float* result, float* input, int num_cols, int num_rows)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= num_rows)
        return;

    for (int x = 0; x < num_cols; x++)
    {
        int j1 = x - HALF_WIDTH;
        int j2 = x + HALF_WIDTH;

        if (j1 < 0)
            j1 = 0;
        if (j2 >= num_cols)
            j2 = num_cols - 1;

        int i1 = j1 - x;

        j1 = j1 - x + HALF_WIDTH;
        j2 = j2 - x + HALF_WIDTH;

        result[thread_id + num_cols * x] = 0.0f;
        for (int i = i1, j = j1; j <= j2; j++, i++)
            result[thread_id + (num_cols * x)] += kernel_c[j] * input[thread_id * num_cols + x + i];
    }
}

__global__ void convolve_columns_kernel_optimized(float* result, float* input, int num_cols, int num_rows)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= num_rows)
        return;

    for (int x = 0; x < num_cols; x++)
    {
        int j1 = thread_id - HALF_WIDTH;
        int j2 = thread_id + HALF_WIDTH;

        if (j1 < 0)
            j1 = 0;
        if (j2 >= num_rows)
            j2 = num_rows - 1;

        int i1 = j1 - thread_id;
        j1 = j1 - thread_id + HALF_WIDTH;
        j2 = j2 - thread_id + HALF_WIDTH;

        result[thread_id * num_cols + x] = 0.0f;
        for (int i = i1, j = j1; j <= j2; j++, i++)
            result[thread_id * num_cols + x] += kernel_c[j] * input[thread_id + (num_cols * x) + i];
    }
}
