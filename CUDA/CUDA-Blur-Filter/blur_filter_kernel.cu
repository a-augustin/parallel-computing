/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void blur_filter_kernel (const float *in, float *out, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * size + col;

    /* Check if the thread is within the image bounds */
    if (row < size && col < size) {
        /* Perform the box blur on the current pixel */
        int num_neighbors = 0;
        float sum = 0.0f;

        /* Iterate over the neighbors of the current pixel */
        for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
            for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
                int neighbor_row = row + i;
                int neighbor_col = col + j;

                /* Check if the neighbor is within the image bounds */
                if (neighbor_row >= 0 && neighbor_row < size && neighbor_col >= 0 && neighbor_col < size) {
                    num_neighbors++;
                    sum += in[neighbor_row * size + neighbor_col];
                }
            }
        }

        /* Calculate the average value for the current pixel */
        out[index] = sum / num_neighbors;
    }
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
