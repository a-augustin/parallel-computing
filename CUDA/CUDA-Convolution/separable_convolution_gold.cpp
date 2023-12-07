/* Reference code for separable convolution.
 * 
 * Author: Naga Kandasamy
 * Date modified: May 26, 2020
 *
 * Note: Original code from Pastimatch source (www.plastimatch.org)
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" void compute_gold(float *, float *, int, int, int);
void convolve_rows(float *, float *, float *, int, int, int);
void convolve_columns(float *, float *, float *, int, int, int);
extern "C" float *create_kernel(float, int);


/* Create convolution kernel with specified half width */
float *create_kernel(float coeff, int HALF_WIDTH)
{
    int width = 2 * HALF_WIDTH + 1;
		  
    printf("Creating Gaussian convolution kernel of width = %d\n", width); 
    
    float *gaussian_kernel = (float *)malloc(width * sizeof(float));
    float sum = 0.0;
    int i;	  
    int j = 0;	  
    for (i = (-HALF_WIDTH); i <= HALF_WIDTH; i++) {
        gaussian_kernel[j] = exp((((float)(-(i*i)))/(2 * coeff * coeff)));
        sum += gaussian_kernel[j];
        j++;
    }
		  
    for (i = 0; i < width; i++)
        gaussian_kernel[i] /= sum;

    return gaussian_kernel;
}

/* Row convolution filter */
void convolve_rows(float *result, float *input, float *kernel,\
              int num_cols, int num_rows, int HALF_WIDTH)
{
    int i, i1;
    int j, j1, j2;
    int x, y;

    for (y = 0; y < num_rows; y++) {
        for (x = 0; x < num_cols; x++) {
            j1 = x - HALF_WIDTH;
            j2 = x + HALF_WIDTH;
            /* Clamp at the edges of the matrix */
            if (j1 < 0) 
                j1 = 0;
            if (j2 >= num_cols) 
                j2 = num_cols - 1;

            /* Obtain relative position of starting element from element being convolved */
            i1 = j1 - x; 
            
            j1 = j1 - x + HALF_WIDTH; /* Obtain operating width of the kernel */
            j2 = j2 - x + HALF_WIDTH;

            /* Convolve along row */
            result[y * num_cols + x] = 0.0f;
            for(i = i1, j = j1; j <= j2; j++, i++)
                result[y * num_cols + x] += 
                    kernel[j] * input[y * num_cols + x + i];
        }
    }

    return;
}

/* Column convolution filter */
void convolve_columns(float *result, float *input, float *kernel,\
                  int num_cols, int num_rows, int HALF_WIDTH)
{
    int i, i1;
    int j, j1, j2;
    int x, y;

    for (y = 0; y < num_rows; y++) {
        for(x = 0; x < num_cols; x++) {
            j1 = y - HALF_WIDTH;
            j2 = y + HALF_WIDTH;
            /* Clamp at the edges of the matrix */
            if (j1 < 0) 
                j1 = 0;
            if (j2 >= num_rows) 
                j2 = num_rows - 1;

            /* Obtain relative position of starting element from element being convolved */
            i1 = j1 - y; 
            
            j1 = j1 - y + HALF_WIDTH; /* Obtain the operating width of the kernel.*/
            j2 = j2 - y + HALF_WIDTH;

            /* Convolve along column */            
            result[y * num_cols + x] = 0.0f;
            for (i = i1, j = j1; j <= j2; j++, i++)
                result[y * num_cols + x] += 
                    kernel[j] * input[y * num_cols + x + (i * num_cols)];
        }
    }

    return;
}

/* Perform separable convolution on CPU */
void compute_gold(float *matrix_a, float *kernel,\
              int num_cols, int num_rows, int HALF_WIDTH)
{
    int num_elements = num_rows * num_cols;
    float *matrix_b = (float *)malloc(sizeof(float) * num_elements);

    /* Convolve over rows: matrix_a is the input matrix and 
     * convolved matrix is stored in matrix_b
    */	 
    printf("Convolving over rows\n");
    convolve_rows(matrix_b, matrix_a, kernel, num_cols, num_rows, HALF_WIDTH);
	  
    /* Convolve over columns: matrix_b is the input matrix and 
     * convolved matrix is stored in matrix_a
     */
    printf("Convolving over columns\n");
    convolve_columns(matrix_a, matrix_b, kernel, num_cols, num_rows, HALF_WIDTH);

    return;
}
