/* Given two vectors A and B, calculate C[i] = sqrt(A[i]^2 + B[i]^2) + 0.5 using SSE2 extensions.
 * Elements of the vectors are single-precision floating point walues (32 bits each)
 * 
 * Author: Naga Kandasamy
 * Date modified: June 4, 2020
 * Compile as follows: gcc first_sse_example.c -o first_sse_example -std=c99 -lm
 *
 * A list of SSE compiler intrinsics can be obtained from:
 * http://msdn.microsoft.com/en-US/library/yc6byew8(v=VS.80).aspx
 * http://msdn.microsoft.com/en-us/library/t467de55(v=vs.90).aspx
 * http://msdn.microsoft.com/en-us/library/4atda1f2(v=VS.90).aspx
 * http://msdn.microsoft.com/en-us/library/a2050yhk(v=vs.80).aspx
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <malloc.h>
#include <xmmintrin.h>

#define NUM_ELEMENTS 64 * 4 * 1000000

void compute_gold(float *, float *, float *, int);
void compute_using_sse_intrinsics(float *, float *, float *, int);
void check_results(float *, float *, int);

int main(int argc, char **argv)
{
    int num_elements = NUM_ELEMENTS;
    int status;
    float *vector_a, *vector_b, *reference, *result;
    void *allocation;

    /* Create the vectors A and B and fill them with random numbers between [-.5, .5]. 
     * IMPORTANT: Each float array processed by SSE instructions should have 16 byte alignment. 
     */
 
    printf("\nCreating the vectors\n");
    status = posix_memalign(&allocation, 16, sizeof(float) * num_elements);
    if (status != 0) {
        printf("Error allocating aligned memory\n");
        exit(EXIT_FAILURE);
    }
    vector_a = (float *)allocation;

    status = posix_memalign(&allocation, 16, sizeof(float) * num_elements);
    if (status != 0) {
        printf("Error allocating aligned memory\n");
        exit(EXIT_FAILURE);
    }
    vector_b = (float *)allocation;

    srand(time(NULL));		
  
    int i;
    for (i = 0; i < num_elements; i++) {
        vector_a[i] = rand() / (float)RAND_MAX - 0.5;
        vector_b[i] = rand() / (float)RAND_MAX - 0.5;
    }

    /* Compute vector C using the reference, single-threaded solution */
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    printf("\nComputing the reference solution\n");
  
    status = posix_memalign(&allocation, 16, sizeof(float) * num_elements);
    if (status != 0) {
        printf("Error allocating aligned memory\n");
        exit(EXIT_FAILURE);
    }
  
    reference = (float *)allocation;
    
    compute_gold(vector_a, vector_b, reference, num_elements);
    
    gettimeofday(&stop, NULL);
    printf("Execution time = %fs\n", 
            (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

  /* Compute the vector using SSE compiler instrinsics */
    gettimeofday(&start, NULL);
    printf("\nComputing the result vector using SSE compiler intrinsics\n");
    
    status = posix_memalign(&allocation, 16, sizeof(float) * num_elements);
    if (status != 0) {
        printf("Error allocating aligned memory\n");
        exit(EXIT_FAILURE);
    }

    result = (float *)allocation;
    compute_using_sse_intrinsics(vector_a, vector_b, result, num_elements);
  
    gettimeofday(&stop, NULL);
    printf("Execution time = %fs\n",
            (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    /* Check the two solutions */
    printf("\nChecking results\n");
    check_results(reference, result, num_elements);

    /* Free memory here */ 
    free((void *)vector_a);
    free((void *)vector_b);
    free((void *)reference);
    free((void *)result);

    exit(EXIT_SUCCESS);
}

/* The reference soution */
void compute_gold(float *vector_a, float *vector_b, 
                   float *vector_c, int num_elements)  
{
    int i;
    for (i = 0; i < num_elements; i++)
        vector_c[i] = sqrt(vector_a[i] * vector_a[i] + vector_b[i] * vector_b[i]) + 0.5;
}

/* Calculate using SSE compiler intrinsics */
void compute_using_sse_intrinsics(float *vector_a, float *vector_b, 
                                  float *vector_c, int num_elements)  
{
    int loop_bound = num_elements / 4;          /* Assumption that loop_boud is a multiple of 4 */
    __m128 m1, m2, m3, m4;	                    /* 128 bit SSE registers */
    __m128 *src_1 = (__m128 *)vector_a;	        /* Pointers to 128 bits at a time */
    __m128 *src_2 = (__m128 *)vector_b;
    __m128 *dest = (__m128 *)vector_c;
  
    __m128 m0_5 = _mm_set_ps1(0.5f);	        /* Set m0_5[0, 1, 2, 3,] = 0.5 */

    int i; 
    for (i = 0; i < loop_bound; i++) {
        m1 = _mm_mul_ps(*src_1, *src_1);	    /* m1 = (*src_1) * (*src_1) */
        m2 = _mm_mul_ps(*src_2, *src_2);	    /* m2 = (*src_2) * (*src_2) */
        m3 = _mm_add_ps(m1, m2);	            /* m3 = m1 + m2 */
        m4 = _mm_sqrt_ps(m3);	                /* m4 = sqrt(m3) */
        *dest = _mm_add_ps(m4, m0_5);	        /* *dest = m4 + 0.5 */
      
        /* Move on to the next set of four floats */
        src_1++;
        src_2++;
        dest++;
    }
}

/* Check reference and SSE results */
void check_results(float *a, float *b, int num_elements) 
{
    double total_diff = 0.0;
    double temp;
    double max_diff = 0.0;

    int i;
    for (i = 0; i < num_elements; i++) {
        temp = fabs(a[i] - b[i]);
        if(temp > max_diff)
            max_diff = temp;

        total_diff += temp; 
    }

    printf("Average difference between computed elements = %f.\n", total_diff/(float)num_elements);
    printf("Maximum difference between computed elements = %f. \n", max_diff);
}
