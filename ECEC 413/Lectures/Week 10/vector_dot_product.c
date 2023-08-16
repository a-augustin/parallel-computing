/* Vector dot product A.B using OpenMP and SSE extensions.
 * 
 * Author: Naga Kandasamy
 * Date modified: June 4, 2020
 *
 * Compile as follows: gcc -fopenmp vector_dot_product.c -o vector_dot_product -std=c99 -lm -O3
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
#include <omp.h>
#include <xmmintrin.h>

#define NUM_ELEMENTS  256 * 1000000
#define NUM_THREADS 16

float compute_gold(float *, float *, int);
float compute_gold_using_sse(float *, float *, int);
float compute_using_openmp(float *, float *, int);
float compute_using_openmp_plus_sse(float *, float *, int);
void *dot_product(void *);

int main(int argc, char **argv)
{
    int num_elements = NUM_ELEMENTS;
	
    /* Create the vectors A and B and fill them with random numbers between [-.5, .5]. */
	float *vector_a, *vector_b;
	void *allocation;
	int status;

    printf ("\nCreating input vectors\n");

    /* Allocate sizeof(float) * num_elements bytes and place address of 
     * allocated memory in allocation.  The address of the allocated memory is a 
     * multiple of alignment, which is a power of two and multiple of sizeof(float). 
     * Alignment is on 16-byte boundaries (or 4 floats) in our case.
       */
	status = posix_memalign(&allocation, 16, sizeof(float) * num_elements);
	if (status != 0) {
			  printf("Error allocating aligned memory\n");
			  exit(EXIT_FAILURE);
	}
	vector_a = (float *)allocation;

	status = posix_memalign(&allocation, 16, sizeof(float) * num_elements);
	if (status != 0) {
			  printf("Error allocating aligned memory. \n");
			  exit(EXIT_FAILURE);
	}
	vector_b = (float *)allocation;

	srand(time(NULL));
    int i;
	for (i = 0; i < num_elements; i++) {
		vector_a[i] = rand()/(float)RAND_MAX - 0.5;
		vector_b[i] = rand()/(float)RAND_MAX - 0.5;
	}

	/* Dot product using the reference, single-threaded solution */
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
    printf("\nComputing dot product with reference solution\n");
	float reference = compute_gold(vector_a, vector_b, num_elements); 
	gettimeofday(&stop, NULL);

	printf("Reference solution = %f\n", reference);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Dot product using single-threaded solution with SSE extensions */
	gettimeofday(&start, NULL);
	printf("\nComputing dot product with SSE extensions\n");
	float reference_sse = compute_gold_using_sse(vector_a, vector_b, num_elements); 
	gettimeofday(&stop, NULL);

	printf("Reference solution with SSE = %f \n", reference_sse);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));


    /* Dot product using OpenMP */
	gettimeofday(&start, NULL);
	printf("\nComputing dot product using OMP\n");
	float omp_result = compute_using_openmp(vector_a, vector_b, num_elements);
	gettimeofday(&stop, NULL);

	printf("OpenMP solution = %f\n", omp_result);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Dot product using OpenMP and SSE */
	gettimeofday(&start, NULL);
	printf("\nComputing dot product using OMP with SSE extensions\n");
	float omp_result_sse = compute_using_openmp_plus_sse(vector_a, vector_b, num_elements);
	gettimeofday(&stop, NULL);

	printf("OpenMP solution with SSE = %f\n", omp_result_sse);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

	free((void *)vector_a);
	free((void *)vector_b);

    exit(EXIT_SUCCESS);
}

/* Reference soution using a single thread */
float compute_gold(float *vector_a, float *vector_b, int num_elements) 
{
	double sum = 0.0;
    int i;

	for (i = 0; i < num_elements; i++)
        sum += vector_a[i] * vector_b[i];
	
	return (float)sum;
}

/* Reference soution using a single thread with SSE extensions */
float compute_gold_using_sse(float *vector_a, float *vector_b, int num_elements)  {
    __m128 partial_sum;                                     /* 128 bit register to store the partial sum */
	__m128 m0, m1, m2, m3, m4;                              /* 128 bit temporary registers */
	float tmp[4] __attribute__ ((aligned(16)));             /* FP values aligned on a 16-byte boundary */
    float sum = 0.0;
    int i;

	if ((num_elements % 4) == 0) {                      /* If the number of elements is a multiple of four, use SSE */
        partial_sum = _mm_load_ss(&sum);                /* Load sum into the low word and clear the upper words; r0 = *sum, r3 = r2 = r1 = 0.0 */
        
        for (i = 0; i < num_elements / 4; i++) {
            m0 = _mm_load_ps(&vector_a[4*i]);           /* Load four single precision FP values that are byte aligned */
            m1 = _mm_load_ps(&vector_b[4*i]);
            m2 = _mm_mul_ps(m0, m1);                    /* Multiply packed scalar */
            partial_sum = _mm_add_ps(partial_sum, m2);  /* Accumulate into the partial_sum register */
        }
			  
        /* partial_sum = {r3, r2, r1, r0} */
        m3 = _mm_movelh_ps(partial_sum, partial_sum);   /* Move lower two FP values of the input registers to the result; m3 = {r1, r0, r1, r0} */
        m4 = _mm_movehl_ps(partial_sum, partial_sum);   /* Move the higher two FP values of the input registers to the result; m4 = {r3, r2, r3, r2} */
        partial_sum = _mm_add_ps(m3, m4);
        _mm_store_ps(tmp, partial_sum);                 /* tmp = {r3, r2, r1, r0} */

        sum = tmp[0] + tmp[1];
    
    }
	else {
        printf("Number of elements in the vector is not a multiple of four. I'm lazy; so, using refrence implementation without SSE extensions\n");
        /* If the number of elements n is not a multiple of 4, you would use loop strip mining or sectioning in which you'll create two loops:
         * 1. The first loop will process four floating-point data items per iteration i using SSE until i < (n - n % 4).
         * 2. The second loop will process any left over elements without using SSE; so called scalar cleanup.
         */
        sum = compute_gold(vector_a, vector_b, num_elements);
	}

	return sum;
}

/* Dot product using multiple threads */
float compute_using_openmp (float *vector_a, float *vector_b, int num_elements) 
{
	int i;
	double sum = 0.0;

	omp_set_num_threads(NUM_THREADS); /* Set the number of threads */

#pragma omp parallel for reduction(+:sum)
	for (i = 0; i < num_elements; i++) {
		  sum = sum + vector_a[i] * vector_b[i];
	}
	
	return (float)sum;	
}

/* Dot product using multiple threads in which each thread uses SSE extensions. 
 * OpenMP is used to extract coarse-level parallelism. 
 * SSE is used within each OpenMP threads to extract any fine-grained parallelism 
 * */
float compute_using_openmp_plus_sse(float *vector_a, float *vector_b, int num_elements)   
{
	int i;
	double sum = 0.0;
	int chunk_size = floorf(num_elements/(float)NUM_THREADS);   /* Chunk size for each thread */
	
	if (chunk_size % 4 == 0) {
        omp_set_num_threads(NUM_THREADS); 

#pragma omp parallel private(i) shared(chunk_size, sum)      
        {		 
            float z = 0;
            __m128 partial_sum = _mm_load_ss(&z);           /* Load z into the low word and clear the upper words; r0 = *sum, r3 = r2 = r1 = 0.0 */
            __m128 m0, m1, m2;                              /* 128 bit temporary registers */ 
            float tmp[4] __attribute__ ((aligned(16)));     /* Array of four FP values aligned on a 16 byte boundary */

            // int tid = omp_get_thread_num();
            // printf("Thread %d is processing\n", tid); 

#pragma omp for schedule(static, chunk_size)
            for (i = 0; i < num_elements / 4; i++) {
                m0 = _mm_load_ps(&vector_a[4*i]);           /* Load four single precision FP values that are byte aligned */
                m1 = _mm_load_ps(&vector_b[4*i]);
                m2 = _mm_mul_ps(m0, m1); 
                partial_sum = _mm_add_ps(partial_sum, m2);  /* Accumulate into the partial_sum register */
            }
            
            /* partial_sum = {r3, r2, r1, r0} */
            _mm_store_ps(tmp, partial_sum); /* tmp = {r3, r2, r1, r0} */

#pragma omp critical
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }
	}
	else {
        printf("Chunk size for the threads is not a multiple of four. Using OpenMP without SSE\n");
        /* Use strip mining as discussed earlier */
        sum = compute_using_openmp(vector_a, vector_b, num_elements);
	}

	return (float)sum;	
}


