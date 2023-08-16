/* Examples of histogram generation using the critical and atomic directives in OpenMP. 
 *
 * Compile as follows: gcc -o histogram histogram.c -fopenmp -std=c99 -Wall -O3 -lm 
 *
 * Author: Naga Kandasamy
 * Date created: May 3, 2023
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Number of bins in histogram */
#define NUM_BINS 1024

/* Function protoypes */
int *compute_gold(int *input, int num_elements, int num_bins);
int *compute_using_omp_v1(int *input, int num_elements, int num_bins, int thread_count);
int *compute_using_omp_v2(int *input, int num_elements, int num_bins, int thread_count);
int check_histogram(int *histogram, int num_elements, int num_bins);
int check_results(int *A, int *B, int num_bins);

int main(int argc, char **argv) 
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s num-elements num-threads\n", argv[0]);
		exit(EXIT_SUCCESS);	
	}

	int num_elements = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
	
    /* Generate input data to be integer values between 0 and (NUM_BINS - 1) */
    fprintf(stderr, "Generating input data\n");
	int *input = (int *)malloc(sizeof(int) * num_elements);
    int i;
    srand(time(NULL));
    for (i = 0; i < num_elements; i++)
        input[i] = floorf((NUM_BINS - 1) * (rand()/(float)RAND_MAX));

	fprintf(stderr, "\nGenerating histogram using reference implementation\n");
    struct timeval start, stop;	
	gettimeofday(&start, NULL);

    int *histogram_v1;
	histogram_v1 = compute_gold(input, num_elements, NUM_BINS);
    if (histogram_v1 == NULL) {
        fprintf(stderr, "Error generating histogram\n");
        exit(EXIT_FAILURE);
    } else {
        fprintf(stderr, "Histogram generated successfully\n");
    }

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Eexcution time = %f\n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);

    fprintf(stderr, "\nGenerating histogram using omp using critical section\n");
	gettimeofday(&start, NULL);
    
    int *histogram_v2;
	histogram_v2 = compute_using_omp_v1(input, num_elements, NUM_BINS, num_threads);
    if (histogram_v2 == NULL) {
        fprintf(stderr, "Error generating histogram\n");
        exit(EXIT_FAILURE);
    } else {
        fprintf(stderr, "Histogram generated successfully\n");
    }

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Eexcution time = %f\n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
    if (check_results(histogram_v1, histogram_v2, NUM_BINS) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else
        fprintf(stderr, "TEST FAILED\n");

    fprintf(stderr, "\nGenerating histogram using pthread implementation, version 2\n");
	gettimeofday(&start, NULL);

    int *histogram_v3;
	histogram_v3 = compute_using_omp_v2(input, num_elements, NUM_BINS, num_threads);
    if (histogram_v3 == NULL) {
        fprintf(stderr, "Error generating histogram\n");
        exit(EXIT_FAILURE);
    } else {
        fprintf(stderr, "Histogram generated successfully\n");
    }

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Eexcution time = %f\n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
    if (check_results(histogram_v1, histogram_v3, NUM_BINS) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else
        fprintf(stderr, "TEST FAILED\n");

    free((void *)input);
    free((void *)histogram_v1);
    free((void *)histogram_v2);
    free((void *)histogram_v3);

    exit(EXIT_SUCCESS);
}

/* Reference implementation */
int *compute_gold(int *input, int num_elements, int num_bins)
{
	int *histogram = (int *)malloc(sizeof(int) * num_bins); 
    if (histogram == NULL)
        return NULL;
    memset(histogram, 0, sizeof(int) * num_bins);    

    /* Generate histogram */
    for (int i = 0; i < num_elements; i++)
        histogram[input[i]]++;

    /* Check correctness */
    if (check_histogram(histogram, num_elements, num_bins) < 0)
        return NULL;
    else
        return histogram;
}

/* OpenMP implementation, version 1: threads accumulate values into one shared histogram which is treated as a critical section.
 * Note: this is a bad implementation from a performance view point since threads are serialized through the critical section. 
 */
int *compute_using_omp_v1(int *input, int num_elements, int num_bins, int thread_count)
{
    int *histogram = (int *)malloc(sizeof(int) * num_bins);                         /* Create shared histogram */
    memset(histogram, 0, sizeof(int) * num_bins);
    
    int i; 
#pragma omp parallel private(i) shared(histogram, input, num_elements) num_threads(thread_count)
    {
#pragma omp for 
        for (i = 0; i < num_elements; i++)
         {
#pragma omp critical
             {
              histogram[input[i]]++;    
             }
         } 
    }
    
    if (check_histogram(histogram, num_elements, num_bins) < 0) 
        return NULL;
    else
        return histogram;
}

/* OpenMP implementation, version 2: threads accumulate values into one shared histogram using atomic add.
 * Note: this achieves better performance compared to version 1 since threads can operate concurrently on different 
 * histogram bins using atomic ALU operations supported by the processor. Serialization will only occur if multiple threads
 * attempt to update the same bin concurrently. 
 */
int *compute_using_omp_v2(int *input, int num_elements, int num_bins, int thread_count)
{
    int *histogram = (int *)malloc(sizeof(int) * num_bins);                         /* Create shared histogram */
    memset(histogram, 0, sizeof(int) * num_bins);
    
    int i; 
#pragma omp parallel private(i) shared(histogram, input, num_elements) num_threads(thread_count)
    {
#pragma omp for 
        for (i = 0; i < num_elements; i++)
         {
#pragma omp atomic
              histogram[input[i]]++;    
         } 
    }
    
    if (check_histogram(histogram, num_elements, num_bins) < 0) 
        return NULL;
    else
        return histogram;
}


/* Check correctness of the histogram: sum of the bins must equal number of input elements */
int check_histogram(int *histogram, int num_elements, int num_bins)
{
    int i;
	int sum = 0;
	for (i = 0; i < num_bins; i++)
		sum += histogram[i];

	if (sum == num_elements)
		return 0;
	else
        return -1;
}

/* Check results against the reference solution */
int check_results(int *A, int *B, int num_bins)
{
	int i;
    int diff = 0;
    for (i = 0; i < num_bins; i++)
		diff += abs(A[i] - B[i]);

    if (diff == 0)
        return 0;
    else
        return -1;
}


