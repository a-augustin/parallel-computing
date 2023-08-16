/* Examples, one bad, one good, of parallel histogram generation. 
 *
 * Compile as follows: gcc -o histogram histogram.c -std=c99 -Wall -O3 -lpthread -lm 
 *
 * Author: Naga Kandasamy
 * Date created: April 18, 2020
 * Date modified: 
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

/* Number of bins in histogram */
#define NUM_BINS 1024

/* Data structure defining what to pass to each worker thread */
typedef struct thread_data_s {
    int tid;                                            /* The thread ID */
    int num_threads;                                    /* Number of threads in the pool */
    int num_elements;                                   /* Number of elements in the input */
    int num_bins;                                       /* Number of bins in histogram */
    int *input;                                         /* Address of input array */
    int *histogram;                                     /* Address of shared histogram */
    pthread_mutex_t *mutex_for_histogram;               /* Address of lock variable protecting shared histogram */
} thread_data_t;


/* Function protoypes */
int *compute_gold(int *input, int num_elements, int num_bins);
int *compute_using_pthreads_v1(int *input, int num_elements, int num_bins, int num_threads);
int *compute_using_pthreads_v2(int *input, int num_elements, int num_bins, int num_threads);
int *compute_using_pthreads_v3(int *input, int num_elements, int num_bins, int num_threads);
void *worker_v1(void *args);
void *worker_v2(void *args);
void *worker_v3(void *args);
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

    fprintf(stderr, "\nGenerating histogram using pthread implementation, version 1\n");
	gettimeofday(&start, NULL);
    
    int *histogram_v2;
	histogram_v2 = compute_using_pthreads_v1(input, num_elements, NUM_BINS, num_threads);
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
	histogram_v3 = compute_using_pthreads_v2(input, num_elements, NUM_BINS, num_threads);
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

    fprintf(stderr, "\nGenerating histogram using pthread implementation, version 3\n");
	gettimeofday(&start, NULL);

    int *histogram_v4;
	histogram_v4 = compute_using_pthreads_v3(input, num_elements, NUM_BINS, num_threads);
    if (histogram_v4 == NULL) {
        fprintf(stderr, "Error generating histogram\n");
        exit(EXIT_FAILURE);
    } else {
        fprintf(stderr, "Histogram generated successfully\n");
    }

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Eexcution time = %f\n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
    if (check_results(histogram_v1, histogram_v4, NUM_BINS) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else
        fprintf(stderr, "TEST FAILED\n");

    free((void *)input);
    free((void *)histogram_v1);
    free((void *)histogram_v2);
    free((void *)histogram_v3);
    free((void *)histogram_v4);

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

/* Pthread implementation, version 1: threads accumulate values into one shared histogram */
int *compute_using_pthreads_v1(int *input, int num_elements, int num_bins, int num_threads)
{
    pthread_t *tid = (pthread_t *)malloc(num_threads * sizeof(pthread_t));          /* Data structure to store thread IDs */
        
    int *histogram = (int *)malloc(sizeof(int) * num_bins);                         /* Create shared histogram */
    memset(histogram, 0, sizeof(int) * num_bins);
    
    pthread_mutex_t mutex_for_histogram;                                          /* Lock for the shared histogram */
    pthread_mutex_init(&mutex_for_histogram, NULL);                               /* Initialize the mutex */

    /* Fork point: Allocate heap memory for required data structures and create the worker threads */
    int i;
    thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_elements = num_elements;
        thread_data[i].num_bins = num_bins;
        thread_data[i].input = input; 
        thread_data[i].histogram = histogram; 
        thread_data[i].mutex_for_histogram = &mutex_for_histogram;
    }

    for (i = 0; i < num_threads; i++)
        pthread_create(&tid[i], NULL, worker_v1, (void *)&thread_data[i]);
					 
    /* Join point: wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(tid[i], NULL);
		 
    /* Free dynamically allocated data structures */
    pthread_mutex_destroy(&mutex_for_histogram);
    free((void *)tid);
    free((void *)thread_data);

    if (check_histogram(histogram, num_elements, num_bins) < 0) 
        return NULL;
    else
        return histogram;
}

/* Worker thread for pthread implementation, version 1 */
void *worker_v1(void *arg)
{
    thread_data_t *thread_data = (thread_data_t *)arg;     /* Typecast argument to pointer to thread_data_t structure */
    
    int offset = thread_data->tid;
    int stride = thread_data->num_threads;
    
    while (offset < thread_data->num_elements) {
        pthread_mutex_lock(thread_data->mutex_for_histogram);
        thread_data->histogram[thread_data->input[offset]]++;
        pthread_mutex_unlock(thread_data->mutex_for_histogram);

        offset += stride;
    }

    pthread_exit(NULL);
}

/* Pthread implementation, version 2: threads first generate local histograms and then 
 * accumulate into the shared histogram */
int *compute_using_pthreads_v2(int *input, int num_elements, int num_bins, int num_threads)
{
    pthread_t *tid = (pthread_t *)malloc(num_threads * sizeof(pthread_t));          /* Data structure to store thread IDs */
        
    int *histogram = (int *)malloc(sizeof(int) * num_bins);                         /* Create shared histogram */
    memset(histogram, 0, sizeof(int) * num_bins);
    
    pthread_mutex_t mutex_for_histogram;                                          /* Lock for the shared histogram */
    pthread_mutex_init(&mutex_for_histogram, NULL);                               /* Initialize the mutex */

    /* Fork point: Allocate heap memory for required data structures and create the worker threads */
    int i;
    thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_elements = num_elements;
        thread_data[i].num_bins = num_bins;
        thread_data[i].input = input; 
        thread_data[i].histogram = histogram; 
        thread_data[i].mutex_for_histogram = &mutex_for_histogram;
    }

    for (i = 0; i < num_threads; i++)
        pthread_create(&tid[i], NULL, worker_v2, (void *)&thread_data[i]);
					 
    /* Join point: wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(tid[i], NULL);
		 
    /* Free dynamically allocated data structures */
    pthread_mutex_destroy(&mutex_for_histogram);
    free((void *)tid);
    free((void *)thread_data);

    if (check_histogram(histogram, num_elements, num_bins) < 0) 
        return NULL;
    else
        return histogram;
}

/* Worker thread for pthread implementation, version 2. Each thread generates 
 * a local histogram indendently and then accumulates the local histogram into the 
 * global histogram. */
void *worker_v2(void *arg)
{
    thread_data_t *thread_data = (thread_data_t *)arg;     /* Typecast argument to pointer to thread_data_t structure */
    
    int offset = thread_data->tid;
    int stride = thread_data->num_threads;

    /* Allocate local histogram */
    int *local_histogram = (int *)malloc(sizeof(int) * thread_data->num_bins); 
    memset(local_histogram, 0, sizeof(int) * thread_data->num_bins);    

    /* Generate local histogram */
    while (offset < thread_data->num_elements) {
        local_histogram[thread_data->input[offset]]++;
        offset += stride;
    }

    /* Accumulate local histogram into the shared histogram */
    int i;
    pthread_mutex_lock(thread_data->mutex_for_histogram);
    
    for (i = 0; i < thread_data->num_bins; i++) 
        thread_data->histogram[i] += local_histogram[i];

    pthread_mutex_unlock(thread_data->mutex_for_histogram);

    pthread_exit(NULL);
}

/* Pthread implementation, version 3: threads first generate local histograms and then 
 * accumulated into shared histogram. This is the lock-free version. */
int *compute_using_pthreads_v3(int *input, int num_elements, int num_bins, int num_threads)
{
    pthread_t *tid = (pthread_t *)malloc(num_threads * sizeof(pthread_t));          /* Data structure to store thread IDs */
        
    int *histogram = (int *)malloc(sizeof(int) * num_bins);                         /* Create shared histogram */
    memset(histogram, 0, sizeof(int) * num_bins);
    
    /* Fork point: Allocate heap memory for required data structures and create the worker threads */
    thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
    int i;
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_elements = num_elements;
        thread_data[i].num_bins = num_bins;
        thread_data[i].input = input; 
        thread_data[i].histogram = histogram; 
        thread_data[i].mutex_for_histogram = NULL;
    }

    for (i = 0; i < num_threads; i++)
        pthread_create(&tid[i], NULL, worker_v3, (void *)&thread_data[i]);
					 
    /* Join point: wait for the workers to finish */
    int j;
    void *ret;
    int *partial_histogram;
    for (i = 0; i < num_threads; i++) {
        pthread_join(tid[i], &ret);                 /* Thread passes pointer to local histogram as a void * data type */
        partial_histogram = (int *)ret;             /* Typecast the return value correctly, to int * data type */
        for (j = 0; j < num_bins; j++) 
            histogram[j] += partial_histogram[j];   /* Accumulate into the shared histrogram */

        free((void *)partial_histogram);            /* Free the local histogram for the thread */
    }
		 
    /* Free dynamically allocated data structures */
    free((void *)tid);
    free((void *)thread_data);

    if (check_histogram(histogram, num_elements, num_bins) < 0) 
        return NULL;
    else
        return histogram;
}

/* Worker thread for pthread implementation, version 2. Each thread generates 
 * a local histogram indendently and then accumulates the local histogram into the 
 * global histogram in a lock free manner. */
void *worker_v3(void *arg)
{
    thread_data_t *thread_data = (thread_data_t *)arg;     /* Typecast argument to pointer to thread_data_t structure */
    
    int offset = thread_data->tid;
    int stride = thread_data->num_threads;

    /* Allocate local histogram */
    int *local_histogram = (int *)malloc(sizeof(int) * thread_data->num_bins); 

    /* Generate local histogram */
    while (offset < thread_data->num_elements) {
        local_histogram[thread_data->input[offset]]++;
        offset += stride;
    }

    /* Return pointer to local histogram to the caller thread */
    pthread_exit((void *)local_histogram);
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


