/* Matrix vector multiplication Ax = b using omp. 
 * A is m x n matrix and x is a n x 1 vector.
 *
 * Compile as follows: gcc -o matrix_vector_multiply matrix_vector_multiply.c -fopenmp -std=c99 -O3 -Wall
 * 
 * Author: Naga Kandasamy
 * Date created: April 25, 2011
 * Date modified: April 26, 2020
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <omp.h>

/* Function prototypes */
void compute_gold(float *, float *, float *, int, int);
void compute_using_openmp_v1(float *, float *, float *, int, int, int, int);
void compute_using_openmp_v2(float *, float *, float *, int, int, int, int);
void compute_using_openmp_v3(float *, float *, float *, int, int, int, int);

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "%s num-rows num-columns num-threads threading-threshold\n", argv[0]);
        fprintf(stderr, "num-rows, num-columns: Number of rows and columns in the A matrix\n");
        fprintf(stderr, "num-threads: Number of threads to create\n");
        fprintf(stderr, "threading-threshold: Threshold beyond which to use multi-threading\n");
        exit(EXIT_SUCCESS);
    }
    
    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);
    int thread_count = atoi(argv[3]);
    int threshold = atoi(argv[4]);

    /* Allocate memory for A, x, and b, and initialize with random values between [-0.5, 0.5] */
    fprintf(stderr, "Creating the vectors\n");
    float *matrix_a = (float *)malloc(sizeof(float) * num_rows * num_cols); 
    float *vector_x = (float *)malloc(sizeof(float) * num_cols); 
    float *vector_b = (float *)malloc(sizeof(float) * num_rows);
    if ((matrix_a == NULL) || (vector_x == NULL) || (vector_b == NULL)) {
        perror("Malloc");
        exit(EXIT_FAILURE);
    }

    int i, j;
    srand(time(NULL));
    for (i = 0; i < num_rows; i++)
        for (j = 0; j < num_cols; j++)
            matrix_a[i*num_cols + j] = rand()/(float)RAND_MAX - 0.5;
    
    for (i = 0; i < num_cols; i++)
            vector_x[i] =  rand()/(float)RAND_MAX - 0.5;
    
    struct timeval start, stop;	
    fprintf(stderr, "\nVector matrix multiplication using single-threaded version\n");
    gettimeofday(&start, NULL);
    compute_gold(matrix_a, vector_x, vector_b, num_rows, num_cols); 
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	
    fprintf(stderr, "\nVector matrix multiplication using omp, version 1\n");
    gettimeofday(&start, NULL);
    compute_using_openmp_v1(matrix_a, vector_x, vector_b, num_rows, num_cols, thread_count, threshold);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

    fprintf(stderr, "\nVector matrix multiplication using omp, version 2\n");
    gettimeofday(&start, NULL);
    compute_using_openmp_v2(matrix_a, vector_x, vector_b, num_rows, num_cols, thread_count, threshold);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

    fprintf(stderr, "\nVector matrix multiplication using OpenMP, version 3\n");
    gettimeofday(&start, NULL);
    compute_using_openmp_v3(matrix_a, vector_x, vector_b, num_rows, num_cols, thread_count, threshold);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Free memory */ 
    free((void *)matrix_a);
    free((void *)vector_b);
    free((void *)vector_x);

    exit(EXIT_SUCCESS);
}

/* Calculate reference soution using a single thread */
void compute_gold(float *matrix_a, float *vector_x, float *vector_b, int num_rows, int num_cols)
{
    int i, j;
    double sum;
    
    for (i = 0; i < num_rows; i++) {
        sum = 0.0;
        for (j = 0; j < num_cols; j++) {
            sum += matrix_a[i*num_cols + j] * vector_x[j];
        }

        vector_b[i] = (float)sum;
    }

    return;
}

/* Calculate the solution using omp with no bells and whistles */
void compute_using_openmp_v1(float *matrix_a, float *vector_x, float *vector_b, int num_rows, int num_cols, int thread_count, int threshold)
{
    int i, j;
	omp_set_num_threads(thread_count);
    double sum; 
	
#pragma omp parallel for default(none) shared(matrix_a, vector_b, vector_x, num_rows, num_cols) private(i, j, sum)
    for (i = 0; i < num_rows; i++) {
        sum = 0.0;
        for (j = 0; j < num_cols; j++) {
            sum += matrix_a[i*num_cols + j] * vector_x[j];
        }
         
        vector_b[i] = (float)sum;
    } /* End of parallel for */
     
    return;
}

/* Implement a thresholding scheme to decide whether to go multi-threaded or not. 
 * This code may exhibit false sharing since the partial sums are being constantly accumulated directly 
 * into the vector_b variable. 
 * */
void compute_using_openmp_v2(float *matrix_a, float *vector_x, float *vector_b, int num_rows, int num_cols, int thread_count, int threshold)
{
    int i, j;
	omp_set_num_threads(thread_count);

#pragma omp parallel for if(num_rows > threshold) default(none) shared(matrix_a, vector_b, vector_x, num_rows, num_cols) private(i, j)
	 for (i = 0; i < num_rows; i++) {
         vector_b[i] = 0;
         for (j = 0; j < num_cols; j++) {
             vector_b[i] += matrix_a[i*num_cols + j] * vector_x[j];
         }
     }

     return;
}

/* Implement a thresholding scheme to decide whether to go multi-threaded or not. 
 * Also reduces the false sharing problem from the previous verion. 
 * */
void compute_using_openmp_v3(float *matrix_a, float *vector_x, float *vector_b, int num_rows, int num_cols, int thread_count, int threshold)
{
	int i, j;
	double sum;

	omp_set_num_threads(thread_count);
	#pragma omp parallel for if(num_rows > threshold) default(none) shared(matrix_a, vector_b, vector_x, num_rows, num_cols) private(i, j, sum)
	 for (i = 0; i < num_rows; i++) {
         sum = 0;
         for (j = 0; j < num_cols; j++) {
             sum += matrix_a[i*num_cols + j] * vector_x[j];
         }
         vector_b[i] = (float)sum;
     }
}



