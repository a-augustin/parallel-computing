/* Vector addition of two vectors using OpenMP. 
 * Shows the use of the nowait clause. 
 *
 * Compile as follows: gcc -fopenmp vector_sum.c -o vector_sum -std=c99 -Wall -O3
 *
 * Author: Naga Kandasamy
 * Date created: April 4, 2011
 * Date modified: April 26, 2020
 *
 *  */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <omp.h>

/* Function prototypes */
double compute_using_openmp_v1(float *, float *, float *, float *, int, int);
double compute_using_openmp_v2(float *, float *, float *, float *, int, int);

int main(int argc, char **argv)
{
    if (argc < 3){
        fprintf(stderr, "Usage: %s num-elements num-threads\n", argv[0]);
        fprintf(stderr, "num-elements: number of elements in the vectors\n");
        fprintf(stderr, "num-threads: number of threads\n");
        exit(EXIT_FAILURE);
    }
  
    int num_elements = atoi(argv[1]);	
    int num_threads = atoi(argv[2]);

    /* Create vectors A and B and populate them with random numbers between [-.5, .5] */
    fprintf(stderr, "Creating random vectors\n");
    float *vector_a = (float *)malloc(sizeof(float) * num_elements);
    float *vector_b = (float *)malloc(sizeof(float) * num_elements);
    float *vector_c = (float *)malloc(sizeof(float) * num_elements);
    float *vector_d = (float *)malloc(sizeof(float) * num_elements);

    srand(time(NULL));  
    int i;
    for (i = 0; i < num_elements; i++) {
      vector_a[i] = ((float)rand() / (float)RAND_MAX) - 0.5;
      vector_b[i] = ((float)rand() / (float)RAND_MAX) - 0.5;
      vector_c[i] = ((float)rand() / (float)RAND_MAX) - 0.5;
      vector_d[i] = ((float)rand() / (float)RAND_MAX) - 0.5;
    }

    /* Compute the vector sum using OpenMP */   
    double sum;
    struct timeval start, stop;
    fprintf(stderr, "\nComputing the vector sum using OpenMP\n");
  
    gettimeofday(&start, NULL);
  
    sum = compute_using_openmp_v1(vector_a, vector_b, vector_c, vector_d, num_elements, num_threads);
    fprintf(stderr, "Sum = %f\n", sum);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

  /* Compute the vector sum using OpenMP. 
   * Use the nowait clause to speed up the execution by reducing the barrier synchronization points. 
   * */
  fprintf(stderr, "\nComputing the vector sum using OpenMP. Minimizing the barrier sync points using the nowait clause\n");
  gettimeofday(&start, NULL);
  
  sum = compute_using_openmp_v2(vector_a, vector_b, vector_c, vector_d, num_elements, num_threads);
  fprintf(stderr, "Sum = %f\n", sum);
  
  gettimeofday(&stop, NULL);
  fprintf(stderr, "Execution time = %fs. \n \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));
  
  free((void *)vector_a);
  free((void *)vector_b);
  free((void *)vector_c);
  free((void *)vector_d);

  exit(EXIT_SUCCESS);
}

/* Compute the vector sum, version one */
double compute_using_openmp_v1(float *vector_a, float *vector_b, float *vector_c, float *vector_d, int num_elements, int num_threads)
{
  int i, j;
  double sum = 0.0;

  omp_set_num_threads(num_threads);	/* Set number of threads */

#pragma omp parallel private(i, j)
  {
#pragma omp for
    for (i = 0; i < num_elements; i++)
      vector_a[i] = vector_a[i] + vector_b[i];

#pragma omp for
    for (j = 0; j < num_elements; j++)
      vector_c[j] = vector_c[j] + vector_d[j];

#pragma omp for reduction(+: sum)
    for (i = 0; i < num_elements; i++)
      sum = sum + vector_a[i] + vector_c[i];
  }

  return sum;
}

/* Computes vector sum, version uses the nowait clause */
double compute_using_openmp_v2(float *vector_a, float *vector_b, float *vector_c, float *vector_d, int num_elements, int num_threads)
{
  int i, j;
  double sum = 0.0;

  omp_set_num_threads(num_threads);	

#pragma omp parallel private(i, j)
  {
#pragma omp for  nowait
    for (i = 0; i < num_elements; i++)
      vector_a[i] = vector_a[i] + vector_b[i];

#pragma omp for nowait
    for (j = 0; j < num_elements; j++)
      vector_c[j] = vector_c[j] + vector_d[j];
#pragma omp barrier

#pragma omp for reduction(+: sum) nowait
    for (i = 0; i < num_elements; i++)
      sum = sum + vector_a[i] + vector_c[i];
  }

  return sum;
}
