/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: APril 26, 2023
 *
 * Student name(s): Alisha Augustin and Ehi Simon
 * Date modified: May 7th, 2023
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include "jacobi_solver.h"


typedef struct thread_data_s {
	int tid;
	int num_threads;
	matrix_t A;
	matrix_t B;
	matrix_t *x;
	matrix_t *new_x;
	int max_iter;
	int start_index;
	int end_index;
    pthread_barrier_t *barrier;
	pthread_mutex_t *lock;
	int *converged;
	double *diff;
	int *num_iter;
} thread_data_t;

void *compute_chunk(void* args);
void *compute_stride(void* args);
/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s matrix-size num-threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
        fprintf(stderr, "num-threads: number of worker threads to create\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x_v1;      /* Solution computed by pthread code using chunking */
    matrix_t mt_solution_x_v2;      /* Solution computed by pthread code using striding */

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x_v1 = allocate_matrix(matrix_size, 1, 0);
    mt_solution_x_v2 = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    struct timeval start, stop;

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution  time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x_v1 and mt_solution_x_v2.
     * */
	fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using chunking\n");
	gettimeofday(&start, NULL);
	/* Compute using chunking method */
	compute_using_pthreads_v1(A, mt_solution_x_v1, B, max_iter, num_threads);
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution  time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
	display_jacobi_solution(A, mt_solution_x_v1, B); /* Display statistics */

	fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using striding\n");
	gettimeofday(&start, NULL);
	/* Compute using striding method */
	compute_using_pthreads_v2(A, mt_solution_x_v2, B, max_iter, num_threads);
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution  time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
	display_jacobi_solution(A, mt_solution_x_v2, B); /* Display statistics */

    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x_v1.elements);
    free(mt_solution_x_v2.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads using chunking. 
 * Result must be placed in mt_sol_x_v1. */
void compute_using_pthreads_v1(const matrix_t A, matrix_t mt_sol_x_v1, const matrix_t B, int max_iter, int num_threads)
{
	pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	pthread_attr_t attributes;
	pthread_attr_init(&attributes);
	matrix_t new_x = allocate_matrix(A.num_rows, 1, 0);

	double diff = 0.0;
	int num_iter = 0;
	int converged = 0;

	int i;
	int chunk_size = (int)floor(mt_sol_x_v1.num_rows / num_threads);
	int remainder = mt_sol_x_v1.num_rows % num_threads;

	pthread_barrierattr_t barrier_attributes;
	pthread_barrier_t barrier;
	pthread_barrierattr_init(&barrier_attributes);
	pthread_barrier_init(&barrier, &barrier_attributes, num_threads);

	pthread_mutex_t lock;
	pthread_mutex_init(&lock, NULL);

	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
	for (i = 0; i < num_threads; i++) {
		int start_index = i * chunk_size;
		int end_index = (i + 1) * A.num_columns;

		thread_data[i].tid = i;
		thread_data[i].num_threads = num_threads;
		thread_data[i].A = A;
		thread_data[i].B = B;
		thread_data[i].x = &mt_sol_x_v1;
		thread_data[i].new_x = &new_x;
		thread_data[i].max_iter = max_iter;
		thread_data[i].start_index = start_index;
		thread_data[i].end_index = end_index;
		thread_data[i].barrier = &barrier;
		thread_data[i].lock = &lock;
		thread_data[i].diff = &diff;
		thread_data[i].converged = &converged;
		thread_data[i].num_iter = &num_iter;
	}

	for (i = 0; i < num_threads; i++)
		pthread_create(&thread_id[i], &attributes, compute_chunk, (void *)&thread_data[i]);

	for (i = 0; i < num_threads; i++)
		pthread_join(thread_id[i], NULL);

	free(new_x.elements);
	free((void *)thread_data);
	pthread_barrier_destroy(&barrier);

}

void *compute_chunk(void* args)
{
    thread_data_t *thread_data = (thread_data_t *)args;
    int tid = thread_data->tid;
	matrix_t A = thread_data->A;
	matrix_t *x = thread_data->x;
	matrix_t *new_x = thread_data->new_x;
	matrix_t B = thread_data->B;
	int max_iter = thread_data->max_iter;
	int start_index = thread_data->start_index;
	int end_index = thread_data->end_index;
	double *diff = thread_data->diff;
	int *converged = thread_data->converged;
	pthread_barrier_t *barrier = thread_data->barrier;
	pthread_mutex_t *lock = thread_data->lock;
    int *num_iter = thread_data->num_iter;
    int num_cols = A.num_columns;

    int i, j;

    while (!*converged) {
        if (tid == 0) {
            *diff = 0;
            (*num_iter)++;
        }

        pthread_barrier_wait(barrier);

        for (i = start_index; i < end_index; i++) {
            double sum = 0.0;
            for (j = 0; j < num_cols; j++) {
                if (i != j)
                    sum += A.elements[i * num_cols + j] * x->elements[j];
            }
            new_x->elements[i] = (B.elements[i] - sum) / A.elements[i * (num_cols + 1)];
        }

        double pdiff = 0.0;

        for (i = start_index; i < end_index; i++) {
            pdiff += fabs(new_x->elements[i] - x->elements[i]);
        }

        pthread_mutex_lock(lock);
        *diff += pdiff;
        pthread_mutex_unlock(lock);

        pthread_barrier_wait(barrier);

        double mse = sqrt(*diff);

        if ((mse <= THRESHOLD) || (*num_iter == max_iter)) {
            *converged = 1;
            for (i = start_index; i < end_index; i++) {
                x->elements[i] = new_x->elements[i];
            }
        }
		pthread_barrier_wait(barrier);

		matrix_t *tmp = x;
		x = new_x;
		new_x = tmp;

	}

	pthread_exit(NULL);
}

/* Complete this function to perform the Jacobi calculation using pthreads using striding. 
 * Result must be placed in mt_sol_x_v2. */
void compute_using_pthreads_v2(const matrix_t A, matrix_t mt_sol_x_v2, const matrix_t B, int max_iter, int num_threads)
{

	int tid;

	pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	pthread_attr_t attributes;
	pthread_attr_init(&attributes);

	matrix_t new_x = allocate_matrix(A.num_rows, 1, 0);

	int converged = 0; 
	double diff = 0.0;
	int num_iter = 0;
    int num_rows = A.num_rows;
	pthread_barrierattr_t barrier_attributes;
	pthread_barrier_t barrier;
	pthread_barrierattr_init(&barrier_attributes);
	pthread_barrier_init(&barrier, &barrier_attributes, num_threads);

	/*Initialize Mutex Lock*/
	pthread_mutex_t lock;
	pthread_mutex_init(&lock, NULL);

	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);

	for(tid = 0; tid < num_threads; tid++){
		thread_data[tid].tid = tid;
		thread_data[tid].num_threads = num_threads;
		thread_data[tid].A = A;
		thread_data[tid].B = B;
		thread_data[tid].x = &mt_sol_x_v2;
		thread_data[tid].new_x = &new_x;
		thread_data[tid].max_iter = max_iter;
		thread_data[tid].barrier = &barrier;
		thread_data[tid].lock = &lock;
		thread_data[tid].diff = &diff;
		thread_data[tid].converged = &converged;
		thread_data[tid].num_iter = &num_iter;
		thread_data[tid].start_index = tid;
		thread_data[tid].end_index = num_rows;
	}

	int i;
	for (i = 0; i < num_threads; i++){
		pthread_create(&thread_id[i], &attributes, compute_stride, (void *)&thread_data[i]);
	}

	for (i = 0; i < num_threads; i++){
		pthread_join(thread_id[i], NULL);
	}

	free(new_x.elements);
	free((void *)thread_data);
	pthread_barrier_destroy(&barrier);

}

void *compute_stride(void* args){

	thread_data_t *thread_data = (thread_data_t *)args;
	int tid = thread_data->tid;
	int stride = thread_data->num_threads;
	matrix_t A = thread_data->A;
	matrix_t *x = thread_data->x;
	matrix_t *new_x = thread_data->new_x;
	matrix_t B = thread_data->B;
	int max_iter = thread_data->max_iter;
	int start_index = thread_data->start_index;
	int end_index = thread_data->end_index;
	double *diff = thread_data->diff;
	int *converged = thread_data->converged;
	pthread_barrier_t *barrier = thread_data->barrier;
	pthread_mutex_t *lock = thread_data->lock;
	int *num_iter = thread_data->num_iter;
	double mse, sum;
	int i = start_index, j;
	sum = 0.0;
    int num_cols = A.num_columns;

	while(i < end_index){
		x->elements[i] = B.elements[i];
		i = i + stride;
	}

	while(!*converged) {
		if(tid == 0) {
			*diff = 0;
			(*num_iter)++;
		}

		pthread_barrier_wait(barrier);

		i = start_index;

		while(i < end_index){
			sum = 0.0;
			for(j=0; j < num_cols; j++){
				if(i != j)
					sum += A.elements[i * num_cols + j] * x->elements[j];

			}
			new_x->elements[i] = (B.elements[i] - sum) / A.elements[i *(num_cols + 1)];

			i = i + stride;

		}

		double pdiff = 0.0;

		i = start_index;
		while(i < end_index){
			pdiff += fabs(new_x->elements[i] - x->elements[i]);
			i = i + stride;
		}
		pthread_mutex_lock(lock);
		*diff += pdiff;
		pthread_mutex_unlock(lock);
		pthread_barrier_wait(barrier);

		mse = sqrt(*diff);

		if ((mse <= THRESHOLD) || (*num_iter == max_iter)) { 
			*converged = 1;
			for (i = start_index; i <= end_index; i++)
				thread_data->x->elements[i] = new_x->elements[i];
		}

		pthread_barrier_wait(barrier);

		matrix_t *tmp = x;
		x = new_x;
		new_x = tmp;

	}

	pthread_exit(NULL);
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}



