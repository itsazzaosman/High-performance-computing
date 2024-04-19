// Group Members
// 1. Azza Osman
// 2. Aime Barema


/* Exercise 0 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

#define MAXTHREADS 100
#define NTHREADS 4
#define N 1024*1024

float RES[NTHREADS];
pthread_t THREAD_ID[NTHREADS];
float U[N] __attribute__((aligned(16)));



double now() {
   // Retourne l'heure actuelle en secondes
   struct timeval t; double f_t;
   gettimeofday(&t, NULL);
   f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
   return f_t;
}


/* Exercise 2 */
void init(float *U, int n) {
    unsigned int i;
    float x;

    for (i=0; i<n; i++) {
        x = (float)rand() / RAND_MAX;
        U[i] = 1 / (1 + pow(x, 2));
    }
}


/* Exercise 1 */
float norm(float *U, int n) {
    unsigned int i;
    double S;

    S = 0;
    for(i=0; i<n; i++) {
        S += U[i] * U[i];
    }

    return sqrt(S);
}


/* Exercise 4 */
float vect_norm(float *U, int n) {
    unsigned int i;
    __m256 *mm_U = (__m256 *)U;

    __m256 mm_S = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0); 
    for(i=0; i<n/8; i++) {
       mm_S = _mm256_add_ps(mm_S, _mm256_mul_ps(mm_U[i], mm_U[i]));
    }

    float rv[8];
    _mm256_store_ps(rv, mm_S);

    double S = 0;
    for(i=0; i<8; i++) {
        S += rv[i];
    }
    
    return sqrt(S);
}


/* Exercise 5 */
void *thread_norm(void *thread_id) {
    double pnorm = 0;
    int id; // logical id of the thread
    id = (int)thread_id;
    int i;

    for(i = id*(N/NTHREADS); i < (id+1)*(N/NTHREADS); i++) {
        pnorm += U[i] * U[i];
    }
    RES[id] = pnorm;

    pthread_exit(NULL);
}


/* Exercise 6 */
float parallel_norm(float *U, int n) {
    int rc;
    int t;

    for(t=0; t<NTHREADS; t++) {
        rc = pthread_create(&THREAD_ID[t], NULL, thread_norm, (void *)t);

        if (rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for(t=0; t<NTHREADS; t++) {
        rc = pthread_join(THREAD_ID[t], NULL);

        if (rc) {
            printf("ERROR: pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    double S = 0;
    for(t=0; t<NTHREADS; t++) {
        S += RES[t];
    }

    return sqrt(S);
}


/* Exercise 3 and 7 */
int main() {
    double t;
    init(U, N);

    float ENORM;

    t = now();
    ENORM = norm(U, N);
    t = now() - t;

    printf("\n");
    printf("Scalar computing\n");
    printf("----------------\n");
    printf("The Euclidean norm of U equals: %f\n", ENORM);
    printf("Execution time: %f second(s)\n\n", t);

    t = now();
    ENORM = vect_norm(U, N);
    t = now() - t;

    printf("Vector computing\n");
    printf("----------------\n");
    printf("The Euclidean norm of U equals: %f\n", ENORM);
    printf("Execution time: %f second(s)\n\n", t);

    t = now();
    ENORM = parallel_norm(U, N);
    t = now() - t;

    printf("Multi-thread\n");
    printf("------------\n");
    printf("The Euclidean norm of U equals: %f\n", ENORM);
    printf("Execution time: %f second(s)\n", t);
}

// VERY GOOD 


