#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

// C: 4x4 blocking + SIMD 

void scale_c_k5(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void mydgemm_cpu_opt_k5(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k5(C,M,N,LDC,beta);
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            double tmp=C(i,j);
            for (k=0;k<K;k++){
                tmp += alpha*A(i,k)*B(k,j);
            }
            C(i,j) = tmp;
        }
    }
}

void mydgemm_cpu_v5(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k5(C,M,N,LDC,beta);
    int M4=M&-4,N4=N&-4;
    __m256d valpha = _mm256_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    for (i=0;i<M4;i+=4){
        for (j=0;j<N4;j+=4){
            __m256d c0 = _mm256_setzero_pd();
            __m256d c1 = _mm256_setzero_pd();
            __m256d c2 = _mm256_setzero_pd();
            __m256d c3 = _mm256_setzero_pd();
            for (k=0;k<K;k++){
                __m256d a = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i,k)));
                __m256d b0 = _mm256_broadcast_sd(&B(k,j));
                __m256d b1 = _mm256_broadcast_sd(&B(k,j+1));
                __m256d b2 = _mm256_broadcast_sd(&B(k,j+2));
                __m256d b3 = _mm256_broadcast_sd(&B(k,j+3));
                c0 = _mm256_fmadd_pd(a,b0,c0);
                c1 = _mm256_fmadd_pd(a,b1,c1);
                c2 = _mm256_fmadd_pd(a,b2,c2);
                c3 = _mm256_fmadd_pd(a,b3,c3);
            }
            _mm256_storeu_pd(&C(i,j), _mm256_add_pd(c0,_mm256_loadu_pd(&C(i,j))));
            _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(c1,_mm256_loadu_pd(&C(i,j+1))));
            _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(c2,_mm256_loadu_pd(&C(i,j+2))));
            _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(c3,_mm256_loadu_pd(&C(i,j+3))));
        }
    }
    if (M4==M&&N4==N) return;
    // boundary conditions
    if (M4!=M) mydgemm_cpu_opt_k5(M-M4,N,K,alpha,A+M4,LDA,B,LDB,1.0,&C(M4,0),LDC);
    if (N4!=N) mydgemm_cpu_opt_k5(M4,N-N4,K,alpha,A,LDA,&B(0,N4),LDB,1.0,&C(0,N4),LDC);
}
