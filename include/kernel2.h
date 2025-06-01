#define A(i,j) A[(i)*LDA+(j)]
#define B(i,j) B[(i)*LDB+(j)]
#define C(i,j) C[(i)*LDC+(j)]

//baseline + row-first

void scale_c_k2(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void mydgemm_cpu_v2(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k2(C,M,N,LDC,beta);
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){ 
            for (k=0;k<K;k++){
                C(i,j) += alpha*A(i,k)*B(k,j);
            } 
        }
    }
}
