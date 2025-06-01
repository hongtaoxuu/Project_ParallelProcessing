#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "utils.h"
#include "mkl.h"


void test_one_kernel(int kernel_num, int m, int n, int k, double alpha, double *A, double *B, double beta, double *C, double *C_ref){
    printf("\nkernel: %d on M: %d N: %d K: %d:\n", kernel_num, m, n, k);
    double t0,t1;
    int NRep = 3;
    if (kernel_num != 0){//not an MKL implementation
        //first verfiy percision
        test_kernel(kernel_num,m,n,k,alpha,A,B,beta,C);
        if (kernel_num == 2){
            cblas_dgemm(CblasRowMajor, CblasNoTrans,CblasNoTrans,m,n,k,alpha,A,m,B,k,beta,C_ref,m);
        }else{
            cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,m,n,k,alpha,A,m,B,k,beta,C_ref,m);
        }
        if (!verify_matrix(C_ref,C,m*n)) {
            printf("Failed to pass the correctness verification against Intel MKL. Exited.\n");
            exit(-3);
        }
    }
    t0=get_sec();
    for (int n_count=0;n_count<NRep;n_count++){
        test_kernel(kernel_num,m,n,k,alpha,A,B,beta,C);
    }
    t1=get_sec();
    printf("Average elasped time: %f second, performance: %f GFLOPS.\n", (t1-t0)/NRep,2.*1e-9*NRep*m*n*k/(t1-t0));
    copy_matrix(C_ref,C,m*n);//sync C with Intel MKl to prepare for the next run
}



int main(int argc, char *argv[]){
    if (argc < 2 || argc > 3) {
        printf("Please select a kernel (range 0 - 19, here 0 is for Intel MKL).\n");
        printf("./dgemm [KernelNum] [OPTIONAL: MatrixNum]\n");
        exit(-1);
    }
    int SIZE[30]={100,200,300,400,500,600,700,800,900,1000,1100,\
                1200,1300,1400,1500,1600,1700,1800,1900,2000,\
                2100,2200,2300,2400,2500,2600,2700,2800,2900,3000};//testing 100-3000 square matrices
    
    // int MSize[4] = {4000, 8, 32, 144};
    // int NSize[4] = {16000, 16, 16000, 144};
    // int KSize[4] = {128, 16000, 16, 144};
    // int TSSize = 4;
    int MSize[8] = {4000, 8, 32, 144, 16, 4, 440, 40};
    int NSize[8] = {16000, 16, 16000, 144, 12344, 54, 193, 1127228};
    int KSize[8] = {128, 16000, 16, 144, 16, 606841, 11, 40};
    int TSSize = 8;
    
    bool tsmm = true;
    int only_test = -1;
    // bool tsmm = false;

    // int MSize2[] = {16, 4, 442, 40};
    // int NSize2[] = {12344, 54, 193, 1127228};
    // int KSize2[] = {16, 606841, 11, 40};

    
    // (m,n,k)=(4000,16000,128) (8,16,16000),(32,16000,16),(144,144,144)
    // (m,n,k)=(16,12344,16),(4,54,606841),(442,193,11),(40,1127228,40)


    printf("argc %d, argv %s, %s, %s", argc, argv[0], argv[1], argv[2]);
    int kernel_num=atoi(argv[1]);
    if(argc == 3){
        only_test=atoi(argv[2]);
        printf("the only test matrix is %d", only_test);
    }
    printf("kernel_num %d", kernel_num);
    if (kernel_num<0||kernel_num>19) {
        printf("Please enter a valid kernel number (0-19).\n");
        exit(-2);
    }
    int m, n, k,max_size=16000;
    int n_count,N=3,upper_limit;
    if (kernel_num<=4&&kernel_num!=0) upper_limit=10;
    else upper_limit=30;
    double *A=NULL,*B=NULL,*C=NULL,*C_ref=NULL;
    // double alpha = 2.0, beta = 0.;//two arbitary input parameters
    double alpha = 2.0, beta = 1;//two arbitary input parameters

    A=(double *)malloc(sizeof(double)*max_size*max_size);
    B=(double *)malloc(sizeof(double)*max_size*max_size);
    C=(double *)malloc(sizeof(double)*max_size*max_size);
    C_ref=(double *)malloc(sizeof(double)*max_size*max_size);

    // randomize_matrix(A,max_size,max_size);randomize_matrix(B,max_size,max_size);randomize_matrix(C,max_size,max_size);copy_matrix(C,C_ref,max_size*max_size);
    if(tsmm == true){
        for(int i_count=0;i_count<TSSize;i_count++){
            if(only_test != -1){
                i_count = only_test;
            }
            m=MSize[i_count];
            n=NSize[i_count];
            k=KSize[i_count];
            printf("\nkernel: %d on M: %d N: %d K: %d:\n", kernel_num, m, n, k);
            test_one_kernel(kernel_num,m,n,k,alpha,A,B,beta,C,C_ref);
            if(only_test != -1){
                break;
            }
        }
        
    }else{
        for(int i_count=0;i_count<upper_limit;i_count++){
            m=n=k=SIZE[i_count];
            test_one_kernel(kernel_num,m,n,k,alpha,A,B,beta,C,C_ref);
            
        }
    }
    free(A);free(B);free(C);free(C_ref);
    return 0;
}