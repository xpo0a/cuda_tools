#include <stdio.h>
#include <math.h>
#include "error.cuh"
#include <time.h>
#define BLOCK_SIZE 16

// 仅使用 global memory 进行 矩阵乘法
__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
// 矩阵A：m*n, 矩阵B：n*k,  *a: A,  *b: B, *c: C = A * B
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

// 使用 shared memory 进行 矩阵乘法 优化
__global__ void gpu_matrix_mult_shared(int *d_a, int *d_b, int *d_result, int n) 
{
    // shared memory 初始化
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // 行 = 前面有几行 * 每行元素数 + 这行的第几个
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    // 列 = 前有几列 * 每列元素数 + 这列第几个
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    // 将两个打矩阵相乘，拆分为 gridDim个 小矩阵对应相乘 再求和，以 sub 0表示：
    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        // 计算第 sub 部分矩阵的行索引
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        // 对 tile_a 进行赋值，即 global memory -> shared memory，多余部分置0
        tile_a[threadIdx.y][threadIdx.x] = row<n && (sub * BLOCK_SIZE + threadIdx.x)<n? d_a[idx]:0;
        // 计算第 sub 部分矩阵的行索引
        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        // 对 tile_b 进行赋值，多余部分置0
        tile_b[threadIdx.y][threadIdx.x] = col<n && (sub * BLOCK_SIZE + threadIdx.y)<n? d_b[idx]:0;


        __syncthreads();  // 要同步，程序别退出
        // 计算 行、列 乘积，同时 对不同 sub 部分进行求和
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();  // 要同步，程序别退出
    }
    // 那么d_result中序号row * n + col的元素值 就为 tmp
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

// 使用 cpu 进行 矩阵乘法
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    // 遍历行
    /*
     A矩阵存在*h_a中，B矩阵存在*h_b中，新矩阵存在*h_result
     */
    for (int i = 0; i < m; ++i) 
    {
        // 遍历 列
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            // 新矩阵中每个元素 需要进行 n次 乘加运算
            for (int h = 0; h < n; ++h) 
            {
                //索引计算：在A矩阵取第i行，B矩阵取第j列进行计算
                tmp += h_a[i * n + h] * h_b[h * k + j];  // 累加A矩阵第i行，B矩阵第j列乘积
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    // 矩阵维度 m*n n*k -> m*k
    int m=1000;
    int n=500;
    int k=1000;

    clock_t start11,stop11;

    // 分配 内存 并检查
    int *h_a, *h_b, *h_c, *h_cc, *h_cs;
    CHECK(cudaMallocHost((void **) &h_a, sizeof(int)*m*n));
    CHECK(cudaMallocHost((void **) &h_b, sizeof(int)*n*k));
    CHECK(cudaMallocHost((void **) &h_c, sizeof(int)*m*k));
    CHECK(cudaMallocHost((void **) &h_cc, sizeof(int)*m*k));
    CHECK(cudaMallocHost((void **) &h_cs, sizeof(int)*m*k));

    // 创建 event 用于统计时间
    cudaEvent_t start, stop,stop_share;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventCreate(&stop_share));


    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = 1;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = 0;
        }
    }
    printf("The matrix A is %dx%d, The matrix B is %dx%d\n",m, k, k, n);
    int *d_a, *d_b, *d_c, *d_c_share;
    // 分配显存 并检查
    CHECK(cudaMalloc((void **) &d_a, sizeof(int)*m*n));
    CHECK(cudaMalloc((void **) &d_b, sizeof(int)*n*k));
    CHECK(cudaMalloc((void **) &d_c, sizeof(int)*m*k));
    CHECK(cudaMalloc((void **) &d_c_share, sizeof(int)*m*k));

    CHECK(cudaEventRecord(start));
    // copy matrix A and B from host to device memory
    CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice));

    // 新矩阵 需要多少个线程
    // 向上取整，保证分配的 线程数 多余被拆分的 小问题
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 定义Grid维度，其中包含的 Grid 数量
    dim3 dimGrid(grid_cols, grid_rows);
    // 定义Block维度，其中包含的 Block 数量
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // 实际执行 use global memor
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m,n,k);    

    // 将新 矩阵，GPU -> CPU
    CHECK(cudaMemcpy(h_c, d_c, (sizeof(int)*m*k), cudaMemcpyDeviceToHost));
    //cudaThreadSynchronize();
    // 结束 stop event
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));


    // 实际执行 use shared memor
    gpu_matrix_mult_shared<<<dimGrid, dimBlock>>>(d_a, d_b, d_c_share, n);
    CHECK(cudaMemcpy(h_cs, d_c_share, (sizeof(int)*m*k), cudaMemcpyDeviceToHost));

    // 结束 stop stop_share
    CHECK(cudaEventRecord(stop_share));
    CHECK(cudaEventSynchronize(stop_share));
    
    float elapsed_time, elapsed_time_share;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CHECK(cudaEventElapsedTime(&elapsed_time_share, stop, stop_share));
    printf("Time_GPU_global = %g ms.\n", elapsed_time);
    printf("Time_GPU_share = %g ms.\n", elapsed_time_share);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));    

    // 实际执行 CPU
    start11 = time(NULL);
    cpu_matrix_mult(h_a, h_b, h_c, m, n, k);
    stop11 = time(NULL);
    printf("Time_CPU = %g s.\n", difftime(stop11, start11));
    int ok = 1;
    for (int i = 0; i < m; ++i)
    { 
        for (int j = 0; j < k; ++j)
        {
            if(fabs(h_cs[i*k + j] - 0)>(1.0e-10))
            {
                printf("hcs: %d hc: %d  ",h_cs[i*k + j], h_c[i*k + j]);
                ok = 0;
            }
        }
    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }
    
    // free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));
    return 0;
}