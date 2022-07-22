# matrix A mutiple matrix B
1. using only global memory to calculate.
2. using global memory and shared memory to optimize.
---

## how to run
1. change the path of nvcc in MakeFile
2. ```make```
3. ```./matrix_mul```
---

## matrix store in a continuous memory
![image](https://github.com/xpo0a/cuda_tools/blob/main/0_matrixMul/pit/array_2to1.png)
---

## gpu_memory
![image](https://github.com/xpo0a/cuda_tools/blob/main/0_matrixMul/pit/gpu_memory.png)
---

## the result
![image](https://github.com/xpo0a/cuda_tools/blob/main/0_matrixMul/pit/result.png)

you can find more in [My blog](https://blog.csdn.net/qq906194732/article/details/125640426)
