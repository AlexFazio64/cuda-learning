__global__ void vec_add(double *A, double *B, double *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (; i < n; i += step)
        C[i] = A[i] + B[i];
}

__global__ void vec_init(double *v, int n, int value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (; i < n; i += step)
        v[i] = value;
}

int main(int argc, char const *argv[])
{
    int n = 2 << 24;
    double *dA, *dB, *dC;
    auto size = sizeof(double) * n;

    /* allocation on GPU */
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    // grid sizing
    int blocks, blocksize;
    blocksize = 32 * 8;
    blocks = (n + blocksize - 1) / blocksize;

    // initialization
    vec_init<<<blocks, blocksize>>>(dA, n, 1.0f);
    vec_init<<<blocks, blocksize>>>(dB, n, 2.0f);
    vec_init<<<blocks, blocksize>>>(dC, n, 0.0f);
    cudaDeviceSynchronize();

    // computation
    vec_add<<<blocks, blocksize>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    // clean-up device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}