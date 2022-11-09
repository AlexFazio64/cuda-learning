__global__ void vec_add(double *A, double *B, double *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

__global__ void vec_init(double *v, int n, int value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        v[i] = value;
}

int main(int argc, char const *argv[])
{
    // global
    int n = 2 << 24;
    double *gA, *gB, *gC;
    auto size = sizeof(double) * n;

    // memory allocation
    cudaMallocManaged(&gA, size);
    cudaMallocManaged(&gB, size);
    cudaMallocManaged(&gC, size);

    // grid sizing
    int blocks, blocksize;
    blocksize = 32 * 4;
    blocks = (n + blocksize - 1) / blocksize;

    // Gpu-side initialization
    vec_init<<<blocks, blocksize>>>(gA, n, 1.0f);
    vec_init<<<blocks, blocksize>>>(gB, n, 2.0f);
    vec_init<<<blocks, blocksize>>>(gC, n, 0.0f);
    cudaDeviceSynchronize();

    // computation
    vec_add<<<blocks, blocksize>>>(gA, gB, gC, n);
    cudaDeviceSynchronize();

    // memory clean-up
    cudaFree(gA);
    cudaFree(gB);
    cudaFree(gC);

    delete[] gA, gB, gC;

    return 0;
}