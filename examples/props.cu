#include <stdio.h>

int main()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	printf("Number of devices: %d\n", nDevices);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	printf("%s\n", prop.name);
	printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
	printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
	printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
	printf("  Warp-size: %d\n\n", prop.warpSize);

	printf("  Streaming Multiprocessors: %d\n", prop.multiProcessorCount);
	printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("  Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
	printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
}