#include "../libs/utils.h"
#include <stdio.h>
#include <chrono>
using namespace std;

int main(int argc, char const *argv[])
{
	int N = 2 << 24;
	double *A, *B, *C;

	A = new double[N];
	B = new double[N];
	C = new double[N];

	for (int i = 0; i < N; ++i)
	{
		A[i] = 1.0f;
		B[i] = 2.0f;
	}

	long before = timeMillis();

	for (int i = 0; i < N; ++i)
		C[i] = A[i] + B[i];

	long after = timeMillis();

	printf("it took %ld ms", after - before);

	delete[] A, B, C;
	return 0;
}