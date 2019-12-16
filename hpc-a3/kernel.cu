#include <iostream>
#include <math.h>
#include <stdlib.h> // random
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BASE_SEED 0x1234abcd

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__
void swap(int &a, int &b) {
	int tmp = a;
	a = b;
	b = tmp;
}

__device__
void compAndSwap(int* a, const int startIdx, const int endIdx, const bool dirAscending) {
	if (dirAscending) {
		if (a[startIdx] > a[endIdx]) {
			swap(a[startIdx], a[endIdx]);
		}
	} else { // descending
		if (a[startIdx] < a[endIdx]) {
			swap(a[startIdx], a[endIdx]);
		}
	}
}

__global__
void mergeStep(int *a, const int aLength, const int k, const int l) {
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int runId = tId; runId < aLength; runId += stride) {
		int startIdx = (runId / l) * 2*l + (runId % l);
		int endIdx = startIdx + l;
		if (endIdx < aLength) {
			int zoneId = startIdx / (2 * k);
			bool dirAscending = !(zoneId % 2);
			compAndSwap(a, startIdx, endIdx, dirAscending);
		}
	}
}

inline void launchMergeStepKernel(int *a, const int aLength, const int k, const int l) {
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the 
					 // maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
		mergeStep, 0, 0);
	// Round up according to array size 
	gridSize = (aLength + blockSize - 1) / blockSize;
	mergeStep << <gridSize, blockSize >> > (a, aLength, k, l);
}

void checkArraySorted(int *a, const int N) {
	bool sorted = true;
	for (int i = 1; i < N; i++) {
		if (a[i-1] > a[i]) {
			std::cout << "ERROR: Array not sorted at positions " << i - 1 << " and " << i << "\n";
			sorted = false;
			break;
		}
	}

	if (sorted) {
		std::cout << "Array sorted successfully\n";
	}
}

int main(int argc, char *argv[]) {
	int N = 1 << 20;
	int *a;

	std::cout << "Allocating unified memory with " << N << " elements\n";

	// Allocate Unified Memory – accessible from CPU or GPU
	gpuErrChk(cudaMallocManaged(&a, N * sizeof(int)));

	/* Initialize the random number generator for the given BASE_SEED
	* plus an offset for the MPI rank of the node, such that on every
	* node different numbers are generated.
	*/
	int rank = 0; // MPI placeholder
	srand(BASE_SEED + rank);

	std::cout << "Initializing array\n";

	for (int i = 0; i < N; i++) {
		a[i] = rand();
	}

	std::cout << "Sorting array\n";

	// Run kernel on 1M elements on the GPU
	/*int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;*/
	for (int k = 1; k <= N / 2; k *= 2) {
		for (int l = k; l >= 1; l /= 2) {
			launchMergeStepKernel(a, N, k, l);
			gpuErrChk(cudaDeviceSynchronize());
		}
	}
	
	std::cout << "Validating results\n";
	// Check for errors (array should be sorted in ascending order)
	checkArraySorted(a, N);

	// Free memory
	gpuErrChk(cudaFree(a));

	std::cout << "Execution finished\n";

	return 0;
}