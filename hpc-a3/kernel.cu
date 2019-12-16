#include <iostream>
#include <math.h>
#include <stdlib.h> // random
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BASE_SEED 0x1234abcd

static int rank = -1;
static int numProcs = -1;

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
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

	int globalN = 1 << 20;
	const int localN = (globalN + numProcs - 1) / numProcs;
	int *globalA = nullptr, *localA = nullptr;

	std::cout << "P" << rank << ": Allocating unified memory with " << localN << " elements\n";

	// Allocate Unified Memory – accessible from CPU or GPU
	gpuErrChk(cudaMallocManaged(&localA, localN * sizeof(int)));

	/* Initialize the random number generator for the given BASE_SEED
	* plus an offset for the MPI rank of the node, such that on every
	* node different numbers are generated.
	*/
	srand(BASE_SEED + rank);

	std::cout << "P" << rank << ": Initializing array\n";

	for (int i = 0; i < localN; i++) {
		localA[i] = rand();
	}

	std::cout << "P" << rank << ": Sorting array\n";

	for (int k = 1; k <= localN / 2; k *= 2) {
		for (int l = k; l >= 1; l /= 2) {
			launchMergeStepKernel(localA, localN, k, l);
			gpuErrChk(cudaDeviceSynchronize());
		}
	}
	
	std::cout << "P" << rank << ": Validating results\n";
	// Check for errors (array should be sorted in ascending order)
	checkArraySorted(localA, localN);
	if (rank == 0) {
		// TODO: merge array
		checkArraySorted(globalA, globalN);
	}

	// Free memory
	gpuErrChk(cudaFree(localA));

	std::cout << "P" << rank << ": Execution finished\n";

	MPI_Finalize();
	return 0;
}