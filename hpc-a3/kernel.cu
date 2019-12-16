#include <iostream>
#include <math.h>
#include <stdlib.h> // random
#include "assert.h"
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BASE_SEED 0x1234abcd

static int rank = -1;
static int numProcs = -1;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline unsigned int genRandomNumber(unsigned int lowBound, unsigned int highBound) {
	return rand() % (highBound - lowBound + 1) + lowBound;
}

template<typename T>
__device__
void swap(T &a, T &b) {
	T tmp = a;
	a = b;
	b = tmp;
}

// It must be a power of 2 greater or equal to the number of processes.
// Power of 2 test checks if only one bit is set and is adapted from https://stackoverflow.com/a/108360/11539572
void assertInputConditions(unsigned int globalN) {
	assert(globalN > 0 && globalN >= numProcs
		&& (globalN & (globalN - 1)) == 0);
}

__device__
void compAndSwap(unsigned int* a, const int startIdx, const int endIdx, const bool dirAscending) {
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
void mergeStep(unsigned int *a, const int aLength, const int k, const int l) {
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

inline void launchMergeStepKernel(unsigned int *a, const int aLength, const int k, const int l) {
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

bool isArraySorted(unsigned int *a, const int N) {
	for (int i = 1; i < N; i++) {
		if (a[i-1] > a[i]) {
			std::cout << "ERROR: Array not sorted at [" << i - 1 << "] = " << a[i-1] << " and [" << i << "] = " << a[i] << std::endl;
			return false;
		}
	}
	return true;
}

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

	int globalN = 1 << 20;
	assertInputConditions(globalN);

	// Round up, although redundant when globalN and numProcs are both powers of 2
	const int localN = (globalN + numProcs - 1) / numProcs;
	unsigned int *globalA = nullptr, *localA = nullptr;

	if (rank == 0) {
		globalA = new unsigned int[globalN];
	}

	std::cout << "P" << rank << ": Allocating unified memory with " << localN << " elements" << std::endl;

	// Allocate Unified Memory – accessible from CPU or GPU
	gpuErrChk(cudaMallocManaged(&localA, localN * sizeof(unsigned int)));

	/* Initialize the random number generator for the given BASE_SEED
	* plus an offset for the MPI rank of the node, such that on every
	* node different numbers are generated.
	*/
	srand(BASE_SEED + rank);

	std::cout << "P" << rank << ": Initializing array" << std::endl;

	unsigned int lowBound = (rank + 0.0) / numProcs * ((unsigned) RAND_MAX + 1);
	unsigned int highBound = (rank + 1.0) / numProcs * ((unsigned) RAND_MAX + 1);
	for (int i = 0; i < localN; i++) {
		localA[i] = genRandomNumber(lowBound, highBound);
	}

	std::cout << "P" << rank << ": Sorting array" << std::endl;

	for (int k = 1; k <= localN / 2; k *= 2) {
		for (int l = k; l >= 1; l /= 2) {
			launchMergeStepKernel(localA, localN, k, l);
			gpuErrChk(cudaDeviceSynchronize());
		}
	}
	
	std::cout << "P" << rank << ": Validating results" << std::endl;
	// Check for errors (array should be sorted in ascending order)
	if (isArraySorted(localA, localN)) {
		std::cout << "P" << rank << ": Local array sorted successfully" << std::endl;
	} else {
		std::cout << "P" << rank << ": Local array sort failed" << std::endl;
	}

	MPI_Gather(localA, localN, MPI_UINT32_T, globalA, localN, MPI_UINT32_T, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		if (isArraySorted(globalA, globalN)) {
			std::cout << "Global array sorted successfully" << std::endl;
		} else {
			std::cout << "Global array sort failed" << std::endl;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	gpuErrChk(cudaFree(localA));

	std::cout << "P" << rank << ": Execution finished" << std::endl;

	MPI_Finalize();
	return 0;
}