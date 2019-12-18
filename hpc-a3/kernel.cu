#include <iostream>
#include <math.h>
#include <stdlib.h> // random
#include <sstream>
#include <string>
#include "assert.h"
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PRINT_INTERVAL 10000
#define LINE_SEPARATOR "-------------------------------------\n"

static int rank = -1;
static int numProcs = -1;

#define ASCENDING true
#define DESCENDING false

// Program arguments.
enum ExecMode {
	CPU,
	GPU
};
static ExecMode execMode = GPU;
static uint64_t globalN = 0;
static unsigned int baseSeed = 0;

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
__host__ __device__
void swap(T &a, T &b) {
	T tmp = a;
	a = b;
	b = tmp;
}

// It must be a power of 2 greater or equal to the number of processes.
// Power of 2 test checks if only one bit is set and is adapted from https://stackoverflow.com/a/108360/11539572
void assertInputConditions(uint64_t globalN) {
	assert(globalN > 0 && globalN >= numProcs
		&& (globalN & (globalN - 1)) == 0);
}

__host__ __device__
void compAndSwap(unsigned int* a, const uint64_t startIdx, const uint64_t endIdx, const bool dirAscending) {
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
void mergeStep(unsigned int *a, const int aLength, const uint64_t k, const uint64_t l) {
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (uint64_t runId = tId; runId < aLength; runId += stride) {
		uint64_t startIdx = (runId / l) * 2*l + (runId % l);
		uint64_t endIdx = startIdx + l;
		if (endIdx < aLength) {
			uint64_t zoneId = startIdx / (2 * k);
			bool dirAscending = !(zoneId % 2);
			compAndSwap(a, startIdx, endIdx, dirAscending);
		}
	}
}

inline void launchMergeStepKernel(unsigned int *a, const int aLength, const uint64_t k, const uint64_t l) {
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

bool isArraySorted(unsigned int *a, const uint64_t n) {
	for (int i = 1; i < n; i++) {
		if (a[i-1] > a[i]) {
			std::cout << "ERROR: Array not sorted at [" << i - 1 << "] = " << a[i-1] << " and [" << i << "] = " << a[i] << std::endl;
			return false;
		}
	}
	return true;
}

void handleUsage(int argc) {
	if (argc != 4) {
		if (rank == 0) {
			printf("Usage:\n\tmpirun [-np X] bitonic.out mode exponent base_seed\n");
		}
		MPI_Finalize();
		exit(1);
	}
}

/**
 * Initialize MPI and related variables and print header.
 */
void init(int *argc, char **argv[]) {
	int  namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(argc, argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &namelen);
	handleUsage(*argc);

	execMode = ((*argv)[1] == std::string("gpu")) ? GPU : CPU;
	std::istringstream iss1((*argv)[2]);
	unsigned int exponent = 0;
	iss1 >> exponent;
	globalN = (uint64_t) 1 << exponent;
	std::istringstream iss2((*argv)[3]);
	iss2 >> std::hex >> baseSeed;

	if (rank == 0) {
		std::cout << "Name: " << processor_name << "\n";
		std::cout << "Number of processes: " << numProcs << "\n";
		std::cout << "Execution mode: " << (execMode == GPU ? "GPU\n" : "CPU\n");
		std::cout << "Array size: 2^" << exponent << " = " << globalN << "\n";
		std::cout << "Base seed: 0x" << std::hex << baseSeed << std::dec << "\n";
		std::cout << LINE_SEPARATOR;
	}

	assertInputConditions(globalN);
	MPI_Barrier(MPI_COMM_WORLD);
}

void printResults(unsigned int *a, uint64_t n, const double startTime, const double localProcTime, const double endTime) {
	std::cout << LINE_SEPARATOR;
	std::cout << "Sorted array:\n";
	for (int i = 0; i < n; i += PRINT_INTERVAL) {
		std::cout << a[i] << " ";
	}
	std::cout << "\n" << LINE_SEPARATOR;
	std::cout << "Local (P0) processing time: " << localProcTime - startTime << "s\n";
	std::cout << "Global processing time: " << endTime - startTime << "s\n";
	std::cout << LINE_SEPARATOR;
}

void localSortGPU(unsigned int *localA, uint64_t localN) {
	for (uint64_t k = 1; k <= localN / 2; k *= 2) {
		for (uint64_t l = k; l >= 1; l /= 2) {
			launchMergeStepKernel(localA, localN, k, l);
			gpuErrChk(cudaDeviceSynchronize());
		}
	}
}

void seqBitonicMerge(unsigned int *a, uint64_t i, uint64_t n, bool dir) {
	if (n > 1) {
		uint64_t m = n / 2;
		for (uint64_t j = i; j < i + m; j++) {
			compAndSwap(a, j, j + m, dir);
		}
		seqBitonicMerge(a, i, m, dir);
		seqBitonicMerge(a, i + m, m, dir);
	}
}

void seqBitonicSort(unsigned int *a, uint64_t i, uint64_t n, bool dir) {
	if (n > 1) {
		uint64_t m = n / 2;
		seqBitonicSort(a, i, m, ASCENDING);
		seqBitonicSort(a, i + m, m, DESCENDING);
		seqBitonicMerge(a, i, n, dir);
	}
}

void localSortCPU(unsigned int *localA, uint64_t localN) {
	seqBitonicSort(localA, 0, localN, ASCENDING);
}

void freeResources(unsigned int* globalA, unsigned int *localA) {
	if (execMode == GPU) {
		gpuErrChk(cudaFree(localA));
	} else {
		delete[] localA;
	}

	if (rank == 0) {
		delete[] globalA;
	}
}

int main(int argc, char *argv[]) {
	init(&argc, &argv);
	double startTime = -1.0, localProcTime = -1.0, endTime = -1.0;

	// Round up, although redundant when globalN and numProcs are both powers of 2
	const uint64_t localN = (globalN + numProcs - 1) / numProcs;
	unsigned int *globalA = nullptr, *localA = nullptr;

	if (rank == 0) {
		globalA = new unsigned int[globalN];
	}

	std::cout << "P" << rank << ": Allocating "
		<< (execMode == GPU ? "unified" : "system")
		<< " memory with " << localN << " elements" << std::endl;

	// Allocate Unified Memory - accessible from CPU or GPU
	if (execMode == GPU) {
		gpuErrChk(cudaMallocManaged(&localA, localN * sizeof(unsigned int)));
	} else {
		localA = new unsigned int[localN];
	}

	/* Initialize the random number generator for the given BASE_SEED
	* plus an offset for the MPI rank of the node, such that on every
	* node different numbers are generated.
	*/
	srand(baseSeed + rank);

	std::cout << "P" << rank << ": Initializing array" << std::endl;

	unsigned int lowBound = (rank + 0.0) / numProcs * ((unsigned) RAND_MAX + 1);
	unsigned int highBound = (rank + 1.0) / numProcs * ((unsigned) RAND_MAX + 1);
	for (int i = 0; i < localN; i++) {
		localA[i] = genRandomNumber(lowBound, highBound);
	}

	std::cout << "P" << rank << ": Sorting array" << std::endl;

	startTime = MPI_Wtime();
	// Calculate parts of array in each node
	switch (execMode) {
	case GPU:
		localSortGPU(localA, localN);
		break;
	case CPU:
		localSortCPU(localA, localN);
		break;
	default:
		std::cerr << "P" << rank << ": wat\n";
	}
	localProcTime = MPI_Wtime();
	// Merge arrays from all nodes
	MPI_Gather(localA, localN, MPI_UINT32_T, globalA, localN, MPI_UINT32_T, 0, MPI_COMM_WORLD);
	endTime = MPI_Wtime();

	std::cout << "P" << rank << ": Validating results" << std::endl;
	// Check for local errors (array should be sorted in ascending order)
	if (isArraySorted(localA, localN)) {
		std::cout << "P" << rank << ": Local array sorted successfully" << std::endl;
	} else {
		std::cout << "P" << rank << ": Local array sort failed" << std::endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	std::cout << "P" << rank << ": Execution finished" << std::endl;
	MPI_Barrier(MPI_COMM_WORLD);

	// Check for global errors in merged array (should be sorted in ascending order)
	if (rank == 0) {
		if (isArraySorted(globalA, globalN)) {
			std::cout << "Global array sorted successfully" << std::endl;
		} else {
			std::cout << "Global array sort failed" << std::endl;
		}
		printResults(globalA, globalN, startTime, localProcTime, endTime);
	}
	
	freeResources(globalA, localA);
	MPI_Finalize();
	return 0;
}