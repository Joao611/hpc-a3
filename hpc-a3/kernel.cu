#include <iostream>
#include <math.h>
#include <stdlib.h> // random
#include <sstream>
#include <string>
#include <algorithm>
#include "assert.h"
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PRINT_INTERVAL 10000
#define LINE_SEPARATOR "-------------------------------------"

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

struct PsrsData {
	uint64_t numChunks, chunkSize;
	uint64_t sampleSize, sampleInterval;
	uint64_t numPivots;

	PsrsData(const uint64_t localN, const uint64_t localEffVram) {
		chunkSize = localEffVram / sizeof(unsigned int);
		numChunks = localN / chunkSize;
		sampleSize = numChunks * numChunks;
		sampleInterval = localN / (numChunks * numChunks);
		numPivots = numChunks - 1;
	}
};

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

uint64_t closestLowerPowerOf2(uint64_t n) {
	uint64_t p = log2(n);
	return pow(2, p);
}

uint64_t closestHigherPowerOf2(uint64_t n) {
	uint64_t p = log2(n);
	if (pow(2, p) == n) {
		return pow(2, p);
	} else {
		return pow(2, p + 1);
	}
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
		std::cout << LINE_SEPARATOR << std::endl;
	}

	assertInputConditions(globalN);
	MPI_Barrier(MPI_COMM_WORLD);
}

void printResults(unsigned int *a, uint64_t n, const double startTime, const double localProcTime, const double endTime) {
	std::cout << LINE_SEPARATOR << "\n";
	std::cout << "Sorted array:\n";
	for (int i = 0; i < n; i += PRINT_INTERVAL) {
		std::cout << a[i] << " ";
	}
	std::cout << "\n" << LINE_SEPARATOR << "\n";
	std::cout << "Local (P0) processing time: " << localProcTime - startTime << "s\n";
	std::cout << "Global processing time: " << endTime - startTime << "s\n";
	std::cout << LINE_SEPARATOR << std::endl;
}


void localBitonicSortGPU(unsigned int *localA, uint64_t localN) {
	for (uint64_t k = 1; k <= localN / 2; k *= 2) {
		for (uint64_t l = k; l >= 1; l /= 2) {
			launchMergeStepKernel(localA, localN, k, l);
			gpuErrChk(cudaDeviceSynchronize());
		}
	}
}

void sortPsrsChunksGPU(unsigned int *localA, uint64_t localN, const PsrsData &psrsData) {
	unsigned int *chunkGPU = nullptr;
	const uint64_t chunkSizeBytes = psrsData.chunkSize * sizeof(unsigned int);
	gpuErrChk(cudaMalloc(&chunkGPU, chunkSizeBytes));
	for (uint64_t chunkStart = 0; chunkStart < localN; chunkStart += psrsData.chunkSize) {
		unsigned int *chunkStartCPU = localA + chunkStart;
		gpuErrChk(cudaMemcpy(chunkGPU, chunkStartCPU, chunkSizeBytes, cudaMemcpyHostToDevice));
		localBitonicSortGPU(chunkGPU, psrsData.chunkSize);
		gpuErrChk(cudaMemcpy(chunkStartCPU, chunkGPU, chunkSizeBytes, cudaMemcpyDeviceToHost));
	}
	gpuErrChk(cudaFree(chunkGPU));
}

void createPsrsSample(unsigned int *sample, unsigned int *localA, const uint64_t localN, const PsrsData &psrsData) {
	const uint64_t samplesPerChunk = psrsData.numChunks;
	for (uint64_t chunk = 0; chunk < psrsData.numChunks; chunk++) {
		for (uint64_t i = 0; i < psrsData.chunkSize; i += psrsData.sampleInterval) {
			sample[chunk * samplesPerChunk + i / psrsData.sampleInterval] = localA[chunk * psrsData.chunkSize + i];
		}
	}
}

void sortPsrsSampleGPU(unsigned int *sample, const PsrsData &psrsData) {
	unsigned int *sampleGPU = nullptr;
	const uint64_t sampleSizeBytes = psrsData.sampleSize * sizeof(unsigned int);
	gpuErrChk(cudaMalloc(&sampleGPU, sampleSizeBytes));
	gpuErrChk(cudaMemcpy(sampleGPU, sample, sampleSizeBytes, cudaMemcpyHostToDevice));

	localBitonicSortGPU(sampleGPU, psrsData.sampleSize);

	gpuErrChk(cudaMemcpy(sample, sampleGPU, sampleSizeBytes, cudaMemcpyDeviceToHost));
	gpuErrChk(cudaFree(sampleGPU));
}

void pickPivots(unsigned int *pivots, const unsigned int *sample, const PsrsData &psrsData, const uint64_t localN) {
	const uint64_t p = psrsData.numChunks / 2;
	for (int i = 0; i < psrsData.numPivots; i++) {
		pivots[i] = sample[psrsData.numChunks * (i + 1) + p - 1];
	}
}

/**
 * Partitions as a matrix of indexes. One line for each chunk,
 * where each element is the index of the first element of the next partition.
 */
void partitionPsrsChunks(uint64_t **partitions, const unsigned int *pivots, const unsigned int *localA, const PsrsData &psrsData) {
	for (uint64_t chunk = 0; chunk < psrsData.numChunks; chunk++) {
		partitions[chunk] = new uint64_t[psrsData.numPivots];
		const unsigned int *chunkStart = localA + chunk * psrsData.chunkSize;
		const unsigned int *chunkEnd = localA + (chunk + 1) * psrsData.chunkSize;
		for (unsigned int pivotInd = 0; pivotInd < psrsData.numPivots; pivotInd++) {
			auto it = std::upper_bound(chunkStart, chunkEnd, pivots[pivotInd]);
			//if (it != chunkEnd) {
				const uint64_t ind = std::distance(chunkStart, it);
				partitions[chunk][pivotInd] = ind;
			//} else { // pivot is at the end of chunk

			//}
		}
	}
}

uint64_t calcPartitionSize(const uint64_t partition, const uint64_t chunk, uint64_t **partitions, const PsrsData &psrsData) {
	uint64_t startPartInd = 0, nextPartInd = 0;
	if (partition == 0) { // first partition of old chunk
		startPartInd = 0;
		nextPartInd = partitions[chunk][partition];
	} else if (partition == psrsData.numChunks - 1) { // last partition of old chunk
		startPartInd = partitions[chunk][partition - 1];
		nextPartInd = psrsData.chunkSize;
	} else {
		startPartInd = partitions[chunk][partition - 1];
		nextPartInd = partitions[chunk][partition];
	}
	return nextPartInd - startPartInd;
}

uint64_t calcNewChunkSize(const uint64_t newChunk, uint64_t **partitions, const PsrsData &psrsData) {
	uint64_t sum = 0;
	for (uint64_t chunk = 0; chunk < psrsData.numChunks; chunk++) {
		sum += calcPartitionSize(newChunk, chunk, partitions, psrsData);
	}
	return sum;
}

/**
 * Partitions are conceptually reorganized into "new chunks".
 */
//void normalizePsrsPartitions(uint64_t **partitions, const unsigned int *localA, const PsrsData &psrsData) {
//	for (uint64_t newChunk = 0; newChunk < psrsData.numChunks; newChunk++) {
//		uint64_t newChunkSize = calcNewChunkSize(0, partitions, psrsData);
//		// Only one of the two following while's will be entered.
//		while (newChunkSize < psrsData.chunkSize) {
//			moveToPartitions(newChunk, partitions, psrsData);
//			newChunkSize++;
//		}
//		while (newChunkSize > psrsData.chunkSize) {
//			moveFromPartitions(newChunk, partitions, psrsData);
//			newChunkSize--;
//		}
//	}
//}

//void reorderPsrsPartitions(uint64_t **partitions, const unsigned int *localA, const PsrsData &psrsData) {
//	unsigned int *swapTemp = new unsigned int[];
//	for (uint64_t chunk1 = 0; chunk1 < psrsData.numChunks; chunk1++) {
//		for (uint64_t partition1 = chunk1 + 1; partition1 < psrsData.numChunks; partition1++) {
//			uint64_t chunk2 = partition1;
//			uint64_t partition2 = chunk1;
//			//swapPsrsPartitions(partition1, chunk1, partition2, chunk2, swapTemp, partitions, localA);
//		}
//	}
//	delete[] swapTemp;
//}

//void sortPsrsNewChunksGPU(unsigned int *localA, uint64_t localN, uint64_t **partitions, const PsrsData &psrsData) {
//	unsigned int *chunkGPU = nullptr;
//	const uint64_t chunkSizeBytes = psrsData.chunkSize * sizeof(unsigned int);
//	gpuErrChk(cudaMalloc(&chunkGPU, chunkSizeBytes));
//	for (uint64_t chunk = 0; chunk < localN; chunk++) {
//		uint64_t chunkStart = chunk * psrsData.chunkSize;
//		unsigned int *chunkStartCPU = localA + chunkStart;
//		uint64_t pastPartitionsSize = 0;
//		for (uint64_t partition = 0; partition < psrsData.numChunks; partition++) {
//			uint64_t partitionSize = calcPartitionSize(partition, chunk, partitions, psrsData);
//			gpuErrChk(cudaMemcpy(chunkGPU + pastPartitionsSize, chunkStartCPU + pastPartitionsSize, partitionSize, cudaMemcpyHostToDevice));
//			pastPartitionsSize += partitionSize;
//		}
//		localBitonicSortGPU(chunkGPU, psrsData.chunkSize);
//		gpuErrChk(cudaMemcpy(chunkStartCPU, chunkGPU, chunkSizeBytes, cudaMemcpyDeviceToHost));
//	}
//	gpuErrChk(cudaFree(chunkGPU));
//}

struct NewChunk {
	unsigned int *chunk;
	uint64_t powerN;
	uint64_t realN;
	unsigned int dummyElem;
};

void mergePartitions(NewChunk *newChunks, uint64_t **partitions, unsigned int *localA, const PsrsData &psrsData) {
	for (uint64_t newChunk = 0; newChunk < psrsData.numChunks; newChunk++) {
		uint64_t newChunkSize = calcNewChunkSize(newChunk, partitions, psrsData);
		uint64_t newChunkPowerSize = closestHigherPowerOf2(newChunkSize);
		newChunks[newChunk].chunk = new unsigned int[newChunkPowerSize];
		newChunks[newChunk].realN = newChunkSize;
		newChunks[newChunk].powerN = newChunkPowerSize;
		unsigned int maxVal = 0;
		uint64_t pastPartitionSizes = 0;
		for (uint64_t chunk = 0; chunk < psrsData.numChunks; chunk++) {
			uint64_t partitionSize = calcPartitionSize(newChunk, chunk, partitions, psrsData);
			unsigned int *partitionStart = localA + chunk * psrsData.chunkSize + (newChunk > 0 ? partitions[chunk][newChunk - 1] : 0);
			unsigned int *partitionEnd = partitionStart + partitionSize;
			maxVal = std::max(maxVal, *(partitionEnd - 1));
			std::copy(partitionStart, partitionEnd, newChunks[newChunk].chunk + pastPartitionSizes);
			pastPartitionSizes += partitionSize;
		}
		newChunks[newChunk].dummyElem = maxVal;
		std::fill(&(newChunks[newChunk].chunk[newChunkSize]), &(newChunks[newChunk].chunk[newChunkPowerSize]), maxVal);
	}
}

void sortPsrsNewChunksGPU(NewChunk *newChunks, const PsrsData &psrsData) {
	for (uint64_t newChunk = 0; newChunk < psrsData.numChunks; newChunk++) {
		unsigned int *chunkGPU = nullptr;
		const uint64_t chunkSizeBytes = newChunks[newChunk].powerN * sizeof(unsigned int);
		std::cout << "Chunk size bytes = " << chunkSizeBytes << std::endl; // TODO REMOVE
		gpuErrChk(cudaMalloc(&chunkGPU, chunkSizeBytes));

		gpuErrChk(cudaMemcpy(chunkGPU, newChunks[newChunk].chunk, chunkSizeBytes, cudaMemcpyHostToDevice));
		localBitonicSortGPU(chunkGPU, newChunks[newChunk].powerN);
		gpuErrChk(cudaMemcpy(newChunks[newChunk].chunk, chunkGPU, chunkSizeBytes, cudaMemcpyDeviceToHost));

		gpuErrChk(cudaFree(chunkGPU));
	}
}

void movePsrsNewChunksToOutput(NewChunk *newChunks, unsigned int *localA, const PsrsData &psrsData) {
	uint64_t pastNewChunkSizes = 0;
	for (uint64_t newChunk = 0; newChunk < psrsData.numChunks; newChunk++) {
		unsigned int *newChunkEnd = newChunks[newChunk].chunk + newChunks[newChunk].realN;
		std::copy(newChunks[newChunk].chunk, newChunkEnd, localA + pastNewChunkSizes);
		pastNewChunkSizes += newChunks[newChunk].realN;
	}
}

/** Due to lack of VRAM, Parallel Sort by Regular Sampling is implemented.
 * It further divides the array into chunks which are individually sorted by the GPU,
 * requiring less VRAM at each instant.
 */
void localPsrsGPU(unsigned int *localA, uint64_t localN) {
	uint64_t localVram = 0;
	gpuErrChk(cudaMemGetInfo(&localVram, nullptr));
	const uint64_t localEffVram = localN * sizeof(unsigned int) / 4; //closestLowerPowerOf2(localVram) / 2;

	PsrsData psrsData(localN, localEffVram);
	unsigned int *sample = new unsigned int[psrsData.sampleSize];
	unsigned int *pivots = new unsigned int[psrsData.numPivots];

	sortPsrsChunksGPU(localA, localN, psrsData);
	createPsrsSample(sample, localA, localN, psrsData);
	sortPsrsSampleGPU(sample, psrsData);
	pickPivots(pivots, sample, psrsData, localN);
	uint64_t **partitions = new uint64_t*[psrsData.numChunks];
	partitionPsrsChunks(partitions, pivots, localA, psrsData);
	NewChunk *newChunks = new NewChunk[psrsData.numChunks];
	mergePartitions(newChunks, partitions, localA, psrsData);
	sortPsrsNewChunksGPU(newChunks, psrsData);
	movePsrsNewChunksToOutput(newChunks, localA, psrsData);

	for (uint64_t chunk = 0; chunk < psrsData.numChunks; chunk++) {
		delete[] newChunks[chunk].chunk;
		delete[] partitions[chunk];
	}
	delete[] newChunks;
	delete[] partitions;
	delete[] pivots;
	delete[] sample;
}

void sortEntireArrayGPU(unsigned int *localA, const uint64_t localN) {
	unsigned int *chunkGPU = nullptr;
	const uint64_t localBytes = localN * sizeof(unsigned int);
	gpuErrChk(cudaMalloc(&chunkGPU, localBytes));
	gpuErrChk(cudaMemcpy(chunkGPU, localA, localBytes, cudaMemcpyHostToDevice));
	localBitonicSortGPU(chunkGPU, localN);
	gpuErrChk(cudaMemcpy(localA, chunkGPU, localBytes, cudaMemcpyDeviceToHost));
	gpuErrChk(cudaFree(chunkGPU));
}

bool fitsInVram(const uint64_t localN) {
	uint64_t localVram = 0;
	gpuErrChk(cudaMemGetInfo(&localVram, nullptr));
	const uint64_t localBytes = localN * sizeof(unsigned int);
	return localBytes <= localVram;
}

void localSortGPU(unsigned int *localA, const uint64_t localN) {
	if (false) { //fitsInVram(localN)) {
		sortEntireArrayGPU(localA, localN);
	} else {
		localPsrsGPU(localA, localN);
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
	delete[] localA;

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

	std::cout << "P" << rank << ": Allocating memory with " << localN << " elements" << std::endl;

	// Allocate local memory in CPU
	localA = new unsigned int[localN];

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