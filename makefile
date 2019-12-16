das-5:
	nvcc -L$(MPI_HOME)/lib64 -lmpi hpc-a3/kernel.cu -o bitonic.out

windows:
	nvcc -I"$env:MSMPI_INC" -L"$env:MSMPI_LIB64" -lmsmpi .\hpc-a3\kernel.cu -o bitonic
