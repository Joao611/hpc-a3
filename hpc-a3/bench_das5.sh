#!/bin/bash
sbatch --nodes=1 main.job 18
sbatch --nodes=1 main.job 21
sbatch --nodes=1 main.job 26
sbatch --nodes=1 main.job 34

sbatch --nodes=2 main.job 18
sbatch --nodes=2 main.job 21
sbatch --nodes=2 main.job 26
sbatch --nodes=2 main.job 34

sbatch --nodes=4 main.job 18
sbatch --nodes=4 main.job 21
sbatch --nodes=4 main.job 26
sbatch --nodes=4 main.job 34

sbatch --nodes=8 main.job 18
sbatch --nodes=8 main.job 21
sbatch --nodes=8 main.job 26
sbatch --nodes=8 main.job 34

sbatch --nodes=16 main.job 18
sbatch --nodes=16 main.job 21
sbatch --nodes=16 main.job 26
sbatch --nodes=16 main.job 34

squeue