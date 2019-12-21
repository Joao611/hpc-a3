#!/bin/bash
# Parameter: number of nodes

# CPU
sbatch --nodes=${1} main.job cpu 18 0x1234abcd
sbatch --nodes=${1} main.job cpu 18 0x10203040
sbatch --nodes=${1} main.job cpu 18 0x40e8c724
sbatch --nodes=${1} main.job cpu 18 0x79cbba1d
sbatch --nodes=${1} main.job cpu 18 0xac7bd459

sbatch --nodes=${1} main.job cpu 21 0x1234abcd
sbatch --nodes=${1} main.job cpu 21 0x10203040
sbatch --nodes=${1} main.job cpu 21 0x40e8c724
sbatch --nodes=${1} main.job cpu 21 0x79cbba1d
sbatch --nodes=${1} main.job cpu 21 0xac7bd459

sbatch --nodes=${1} main.job cpu 26 0x1234abcd
sbatch --nodes=${1} main.job cpu 26 0x10203040
sbatch --nodes=${1} main.job cpu 26 0x40e8c724
sbatch --nodes=${1} main.job cpu 26 0x79cbba1d
sbatch --nodes=${1} main.job cpu 26 0xac7bd459

sbatch --nodes=${1} main.job cpu 34 0x1234abcd
sbatch --nodes=${1} main.job cpu 34 0x10203040
sbatch --nodes=${1} main.job cpu 34 0x40e8c724
sbatch --nodes=${1} main.job cpu 34 0x79cbba1d
sbatch --nodes=${1} main.job cpu 34 0xac7bd459

# GPU
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 18 0x1234abcd
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 18 0x10203040
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 18 0x40e8c724
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 18 0x79cbba1d
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 18 0xac7bd459

sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 21 0x1234abcd
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 21 0x10203040
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 21 0x40e8c724
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 21 0x79cbba1d
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 21 0xac7bd459

sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 26 0x1234abcd
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 26 0x10203040
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 26 0x40e8c724
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 26 0x79cbba1d
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 26 0xac7bd459

sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 34 0x1234abcd
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 34 0x10203040
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 34 0x40e8c724
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 34 0x79cbba1d
sbatch --nodes=${1} --gres=gpu:1 -C TitanX main.job gpu 34 0xac7bd459

squeue