#!/bin/bash
#SBATCH --job-name=su8_grmhd_grrt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/su8_%j.out
#SBATCH --error=logs/su8_%j.err

module load cuda
module load mpi

# GRMHD run
srun ./harm_su8_solver \
    --resolution 512 \
    --spin 0.94 \
    --topology MAD \
    --kappa_ac 0.4 \
    --gamma_ac 0.8 \
    --alpha_su8 0.1 \
    --lambda_ac 0.2 \
    --output-dir data/grmhd_run_01

# GRRT run
srun ./ipole_su8 \
    --input data/grmhd_run_01 \
    --freq 230e9 \
    --inclination 17 \
    --output-dir data/grrt_run_01
