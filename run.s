#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=GPUDemo
#SBATCH --mail-type=END
#SBATCH --mail-user=ij405@nyu.edu
#SBATCH --output=slurm_%j.out

module purge

source activate py2

cd ~/ms_det/scripts/
#python ~/ms_det/trouble/test_slurm.py
#python ~/ms_det/scripts/trouble.py
python ~/ms_det/trouble/eval_perf.py
#python ~/ms_det/trouble/eval_perf.py
