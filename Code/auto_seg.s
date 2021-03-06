#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=p100_4 
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --job-name= $1 $2 $3
#SBATCH --mail-type=END
#SBATCH --mail-user=ij405@nyu.edu
#SBATCH --output=slurm_%j.out

module purge

module load cuda/9.0.176 cudnn/9.0v7.0.5

source activate py2_gpu

cd ~/ms_det/scripts/
#python ~/ms_det/trouble/test_slurm.py
#python ~/ms_det/scripts/trouble.py
python ~/ms_det/scripts/automated_segment.py $1 $2 $3 $4
#python ~/ms_det/trouble/eval_perf.py
