#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=32:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=segment
#SBATCH --mail-type=END
#SBATCH --mail-user=ij405@nyu.edu
#SBATCH --output=slurm_%j.out

module purge

source activate py2

cd ~/ms_det/scripts/
#python ~/ms_det/trouble/test_slurm.py
#python ~/ms_det/scripts/trouble.py
python ~/ms_det/scripts/segment.py
#python ~/ms_det/trouble/eval_perf.py
