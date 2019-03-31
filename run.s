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

module load python/intel/2.7.12
module load pandas/intel/py2.7/0.20.3 
module load scikit-learn/intel/0.18.1
module load numpy/python2.7/intel/1.14.0

source ~/ms_det/py2_ms/bin/activate

cd ~/ms_det/scripts/
python ~/ms_det/scripts/test_script_all_patients.py

