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
module load nibabel/2.1.0
module load keras/2.0.2
module load scikit-learn/intel/0.18.1
module load numpy/python2.7/intel/1.14.0

source ~/ms_det/py2_ms/bin/activate

cd ~/scratch/ij405/test
python ~/ms_det/3d_cnn.py

