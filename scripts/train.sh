#!/bin/bash
#SBATCH --job-name=TrainUnet
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2 
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicholb@crick.ac.uk
 
ml purge
ml TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
source /camp/lab/vincentj/home/users/nicholb/venv/bin/activate
python3 morgana/CLI/train.py /camp/lab/vincentj/home/users/nicholb/organoid_data/model_mlp --epochs 50
