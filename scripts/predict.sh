#!/bin/bash
#SBATCH --job-name=Predict
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2 
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicholb@crick.ac.uk
DATA=/camp/lab/vincentj/home/users/nicholb/organoid_data
ml purge
ml TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
source /camp/lab/vincentj/home/users/nicholb/venv/bin/activate
python3 morgana/CLI/predict.py ${DATA}/model_unet ${DATA}/240930_saving/image_1_MMStack_control_DMSO_1-1.ome_restacked/ROI1