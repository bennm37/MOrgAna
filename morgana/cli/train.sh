
#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=simple-test
#SBATCH --ntasks=1
#SBATCH --time=5:00
#SBATCH --mem-per-cpu=128
#SBATCH --partition=ncpu
 
srun python train.py