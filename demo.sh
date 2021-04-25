#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err


singularity exec --nv --bind /scratch --overlay /scratch/vvb238/overlay-50G-10M.ext3:ro --overlay /scratch/vvb238/DL21SP/student_dataset.sqsh:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate dev
cd loSSLess
python barlow.py
"
