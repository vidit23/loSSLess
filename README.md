# loSSLess
Self-supervised learning for Image Classification


1. srun --nodes=1 --tasks-per-node=1 --cpus-per-task=4 --mem=32GB --time=10:00:00 --gres=gpu:1 --pty /bin/bash

2. singularity exec --nv --bind /scratch --overlay /scratch/vvb238/overlay-50G-10M.ext3 --overlay /scratch/vvb238/DL21SP/student_dataset.sqsh:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash

3. source /ext3/env.sh

4. conda activate dev

5. jupyter lab --ip 0.0.0.0 --port 8965 --no-browser

6. ssh -L 8965:gr030.nyu.cluster:8965 greene -N 
