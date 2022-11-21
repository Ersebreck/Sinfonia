#!/bin/sh
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=20G
#SBATCH -t 01:00:00
#SBATCH --mail-user=es.lozano@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH --job-name=frame_d
#SBATCH -o frame_ps.log
echo "Soy un JOB de prueba en GPU"
nvidia-smi
python YoloRos.py