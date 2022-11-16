#!/bin/sh
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=10G
#SBATCH -t 00:10:00
#SBATCH --mail-user=es.lozano@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH --job-name=lectura
#SBATCH -o lectura.log
echo "Soy un JOB de prueba en GPU"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python lectura.py