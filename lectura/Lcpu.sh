#!/bin/bash
####### Zona de Par�metros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=TestJOB		#Nombre del job
#SBATCH -p short			#Cola a usar, Default=short (Ver colas y l�mites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1				#Nodos requeridos, Default=1
#SBATCH -n 1				#Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=1		#Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=2000		#Memoria en Mb por CPU, Default=2048
#SBATCH --time=00:10:00			#Tiempo m�ximo de corrida, Default=2 horas
#SBATCH --mail-user=USER@uniandes.edu.co
#SBATCH --mail-type=ALL			
#SBATCH -o TEST_job.o%j			#Nombre de archivo de salida
#
########################################################################################
# ################## Zona Carga de M�dulos ############################################
module load  anaconda/python3.9
########################################################################################
# ###### Zona de Ejecuci�n de c�digo y comandos a ejecutar secuencialmente #############
sleep 60
echo "Soy un JOB de prueba"
########################################################################################