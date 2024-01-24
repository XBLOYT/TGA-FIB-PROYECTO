#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=MultiGPU
#SBATCH -D .
#SBATCH --output=submit-FILTRO.o%j
#SBATCH --error=submit-FILTRO.e%j
#SBATCH -A cuda
#SBATCH -p cuda
### Se pide 1 GPU 
#SBATCH --gres=gpu:1

export PATH=/Soft/cuda/11.2.1/bin:$PATH



#./filtrarBW.exe IMG03.jpg OutBW.jpg
#./filtrarBW.exe IMG03.jpg OutBW.jpg
#./filtrarBW.exe IMG03.jpg OutBW.jpg
#./filtrarBW.exe IMG03.jpg OutBW.jpg
#./filtrarBW.exe IMG03.jpg OutBW.jpg
./filtrarBWP.exe IMG03.jpg OutBWP.jpg
#./filtrarBWP.exe IMG03.jpg OutBWP.jpg
#./filtrarBWP.exe IMG03.jpg OutBWP.jpg
#./filtrarBWP.exe IMG03.jpg OutBWP.jpg
#./filtrarBWP.exe IMG03.jpg OutBWP.jpg
