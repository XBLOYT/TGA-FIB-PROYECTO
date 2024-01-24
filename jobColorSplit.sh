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



#./filtrarColorSplit.exe IMG03.jpg OutSplit.jpg
#./filtrarColorSplit.exe IMG03.jpg OutSplit.jpg
#./filtrarColorSplit.exe IMG03.jpg OutSplit.jpg
#./filtrarColorSplit.exe IMG03.jpg OutSplit.jpg
#./filtrarColorSplit.exe IMG03.jpg OutSplit.jpg
./filtrarColorSplitP.exe IMG03.jpg OutSplitP.jpg
./filtrarColorSplitP.exe IMG03.jpg OutSplitP.jpg
./filtrarColorSplitP.exe IMG03.jpg OutSplitP.jpg
./filtrarColorSplitP.exe IMG03.jpg OutSplitP.jpg
./filtrarColorSplitP.exe IMG03.jpg OutSplitP.jpg
