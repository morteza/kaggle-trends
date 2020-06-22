#!/bin/bash

#SBATCH --job-name=all_on_gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu
#SBATCH --ntasks=1
#SBATCH --mem=1gb
# --partition=gpu
#SBATCH --time=00:01:00
#SBATCH --output=logs/all_on_gpu_%j.log

# to enable access to module
. /etc/profile.d/lmod.sh


#module load tools/EasyBuild
#module use $HOME/.local/easybuild/modules/all
#module load lang/Miniconda3

conda env create -f environment.yml
conda activate kaggle-trends

python workspace/kaggle-trends/src/automl_loading.py