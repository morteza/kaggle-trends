#!/bin/bash

#SBATCH --job-name=all_on_gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu
#SBATCH --ntasks=1
#SBATCH --mem=1gb
# --partition=gpu
#SBATCH --time=00:01:00
#SBATCH --output=all_on_gpu_%j.log

export LOCAL_MODULES=$HOME/.local/easybuild/modules/all

source module use $LOCAL_MODULES
source module load tools/EasyBuild
source module load lang/Miniconda3


source conda env create -f environment.yml
source conda activate kaggle-trends

python src/automl_loading.py