#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=dlp_caller
#SBATCH --gpus=1
#SBATCH --mem=50gb
#SBATCH --time=12:00:00
#SBATCH --output=dlp_caller_%j.ou

source /orfeo/cephfs/scratch/cdslab/ebusca00//envs/virtualenv/jupyter/bin/activate

python binomial_lik/run_dlp.py process_dlp_data/high_cov_20x

python binomial_lik/run_dlp.py process_dlp_data/high_cov_300x

python binomial_lik/run_dlp.py process_dlp_data/high_cov_1500x
