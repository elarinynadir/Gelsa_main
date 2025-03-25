#!/bin/bash
#SBATCH --job-name=g033r
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-33
#SBATCH --mem=16G
#SBATCH --output=Results/logs/galaxy_%A_%a.out 


SIRPACK="sir_pack.pickle"
GOLD_SAMPLE="gold_sample.fits"

python _redshift-fit.py --sirpack $SIRPACK --gold_sample $GOLD_SAMPLE --index $SLURM_ARRAY_TASK_ID
