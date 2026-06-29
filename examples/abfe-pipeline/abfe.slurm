#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -t 24:00:00
#SBATCH -J EASYBFE_ABFE
#SBATCH -o EASYBFE_ABFE.o%j
#SBATCH -A m2834_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -G 4
#SBATCH --gpu-bind=none
#SBATCH --qos=premium
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ericwangyz@berkeley.edu

set -o pipefail

source /global/u2/e/eric6/env/easybfe.env

easybfe abfe pipeline config_5ns.yaml -p ./9qdz_fixed_dry.pdb -l ./1508.sdf -o ./run 