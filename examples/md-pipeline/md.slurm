#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -t 1:00:00
#SBATCH -J EASYBFE_MD
#SBATCH -o EASYBFE_MD.o%j
#SBATCH -A m2834_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -G 4
#SBATCH --gpu-bind=none
#SBATCH --qos=regular

# Runs the three kinds of plain MD side by side, one GPU each.
set -o pipefail
source /global/u2/e/eric6/env/easybfe.env

CONFIG=config_md_1ns.yaml
CUDA_VISIBLE_DEVICES=0 easybfe md pipeline $CONFIG -p tyk2_pdbfixer_dry.pdb -l jmc_23.sdf -o run_complex > run_complex.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 easybfe md pipeline $CONFIG -p tyk2_pdbfixer_dry.pdb              -o run_protein > run_protein.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 easybfe md pipeline $CONFIG                          -l jmc_23.sdf -o run_ligand  > run_ligand.out 2>&1 &

status=0
for job in $(jobs -p); do
    wait "$job" || status=1
done
exit $status
