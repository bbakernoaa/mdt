#!/bin/bash
#SBATCH -J mdt-alloc-smoke
#SBATCH -M c6
#SBATCH -p batch
#SBATCH -A bil-fire3
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -o allocation_smoke_%j.out

cd /gpfs/f6/bil-fire3/scratch/Barry.Baker/python/mdt
srun -n 4 /gpfs/f6/bil-fire3/world-shared/python/envs/mdt/bin/python -m mdt.allocation docs/examples/gefs_aeronet_gaea_prep.yaml
echo "ALLOC_EXIT=$?"