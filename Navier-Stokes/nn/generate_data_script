#!/bin/bash

#Submit this script with: sbatch calibrate_script

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=20       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -J "NS"    # job name

#SBATCH --mem-per-cpu=16G 

module purge
module load julia/1.6.0 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1

julia -p 20 NN-Data-Par.jl
