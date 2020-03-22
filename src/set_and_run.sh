#!/usr/bin/env bash
#PBS -l walltime=05:00:00
#PBS -l nodes=1:ppn=2
#PBS -N par_sweep
#PBS -j oe


cd $PBS_O_WORKDIR

module load anaconda3-2019.07-gcc-4.8.5-pqb5ojq

python parameter_sweep.py
