#!/bin/bash
# Sample PBS script
#PBS -N xxNameOfJobxx
# Cores requested: nodes, ppn (cpus per node), gpus
#PBS -l nodes=1:ppn=1
# Memory per core
#PBS -l pmem=2000MB
# Minimum acceptable duration for the job, format HH:MM:SS, or DD:HH:MM:SS for longer periods
#PBS -l walltime=36:00:00
# email user if job aborts (a) begins (b) or ends (e)
# Set output name:
#PBS -o joboutput.txt
#PBS -e joberror.txt
#PBS -j oe
#PBS -V

# Change directory to directory the application was launched from
cd $PBS_O_WORKDIR

# Show the host on which the job ran
hostname

# Show what PBS ennvironment variables our environment has
env | grep PBS

# Launch the Matlab job
matlab -nodisplay -r "cd $PBS_O_WORKDIR; disp(pwd); HCTSA_Runscript; exit"


