#!/bin/csh
#PBS -N timeseries
#PBS -o PBS_stdout
#PBS -j oe
#PBS -l select=1:ncpus=1:mem=2GB
#PBS -l walltime=10:00:00
#PBS -m ea
#PBS -M bhar9988@uni.sydney.edu.au
#PBS -V
#PBS -J 1-100
cd "$PBS_O_WORKDIR"
module load Matlab2017b
touch "TS_log-$PBS_ARRAY_INDEX.txt"
matlab -nodisplay -singleCompThread -r "home_dir = pwd; cd('~/hctsa_v1'), startup, cd(home_dir), TS_compute(0, [], [], [], 'HCTSA_$PBS_ARRAY_INDEX.mat', 0); exit" >& "TS_log-$PBS_ARRAY_INDEX.txt"
exit
