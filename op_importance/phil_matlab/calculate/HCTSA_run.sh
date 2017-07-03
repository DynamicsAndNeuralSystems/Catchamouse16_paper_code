#!/bin/bash
# Set range of ts_ids to calculate
tsmin=1
tsmax=75182
#tsmax=4000
# Set number of time series to calculate per job
NumPerJob=100

# Calculate the number of jobs required
NumJobs=$((($tsmax-$tsmin)/$NumPerJob+1))

# 
# Start by writing the directory structure
# Stored in {DirNames[]} array
TheTS=$tsmin
for (( i=0; i<$NumJobs; i++)); do
    #HereMin=$TheTS
    printf -v HereMin '%06d' $TheTS
    if [ $i -eq $(($NumJobs-1)) -a $(($NumJobs*$NumPerJob)) -gt $tsmax ]; then
        #HereMax=$tsmax
        printf -v HereMax '%06d' $tsmax
    else
        #HereMax=$(($TheTS+$NumPerJob-1))
        printf -v HereMax '%06d' $(($TheTS+$NumPerJob-1))
    fi
    DirNames[i]="tsids_$HereMin-$HereMax" # Store the directory names
    JobNames[i]="tsids-$HereMin-$HereMax" # Make names for PBS jobs
    MinIDS[i]=$HereMin # Also store the minimum ts_id
    MaxIDS[i]=$HereMax # Also store the maximum ts_id
    mkdir ${DirNames[i]}
    TheTS=$(($TheTS+$NumPerJob))
done

#
# Next we want to go into each directory and create
# a PBS script with a suitable job name
# The file hctsa_pbs_job.sh must be in the base directory
for ((i=0; i<$NumJobs; i++)); do
    # Define the script location:
    ScriptLocation="${DirNames[i]}/hctsa_pbs_job.sh"
    # echo $ScriptLocation
    # Use sed to replace the wildcard NameOfJob with the directory name
    sed "s/xxNameOfJobxx/${JobNames[i]}/g" hctsa_pbs_job.sh > $ScriptLocation
    # sed -e "s/xxNameOfJobxx/${DirNames[i]}/g" -e "s/xxNameOfDirectoryxx/${DirNames[i]}/g" hctsa_pbs_job.sh > $ScriptLocation
done

#
# Ok, so now we have all the PBS shell scripts for the jobs we want to
# run in their respective directories.
# Now we need to copy Matlab runscripts with the right range of ts_ids in them
# (into each directory)
for ((i=0; i<$NumJobs; i++)); do
    # Define the script location:
    RunScriptLocation="${DirNames[i]}/HCTSA_Runscript.m"
    # echo $ScriptLocation
    # Use sed to replace the wildcard NameOfJob with the directory name
    sed -e "s/xxTSIDMINxx/${MinIDS[i]}/g" -e "s/xxTSIDMAXxx/${MaxIDS[i]}/g" HCTSA_Runscript.m > $RunScriptLocation
done

# 
# Ok, so now we want to go through and actually submit all the PBS scripts as jobs
for ((i=0; i<$NumJobs; i++)); do
    cd ${DirNames[i]}
    JobNumber=$(qsub hctsa_pbs_job.sh) # Take note of the job number
    echo "Job submitted for tsids between ${MinIDS[i]} and ${MaxIDS[i]} as $JobNumber"
    # Make a file for the job number
    echo ${JobNumber} > "${JobNumber%%.*}.txt"
    cd ../
done

