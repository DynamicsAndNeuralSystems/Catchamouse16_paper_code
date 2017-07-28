## interpreter directive - this is a shell script
#!/bin/sh
## ask PBS for time (format hh:mm:ss)
#PBS -l walltime=xxxJOBTIMExxx
## ask for one node with some cpus and memory (per node)
#PBS -l select=1:ncpus=xxxCPUSxxx:mem=xxxRAMGBxxxgb

##load anaconda python module
module purge module load anaconda3/personal

##command line
cd ~/op_importance/op_importance/workflow_classes/
/home/ss7412/anaconda3/envs/py27/bin/python Workflow.py "xxxRUNTYPExxx" "xxxTASKSxxx"
