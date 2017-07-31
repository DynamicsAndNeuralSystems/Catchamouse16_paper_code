## interpreter directive - this is a bash script
#!/bin/bash

cpusPerJob=8
memPerJob=8

## declare an array variable
declare -a runtypes=("svm_maxmin" "svm_maxmin_null" "dectree_maxmin" "dectree_maxmin_null")

## declare an array variable
declare -a longestTasks=("ElectricDevices" "StarLightCurves")

submitFolder="submit_scripts"
mkdir $submitFolder

## first loop through the runtypes
for rtype in "${runtypes[@]}"
do
  ## now loop through the tasks
  for i in "${longestTasks[@]}"
  do
    # dectree takes longer. some take especially long...
    timePerJob=12:00:00

    ScriptLocation="submit_scripts/job-$i-$rtype.sh"
    sed -e "s/xxxJOBTIMExxx/$timePerJob/g" -e "s/xxxCPUSxxx/$cpusPerJob/g" -e "s/xxxRAMGBxxx/$memPerJob/g" -e "s/xxxRUNTYPExxx/$rtype/g" -e "s/xxxTASKSxxx/$i/g" workflow_job_template.sh > $ScriptLocation
    qsub $ScriptLocation
    echo "Sumbmitted job $ScriptLocation"
  done
done

rm -r $submitFolder
