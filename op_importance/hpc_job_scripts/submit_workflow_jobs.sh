## interpreter directive - this is a bash script
#!/bin/bash

cpusPerJob=8
memPerJob=8

## declare an array variable
declare -a runtypes=("svm_maxmin" "svm_maxmin_null" "dectree_maxmin" "dectree_maxmin_null")

## declare an array variable
declare -a tasks=("50words" "Adiac" "ArrowHead" "Beef" "BeetleFly" "BirdChicken" "CBF" "Car" "ChlorineConcentration" "CinC_ECG_torso" "Coffee" "Computers" "Cricket_X" "Cricket_Y" "Cricket_Z" "DiatomSizeReduction" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxOutlineCorrect" "DistalPhalanxTW" "ECG200" "ECG5000" "ECGFiveDays" "Earthquakes" "ElectricDevices" "FISH" "FaceAll" "FaceFour" "FacesUCR" "FordA" "FordB" "Gun_Point" "Ham" "HandOutlines" "Haptics" "Herring" "InlineSkate" "InsectWingbeatSound" "ItalyPowerDemand" "LargeKitchenAppliances" "Lighting2" "Lighting7" "MALLAT" "Meat" "MedicalImages" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxOutlineCorrect" "MiddlePhalanxTW" "MoteStrain" "NonInvasiveFatalECG_Thorax1" "NonInvasiveFatalECG_Thorax2" "OSULeaf" "OliveOil" "PhalangesOutlinesCorrect" "Phoneme" "Plane" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxOutlineCorrect" "ProximalPhalanxTW" "RefrigerationDevices" "ScreenType" "ShapeletSim" "ShapesAll" "SmallKitchenAppliances" "SonyAIBORobotSurface" "SonyAIBORobotSurfaceII" "StarLightCurves" "Strawberry" "SwedishLeaf" "Symbols" "ToeSegmentation1" "ToeSegmentation2" "Trace" "TwoLeadECG" "Two_Patterns" "UWaveGestureLibraryAll" "Wine" "WordsSynonyms" "Worms" "WormsTwoClass" "synthetic_control" "uWaveGestureLibrary_X" "uWaveGestureLibrary_Y" "uWaveGestureLibrary_Z" "wafer" "yoga")

declare -a longerTasks=("ECG5000" "ElectricDevices" "FordA" "NonInvasiveFatalECG_Thorax1" "NonInvasiveFatalECG_Thorax2" "StarLightCurves" "wafer")

submitFolder="submit_scripts"
mkdir $submitFolder

## first loop through the runtypes
for rtype in "${runtypes[@]}"
do
  ## now loop through the tasks
  for i in "${tasks[@]}"
  do
    # dectree takes longer. some take especially long...
    timePerJob=1:00:00
    if [[ $rtype == *"dectree"* ]]; then
      echo "dectree"
      timePerJob=2:00:00
      if [[ " ${longerTasks[@]} " =~ " ${i} " ]]; then
        echo 'longerTasks';
        timePerJob=4:00:00
      fi
    fi

    ScriptLocation="submit_scripts/job-$i-$rtype.sh"
    sed -e "s/xxxJOBTIMExxx/$timePerJob/g" -e "s/xxxCPUSxxx/$cpusPerJob/g" -e "s/xxxRAMGBxxx/$memPerJob/g" -e "s/xxxRUNTYPExxx/$rtype/g" -e "s/xxxTASKSxxx/$i/g" workflow_job_template.sh > $ScriptLocation
    qsub $ScriptLocation
    echo "Sumbmitted job $ScriptLocation"
  done
done

rm -r $submitFolder
