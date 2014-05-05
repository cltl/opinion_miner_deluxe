#!/bin/bash

# This script runs a bunch of experiments given a list of KAF files
# Input:
#  $1 --> folder where store all the models 
#  $2 --> list of kaf files for training
# Output:
#  standard output --> latex table with the results
#  standard error  --> progress information
# The script looks for all the subfolders called exp1 exp2 exp3 within the $exps_folder folder

general_folder=$1

if [ ! -d $general_folder ]; 
then 
  mkdir $general_folder;
fi

list_files=$2
exps_folder=experiments

echo "Output folder: $general_folder"
echo "List of files: $list_files"
echo "\begin{table}"
echo "\begin{tabular}{c|c|c|c|c|c|c||c|c|c|c}"
echo "\hline"
echo "Type & \multicolumn{2}{|c|}{Expression} & \multicolumn{2}{|c|}{Target} & \multicolumn{2}{|c||}{Holder} & \multicolumn{2}{|c|}{Exp-Tar} & \multicolumn{2}{|c|}{Exp-Hol} \\\\"
echo "\hline"
echo "& P & R &  P & R &  P & R &  P & R &  P & R \\\\"

for exp in $exps_folder/exp*
do
  #id=$1  folder=$2  list_files=$3  experiment_folder=$4 
  echo `date +%T` starting experiment $exp >> /dev/stderr
  id=`basename $exp`
  outfolder=$general_folder/$id
  # If the output folder not exists already
  if [ ! -d $outfolder ]; then 
    run_experiment.sh $id $outfolder $list_files $exp
    echo `date +%T` Done >> /dev/stderr
    echo "Done experiment $exp Files: $list_files Out: $general_folder" | mail -s "Experiment done" ruben.izquierdobevia@vu.nl
  else 
    echo "The experiment $exp on $outfolder already exists, skipped" >> /dev/stderr
  fi
done

echo "\end{tabular}"
echo "\end{table}"
  