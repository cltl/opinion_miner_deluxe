#!/bin/bash

# This script runs a given experiment
# Input:
# $1 the id used for the table latex for the experiment
# $2 the output folder to store all the models and folds (for the validation)
# $3 the list of files for the training
# $4 the folder with the experiment (must contain a file called config.cfg with the configuration
#
# Output
# Standard out --> 2 rows of the latex table (for the basic and deluxe)
# Standard err --> progress of the program
# The log of training/evaluation will be on the same folder there the exp, called $id.log
id=$1
folder=$2
list_files=$3
experiment_folder=$4

numfolds=10
base_out_folder=`dirname $folder`
err_file=$base_out_folder/$id.log
out_per_folds=$base_out_folder/$id.out_per_fold.tex

tmpconfig=`mktemp`

echo "[general]" > $tmpconfig
echo "output_folder = $folder" >> $tmpconfig 
echo "filename_training_list = $list_files" >> $tmpconfig 
echo >> $tmpconfig 
echo "[feature_templates]" >> $tmpconfig
echo "expression = $experiment_folder/templates_exp.txt " >> $tmpconfig
echo "holder = $experiment_folder/templates_hol.txt" >> $tmpconfig
echo "target = $experiment_folder/templates_tar.txt" >> $tmpconfig
echo >> $tmpconfig
cat $experiment_folder/config.cfg >> $tmpconfig 

echo Running experiment $id Logs: $err_file Out per fold: $out_per_folds >> /dev/stderr
#Output to standard output
cross_validation.py -n $numfolds -f $tmpconfig -id $id -out_folds $out_per_folds 2> $err_file

rm $tmpconfig

