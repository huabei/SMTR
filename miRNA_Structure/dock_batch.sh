#!/usr/bin/bash
# Usage
# dock_batch.sh receptor.pdbqt ligand.pdbqt config.txt output_folder
# $PWD
receptor_pdbqt=$1
ligand_folder=$2
config_txt=$3
output_folder=$4
# take count of processes
number=0
# 
for file in "$ligand_folder"/*.pdbqt;do
    echo $file
    # break
	number=$(expr $number + 1)
	echo "process:" $number
	# if [ $number -le $1 ]; then continue;fi
	output_file=${file/.pdbqt/_out.pdbqt}
	echo  "write to $output_file"
    break
	nohup rna_denovo -fasta $file -secstruct_file ${input_file/.fa/_sec.fold} -nstruct 100 -out:file:silent ${output_file/.fa/.100.out} -minimize_rna > ${output_file/.fa/.nohup.out} 2>&1 &
	working_num=`jobs -l | wc -l`
	echo "working_num is $working_num"
	sleep 1s
	while [ $working_num -gt 7 ]
	do
	echo "working number large than 7, sleep 1h" 
	sleep 1h
	echo "---*---*---*---*---"
	working_num=`jobs -l | wc -l`
	echo "working_num is $working_num"
	done
	
done