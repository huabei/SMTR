number=0
for file in ./hairpin/*
do
	number=$(expr $number + 1)
	echo $number
	if [ $number -le $1 ]; then continue;fi
	input_file=${file/hairpin/hairpin_sec}
	output_file=${file/hairpin/pre-miRNA_3d_structure}
	echo  "write to $output_file"
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