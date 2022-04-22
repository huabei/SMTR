#!/usr/bin/bash
# use FARFAR2 generate mir structures
# sec_file=/mnt/e/Research/SM_miRNA/Data/miR_Sequence/mirgenedb/hairpin_sec/mature/3p/Hsa-Mir-21_3p_star_sec.fold
# fa_file=/mnt/e/Research/SM_miRNA/Data/miR_Sequence/mirgenedb/hairpin/mature/3p/Hsa-Mir-21_3p_star.fa
# out_file=/home/huabei/soft_folder/smtr_data/miR_Sequence/mirgenedb/Struc/mir-21/mature/3p/matu_mir_21_3p.out
# nohup rna_denovo -fasta $fa_file -secstruct_file $sec_file -nstruct 100 -out:file:silent $out_file -minimize_rna > ${out_file/.out/.nohup.out} 2>&1 &

sec_file=$2
fa_file=$1
out_file=$3
nohup rna_denovo -fasta $fa_file -secstruct_file $sec_file -nstruct 100 -out:file:silent $out_file -minimize_rna > ${out_file/.out/.nohup.out} 2>&1 &