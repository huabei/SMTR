#!/usr/bin/bash
# run
Root_dir=$PWD
model=~/soft_folder/smtr_code/miRNA_Structure/dock_batch.py
receptor_file=~/soft_folder/smtr_data/Dock/miRNA/mir-21/pri/pri_mir_21_top_1.pdbqt
ligand_folder=~/soft_folder/smtr_data/Dock/SM/mk_prepare_ligand_fda
config_file=~/soft_folder/smtr_data/Dock/mir_21_vina_config.txt
out_folder=~/soft_folder/smtr_data/Dock/miRNA/mir-21/pri/mk_fda_Dock_results_2/
python  $model $receptor_file $ligand_folder $config_file $out_folder
