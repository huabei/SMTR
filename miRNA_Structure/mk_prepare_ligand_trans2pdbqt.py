#!/usr/bin/python3
# trans pdb(mol2\pdbq) to pdbqt use autodock tools in batch

import os
import sys
import pathlib
# prepare files path
pyfile_name, input_folder, output_folder = sys.argv
# root_dir = pathlib.Path(__file__).parent.absolute()
root_dir = os.getcwd()
input_path = os.path.join(root_dir, input_folder)
output_path = os.path.join(root_dir, output_folder)
os.chdir(input_path)
for file in os.listdir(input_path):
    # print(os.path.abspath(input_path))
    
    # file_path = os.path.join(input_path, file)
    suffix = os.path.splitext(file)[-1]
    command = 'pythonsh ~/bin/prepare_ligand4 -l ' + file + ' -o ' + os.path.join(output_path, file.replace(suffix, '.pdbqt'))
    # print(command)
    os.system(command)
    # break


