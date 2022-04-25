#!/usr/bin/python3
import os
from os import system
from sys import argv
import numpy as np
import pathlib
# files path prepare
pyfile_name, input_file, output_folder, ares_scores_file, top_n = argv

# root_dir = pathlib.Path(__file__).parent.absolute()
root_dir = os.getcwd()
input_path = os.path.join(root_dir, input_file)
output_path = os.path.join(root_dir, output_folder)
ares_scores_path = os.path.join(root_dir, ares_scores_file)

# get scores
scores = np.loadtxt(ares_scores_path)
sort_arg_scores = np.argsort(scores)
tags = list()
top = 0
mv_command = ''
for tag_n in sort_arg_scores[:int(top_n)]:
    top += 1
    tag_n += 1000001
    
    tag = 'S_' + str(tag_n)[1:]
    new_tag = 'Top_' + str(top) + '_S_' + str(tag_n)[1:] + '.pdb'
    tags.append(tag)
    mv_command += 'mv '
    mv_command += tag + '.pdb '
    mv_command += os.path.join(output_path, new_tag)
    mv_command += ';'
    # new_tags.append(new_tag)
    
EXE = 'extract_pdbs'
extract_command = '{0} -in:file:silent {1} -in:file:tags {2}'.format(EXE, input_path, " ".join(tags))
system(extract_command)

# mv_command = 'mv {0} {1}'.format(' '.join(tags), output_path)
system(mv_command)

