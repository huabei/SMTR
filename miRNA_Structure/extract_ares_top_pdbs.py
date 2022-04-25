#!/usr/bin/python3
import os
from os import system
from sys import argv
import numpy as np

# files path prepare
pyfile_name, input_file, output_folder, ares_scores_file, top_n = argv

# root_dir = pathlib.Path(__file__).parent.absolute()
root_dir = '..'
input_path = os.path.join(root_dir, input_file)
output_path = os.path.join(root_dir, output_folder)
ares_scores_path = os.path.join(root_dir, ares_scores_file)

# get scores
scores = np.loadtxt(ares_scores_path)
sort_arg_scores = np.argsort(scores)
tags = list()
for tag_n in sort_arg_scores[:3]:
    tag_n += 1000001
    tag = 'S_' + str(tag_n)[1:]
    tags.append(tag)
    
EXE = 'extract_pdbs'
command = '%s -in:file:silent %s -in:file:tags %s' % (EXE, outfilename, " ".join(tags))
system(command)

