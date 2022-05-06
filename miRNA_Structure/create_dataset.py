#!/usr/bin/env python
# coding: utf-8


''' take auto dock results score'''


import sys
import os
# import numpy as np
# from scipy.io import savemat
from tqdm.auto import tqdm

def get_pdb_info(file_path):
    """get [[atom_type, x, y, z], ...]"""
    f = open(file_path, 'r')
    pdb_info_list = list()
    for row in f:
        if row[:6] in ['HETATM', 'ATOM  '] :
            x = str(float(row[30:38]))
            y = str(float(row[38:46]))
            z = str(float(row[46:54]))
            atom_type = row[76:78]
            pdb_info_list.append((atom_type, x, y, z))
    f.close()
    return pdb_info_list


def get_autodock_info(file_path):
    '''return [[energy(kcal/mol), lb(rmsd), ub(rmsd)], ...]'''
    
    if file_path[-6:] == '.pdbqt':
        f = open(file_path, 'r')
        dock_data = list()
        for line in f:
            if line[:18] == 'REMARK VINA RESULT':
                x = line.split(' ')
                x = [i for i in x if i != ''][3:]
                x = [float(i) for i in x]
                dock_data.append(x)
        f.close()
    else:
        print(' Nonsupport file {}'.format(file_path[-6:]))
    return dock_data


def get_autodock_best_score(file_path):
    # other method
    f = open(file_path, 'r')
    for line in f:
        if line[:18] == 'REMARK VINA RESULT':
            score = float(line[24:30])
            break
    f.close()
    return str(score)

def write_data(f, data_dict):
    f.write('\n')
    name = data_dict['id']
    pos = data_dict['pos']
    score = data_dict['score']
    f.write(name)
    f.write('\n')
    for line in pos:
        f.write(' '.join(line))
        f.write('\n')
    f.write(str(score))
    f.write('\n')
    return f


# prepare files folder path
pyfile_path, struc_file_path, results_file_path, output_file = sys.argv
root_dir = os.getcwd()
struc_path = os.path.join(root_dir, struc_file_path)
results_path = os.path.join(root_dir, results_file_path)
output_file = os.path.join(root_dir, output_file)

total_data = dict()
f = open(output_file, 'w')
f.write('e(kcalmol^-1)\n')
for file_name in tqdm(os.listdir(results_path), desc='prepare dataset'):
    # print(f'write file {file_name[:-6]}')
    # prepare file path
    results_file = os.path.join(results_path, file_name)
    struc_file_name = file_name.replace('_out.pdbqt', '.pdb')
    struc_file = os.path.join(struc_path, struc_file_name)
    # if os.path.exists()
    # get_data
    total_data['id'] = file_name[:-6]
    total_data['pos'] = get_pdb_info(struc_file)
    total_data['score'] = get_autodock_best_score(results_file)
    # write_data
    write_data(f, total_data)
print('Congratulate')
f.close()
# # scipy.io.savemat(file_name, mdict, appendmat=True)
# if output_file[-4:] != '.mat':
#     output_file += '.mat'
# # print(output_file)
# savemat(output_file, total_dock_data)


