# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 13:16
# @Author  : HuaBei
# @File    : split.py
from sys import argv
import pathlib
import os

# file path prepare
input_file, output_folder = argv[1], argv[2]

root_dir = pathlib.Path(__file__).parent.absolute()
input_path = os.path.join(root_dir, input_file)
output_path = os.path.join(root_dir, output_folder)

f = open(input_path, 'r')
# out_f = open("README.txt", 'r')
try:
    num = 0
    while True:
        num += 1
        line = next(f)
        if line[0] == '>':
            if num != 1:
                out_f.close()
            f_n = os.path.join(output_path, line[1:-1].replace('*', '_star') + '.fa')
            out_f = open(f_n, 'w')
            out_f.write(line)
        else:
            out_f.write(line.strip().lower())

except:
    f.close()
    out_f.close()
    print('----OK---'*5)


