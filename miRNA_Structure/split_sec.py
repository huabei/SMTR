# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 13:16
# @Author  : HuaBei
# @File    : split.py
''' split RNAfold out file'''
from sys import argv
import pathlib
import os

# file path prepare
input_file, output_folder = argv[1], argv[2]

# if input Relative path
root_dir = pathlib.Path(__file__).parent.absolute()
input_path = os.path.join(root_dir, input_file)
output_path = os.path.join(root_dir, output_folder)
# print(input_path, output_path)
# raise ValueError
# work_path = r"E:\Research\SM_miRNA\Data\miR_Squence\mirgenedb"

in_f = open(input_path, 'r')
try:
    while True:
        # 获取元素

        # 三行一个循环
        line_name = next(in_f)
        line_sequence = next(in_f)
        # 二级结构和能量有时候有多余的空格，sec_e是列表
        line_sec, *sec_e = next(in_f).split(' ')

        # 将能量中多余的空格消除
        sec_e = ''.join(sec_e)
        # 输出文件路径
        f_n = os.path.join(output_path, line_name.replace('*', '_star').split(" ")[0][1:-1] + '_sec.fold')
        # f_n = output_path + line_name.split(" ")[0][1:-1] + '_sec.fold'
        out_f = open(f_n, 'w')

        # print(line_name)
        # print(line_sec)
        # 写入文件
        out_f.write(line_sec + '\n')
        out_f.write(line_sequence)
        out_f.write(line_name)
        out_f.write(sec_e)
        out_f.close()
except:
    in_f.close()
    print('----OK---'*5)


