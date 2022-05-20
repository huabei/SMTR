# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 13:16
# @Author  : HuaBei
# @File    : split.py
''' split .fas to singel file and lower letters'''
work_flow = r'E:\Research\SM_miRNA\Data\miR_Squence\mirgenedb'
file_name = 'hsa-pre-miR.fas'
f = open(work_flow + '\\' + file_name, 'r')
out_f = open("README.txt", 'r')
try:
    while True:
        line = next(f)
        if line[0] == '>':
            out_f.close()
            f_n = work_flow + '\\hairpin\\'+line.split(" ")[0][1:] + '.fa'
            out_f = open(f_n, 'w')
            out_f.write(line)
        else:
            out_f.write(line.strip())

except:
    f.close()
    out_f.close()
    print('----OK---'*5)


