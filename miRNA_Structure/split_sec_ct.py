# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 13:16
# @Author  : HuaBei
# @File    : split.py
'''split sec file for generate ct file '''
work_path = r'E:\Research\SM_miRNA\Data\miR_Squence\mirgenedb\hsa-pre-miR-sec.fold'
in_f = open(work_path, 'r')


while True:
    # 获取元素
    line_name = next(in_f)
    line_sequence = next(in_f)
    line_sec, *sec_e = next(in_f).split(' ')
    sec_e = ''.join(sec_e)
    f_n = 'hairpin_sec_ct/sec/'+line_name.split(" ")[0][1:] + '_sec.fold'
    out_f = open(f_n, 'w')
    print(line_name)
    print(line_sec)
    out_f.write(line_name)
    out_f.write(line_sequence)
    out_f.write(line_sec)
    # out_f.write(sec_e)
    out_f.close()
    # break

in_f.close()
print('----OK---'*5)


