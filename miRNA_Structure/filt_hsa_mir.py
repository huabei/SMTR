#coding=gbk
# take all human hairpin from 'hairpin.fa'
# 20220116
# Huabei did

f = open('hairpin.fa', 'r')
out_f = open('hsa_hairpin.fa', 'w')
bol = False
try:
    while True:
        line = next(f)
        if line[0] == '>':
            if line[1:4] == 'hsa':
                bol = True
                out_f.write(line)
            else:
                bol = False
        elif bol:
            out_f.write(line.lower())
except:
    f.close()
    out_f.close()
    print('-------OK------OK------OK-----OK------OK------OK-------')

