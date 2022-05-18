# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 13:54
# @Author  : HuaBei
# @File    : main.py
'''analysis FARFAR2 generate RNA time from stdout'''
import re
import matplotlib.pyplot as plt
import numpy as np

file_path = 'let_7a_1/scores_1.txt'
scores = np.loadtxt(file_path)
rf = np.var(scores)
mean = scores.mean()
print((mean, rf))

file = open(file_path, 'r')

f_str = file.read()
s_time = re.findall('S_(.*?) reported success in (.*?) seconds', f_str, re.S)
tm = [int(x[1]) for x in s_time]
tm_mean = np.array(tm).mean()
plt.plot(tm)
plt.ylim(-1000, 5000)
plt.show()


