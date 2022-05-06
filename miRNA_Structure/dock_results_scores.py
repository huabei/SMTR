#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/python3

''' take auto dock results score'''


# In[1]:


import sys
import os
import numpy as np
from scipy.io import savemat

# In[ ]:


# prepare files folder path
pyfile_path, input_file_path, output_file_path = sys.argv
root_dir = os.getcwd()
input_path = os.path.join(root_dir, input_file_path)
output_file = os.path.join(root_dir, output_file_path)


total_dock_data = dict()
for file in os.listdir(input_path):
    file = os.path.join(input_path, file)
    
    if file[-6:] == '.pdbqt':
        f = open(file)
        dock_data = list()
        for line in f:
            if line[:18] == 'REMARK VINA RESULT':
                x = line.split(' ')
                x = [i for i in x if i != ''][3:]
                x = [float(i) for i in x]
                dock_data.append(x)
        f.close()
        total_dock_data[file[:-6]] = np.array(dock_data)
    else:
        print(' Nonsupport file {}'.format(file[-6:]))
# scipy.io.savemat(file_name, mdict, appendmat=True)
if output_file[-4:] != '.mat':
    output_file += '.mat'
# print(output_file)
savemat(output_file, total_dock_data, appendmat=True)


