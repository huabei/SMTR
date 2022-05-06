# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:44:41 2022

@author: ZNDX
"""
import pandas as pd
from optparse import OptionParser


parser = OptionParser(usage="%prog [options]",
        description=__doc__)
parser.add_option('-c', "--chainname", action="store",dest='chaname',
        help="Set ssRNA chain name")
parser.add_option('-i', "--iputsilent", action="store",dest='iputpdbfp',
        help="Input RNA PDB file.")
parser.add_option('-o', "--oputpdb", action="store", dest='oputpdbfp',
        help="Output RNA PDB file.")
parser.add_option('-s', "--start_num", action="store", dest='start_number',
        help="Which number to start the chain residue.")


web_s = pd.read_csv(r'\\wsl.localhost\Ubuntu\home\huabei\work\ares_release\ares_release\data\silents\pre-miR-21_100.txt', header=None)
local_s = pd.read_csv(r'\\wsl.localhost\Ubuntu\home\huabei\work\ares_release\ares_release\pre_mir_21_output_new_rms_20ep.txt')

# for i in test_data:
#     if np.array(i).shape != (9, 3):
#         print(i)
#         break