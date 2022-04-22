# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:get_inforna.py
@time:2022/03/09
"""
import os
import time
import pandas as pd


root_dir = r'E:\Research\SM_miRNA\Data\rosetta\20220116\hairpin_sec_ct\info_data'
file_list = os.listdir(root_dir)
db_motif = None
# pre_mirna_db = dict()
for file in file_list:
    print('read file' + file)
    file_path = os.path.join(root_dir, file)
    data_tmp = pd.read_csv(file_path, delimiter='\t', encoding='latin1')
    db_motif_tmp = data_tmp[['Motif ID', 'SMILES', 'Fitness Score', 'Loop Nucleotides', 'Motif in Target RNA', 'PMID',
                             'Loop Identifier', 'Dissociation Constant']]
    # 筛选出所有的query motif 保存到集合
    query_motif_tmp = set([list(x)[0] for x in data_tmp[['Query Motif']].values])
    print('query motif')
    print(query_motif_tmp)
    pre_mirna_tmp = dict()
    if data_tmp.empty:
        continue
    # for motif in query_motif_tmp:
    #     id2query_motif = db_motif_tmp[data_tmp['Query Motif'] == motif]
    #     pre_mirna_tmp[motif] = id2query_motif
    # pre_mirna_db[data_tmp.loc[0, 'CT File Name']] = pre_mirna_tmp
    if db_motif is None:
        db_motif = data_tmp
        continue
    db_motif = pd.concat([db_motif, data_tmp])

db_motif.to_csv('inforna_pre-rna_db.csv', index=False)
exact_match = db_motif[db_motif['Motif in Target RNA'].str.lower() == db_motif['Query Motif'].values]
exact_match.to_csv('exact_match.csv', index=False)
exact_match_fit100 = exact_match[exact_match['Fitness Score'] == 100]
exact_match_fit100.to_csv('exact_match_fit100.csv', index=False)



