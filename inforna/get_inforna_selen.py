# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:get_inforna.py
@time:2022/03/09
"""
import requests
import os
import time
from selenium import webdriver

driver = webdriver.Firefox()
url = "https://rna-inforna.florida.scripps.edu/"
passwd = "Sm@11ml$"
driver.get(url)
root_dir = r'E:\Research\SM_miRNA\Data\rosetta\20220116\hairpin_sec_ct\ct'
file_list = os.listdir(root_dir)
num = 0
for file_name in file_list:
    num += 1
    print(num)
    file_path = os.path.join(root_dir, file_name)

    driver.find_element_by_xpath('//*[@id="ext-comp-1002"]').send_keys(file_path)
    driver.find_element_by_id('ext-gen71').click()
    time.sleep(5)


