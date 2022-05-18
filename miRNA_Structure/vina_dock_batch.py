#！/usr/bin/python3
from sys import argv
import os
from tqdm.auto import tqdm

# prepare files
pyfile_name, receptor_file, ligand_folder, config, output_folder = argv
root_dir = os.getcwd()
receptor_path = os.path.join(root_dir, receptor_file)
ligand_path = os.path.join(root_dir, ligand_folder)
output_path = os.path.join(root_dir, output_folder)
config_path = os.path.join(root_dir, config)
process_num = 0
for file_name in tqdm(os.listdir(ligand_path), desc='docking'):
    process_num += 1
    file_path = os.path.join(ligand_path, file_name)
    output_file_path = os.path.join(output_path, file_name.replace('.pdbqt', '_out.pdbqt'))
    # print('starting process {}'.format(process_num))
    if os.path.exists(output_file_path):
        continue
    command = 'vina --config {} --receptor {} --ligand {} --out {}'.format(config_path, receptor_path, file_path, output_file_path)
    if os.system(command):
        print('wrong in docking with {}'.format(file_name))
        print('\n --Try Again---')
        if os.system(command):
            print('Wrong Again\n Next One')
print('Dock Finished, Congratulate!')
    # break