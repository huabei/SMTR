# data prepare
# transfor pdb to lmdb
import click
import logging
import sys, os
import atom3d.datasets.datasets as da
import atom3d.util.file as fi
import atom3d.util.formats as fo
import functools

logger = logging.getLogger(__name__)
root_dir = '/home/huabei/work/ares_release/ares_release'
train_pdb_dir = root_dir + '/data/classics_train_val/example_train'
val_pdb_dir = root_dir + '/data/classics_train_val/example_val'
train_output_lmdb = root_dir + '/data/lmdbs/train/classics_train'
val_output_lmdb = root_dir + '/data/lmdbs/val/classics_val'
filetype = 'pdb'
serialization_format = 'json'


def prepare(item, label_to_use='rms'):
    pdb_file = item['file_path']
    # print(pdb_file)
    f = open(pdb_file)
    scores = {}
    for ft in f:
        if ft[:4] == 'rms ':
            rms_key, rms_value = ft.split(' ')
            scores[rms_key] = float(rms_value)
            item['scores'] = scores
    return item


transform = functools.partial(prepare, label_to_use='rms')

# @click.command()
# @click.argument('input_dir', type=click.Path(exists=True))
# @click.argument('output_lmdb', type=click.Path(exists=False))
# @click.option('-f', '--filetype', type=click.Choice(['pdb', 'silent', 'xyz', 'xyz-gdb']),
#               default='pdb')
# @click.option('-sf', '--serialization_format',
#               type=click.Choice(['msgpack', 'pkl', 'json']),
#               default='json')
# @click.option('--score_path', type=click.Path(exists=True))
input_dir = train_pdb_dir
output_lmdb = train_output_lmdb

fileext = filetype
file_list = da.get_file_list(input_dir, fileext)

dataset = da.load_dataset(file_list, filetype, transform=transform)

da.make_lmdb_dataset(
    dataset, output_lmdb, serialization_format=serialization_format)


