import functools
import logging

import numpy as np
import pandas as pd
import torch

import smtr.datasets as da
# import atom3d.util.rosetta as rose
# import e3nn.point.data_helpers as dh
import torch_geometric as tg

logger = logging.getLogger("lightning")


def create_transform(label_dir):
    # read label

    transform = functools.partial(prepare)

    return transform


def prepare(item, label='0', k=2):
    element_mapping = {
        'C': 0,
        'O': 1,
        'N': 2,
    }
    num_channels = len(element_mapping)
    if type(item['atoms']) != pd.DataFrame:
        item['atoms'] = pd.DataFrame(**item['atoms'])
    coords = item['atoms'][['x', 'y', 'z']].values
    elements = item['atoms']['element'].values

    sel = np.array([i for i, e in enumerate(elements) if e in element_mapping])
    total_atoms = elements.shape[0]
    coords = coords[sel]
    elements = elements[sel]

    # Make one-hot
    elements_int = np.array([element_mapping[e] for e in elements])
    one_hot = np.zeros((elements.size, len(element_mapping)))
    one_hot[np.arange(elements.size), elements_int] = 1

    geometry = torch.tensor(coords, dtype=torch.float32)
    features = torch.tensor(one_hot, dtype=torch.float32)

    ra = geometry.unsqueeze(0)
    rb = geometry.unsqueeze(1)
    # get all atoms distance
    pdist = (ra - rb).norm(dim=2)
    tmp = torch.topk(-pdist, k, axis=1)

    nei_list = []
    geo_list = []
    for source, x in enumerate(tmp.indices):
        cart = geometry[x]
        nei_list.append(
            torch.tensor(
                [[source, dest] for dest in x], dtype=torch.long))
        geo_list.append(cart - geometry[source])
    nei_list = torch.cat(nei_list, dim=0).transpose(1, 0)
    geo_list = torch.cat(geo_list, dim=0)

    r_max = 10  # Doesn't matter since we override
    d = tg.data.Data(x=features, pos=geometry, edge_index=nei_list, edge_attr=geo_list, y=label)
    # d.edge_attr = geo_list
    # d.edge_index = nei_list
    # d.y = label
    d.id = item['id']
    d.file_path = item['file_path']

    return d
