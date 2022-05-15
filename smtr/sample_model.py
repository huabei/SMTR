# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:sample_model.py
@time:2022/05/05
"""

from smtr.data import create_dataset
from sklearn.model_selection import train_test_split
from e3nn.nn.models.v2106.gate_points_networks import NetworkForAGraphWithAttributes
from torch_cluster import radius_graph
import torch
import numpy as np

if __name__ == '__main__':
    data_file = r'D:\Projects\smtr\dataset\data_train.txt'
    dataset, N_atoms = create_dataset(data_file)
    dataset_train, dataset_val = train_test_split(dataset, test_size=0.2, random_state=42)
    data = dataset_train[0]
    num_nodes = len(data[1])
    one_hot = np.zeros((num_nodes, N_atoms))
    one_hot[np.arange(num_nodes), data[0]] = 1
    max_radius = 2.0

    net = NetworkForAGraphWithAttributes(
        irreps_node_input='9x0e',
        irreps_node_attr='1o',
        irreps_edge_attr='1o',
        irreps_node_output='1o',
        max_radius=max_radius,
        num_neighbors=4.0,
        num_nodes=num_nodes
    )
    pos = torch.FloatTensor(data[1])
    edge_index = radius_graph(pos, max_radius)
    edge_src = edge_index[0]
    edge_dst = edge_index[1]
    num_edges = edge_index.shape[1]
    edge_vec = pos[edge_src] - pos[edge_dst]
    net({
        'pos': pos,
        'edge_src': edge_src,
        'edge_dst': edge_dst,
        'node_input': torch.IntTensor(one_hot),
        'node_attr': pos,
        'edge_attr': edge_vec,
    })
    # pass