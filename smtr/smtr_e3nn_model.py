
# 20220518 created by huabei

import torch
from torch.utils.data import random_split
import pytorch_lightning as pl
import collections as col
from functools import partial
from torch_scatter import scatter_mean

from e3nn.kernel import Kernel
from e3nn.linear import Linear
from e3nn import o3
# from e3nn.non_linearities.norm import Norm
from e3nn.non_linearities.norm_activation import NormActivation
from e3nn.point.message_passing import Convolution
from e3nn.radial import GaussianRadialModel
from e3nn.non_linearities.rescaled_act import shiftedsoftplus
from e3nn import rs
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
from torch_geometric.data import Data
import copy

# custom Norm Add by Huabei
class Norm(torch.nn.Module):
    def __init__(self, Rs, normalization='component'):
        super().__init__()

        Rs = rs.simplify(Rs)
        n = sum(mul for mul, _, _ in Rs)
        self.Rs_in = Rs
        self.Rs_out = [(n, 0, +1)]
        self.normalization = normalization

    def forward(self, features):
        '''
        :param features: [..., channels]
        '''
        *size, n = features.size()
        output = []
        index = 0
        for mul, l, _ in self.Rs_in:
            sub = features.narrow(-1, index, mul * (2 * l + 1)).reshape(*size, mul, 2 * l + 1)  # [..., u, i]
            index += mul * (2 * l + 1)

            norms = sub.norm(2, dim=-1, keepdim=True)  # [..., u]
            sub = sub / norms
            if self.normalization == 'component':
                sub = sub / (2 * l + 1) ** 0.5

            output.append(sub)
        assert index == n

        return torch.cat(output, dim=-1).reshape(*size, mul * (2 * l + 1))


class SMTR(pl.LightningModule):

    # @staticmethod
    def __init__(self, learning_rate=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.predictions = col.defaultdict(list)
        self.lr = learning_rate
        # Define the input and output representations
        Rs0 = [(8, 0)]
        Rs1 = [(24, 0)]
        Rs20 = [(24, 0)]
        Rs21 = [(24, 1)]
        Rs22 = [(24, 2)]
        Rs3 = [(12, 0), (12, 1), (12, 2)]
        Rs30 = [(12, 0)]
        Rs31 = [(12, 1)]
        Rs32 = [(12, 2)]
        # To account for multiple output paths of conv
        Rs30_exp = [(3 * 12, 0)]
        Rs31_exp = [(6 * 12, 1)]
        Rs32_exp = [(6 * 12, 2)]
        Rs4 = [(4, 0), (4, 1), (4, 2)]
        Rs40 = [(4, 0)]
        Rs41 = [(4, 1)]
        Rs42 = [(4, 2)]
        Rs40_exp = [(3 * 4, 0)]
        Rs41_exp = [(6 * 4, 1)]
        Rs42_exp = [(6 * 4, 2)]
        # change from 4 to 36
        Ds1 = (4, 4)
        Ds2 = (4, 256)
        Ds3 = (256, 1)
        relu = torch.nn.ReLU()
        # Radial model: R+ -> R^d
        RadialModel = partial(
            GaussianRadialModel, max_radius=5.0, number_of_basis=5, h=12,
            L=1, act=relu)
        
        ssp = shiftedsoftplus
        self.elu = torch.nn.ELU()

        # kernel: composed on a radial part that contains the learned
        # parameters and an angular part given by the spherical hamonics and
        # the Clebsch-Gordan coefficients
        selection_rule = partial(o3.selection_rule_in_out_sh, lmax=2)
        K = partial(
            Kernel, RadialModel=RadialModel, selection_rule=selection_rule
        )
        self.lin1 = Linear(Rs0, Rs1)
        
        self.conv10 = Convolution(K(Rs1, Rs20))
        self.conv11 = Convolution(K(Rs1, Rs21))
        self.conv12 = Convolution(K(Rs1, Rs22))
        
        self.norm10 = Norm(Rs20)
        self.norm11 = Norm(Rs21)
        self.norm12 = Norm(Rs22)

        self.lin20 = Linear(Rs20, Rs20)
        self.lin21 = Linear(Rs21, Rs21)
        self.lin22 = Linear(Rs22, Rs22)
        
        self.nonlin10 = NormActivation(Rs20, activation=ssp)
        self.nonlin11 = NormActivation(Rs21, activation=ssp)
        self.nonlin12 = NormActivation(Rs22, activation=ssp)

        self.lin30 = Linear(Rs20, Rs30)
        self.lin31 = Linear(Rs21, Rs31)
        self.lin32 = Linear(Rs22, Rs32)
        def filterfn_def(x, f):
            return x == f
        
        self.conv2 = torch.nn.ModuleDict()
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    filterfn = partial(filterfn_def, f=f)
                    selection_rule = \
                        partial(o3.selection_rule, lmax=2, lfilter=filterfn)
                    K = partial(Kernel, RadialModel=RadialModel, selection_rule=selection_rule)
                    self.conv2[str((i, f, o))] = \
                        Convolution(K([Rs3[i]], [Rs3[o]]))
        self.norm20 = Norm(Rs30_exp)
        self.norm21 = Norm(Rs31_exp)
        self.norm22 = Norm(Rs32_exp)

        self.lin40 = Linear(Rs30_exp, Rs30)
        self.lin41 = Linear(Rs31_exp, Rs31)
        self.lin42 = Linear(Rs32_exp, Rs32)

        self.nonlin20 = NormActivation(Rs30, activation=ssp)
        self.nonlin21 = NormActivation(Rs31, activation=ssp)
        self.nonlin22 = NormActivation(Rs32, activation=ssp)

        self.lin50 = Linear(Rs30, Rs40)
        self.lin51 = Linear(Rs31, Rs41)
        self.lin52 = Linear(Rs32, Rs42)

        self.conv3 = torch.nn.ModuleDict()
        for i in range(3):
            for f in range(3):
                for o in range(abs(f -i), min(i + f + 1, 3)):
                    filterfn = partial(filterfn_def, f=f)
                    selection_rule = \
                        partial(o3.selection_rule, lmax=2, lfilter=filterfn)
                    K = partial(Kernel, RadialModel=RadialModel, selection_rule=selection_rule)
                    self.conv3[str((i, f, o))] = \
                        Convolution(K([Rs4[i]], [Rs4[o]]))
        self.norm30 = Norm(Rs40_exp)
        self.norm31 = Norm(Rs41_exp)
        self.norm32 = Norm(Rs42_exp)

        self.lin60 = Linear(Rs40_exp, Rs40)
        self.lin61 = Linear(Rs41_exp, Rs41)
        self.lin62 = Linear(Rs42_exp, Rs42)

        self.nonlin30 = NormActivation(Rs40, activation=ssp)
        self.nonlin31 = NormActivation(Rs41, activation=ssp)
        self.nonlin32 = NormActivation(Rs42, activation=ssp)
        
        self.dense1 = torch.nn.Linear(Ds1[0], Ds1[1], bias=True)
        self.dense2 = torch.nn.Linear(Ds2[0], Ds2[1], bias=True)
        self.dense3 = torch.nn.Linear(Ds3[0], Ds3[1], bias=True)


    def forward(self, d):
        out = self.lin1(d.x)

        out0 = self.conv10(out, d.edge_index, d.edge_attr)
        out1 = self.conv11(out, d.edge_index, d.edge_attr)
        out2 = self.conv12(out, d.edge_index, d.edge_attr)
        
        out0 = self.norm10(out0)
        out1 = self.norm11(out1)
        out2 = self.norm12(out2)
        
        out0 = self.lin20(out0)
        out1 = self.lin21(out1)
        out2 = self.lin22(out2)
        
        out0 = self.nonlin10(out0)
        out1 = self.nonlin11(out1)
        out2 = self.nonlin12(out2)

        out0 = self.lin30(out0)
        out1 = self.lin31(out1)
        out2 = self.lin32(out2)

        ins = {0: out0, 1: out1, 2: out2}
        tmp = col.defaultdict(list)
        for i in range(3):
            for f in range(3):
                for o in range(abs(f -i), min(i + f + 1, 3)):
                    curr =self.conv2[str((i, f, o))](
                        ins[i], d.edge_index, d.edge_attr)
                    tmp[o].append(curr)
        
        out0 = torch.cat(tmp[0], axis=1)
        out1 = torch.cat(tmp[1], axis=1)
        out2 = torch.cat(tmp[2], axis=1)

        out0 = self.norm20(out0)
        out1 = self.norm21(out1)
        out2 = self.norm22(out2)

        out0 = self.lin40(out0)
        out1 = self.lin41(out1)
        out2 = self.lin42(out2)

        out0 = self.nonlin20(out0)
        out1 = self.nonlin21(out1)
        out2 = self.nonlin22(out2)

        out0 = self.lin50(out0)
        out1 = self.lin51(out1)
        out2 = self.lin52(out2)

        ins = {0: out0, 1: out1, 2: out2}
        tmp = col.defaultdict(list)
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    curr = self.conv3[str((i, f, o))](
                        ins[i], d.edge_index, d.edge_attr)
                    tmp[o].append(curr)
        out0 = torch.cat(tmp[0], axis=1)
        out1 = torch.cat(tmp[1], axis=1)
        out2 = torch.cat(tmp[2], axis=1)
        
        out0 = self.norm30(out0)
        out1 = self.norm31(out1)
        out2 = self.norm32(out2)

        out0 = self.lin60(out0)
        out1 = self.lin61(out1)
        out2 = self.lin62(out2)

        out0 = self.nonlin30(out0)
        out1 = self.nonlin31(out1)
        out2 = self.nonlin32(out2)

        # out = out0, out1, out2
        # out = torch.cat((out0, out1, out2), dim=-1)
        # Per-channel mean
        out = scatter_mean(out0, d.batch, dim=0)

        out = self.dense1(out)
        out = self.elu(out)
        out = self.dense2(out)
        out = self.dense3(out)
        out = torch.squeeze(out, axis=1)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        y_hat = self(train_batch)
        loss = torch.nn.functional.smooth_l1_loss(y_hat, train_batch.label.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        y_hat = self(val_batch)
        loss = torch.nn.functional.smooth_l1_loss(y_hat, val_batch.label.float())
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = torch.nn.functional.smooth_l1_loss(y_hat, batch.label.float())
        self.predictions['pred'].extend(y_hat.cpu().numpy())
        self.predictions['true'].extend(batch.label.cpu().numpy())
        self.predictions['id'].extend(batch.id)
        return loss


class Ligand_dataset(Dataset):
    '''create dataset for sm ligand'''
    def __init__(self, file_path, transform=None):
        """Load a dataset."""
        self._transform = transform
        with open(file_path, 'r') as f:
            self._property_types = f.readline().strip().split()
            self._data_original = f.read().strip().split('\n\n')
        self._num_examples = len(self._data_original)
        elements_dict = defaultdict(lambda: len(elements_dict))

        self.items = list()
        
        for data in self._data_original:
            data = data.strip().split('\n')
            id = data[0]
            property = float(data[-1].strip())
            atoms, atom_coords = [], []
            for atom_xyz in data[1:-1]:
                atom, x, y, z = atom_xyz.split()
                atoms.append(atom)
                # atom_coords[0].append(float(x))
                # atom_coords[1].append(float(y))
                # atom_coords[2].append(float(z))
                xyz = [float(v) for v in [x, y, z]]
                atom_coords.append(xyz)
            atoms = [elements_dict[a] for a in atoms]
       
            molecule = {
                'elements': atoms,
                'atom_coords': atom_coords
            }
            item = {
                'molecule': molecule,
                'id': id,
                'score': property
            }
            self.items.append(item)
        self._num_elements = len(elements_dict)
    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        item = copy.deepcopy(self.items[index])
        # Make one-hot
        atoms = item['molecule']['elements']
        one_hot = np.zeros((len(atoms), self._num_elements))
        one_hot[np.arange(len(atoms)), np.array(atoms, dtype=np.int8)] = 1
        item['molecule']['elements'] = one_hot
        if self._transform:
            item = self._transform(item)
        return item

    

def create_transform(k=5):
    return partial(prepare, k=k)
def prepare(item, k=5):
    
    coords = item['molecule']['atom_coords']
    features = torch.tensor(item['molecule']['elements'], dtype=torch.float32)
    label = torch.tensor(item['score'])
    geometry = torch.tensor(coords, dtype=torch.float32)

    ra = geometry.unsqueeze(0)
    rb = geometry.unsqueeze(1)
    geo_list = [i.squeeze(0) for i in (ra - rb)]
    geo_list = torch.cat(geo_list, dim=0)
    nei_list = torch.tensor([(c, n) for c in range(len(coords)) for n in range(len(coords))]).transpose(1, 0)

    r_max = 10  # Doesn't matter since we override
    d = Data(features, nei_list, geo_list, y=label, pos=geometry)
    d.edge_attr = geo_list
    d.edge_index = nei_list
    d.label = label
    d.id = item['id']
    return d
