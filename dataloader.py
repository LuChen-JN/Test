import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_DNA


class NTIDataset(data.Dataset):
    def __init__(self, list_IDs, df,predict=False, max_drug_nodes = 180):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.predict = predict

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        # compound  encoder
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')

        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.zeros(num_virtual_nodes, 74)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()
        # Sequence encoding
        v_p = self.df.iloc[index]['Sequence']
        v_p = integer_label_DNA(v_p)
        if self.predict == True:
            y = 0
        else:
            y = self.df.iloc[index]["Y"]
        return v_d, v_p, y


