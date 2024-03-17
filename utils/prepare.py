import os
import ast
import numpy as np

from typing import List, Union


from rdkit import Chem
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from joblib import Parallel, delayed


class FeaturizationParameters:
    def __init__(self):
        self.max_atomic_num = 100
        self.atom_features = {
            'atomic_num': list(range(self.max_atomic_num)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
        }
        self.atom_fdim = sum(len(choices) + 1 for choices in self.atom_features.values()) + 2

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_skipatom_vector(atom_symbol, skipatom_model):
    if atom_symbol in skipatom_model.dictionary:
        return skipatom_model.vectors[skipatom_model.dictionary[atom_symbol]].tolist()
    else:
        return [0] * skipatom_model.vectors.shape[1]


def atom_features(atom, params, skipatom_model=None):
    # Существующие признаки атома
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, params.atom_features['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), params.atom_features['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), params.atom_features['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), params.atom_features['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), params.atom_features['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), params.atom_features['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # Масштабирование массы

    # Признаки SkipAtom
    if skipatom_model is not None:
        atom_symbol = atom.GetSymbol()
        skipatom_features = get_skipatom_vector(atom_symbol, skipatom_model)
        features += skipatom_features

    return features

PARAMS = {
    'BOND_FDIM': 10
}

def bond_features(bond: Chem.rdchem.Bond, skipatom_model=None) -> List[Union[bool, int, float, np.ndarray]]:
    if bond is None:
        fbond = [1] + [0] * (PARAMS['BOND_FDIM'] - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated() if bt is not None else 0,
            bond.IsInRing() if bt is not None else 0
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    # Признаки SkipAtom
    if skipatom_model is not None:
        start_atom_symbol = bond.GetBeginAtom().GetSymbol()
        end_atom_symbol = bond.GetEndAtom().GetSymbol()

        start_atom_vector = get_skipatom_vector(start_atom_symbol, skipatom_model)
        end_atom_vector = get_skipatom_vector(end_atom_symbol, skipatom_model)

        fbond += start_atom_vector + end_atom_vector
    
    return fbond


class MoleculeData:
    def __init__(self, smiles, target, addHs=True, skipatom_model=None):
        self.smiles = smiles
        self.target = torch.tensor(target, dtype=torch.float)
        self.mol = Chem.MolFromSmiles(smiles)
        if addHs:
            self.mol = Chem.AddHs(self.mol)
        self.params = FeaturizationParameters()
        self.skipatom_model = skipatom_model
        self.edge_index, self.edge_attr = self.construct_graph()

    def construct_graph(self):
        edge_index = []
        edge_attr = []
        for bond in self.mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[start, end], [end, start]])
            edge_attr.extend([
                bond_features(bond, self.skipatom_model),
                bond_features(bond, self.skipatom_model)
            ])
        return torch.tensor(edge_index).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float)

    def generate_atom_features(self):
        features = []
        for atom in self.mol.GetAtoms():
            features.append(atom_features(atom, self.params, self.skipatom_model))
        return torch.tensor(features, dtype=torch.float)

class MoleculeDataset(Dataset):
    def __init__(self, dataframe, smiles_column='smiles', target_column='target', addHs=True, n_jobs=-1, skipatom_model=None):
        super(MoleculeDataset, self).__init__()
        self.use_skipatom = skipatom_model is not None
        self.data_list = Parallel(n_jobs=n_jobs)(
            delayed(lambda row: MoleculeData(row[smiles_column], row[target_column], addHs, skipatom_model))(
                row) for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]))



    def len(self): 
        return len(self.data_list)

    def get(self, idx):
        molecule_data = self.data_list[idx]
        x = molecule_data.generate_atom_features()
        edge_index = molecule_data.edge_index
        edge_attr = molecule_data.edge_attr
        y = molecule_data.target
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.smiles = molecule_data.smiles
        
        return data

def convert_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return []

