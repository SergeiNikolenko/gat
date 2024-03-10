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
    """Параметры для фичеризации молекул.
    
    Attributes:
        max_atomic_num (int): Максимальный атомный номер для кодирования.
        atom_features (dict): Словарь с определениями фичей атомов.
        atom_fdim (int): Размерность вектора фичей атома.
    """
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
    """Однократное кодирование с неизвестным значением.
    
    Args:
        value: Значение для кодирования.
        choices: Список возможных значений.
    
    Returns:
        List[int]: Однократно закодированный вектор.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def atom_features(atom, params):
    """Генерирует фичи для одного атома.
    
    Args:
        atom (Chem.rdchem.Atom): Атом для фичеризации.
        params (FeaturizationParameters): Параметры фичеризации.
    
    Returns:
        List[Union[bool, int, float]]: Фичи атома.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, params.atom_features['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), params.atom_features['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), params.atom_features['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), params.atom_features['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), params.atom_features['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), params.atom_features['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features

PARAMS = {
    'BOND_FDIM': 10
}

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """Генерирует фичи для одной связи.
    
    Args:
        bond (Chem.rdchem.Bond): Связь для фичеризации.
    
    Returns:
        List[Union[bool, int, float]]: Фичи связи.
    """
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
    return fbond


class MoleculeData:
    """Данные молекулы для графовой нейросети.
    
    Args:
        smiles (str): SMILES представление молекулы.
        target: Целевое значение для молекулы.
        addHs (bool, optional): Добавлять водороды к молекуле. По умолчанию True.
    
    Attributes:
        smiles (str): SMILES представление молекулы.
        target (torch.Tensor): Целевое значение для молекулы.
        mol (Chem.rdchem.Mol): Объект молекулы RDKit.
        params (FeaturizationParameters): Параметры для фичеризации молекул.
        edge_index (torch.Tensor): Индексы рёбер для графа молекулы.
        edge_attr (torch.Tensor): Атрибуты рёбер для графа молекулы.
    """
    def __init__(self, smiles, target, addHs=True):
        self.smiles = smiles
        self.target = torch.tensor(target, dtype=torch.float)
        self.mol = Chem.MolFromSmiles(smiles)
        if addHs:
            self.mol = Chem.AddHs(self.mol)
        self.params = FeaturizationParameters()
        self.edge_index, self.edge_attr = self.construct_graph()

    def construct_graph(self):
        edge_index = []
        edge_attr = []
        for bond in self.mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[start, end], [end, start]])
            edge_attr.extend([bond_features(bond), bond_features(bond)])  # Добавляем признаки для обеих направлений связи
        return torch.tensor(edge_index).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float)

    def generate_atom_features(self):
        features = []
        for atom in self.mol.GetAtoms():
            features.append(atom_features(atom, self.params))
        return torch.tensor(features, dtype=torch.float)

class MoleculeDataset(Dataset):
    """Датасет молекул для обучения графовых нейросетей.
    
    Args:
        dataframe (pd.DataFrame): DataFrame с данными молекул.
        smiles_column (str, optional): Имя колонки с SMILES представлениями. По умолчанию 'smiles'.
        target_column (str, optional): Имя колонки с целевыми значениями. По умолчанию 'target'.
        addHs (bool, optional): Добавлять водороды к молекулам. По умолчанию True.
        n_jobs (int, optional): Количество параллельных задач. По умолчанию -1 (использовать все ядра).
    
    Attributes:
        data_list (List[MoleculeData]): Список объектов данных молекул.
    """
    def __init__(self, dataframe, smiles_column='smiles', target_column='target', addHs=True, n_jobs=-1):
        super(MoleculeDataset, self).__init__()
        
        self.data_list = Parallel(n_jobs=n_jobs)(
            delayed(lambda row: MoleculeData(row[smiles_column], row[target_column], addHs))(
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





def save_dataset(dataset, file_path):
    """Сохраняет датасет в файл.

    Args:
        dataset (Dataset): Экземпляр датасета для сохранения.
        file_path (str): Путь к файлу для сохранения датасета.
    """
    torch.save(dataset, file_path)
    print(f"Датасет успешно сохранен в {file_path}")



def load_dataset(file_path):
    """Загружает датасет из файла.

    Args:
        file_path (str): Путь к файлу, из которого нужно загрузить датасет.

    Returns:
        Dataset: Загруженный датасет.
    """
    dataset = torch.load(file_path)

    print(dataset)
    print(dataset[0])
    
    print(f"Shape of atom features (x): {dataset[0].x.shape}")
    print(f"Shape of edge index: {dataset[0].edge_index.shape}")
    print(f"Shape of edge attr: {dataset[0].edge_attr.shape}")
    print(f"Target value (y): {dataset[0].y}")
    print(f"Shape of target value: {dataset[0].y.shape}")
    print(f"Number of atoms in the molecule: {dataset[0].x.size(0)}")
    print(f"Number of bonds in the molecule: {dataset[0].edge_index.size(1) // 2}") 

    return dataset

def convert_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return []
