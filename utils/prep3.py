# %%

import ast
import torch
import pandas as pd
import numpy as np
from pathlib import Path

import warnings

from joblib import Parallel, delayed
from tqdm import tqdm

import datamol as dm
from molfeat.calc.atom import AtomCalculator, AtomMaterialCalculator, DGLWeaveAtomCalculator
from molfeat.calc.bond import EdgeMatCalculator, BondCalculator
from molfeat.trans.graph import PYGGraphTransformer

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')

def convert_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return []



# %%
data = pd.read_csv('../data/QM_137k.csv')

# %%
data['CDD'] = data['CDD'].apply(convert_string_to_list)
columns_to_drop = ['hirshfeld_charges', 'hirshfeld_fukui_elec', 'hirshfeld_fukui_neu', 'NMR_SC', 'bond_length_matrix', 'bond_index_matrix']
data = data.drop(columns=columns_to_drop, axis=1)

# %%
smiles_column = "smiles"

def _preprocess(i, row):

    dm.disable_rdkit_log()

    mol = dm.to_mol(row[smiles_column], ordered=True, add_hs=True, kekulize=True)
    if mol is None:
        return None

    mol = dm.fix_mol(mol)
    if mol is None:
        return None
    
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=True, add_hs=True)
    if mol is None:
        return None

    mol = dm.standardize_mol(
        mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True
    )
    if mol is None:
        return None

    row["smiles"] = dm.standardize_smiles(dm.to_smiles(mol))

    row['mol'] = mol
    return row


processed_results = dm.parallelized(_preprocess, data.iterrows(), arg_type="args", n_jobs=-1, progress=True, total=len(data))
processed_results = [result for result in processed_results if result is not None]
data = pd.DataFrame(processed_results)

mols = data["mol"].tolist()


# %%
def process_batch(mols, data, start_idx, end_idx, pyg_trans, skipatom_model):
    results = []
    for idx in range(start_idx, end_idx):
        mol = mols[idx]
        try:
            graph = pyg_trans.transform([mol])[0]
            
            # Process targets
            graph.y = torch.tensor([data['CDD'].iloc[idx]], dtype=torch.float32).squeeze()
            graph.smiles = data['smiles'].iloc[idx]
            
            if skipatom_model:
                atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                atom_features = [skipatom_model.vectors[skipatom_model.dictionary[symbol]] for symbol in atom_symbols]
                atom_features_tensor = torch.tensor(np.array(atom_features, dtype=np.float32))
                graph.x = torch.cat([graph.x, atom_features_tensor], dim=1)
            
            results.append(graph)
        except Exception as e:
            print(f"Err {e}")
    return results

def process_dataset(mols, data, n_jobs=-1, skipatom_model=None, progress_bar=True, batch_size=1000):
    pyg_trans = PYGGraphTransformer(
        atom_featurizer=AtomCalculator(),
        bond_featurizer=BondCalculator(self_loop=True),
        explicit_hydrogens=True,
        self_loop=True,
        canonical_atom_order=True,
        dtype=np.float32,
    )

    total = len(mols)
    tasks = (total + batch_size - 1) // batch_size

    if progress_bar:
        results = Parallel(n_jobs=n_jobs)(delayed(process_batch)(mols, data, i * batch_size, min((i + 1) * batch_size, total), pyg_trans, skipatom_model) for i in tqdm(range(tasks)))
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(process_batch)(mols, data, i * batch_size, min((i + 1) * batch_size, total), pyg_trans, skipatom_model) for i in range(tasks))
    
    dataset = [item for sublist in results for item in sublist]
    return dataset

dataset = process_dataset(mols, data, n_jobs=4, skipatom_model=None, progress_bar=True, batch_size=1000)
print(dataset[0])

# %%
def clean_dataset(dataset):
    inconsistencies = []
    inconsistency_counts = {'size_mismatch': 0}
    indices_to_remove = []

    for idx, data_point in enumerate(dataset):
        if data_point.x.shape[0] != data_point.y.shape[0]:
            inconsistencies.append((idx, data_point.x.shape[0], data_point.y.shape[0], data_point.smiles))
            inconsistency_counts['size_mismatch'] += 1
            indices_to_remove.append(idx)

    dataset_clean = [data_point for idx, data_point in enumerate(dataset) if idx not in indices_to_remove]

    if inconsistencies:
        print("Inconsistencies found in the following dataset elements:")
        for incon in inconsistencies:
            print(f"Index: {incon[0]}, X size: {incon[1]}, Y size: {incon[2]}, SMILES: {incon[3]}")
    else:
        print("All dataset elements are consistent.")

    print("Number of different types of inconsistencies:")
    for key, value in inconsistency_counts.items():
        print(f"{key}: {value}")

    print(f"Removed {len(dataset) - len(dataset_clean)} inconsistent elements. New dataset size: {len(dataset_clean)}")
    return dataset_clean

dataset_clean = clean_dataset(dataset)


# %%
torch.save(dataset_clean, f'../data/QM_137k_atom_bond_self.pt')


