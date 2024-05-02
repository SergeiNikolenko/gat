import torch
import datamol as dm
from tqdm import tqdm
from skipatom import SkipAtomInducedModel
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def add_skipatom_features_to_dataset(dataset,
                                     model_file = "../skipatom/data/mp_2020_10_09.dim30.model",
                                     training_data="../skipatom/data/mp_2020_10_09.training.data",
                                     min_count=2e7,
                                     top_n=4,
                                     device='cpu',
                                     progress_bar=True,
                                     scaler=StandardScaler()):
    """
    Enhances a dataset of molecular graphs with features generated by the SkipAtom model.

    This function processes a given dataset of molecular graphs by adding new features. These features are derived from a pre-trained SkipAtom model which captures contextual relationships between atoms in molecular structures.

    Args:
        dataset (list): A list of molecular graphs to be processed.
        model_file (str): Path to the trained SkipAtom model file.
        training_data (str): Path to the training data used for the SkipAtom model.
        min_count (float, optional): The minimum count threshold for atoms to be included in the model vocabulary. Defaults to 20,000,000.
        top_n (int, optional): The number of top neighboring atoms to consider for feature generation. Defaults to 4.
        device (str, optional): The computation device ('cpu' or 'cuda') on which the model runs. Defaults to 'cpu'.
        progress_bar (bool, optional): If True, displays a progress bar during the processing of the dataset. Defaults to True.
        scaler (sklearn.preprocessing._data.StandardScaler, optional): The scaler to be used for feature scaling. Defaults to StandardScaler().

    Returns:
        list: A new dataset with enhanced features where each graph has additional atom features appended.

    Raises:
        FileNotFoundError: If the model file or training data file does not exist.
        RuntimeError: For errors related to PyTorch operations, particularly when moving tensors to GPU.
    """
    try:
        model_path = Path(model_file)
        data_path = Path(training_data)
        model = SkipAtomInducedModel.load(model_path, data_path, min_count=min_count, top_n=top_n)
        model.vectors = torch.tensor(model.vectors, dtype=torch.float32).to(device)

        new_dataset = []

        progress = tqdm(dataset, desc="Add Skipatom features", disable=not progress_bar)
        with torch.no_grad():
            for graph in progress:
                new_graph = graph.clone() 
                new_graph.x = new_graph.x.to(device)
                mol = dm.to_mol(new_graph.smiles, add_hs=True)
                if mol is None:
                    continue

                atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                atom_indices = [model.dictionary.get(symbol, -1) for symbol in atom_symbols]
                atom_features = model.vectors[atom_indices]

                if len(atom_features) == new_graph.x.size(0):
                    # Применение скалера перед конкатенацией фичей
                    scaler=scaler
                    atom_features_np = atom_features.cpu().numpy()
                    atom_features_scaled = scaler.fit_transform(atom_features_np)
                    atom_features_tensor = torch.tensor(atom_features_scaled, dtype=torch.float32, device=device)
                    new_graph.x = torch.cat([new_graph.x, atom_features_tensor], dim=1)
                else:
                    print(f"Error: Mismatch in features count. Graph features {new_graph.x.size(0)}, Atom features {len(atom_features)}")
                
                new_dataset.append(new_graph)
                
        torch.cuda.empty_cache()

        return new_dataset
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise