##### my_utils.py

import os
import torch
import deepsmiles
import numpy as np
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from rdkit import Chem
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class myCVAE_Dataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 smiles_column,  # 'canonical_smiles' or 'deep_smiles'
                 **kwargs):  # **kwarg format e.g. prop_name = 'col_name' # e.g.  Tg='Tg', Tm='Tm'
        self.df = df
        self.smiles_column = smiles_column
        # extract info from kwargs
        self.properties_info = kwargs

        # calculate mean, std for each column
        self.property_stats = {}
        for prop_name, col_name in self.properties_info.items():
            if col_name not in df.columns:
                raise ValueError(f"DataFrame must contain column '{col_name}' for property '{prop_name}'.")
            mean_val = self.df[col_name].mean()
            std_val = self.df[col_name].std()
            self.property_stats[prop_name] = {'mean': mean_val, 'std': std_val}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        numericalized_smiles = self.df['numericalize_token'].iloc[index]
        smiles = self.df[self.smiles_column].iloc[index]

        # normalize properties into tensor
        normalized_properties = []
        for prop_name, col_name in self.properties_info.items():
            value = self.df[col_name].iloc[index]
            mean_val = self.property_stats[prop_name]['mean']
            std_val = self.property_stats[prop_name]['std']

            normalized_value = (value - mean_val) / std_val
            normalized_properties.append(normalized_value)

        properties_tensor = torch.tensor(normalized_properties, dtype=torch.float32)

        # input, target preparation
        input_tokens = numericalized_smiles[:-1]
        target_tokens = numericalized_smiles[1:]

        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        target_tensor = torch.tensor(target_tokens, dtype=torch.long)
        sequence_length = len(input_tokens)

        return input_tensor, target_tensor, sequence_length, smiles, properties_tensor


def my_collate_fn(batch, pad_idx=0):
    inputs, targets, lengths, smiles, properties = zip(*batch)

    padded_inputs = rnn_utils.pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_targets = rnn_utils.pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    properties_tensor = torch.stack(properties)

    return padded_inputs, padded_targets, torch.tensor(lengths, dtype=torch.long), smiles, properties_tensor


def process_to_canonical_or_deep_smiles(df, smiles_col, deep_smiles=False):
    """
    Args:
        df (pd.DataFrame): DataFrame.
        smiles_col (str): SMILES column name.
        deep_smiles (bool): default = False.

    Returns:
        pd.DataFrame: Mol object, Canonical SMILES, DeepSMILES in DataFrame.
    """

    mol_list = []
    for smiles in tqdm(df[smiles_col], desc="Converting SMILES to Mol"):
        mol_list.append(Chem.MolFromSmiles(smiles))

    df_processed = df.copy()
    df_processed['mol'] = mol_list

    indices_to_drop = df_processed[df_processed['mol'].isnull()].index.tolist()
    if indices_to_drop:
        print(f"Dropping {len(indices_to_drop)} rows due to failed Mol conversion.")
        df_processed = df_processed.drop(indices_to_drop)

    df_processed = df_processed.reset_index(drop=True)  # reset index
    print("\n Starting Canonical SMILES conversion (Mol -> SMILES)...")

    canonical_smiles_list = []
    for mol in tqdm(df_processed['mol'], desc="Converting Mol to Canonical SMILES"):
        canonical_smiles_list.append(Chem.MolToSmiles(mol))

    df_processed['canonical_smiles'] = canonical_smiles_list

    if deep_smiles:
        print("\n Starting DeepSMILES conversion (Canonical SMILES -> DeepSMILES)...")
        try:
            converter = deepsmiles.Converter(rings=True, branches=True)

            deep_smiles_list = []
            for canonical_smiles in tqdm(df_processed['canonical_smiles'], desc="Encoding DeepSMILES"):
                try:
                    encoded = converter.encode(canonical_smiles)
                    deep_smiles_list.append(encoded)
                except Exception as e:
                    print(f"DeepSMILES encoding failed for {canonical_smiles}: {e}. Appending None.")
                    deep_smiles_list.append(None)

            df_processed['deep_smiles'] = deep_smiles_list
            print("DeepSMILES conversion complete.")

        except Exception as e:
            print(f"Warning: DeepSMILES conversion failed completely. Check deepsmiles installation. Error: {e}")

    print("\nProcessing complete. Returning DataFrame.")
    return df_processed


#### KL loss Annealing
class AnnealingSchedules:

    def __init__(self, method=None,  # ('cycle_linear' or 'cycle_sigmoid' or 'cycle_cosine' )
                 update_unit=None,  # ('step' or 'epoch')
                 num_training_steps=None,  # int
                 num_training_steps_per_epoch=None,  # int
                 **kwargs):
        self.method = method
        assert update_unit in ['step', 'epoch']
        self.update_unit = update_unit
        self.num_training_steps = num_training_steps
        self.num_training_steps_per_epoch = num_training_steps_per_epoch
        self.kwargs = kwargs

        self._calculate_annealing_schedule(**self.kwargs)

    def _get_annealing_value(self, w: float) -> float:
        if self.method == 'cycle_linear':
            return w
        elif self.method == 'cycle_sigmoid':
            return 1.0 / (1.0 + np.exp(- (w * 12. - 6.)))
        elif self.method == 'cycle_cosine':
            return .5 - .5 * np.cos(w * np.pi)

    def _calculate_annealing_schedule(self, start_weight: float = 0.0,
                                      stop_weight: float = 1.0,
                                      n_cycle: int = 1,
                                      ratio: float = 1.0, ):
        self.L = np.ones(self.num_training_steps) * stop_weight
        period = self.num_training_steps / n_cycle
        weight_step = (stop_weight - start_weight) / (period * ratio)  # linear schedule

        for c in range(n_cycle):
            w, i = start_weight, 0
            while w <= stop_weight and (int(i + c * period) < self.num_training_steps):
                self.L[int(i + c * period)] = self._get_annealing_value(w)
                w += weight_step
                i += 1

        if self.update_unit == 'epoch':
            for global_step, w in enumerate(self.L):
                quotient = global_step // self.num_training_steps_per_epoch
                self.L[global_step] = self.L[quotient * self.num_training_steps_per_epoch]

    def __call__(self, global_step: int):
        assert global_step < self.num_training_steps
        return self.L[global_step]

    def get_annealing_schedule(self):
        return self.L


