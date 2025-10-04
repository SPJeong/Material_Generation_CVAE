##### my_tokenizer.py

import pandas as pd
import atomInSmiles
import re
import codecs
import os
import json
import transformers
import SmilesPE
from SmilesPE.tokenizer import *
from typing import List, Dict, Union

tokenizer_type_list = ['gpt', 'smilesPE', 'atomwise', 'atomInSmiles']


class SmilesTokenizer:
    def __init__(self, df: pd.DataFrame, smiles_column: str = None, tokenizer_type: str = None):
        self.special_tokens = ["<<PAD>>", "<<SOS>>", "<<EOS>>", "<<UNK>>", ]  # if atomInSmiles, add "*"
        self.tokenizer_type = tokenizer_type
        self.tokenizer_instance = self._get_tokenizer_instance(tokenizer_type)
        self.vocab = self._tokenize_set(df, smiles_column)
        self.char2int = self._numericalize(self.vocab)
        self.int2char = {idx: char for char, idx in self.char2int.items()}

    def _get_tokenizer_instance(self, tokenizer_type: str):
        if tokenizer_type == 'gpt':
            return transformers.GPT2Tokenizer.from_pretrained('gpt2')
        elif tokenizer_type == 'smilesPE':
            codes_file_path = './SPE_ChEMBL.txt'  # pretrained data load
            if not os.path.exists(codes_file_path):
                raise FileNotFoundError(f"Error: The codes file '{codes_file_path}' was not found.")
            spe_vocab = codecs.open(codes_file_path)
            return SPE_Tokenizer(spe_vocab)
        elif tokenizer_type == 'atomwise':
            return re.compile(
                r"(\[[^\]]+\]|Al|Ag|Au|Be|Ba|br|Br|Ca|Cd|Ce|Co|Cr|Cs|Cu|cl|Cl|fe|Fe|Hg|Li|Mg|Mn|Mo|Na|Ni|Os|Pd|Pb|Pt|Se|Si|Sn|Ti|Zn|Zr|C|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
        elif tokenizer_type == 'atomInSmiles':
            return atomInSmiles
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    def _tokenize_set(self, df: pd.DataFrame, smiles_column: str) -> List[str]:
        all_smiles_tokens = set()
        for smiles in df[smiles_column]:
            try:
                if self.tokenizer_type == 'atomInSmiles':
                    token_string = self.tokenizer_instance.encode(smiles)
                    if token_string:
                        tokens = token_string.split()
                elif self.tokenizer_type == 'gpt':
                    tokens = self.tokenizer_instance.tokenize(smiles)
                elif self.tokenizer_type == 'smilesPE':
                    token_string = self.tokenizer_instance.tokenize(smiles)
                    tokens = token_string.split()
                elif self.tokenizer_type == 'atomwise':
                    tokens = self.tokenizer_instance.findall(smiles)
                all_smiles_tokens.update(tokens)
            except Exception as e:
                print(f"Error tokenizing smiles '{smiles}' with {self.tokenizer_type} tokenizer: {e}")
                continue

        return self.special_tokens + list(all_smiles_tokens)

    def _numericalize(self, vocab: List[str]) -> Dict[str, int]:
        return {token: idx for idx, token in enumerate(vocab)}

    def tokenize(self, smiles_string: str) -> List[str]:
        try:
            if self.tokenizer_type == 'atomInSmiles':
                token_string = self.tokenizer_instance.encode(smiles_string)
                if not token_string:
                    tokens = [self.special_tokens[-1]]
                else:
                    tokens = token_string.split()
            elif self.tokenizer_type == 'gpt':
                tokens = self.tokenizer_instance.tokenize(smiles_string)
            elif self.tokenizer_type == 'smilesPE':
                token_string = self.tokenizer_instance.tokenize(smiles_string)
                tokens = token_string.split()
            elif self.tokenizer_type == 'atomwise':
                tokens = self.tokenizer_instance.findall(smiles_string)

            return [self.special_tokens[1]] + tokens + [self.special_tokens[2]]
        except Exception as e:
            print(f"Error tokenizing smiles: {e}")
            return [self.special_tokens[-1]]

    def encode(self, smiles_string: str) -> List[int]:
        tokens = self.tokenize(smiles_string)
        return [self.char2int.get(token, self.char2int["<<UNK>>"]) for token in tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if self.tokenizer_type == 'atomInSmiles':  # designed only for atomInSmiles
            tokens: str = ''
            for token_id in token_ids:
                token = self.int2char.get(int(token_id))
                temp = ' ' + str(token)
                tokens += temp
                filtered_list = tokens[9:-8]  # remove <<SOS>>, <<EOS>>
            print("filtered_list:" + filtered_list)
            return atomInSmiles.decode(
                str(filtered_list))  # need to mol conversion and check the smiles, failed to smile conversion sometimes


        else:
            tokens = []
            for token_id in token_ids:
                token = self.int2char.get(token_id)
                if token and skip_special_tokens and token in self.special_tokens:
                    continue
                if token:
                    tokens.append(token)
            return "".join(tokens)
