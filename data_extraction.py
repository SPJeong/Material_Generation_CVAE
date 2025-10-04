##### data_extraction.py
import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import CONFIG # custom.py

# Data extraction based on PSMILES length
def load_and_process_data(filtered_num= None,
                          random_pick_num= None,
                          data_extraction_folder= None):
    '''
    preprocessing for Parquet data.
    Args:
        filtered_num (int): filtering less than filtered_num of each SMILES length
        random_pick_num (int): total sampling number
        data_extraction_folder (str): folder for saving data extraction file.

    Returns:
        tuple: (raw_df, small_df).
               - raw_df: before sampling
               - small_df: after sampling
    '''

    # raw_df load or start pre-processing
    total_file_name = f'filtered_polyOne_{filtered_num}_total.csv'
    total_file_path = os.path.join(data_extraction_folder, total_file_name)
    filtered_file_name = f'filtered_polyOne_{filtered_num}_num_{random_pick_num}.csv'
    filtered_file_path = os.path.join(data_extraction_folder, filtered_file_name)

    if os.path.exists(total_file_path):
        print('Loading previous pre-processed data file: ' + str(total_file_name))
        raw_df = pd.read_csv(total_file_path)
    else:
        print('pre-processed data file is NOT existed. Starting pre-processing')
        if not os.path.exists(data_extraction_folder):
            os.makedirs(data_extraction_folder)

        alphabet_letters_part1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        alphabet_letters_part2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        df_list_all = []
        for idx, i in tqdm(enumerate(alphabet_letters_part1), total= len(alphabet_letters_part1), desc= "Processing Part 1"):
            for j in tqdm(alphabet_letters_part2, leave= False, desc= f"Processing Part 2 for '{i}'"):
                if i == 'h' and j == 'y':
                    break

                file_path = fr"C:\Users\wisdo\polyOne_Data_Set\polyOne_{i}{j}.parquet"
                if os.path.exists(file_path):
                    df = pq.read_table(file_path).to_pandas()
                    df.drop(['epse_6.0', 'epsc', 'epse_3.0', 'epse_1.78', 'epse_15.0', 'epse_4.0', 'epse_5.0', 'epse_2.0', 'epse_9.0',
                             'epse_7.0'], axis= 1, inplace= True)
                    df['len'] = df['smiles'].apply(len)
                    df = df[df['len'] <= filtered_num]
                    df_list_all.append(df)

        raw_df = pd.concat(df_list_all, ignore_index= True)
        raw_df.to_csv(total_file_path, index= False)
        print(f"New pre-processed file (raw_df) is saved at '{total_file_path}'")

    # sampling

    if os.path.exists(filtered_file_path):
        print('Loading previous pre-processed data file: ' + str(filtered_file_name))
        small_df = pd.read_csv(filtered_file_path)
    else:
        if random_pick_num > len(raw_df):
            print(f"The requested number of samples ({random_pick_num}) is greater than the total number of data points ({len(raw_df)}). The entire dataset will be used.")
            small_df = raw_df.copy()
        else:
            small_df = raw_df.sample(n= random_pick_num, ignore_index= True, axis= 0)
            small_df.to_csv(filtered_file_path, index= False)
            print(f"New pre-processed file (filtered_df) is saved at '{filtered_file_path}'")

    print(f"Data sampling is complete. final data len: {len(small_df):,}")
    return raw_df, small_df

'''
material_property_name = {
    'smiles': 'chemical structure', 
    'Cp': 'heat capacity [Unit: Jg−1K−1]', 
    'Tg': 'glass transition temp [Unit: K]', 
    'Tm': 'melting temp [Unit: K]', 
    'Td': 'degradation temp [Unit: K]', 
    'LOI': 'Limiting oxygen index [Unit: %]', 
    'YM': 'Young's modulus [Unit: MPa]', 
    'TSy': 'tensile strength at yield [Unit: MPa]', 
    'TSb': 'tensile strength at break [Unit: MPa]', 
    'epsb': 'elongation at break [Unit: %]', 
    'CED': 'Cohesive energy density [Unit: cal cm−3]', 
    'Egc': 'bandgap (chain) [Unit: eV]', 
    'Egb': 'bandgap (bulk)' [Unit: eV], 
    'Eib': 'electron injection barrier [Unit: eV]', 
    'Ei': 'ionization energy [Unit: eV]', 
    'Eea': 'electron affinity [Unit: eV]', 
    'nc': 'refractive index (DFT) [Unit: dimensionless]', 
    'ne': 'refractive index (exp) [Unit: dimensionless]', 
    'permH2': 'H2 permeability [Unit: barrer]', 
    'permHe': 'He permeability [Unit: barrer]', 
    'permCH4': 'CH4 permeability [Unit: barrer]', 
    'permCO2': 'CO2 permeability [Unit: barrer]', 
    'permN2': 'N2 permeability [Unit: barrer]', 
    'permO2': 'O2 permeability [Unit: barrer]', 
    'Eat': 'atomization energy [Unit: barrer]', 
    'rho': 'density [Unit: g cm−3]', 
    'Xc': 'crystallization tendency (DFT) [Unit: %]', 
    'Xe': 'crystallization tendency (exp) [Unit: %]'
}
'''

if __name__ == '__main__':
    filtered_num = CONFIG.filtered_num
    random_pick_num = CONFIG.random_pick_num
    data_extraction_folder = CONFIG.data_extraction_folder

    raw_df, small_df = load_and_process_data(filtered_num, random_pick_num, data_extraction_folder)
    print('*'*30 + 'raw_df.head()' + '*'*30 )
    print(f'raw_df shape: {raw_df.shape}')
    print(f'raw_df columns: {raw_df.columns}')
    print(raw_df.head())
    print('*'*30 + 'small_df.head()' + '*'*30 )
    print(f'small_df shape: {raw_df.shape}')
    print(small_df.head())
