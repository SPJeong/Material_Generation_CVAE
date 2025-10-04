##### chemical_feature_extraction.py
import os
import numpy as np
import pandas as pd
import joblib
import rdkit
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.Chem.GraphDescriptors import (BalabanJ, BertzCT, Chi0, Chi0n, Chi0v, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3n,
                                         Chi3v, Chi4n, Chi4v, HallKierAlpha, Ipc, Kappa1, Kappa2, Kappa3)
from rdkit.Chem.EState.EState_VSA import (EState_VSA1, EState_VSA10, EState_VSA11, EState_VSA2, EState_VSA3,
                                          EState_VSA4, EState_VSA5, EState_VSA6, EState_VSA7, EState_VSA8, EState_VSA9,
                                          VSA_EState1, VSA_EState10, VSA_EState2, VSA_EState3, VSA_EState4, VSA_EState5,
                                          VSA_EState6, VSA_EState7, VSA_EState8, VSA_EState9, )
from rdkit.Chem.Descriptors import (ExactMolWt, MolWt, HeavyAtomMolWt, NumRadicalElectrons, NumValenceElectrons)
from rdkit.Chem.EState.EState import (MaxAbsEStateIndex, MaxEStateIndex, MinAbsEStateIndex, MinEStateIndex, )
from rdkit.Chem.Lipinski import (FractionCSP3, HeavyAtomCount, NHOHCount, NOCount, NumAliphaticCarbocycles,
                                 NumAliphaticHeterocycles, NumAliphaticRings, NumAromaticCarbocycles,
                                 NumAromaticHeterocycles, NumAromaticRings, NumHAcceptors, NumHDonors, NumHeteroatoms,
                                 RingCount, NumRotatableBonds, NumSaturatedCarbocycles, NumSaturatedHeterocycles,
                                 NumSaturatedRings, )
from rdkit.Chem.Crippen import (MolLogP, MolMR, )
from rdkit.Chem.MolSurf import (LabuteASA, PEOE_VSA1, PEOE_VSA10, PEOE_VSA11, PEOE_VSA12, PEOE_VSA13, PEOE_VSA14,
                                PEOE_VSA2, PEOE_VSA3, PEOE_VSA4, PEOE_VSA5, PEOE_VSA6, PEOE_VSA7, PEOE_VSA8, PEOE_VSA9,
                                SMR_VSA1, SMR_VSA10, SMR_VSA2, SMR_VSA3, SMR_VSA4, SMR_VSA5, SMR_VSA6, SMR_VSA7,
                                SMR_VSA8, SMR_VSA9, SlogP_VSA1, SlogP_VSA10, SlogP_VSA11, SlogP_VSA12, SlogP_VSA2,
                                SlogP_VSA3, SlogP_VSA4, SlogP_VSA5, SlogP_VSA6, SlogP_VSA7, SlogP_VSA8, SlogP_VSA9,
                                TPSA, )
from rdkit.Chem.Fragments import (fr_Al_COO, fr_Al_OH, fr_Al_OH_noTert, fr_ArN, fr_Ar_COO, fr_Ar_N, fr_Ar_NH, fr_Ar_OH,
                                  fr_COO, fr_COO2, fr_C_O, fr_C_O_noCOO, fr_C_S, fr_HOCCN, fr_Imine, fr_NH0, fr_NH1,
                                  fr_NH2, fr_N_O, fr_Ndealkylation1, fr_Ndealkylation2, fr_Nhpyrrole, fr_SH,
                                  fr_aldehyde, fr_alkyl_carbamate, fr_alkyl_halide, fr_allylic_oxid, fr_amide,
                                  fr_amidine, fr_aniline, fr_aryl_methyl, fr_azide, fr_azo, fr_barbitur, fr_benzene,
                                  fr_benzodiazepine, fr_bicyclic, fr_diazo, fr_dihydropyridine, fr_epoxide, fr_ester,
                                  fr_ether, fr_furan, fr_guanido, fr_halogen, fr_hdrzine, fr_hdrzone, fr_imidazole,
                                  fr_imide, fr_isocyan, fr_isothiocyan, fr_ketone, fr_ketone_Topliss, fr_lactam,
                                  fr_lactone, fr_methoxy, fr_morpholine, fr_nitrile, fr_nitro, fr_nitro_arom,
                                  fr_nitro_arom_nonortho, fr_nitroso, fr_oxazole, fr_oxime, fr_para_hydroxylation,
                                  fr_phenol, fr_phenol_noOrthoHbond, fr_phos_acid, fr_phos_ester, fr_piperdine,
                                  fr_piperzine, fr_priamide, fr_prisulfonamd, fr_pyridine, fr_quatN, fr_sulfide,
                                  fr_sulfonamd, fr_sulfone, fr_term_acetylene, fr_tetrazole, fr_thiazole, fr_thiocyan,
                                  fr_thiophene, fr_unbrch_alkane, fr_urea)
from rdkit.Chem.rdMolDescriptors import CalcNumBridgeheadAtoms
from sklearn import preprocessing

import CONFIG  # custom.py
import data_extraction  # custom.py


# Define calculate descriptors # 193 features extraction
def descriptors_calculation(mol):
    if mol is None:
        return None
    try:
        descriptors = [
            BalabanJ(mol), BertzCT(mol), Chi0(mol), Chi0n(mol), Chi0v(mol), Chi1(mol), Chi1n(mol), Chi1v(mol),
            Chi2n(mol), Chi2v(mol), Chi3n(mol), Chi3v(mol), Chi4n(mol), Chi4v(mol), HallKierAlpha(mol), Ipc(mol),
            Kappa1(mol), Kappa2(mol), Kappa3(mol),

            EState_VSA1(mol), EState_VSA10(mol), EState_VSA11(mol), EState_VSA2(mol), EState_VSA3(mol),
            EState_VSA4(mol), EState_VSA5(mol), EState_VSA6(mol), EState_VSA7(mol), EState_VSA8(mol), EState_VSA9(mol),
            VSA_EState1(mol), VSA_EState10(mol), VSA_EState2(mol), VSA_EState3(mol), VSA_EState4(mol), VSA_EState5(mol),
            VSA_EState6(mol), VSA_EState7(mol), VSA_EState8(mol), VSA_EState9(mol),

            ExactMolWt(mol), MolWt(mol), HeavyAtomMolWt(mol), NumRadicalElectrons(mol), NumValenceElectrons(mol),

            MaxAbsEStateIndex(mol), MaxEStateIndex(mol), MinAbsEStateIndex(mol), MinEStateIndex(mol),

            FractionCSP3(mol), HeavyAtomCount(mol), NHOHCount(mol), NOCount(mol), NumAliphaticCarbocycles(mol),
            NumAliphaticHeterocycles(mol), NumAliphaticRings(mol), NumAromaticCarbocycles(mol),
            NumAromaticHeterocycles(mol), NumAromaticRings(mol), NumHAcceptors(mol), NumHDonors(mol),
            NumHeteroatoms(mol), RingCount(mol), NumRotatableBonds(mol), NumSaturatedCarbocycles(mol),
            NumSaturatedHeterocycles(mol), NumSaturatedRings(mol),

            MolLogP(mol), MolMR(mol),

            LabuteASA(mol), PEOE_VSA1(mol), PEOE_VSA10(mol), PEOE_VSA11(mol), PEOE_VSA12(mol), PEOE_VSA13(mol),
            PEOE_VSA14(mol), PEOE_VSA2(mol), PEOE_VSA3(mol), PEOE_VSA4(mol), PEOE_VSA5(mol), PEOE_VSA6(mol),
            PEOE_VSA7(mol), PEOE_VSA8(mol), PEOE_VSA9(mol), SMR_VSA1(mol), SMR_VSA10(mol), SMR_VSA2(mol), SMR_VSA3(mol),
            SMR_VSA4(mol), SMR_VSA5(mol), SMR_VSA6(mol), SMR_VSA7(mol), SMR_VSA8(mol), SMR_VSA9(mol), SlogP_VSA1(mol),
            SlogP_VSA10(mol), SlogP_VSA11(mol), SlogP_VSA12(mol), SlogP_VSA2(mol), SlogP_VSA3(mol), SlogP_VSA4(mol),
            SlogP_VSA5(mol), SlogP_VSA6(mol), SlogP_VSA7(mol), SlogP_VSA8(mol), SlogP_VSA9(mol), TPSA(mol),

            fr_Al_COO(mol), fr_Al_OH(mol), fr_Al_OH_noTert(mol), fr_ArN(mol), fr_Ar_COO(mol), fr_Ar_N(mol),
            fr_Ar_NH(mol), fr_Ar_OH(mol), fr_COO(mol), fr_COO2(mol), fr_C_O(mol), fr_C_O_noCOO(mol), fr_C_S(mol),
            fr_HOCCN(mol), fr_Imine(mol), fr_NH0(mol), fr_NH1(mol), fr_NH2(mol), fr_N_O(mol), fr_Ndealkylation1(mol),
            fr_Ndealkylation2(mol), fr_Nhpyrrole(mol), fr_SH(mol), fr_aldehyde(mol), fr_alkyl_carbamate(mol),
            fr_alkyl_halide(mol), fr_allylic_oxid(mol), fr_amide(mol), fr_amidine(mol), fr_aniline(mol),
            fr_aryl_methyl(mol), fr_azide(mol), fr_azo(mol), fr_barbitur(mol), fr_benzene(mol), fr_benzodiazepine(mol),
            fr_bicyclic(mol), fr_diazo(mol), fr_dihydropyridine(mol), fr_epoxide(mol), fr_ester(mol), fr_ether(mol),
            fr_furan(mol), fr_guanido(mol), fr_halogen(mol), fr_hdrzine(mol), fr_hdrzone(mol), fr_imidazole(mol),
            fr_imide(mol), fr_isocyan(mol), fr_isothiocyan(mol), fr_ketone(mol), fr_ketone_Topliss(mol), fr_lactam(mol),
            fr_lactone(mol), fr_methoxy(mol), fr_morpholine(mol), fr_nitrile(mol), fr_nitro(mol), fr_nitro_arom(mol),
            fr_nitro_arom_nonortho(mol), fr_nitroso(mol), fr_oxazole(mol), fr_oxime(mol), fr_para_hydroxylation(mol),
            fr_phenol(mol), fr_phenol_noOrthoHbond(mol), fr_phos_acid(mol), fr_phos_ester(mol), fr_piperdine(mol),
            fr_piperzine(mol), fr_priamide(mol), fr_prisulfonamd(mol), fr_pyridine(mol), fr_quatN(mol), fr_sulfide(mol),
            fr_sulfonamd(mol), fr_sulfone(mol), fr_term_acetylene(mol), fr_tetrazole(mol), fr_thiazole(mol),
            fr_thiocyan(mol), fr_thiophene(mol), fr_unbrch_alkane(mol), fr_urea(mol),

            CalcNumBridgeheadAtoms(mol),
        ]
        return descriptors
    except Exception as e:
        print(f"Error calculating descriptors for a molecule: {e}")
        return None


# Define chemical feature extraaction from smiles
def run_feature_extraction(filtered_num=CONFIG.filtered_num,
                           random_pick_num=CONFIG.random_pick_num,
                           data_extraction_folder=CONFIG.data_extraction_folder,
                           ecfp=False,
                           descriptors=False,
                           scale_descriptors=False,
                           ecfp_radius=None,
                           ecfp_nbits=None,
                           chemical_feature_extraction_folder=CONFIG.chemical_feature_extraction_folder,
                           inference_mode: bool = False,
                           new_smiles_list: list = None):
    """
    Performs molecular feature extraction based on specified options for training or inference.

    Args:
    filtered_num (int): SMILES length filtering criterion.
    random_pick_num (int): Number of molecules to sample.
    data_extraction_folder (str): Folder to save/load data.
    ecfp (bool): Whether to generate ECFP features.
    descriptors (bool): Whether to generate RDKit descriptors.
    scale_descriptors (bool): Whether to Min-Max scale descriptors.
    ecfp_radius (int): Radius for ECFP calculation.
    ecfp_nbits (int): Number of bits for ECFP bit vector.
    chemical_feature_extraction_folder (str): Folder to save/load data after run_feature_extraction function
    inference_mode (bool): If True, runs in inference mode on new_smiles_list.
    new_smiles_list (list): A list of SMILES strings for inference. e.g. ['[*]Oc1cc(OC(=O)CNCC)sc1[*]', '[*]CCCCCCCNCc1nnc(C[*])o1', ...]
    """
    # 1. Load the dataframe
    if inference_mode:
        print("\n Starting chemical feature extraction in INFERENCE MODE...")
        if not new_smiles_list:
            raise ValueError("new_smiles_list must be provided in inference mode.")
        small_df = pd.DataFrame({'smiles': new_smiles_list})
    else:
        print("\n Starting chemical feature extraction in TRAINING MODE...")
        raw_df, small_df = data_extraction.load_and_process_data(filtered_num=filtered_num,
                                                                 random_pick_num=random_pick_num,
                                                                 data_extraction_folder=data_extraction_folder)

    # 2. Mol conversion and filtering invalid SMILES
    small_df['mol'] = [Chem.MolFromSmiles(smiles) for smiles in
                       tqdm(small_df['smiles'], desc='Converting SMILES to Mol')]
    initial_count = len(small_df)

    invalid_smiles_mask = small_df['mol'].isnull()
    invalid_smiles_df = small_df[invalid_smiles_mask]

    if not invalid_smiles_df.empty:
        print("\n The following SMILES could not be converted to mol objects:")
        for index, row in invalid_smiles_df.iterrows():
            print(f"Index: {index}, SMILES: '{row['smiles']}'")

    small_df.dropna(subset=['mol'], inplace=True)
    small_df.reset_index(drop=True, inplace=True)

    final_count = len(small_df)
    if initial_count > final_count:
        print(f"\n Removed {initial_count - final_count} invalid SMILES from the dataset.")

    # 3. Initialize a list for final dataframes
    feature_dfs = [small_df.drop('mol', axis=1)]

    # 4. ECFP conversion (Optional)
    if ecfp:
        print("\n Generating ECFP features...")
        ECFP_list = []
        for mol in tqdm(small_df['mol'], desc='Generating ECFPs'):
            fp_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius=ecfp_radius, nBits=ecfp_nbits)
            fp_array = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp_vector, fp_array)
            ECFP_list.append(fp_array)
        ECFP_df = pd.DataFrame(ECFP_list, index=small_df.index)
        feature_dfs.append(ECFP_df)

    # 5. Calculate and process RDKit descriptors (Optional)
    if descriptors:
        print("\n Calculating RDKit descriptors...")
        descriptors_names = 'BalabanJ(mol), BertzCT(mol), Chi0(mol), Chi0n(mol), Chi0v(mol), Chi1(mol), Chi1n(mol), Chi1v(mol), Chi2n(mol), Chi2v(mol), Chi3n(mol), Chi3v(mol), Chi4n(mol), Chi4v(mol), HallKierAlpha(mol), Ipc(mol), Kappa1(mol), Kappa2(mol), Kappa3(mol), EState_VSA1(mol), EState_VSA10(mol), EState_VSA11(mol), EState_VSA2(mol), EState_VSA3(mol), EState_VSA4(mol), EState_VSA5(mol), EState_VSA6(mol), EState_VSA7(mol), EState_VSA8(mol), EState_VSA9(mol), VSA_EState1(mol), VSA_EState10(mol), VSA_EState2(mol), VSA_EState3(mol), VSA_EState4(mol), VSA_EState5(mol), VSA_EState6(mol), VSA_EState7(mol), VSA_EState8(mol), VSA_EState9(mol), ExactMolWt(mol), MolWt(mol), HeavyAtomMolWt(mol), NumRadicalElectrons(mol), NumValenceElectrons(mol), MaxAbsEStateIndex(mol), MaxEStateIndex(mol), MinAbsEStateIndex(mol), MinEStateIndex(mol), FractionCSP3(mol), HeavyAtomCount(mol), NHOHCount(mol), NOCount(mol), NumAliphaticCarbocycles(mol), NumAliphaticHeterocycles(mol), NumAliphaticRings(mol), NumAromaticCarbocycles(mol), NumAromaticHeterocycles(mol), NumAromaticRings(mol), NumHAcceptors(mol), NumHDonors(mol), NumHeteroatoms(mol), RingCount(mol), NumRotatableBonds(mol), NumSaturatedCarbocycles(mol), NumSaturatedHeterocycles(mol), NumSaturatedRings(mol), MolLogP(mol), MolMR(mol), LabuteASA(mol), PEOE_VSA1(mol), PEOE_VSA10(mol), PEOE_VSA11(mol), PEOE_VSA12(mol), PEOE_VSA13(mol), PEOE_VSA14(mol), PEOE_VSA2(mol), PEOE_VSA3(mol), PEOE_VSA4(mol), PEOE_VSA5(mol), PEOE_VSA6(mol), PEOE_VSA7(mol), PEOE_VSA8(mol), PEOE_VSA9(mol), SMR_VSA1(mol), SMR_VSA10(mol), SMR_VSA2(mol), SMR_VSA3(mol), SMR_VSA4(mol), SMR_VSA5(mol), SMR_VSA6(mol), SMR_VSA7(mol), SMR_VSA8(mol), SMR_VSA9(mol), SlogP_VSA1(mol), SlogP_VSA10(mol), SlogP_VSA11(mol), SlogP_VSA12(mol), SlogP_VSA2(mol), SlogP_VSA3(mol), SlogP_VSA4(mol), SlogP_VSA5(mol), SlogP_VSA6(mol), SlogP_VSA7(mol), SlogP_VSA8(mol), SlogP_VSA9(mol), TPSA(mol), fr_Al_COO(mol), fr_Al_OH(mol), fr_Al_OH_noTert(mol), fr_ArN(mol), fr_Ar_COO(mol), fr_Ar_N(mol), fr_Ar_NH(mol), fr_Ar_OH(mol), fr_COO(mol), fr_COO2(mol), fr_C_O(mol), fr_C_O_noCOO(mol), fr_C_S(mol), fr_HOCCN(mol), fr_Imine(mol), fr_NH0(mol), fr_NH1(mol), fr_NH2(mol), fr_N_O(mol), fr_Ndealkylation1(mol), fr_Ndealkylation2(mol), fr_Nhpyrrole(mol), fr_SH(mol), fr_aldehyde(mol), fr_alkyl_carbamate(mol), fr_alkyl_halide(mol), fr_allylic_oxid(mol), fr_amide(mol), fr_amidine(mol), fr_aniline(mol), fr_aryl_methyl(mol), fr_azide(mol), fr_azo(mol), fr_barbitur(mol), fr_benzene(mol), fr_benzodiazepine(mol), fr_bicyclic(mol), fr_diazo(mol), fr_dihydropyridine(mol), fr_epoxide(mol), fr_ester(mol), fr_ether(mol), fr_furan(mol), fr_guanido(mol), fr_halogen(mol), fr_hdrzine(mol), fr_hdrzone(mol), fr_imidazole(mol), fr_imide(mol), fr_isocyan(mol), fr_isothiocyan(mol), fr_ketone(mol), fr_ketone_Topliss(mol), fr_lactam(mol), fr_lactone(mol), fr_methoxy(mol), fr_morpholine(mol), fr_nitrile(mol), fr_nitro(mol), fr_nitro_arom(mol), fr_nitro_arom_nonortho(mol), fr_nitroso(mol), fr_oxazole(mol), fr_oxime(mol), fr_para_hydroxylation(mol), fr_phenol(mol), fr_phenol_noOrthoHbond(mol), fr_phos_acid(mol), fr_phos_ester(mol), fr_piperdine(mol), fr_piperzine(mol), fr_priamide(mol), fr_prisulfonamd(mol), fr_pyridine(mol), fr_quatN(mol), fr_sulfide(mol), fr_sulfonamd(mol), fr_sulfone(mol), fr_term_acetylene(mol), fr_tetrazole(mol), fr_thiazole(mol), fr_thiocyan(mol), fr_thiophene(mol), fr_unbrch_alkane(mol), fr_urea(mol), CalcNumBridgeheadAtoms(mol)'
        descriptors_name_list = [name.strip().replace('(mol)', '') for name in descriptors_names.split(',')]
        descriptor_lists = [np.array(descriptors_calculation(mol)).astype(float) for mol in
                            tqdm(small_df['mol'], desc='Calculating Descriptors')]
        descriptor_df = pd.DataFrame(descriptor_lists, columns=descriptors_name_list, index=small_df.index)

        # Handle Inf/NaN values
        descriptor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        descriptor_df.fillna(0, inplace=True)

        # 6. Scale descriptors (Optional)
        if scale_descriptors:
            print("\n Scaling descriptors with MinMaxScaler...")
            scaler_file_path = os.path.join(chemical_feature_extraction_folder,
                                            f'filtered_polyOne_{filtered_num}_num_{random_pick_num}_min_max_scaler.pkl')

            if inference_mode:
                # Load the pre-trained scaler in inference mode
                if not os.path.exists(scaler_file_path):
                    raise FileNotFoundError(
                        f"Scaler file not found at: {scaler_file_path}. Cannot perform scaling in inference mode without a trained scaler.")
                min_max_scaler = joblib.load(scaler_file_path)
                scaled_descriptor_df = pd.DataFrame(min_max_scaler.transform(descriptor_df),
                                                    columns=descriptor_df.columns, index=descriptor_df.index)
            else:
                # Train and save the scaler in training mode
                min_max_scaler = preprocessing.MinMaxScaler()
                scaled_descriptor_df = pd.DataFrame(min_max_scaler.fit_transform(descriptor_df),
                                                    columns=descriptor_df.columns, index=descriptor_df.index)
                os.makedirs(chemical_feature_extraction_folder, exist_ok=True)
                joblib.dump(min_max_scaler, scaler_file_path)
                print(f"MinMaxScaler object is saved at '{scaler_file_path}'")

            feature_dfs.append(scaled_descriptor_df)
        else:
            feature_dfs.append(descriptor_df)

    # 7. Combine all features and save/return
    combined_TOTAL_df = pd.concat(feature_dfs, axis=1)

    if not inference_mode:
        combined_file_folder = chemical_feature_extraction_folder
        combined_file_name = f'chemical_feature_extraction_len_{filtered_num}_num_{random_pick_num}_scaled_{scale_descriptors}_ECFP_{ecfp}_desc_{descriptors}.csv'
        os.makedirs(combined_file_folder, exist_ok=True)
        combined_file_path = os.path.join(combined_file_folder, combined_file_name)
        combined_TOTAL_df.to_csv(combined_file_path, index=False)
        print(f"\n All features combined and saved to '{combined_file_path}'")
    else:
        print("\n Features for new SMILES generated.")

    return combined_TOTAL_df


if __name__ == "__main__":
    # choose one between two belows.
    # Run in training mode
    train_df = run_feature_extraction(filtered_num=30,
                                      random_pick_num=100,
                                      data_extraction_folder=fr"C:\Users\wisdo\polyOne_Data_Set\data_extraction",
                                      ecfp=True,
                                      descriptors=True,
                                      scale_descriptors=True,
                                      ecfp_radius=2,
                                      ecfp_nbits=1024,
                                      chemical_feature_extraction_folder=fr"C:\Users\wisdo\polyOne_Data_Set\chemical_feature_extraction",
                                      inference_mode=False,
                                      new_smiles_list=None)

    # Run in inference mode with user-provided SMILES list
    # Inference mode should be done after generating file for min_max scaler pickle!
    inference_df = run_feature_extraction(filtered_num=30,
                                          random_pick_num=100,
                                          data_extraction_folder=fr"C:\Users\wisdo\polyOne_Data_Set\data_extraction",
                                          ecfp=True,
                                          descriptors=True,
                                          scale_descriptors=True,
                                          ecfp_radius=2,
                                          ecfp_nbits=1024,
                                          chemical_feature_extraction_folder=fr"C:\Users\wisdo\polyOne_Data_Set\chemical_feature_extraction",
                                          inference_mode=True,
                                          new_smiles_list=['[*]Oc1cc(OC(=O)CNCC)sc1[*]', '[*]CCCCCCCNCc1nnc(C[*])o1',
                                                           'invalid_smiles'])