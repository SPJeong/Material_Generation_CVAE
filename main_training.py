##### main_training.py (conditional VAE)

# import
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

import CONFIG  # custom.py
import chemical_feature_extraction  # custom.py
import model  # custom.py
import my_utils # custom.py
import my_tokenizer # custom.py
import model_trainer # custom.py


# parameter setting
filtered_num = CONFIG.filtered_num
random_pick_num = CONFIG.random_pick_num
data_extraction_folder = CONFIG.data_extraction_folder
chemical_feature_extraction_folder = CONFIG.chemical_feature_extraction_folder
plot_save_folder = CONFIG.plot_save_folder
model_save_folder = CONFIG.model_save_folder
os.makedirs(chemical_feature_extraction_folder, exist_ok=True)
os.makedirs(plot_save_folder, exist_ok=True)
os.makedirs(model_save_folder, exist_ok=True)
model_name = CONFIG.model_name
batch_size = CONFIG.batch_size
learning_rate = CONFIG.learning_rate
epochs = CONFIG.epochs
device = CONFIG.device
ROnPlateauLR_mode = CONFIG.ROnPlateauLR_mode
ROnPlateauLR_factor = CONFIG.ROnPlateauLR_factor
ROnPlateauLR_patience = CONFIG.ROnPlateauLR_patience

embedding_dim = CONFIG.embedding_dim # 300
hidden_dim = CONFIG.hidden_dim # 512
num_layers = CONFIG.num_layers # 2
dropout = CONFIG.dropout # 0.2
latent_dim = CONFIG.latent_dim #256
encoder_property_dim= CONFIG.encoder_property_dim # e.g. 0,  # if 0 -> VAE, e.g. if two properties, then the dim = 2 (CVAE)
decoder_property_dim= CONFIG.decoder_property_dim  # must have at least more than 1. e.g. if two properties, then the dim = 2

tokenizer_type = CONFIG.tokenizer_type  # tokenizer_type_list = ['gpt', 'smilesPE', 'atomwise', 'atomInSmiles']
smiles_column = CONFIG.smiles_column # 'canonical_smiles' or 'deep_smiles' for tokenizing and numericalizing

KL_Annealing_method = CONFIG.KL_Annealing_method  # 'cycle_sigmoid' # ('cycle_linear' or 'cycle_sigmoid' or 'cycle_cosine' )
KL_Annealing_update_unit = CONFIG.KL_Annealing_update_unit  # 'epoch' # ('step' or 'epoch')
KL_Annealing_start_weight = CONFIG.KL_Annealing_start_weight  # 0.0
KL_Annealing_stop_weight = CONFIG.KL_Annealing_stop_weight  # 1.0

max_length= CONFIG.max_length # for max sequence length during testing function e.g. 50
num_return_sequences= CONFIG.num_return_sequences # for number of smiles generation during testing function e.g. 100



'''
# property parameters
thermal_df_list = ['Cp', 'Tg', 'Tm', 'Td', 'LOI']
mechanical_df_list = ['YM', 'TSy', 'TSb', 'epsb', 'CED']
electronic_optical_df_list = ['Egc', 'Egb', 'Eib', 'Ei', 'Eea', 'nc', 'ne']
gas_permeation_df_list = ['permH2', 'permHe', 'permCH4', 'permCO2', 'permN2', 'permO2']
structural_df_list = ['Eat', 'rho', 'Xc', 'Xe']
'''

# load file
file_folder = chemical_feature_extraction_folder
file_name = f'chemical_feature_extraction_len_{filtered_num}_num_{random_pick_num}_scaled_False_ECFP_False_desc_False.csv'
file_raw_path = os.path.join(file_folder, file_name)

if os.path.exists(file_raw_path):
    print(f"Loading existing file from: {file_raw_path}")
    file_raw = pd.read_csv(file_raw_path)

else:
    print(f"File not found. Generating data and saving to: {file_raw_path}")
    file_raw = chemical_feature_extraction.run_feature_extraction(filtered_num= filtered_num,
                                                                  random_pick_num= random_pick_num,
                                                                  data_extraction_folder= data_extraction_folder,
                                                                  ecfp= False,
                                                                  descriptors= False,
                                                                  scale_descriptors= False,
                                                                  ecfp_radius= None,
                                                                  ecfp_nbits= None,
                                                                  chemical_feature_extraction_folder= chemical_feature_extraction_folder,
                                                                  inference_mode= False,
                                                                  new_smiles_list= None)


# processing for canonical_smiles & deep_smiles
X_file_processed = my_utils.process_to_canonical_or_deep_smiles(df= file_raw , smiles_col= 'smiles', deep_smiles= True)
smiles_preprocessing = X_file_processed[['smiles', 'canonical_smiles', 'deep_smiles']].copy()

# tokenizing and numericalizing
tokenizer = my_tokenizer.SmilesTokenizer(df= X_file_processed, smiles_column= smiles_column, tokenizer_type= tokenizer_type)
vocab_size = len(tokenizer.vocab)
sos_token_id = tokenizer.char2int["<<SOS>>"]  # start_token_id
wildcard_token_id = tokenizer.char2int["*"]  # start_token_id_for_generation
eos_token_id = tokenizer.char2int["<<EOS>>"]  # end_token_id
padding_value = tokenizer.char2int["<<PAD>>"]

X_file_processed['numericalize_token'] = smiles_preprocessing[str(smiles_column)].apply(lambda x: tokenizer.encode(x))


# custom dataset and dataloader
dataset = my_utils.myCVAE_Dataset(df= X_file_processed, smiles_column= smiles_column, Tg='Tg', Tm='Tm')
# print(dataset.property_stats) # for getting mean and std of each target value

train_data, val_test_data_temp = train_test_split(dataset, test_size=0.2, random_state=777)
val_data, test_data = train_test_split(val_test_data_temp, test_size=0.5, random_state=777)

train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle= True, collate_fn= my_utils.my_collate_fn)
val_loader = torch.utils.data.DataLoader(val_data, batch_size= batch_size, shuffle= True, collate_fn= my_utils.my_collate_fn)
test_loader = torch.utils.data.DataLoader(test_data, batch_size= batch_size, shuffle= False, collate_fn= my_utils.my_collate_fn)
print(f"train_loader len: {len(train_loader)} | val_loader len: {len(val_loader)} | test_loader len: {len(test_loader)}")

# dataloader test
for batch_idx, (inputs, targets, lengths, smiles, properties_tensor) in enumerate(train_loader):
    print(f"--- train_loader----Batch {batch_idx + 1} ---")
    print(f"Inputs (padded) shape: {inputs.shape}")
    print(f"Targets (padded) shape: {targets.shape}")
    print(f"Lengths: \n{lengths}\n")
    print(f'smiles[0:10]: {smiles[0:10]}')
    print(f'properties_tensor len: {len(properties_tensor)}')
    print(f'properties_tensor shape: {properties_tensor.shape}')
    print(f'properties_tensor: {properties_tensor[0:10]}')
    break

# model define
my_model = model.CVAEModel(vocab_size= vocab_size,
                            embedding_dim= embedding_dim,
                            hidden_dim= hidden_dim,
                            num_layers= num_layers,
                            dropout= dropout,
                            latent_dim= latent_dim,
                            padding_value= padding_value,
                            encoder_property_dim= encoder_property_dim, # if 0 -> VAE, e.g. if two properties, then the dim = 2 (CVAE)
                            decoder_property_dim= decoder_property_dim, # must have at least more than 1. e.g. if two properties, then the dim = 2
                            device= device)


reconstruction_loss_fn = nn.CrossEntropyLoss(ignore_index= padding_value)
optimizer = optim.Adam(my_model.parameters(), lr= learning_rate)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode= ROnPlateauLR_mode,
                                                          factor= ROnPlateauLR_factor,
                                                          patience= ROnPlateauLR_patience,
                                                          verbose=False)


kl_annealing = my_utils.AnnealingSchedules(method= KL_Annealing_method,
                                           update_unit= KL_Annealing_update_unit,
                                           num_training_steps= epochs * len(train_loader),
                                           num_training_steps_per_epoch= len(train_loader),
                                           start_weight= KL_Annealing_start_weight,
                                           stop_weight= KL_Annealing_stop_weight,)


# Call the training function
model_trainer.train(my_model, train_loader, val_loader, reconstruction_loss_fn, optimizer, lr_scheduler, device,
                    num_epochs= epochs, output_dir= model_save_folder, tokenizer= tokenizer,
                    model_name= model_name, filtered_num=filtered_num, random_pick_num= random_pick_num, tokenizer_type= tokenizer_type, kl_annealing= kl_annealing)



# Call the test function
start_token_id = sos_token_id
start_token_id_for_generation = wildcard_token_id
end_token_id = eos_token_id

model_trainer.test(my_model, test_loader, reconstruction_loss_fn, tokenizer, max_length, num_return_sequences,
         start_token_id, start_token_id_for_generation, end_token_id, device)


