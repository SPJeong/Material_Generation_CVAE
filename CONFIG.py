##### CONFIG.py
import torch

filtered_num = 30 # filtering num of SMILES
random_pick_num = 20000 # num_pick
data_extraction_folder = fr"C:\Users\wisdo\polyOne_Data_Set\data_extraction"
chemical_feature_extraction_folder = fr"C:\Users\wisdo\polyOne_Data_Set\chemical_feature_extraction"
model_save_folder = fr"C:\Users\wisdo\polyOne_Data_Set\models"
plot_save_folder = fr"C:\Users\wisdo\polyOne_Data_Set\plot"

model_name = 'CVAE'
batch_size = 128
learning_rate = 3e-4
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_dim = 300
hidden_dim = 512
num_layers = 2
dropout = 0.2
latent_dim = 256
encoder_property_dim= 0
decoder_property_dim= 2



tokenizer_type ='atomwise' # tokenizer_type_list = ['gpt', 'smilesPE', 'atomwise', 'atomInSmiles']
smiles_column = 'canonical_smiles' # 'canonical_smiles' or 'deep_smiles' for tokenizing and numericalizing

ROnPlateauLR_mode = 'min'
ROnPlateauLR_factor = 0.2
ROnPlateauLR_patience = 5

KL_Annealing_method = 'cycle_sigmoid' # ('cycle_linear' or 'cycle_sigmoid' or 'cycle_cosine' )
KL_Annealing_update_unit = 'epoch' # ('step' or 'epoch')
KL_Annealing_start_weight = 0.0
KL_Annealing_stop_weight = 1.0

max_length= 50 # for max sequence length during testing function e.g. 50
num_return_sequences= 100 # for number of smiles generation during testing function e.g. 100
