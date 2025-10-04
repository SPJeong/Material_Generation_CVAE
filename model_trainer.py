##### model_trainer.py

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.cuda.amp import autocast, GradScaler
from rdkit import Chem


def save_model(model,
               save_model_path,
               epoch=None,
               global_step=None,
               optimizer=None,
               lr_scheduler=None,
               best_loss=None):
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    checkpoint = {'model_state_dict': model.state_dict(), }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if global_step is not None:
        checkpoint['global_step'] = global_step
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss

    torch.save(checkpoint, save_model_path)
    print(f'Saved model checkpoint to {save_model_path}.')


def calculate_generation_metrics(generated_smiles_list,
                                 test_smiles_set) -> dict:  # for use in test function

    valid_smiles = set()
    total_count = len(generated_smiles_list)

    for smiles in generated_smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            standard_smiles = Chem.MolToSmiles(mol, canonical=True)
            valid_smiles.add(standard_smiles)

    valid_count = len(valid_smiles)

    # 1. Validity check (real molecular structure)
    valid_percentage = (valid_count / total_count) * 100 if total_count > 0 else 0.0

    # 2. Uniqueness check (no overlapped, repeated molecule)
    unique_percentage = (valid_count / total_count) * 100 if total_count > 0 else 0.0

    # 3. Novelty check: new appeared molecule which is not seen in train/test set
    if test_smiles_set:
        novel_smiles = valid_smiles - test_smiles_set
        novel_percentage = (len(novel_smiles) / valid_count) * 100 if valid_count > 0 else 0.0
    else:
        novel_percentage = 0.0  # if no test_smile set, then just set as zero

    return {'valid_percentage': valid_percentage,
            'unique_percentage': unique_percentage,
            'novel_percentage': novel_percentage,
            'valid_smiles_count': valid_count,
            'total_generated_count': total_count}


def validate(my_model, val_loader, reconstruction_loss_fn, device):
    my_model.eval().to(device)
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, lengths, _, properties) in enumerate(val_loader):
            inputs, targets, lengths, norm_properties = inputs.to(device), targets.to(device), lengths.to(
                device), properties.to(device)

            with autocast():
                outputs, z_mu, z_logvar = my_model(inputs, lengths, norm_properties)
                reconstruction_loss = reconstruction_loss_fn(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
                kl_loss = .5 * (torch.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar).sum(1).mean()

                total_loss_re_kl = reconstruction_loss + kl_loss

            total_loss += total_loss_re_kl.item()
    avg_val_loss = total_loss / len(val_loader)

    return avg_val_loss


def test(my_model, test_loader, reconstruction_loss_fn, tokenizer, max_length, num_return_sequences,
         start_token_id, start_token_id_for_generation, end_token_id, device):
    my_model.eval().to(device)
    total_loss = 0.0

    test_smiles_set = set()

    with torch.no_grad():
        for batch_idx, (inputs, targets, lengths, original_smiles, properties) in enumerate(test_loader):
            inputs, targets, lengths, norm_properties = inputs.to(device), targets.to(device), lengths.to(
                device), properties.to(device)

            test_smiles_set.update(original_smiles)

            with autocast():
                outputs, z_mu, z_logvar = my_model(inputs, lengths, norm_properties)
                reconstruction_loss = reconstruction_loss_fn(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
                kl_loss = .5 * (torch.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar).sum(1).mean()

                total_loss_re_kl = reconstruction_loss + kl_loss

            total_loss += total_loss_re_kl.item()

    avg_test_loss = total_loss / len(test_loader)
    print(f"\n=====> Test Loss: {avg_test_loss:.4g} <=====")

    # CVAE generation evaluation
    if 'norm_properties' in locals():
        # Last batch's normalized properties are used to define the generation target for testing
        target_property_vector = norm_properties.mean(dim=0)
    else:
        print("Could not find 'norm_properties'. Using a dummy property tensor.")
        property_dim = my_model.decoder_property_dim
        target_property_vector = torch.zeros(property_dim, device=device)

    generation_properties = target_property_vector.unsqueeze(0).repeat(num_return_sequences, 1)

    print(f"Generating {num_return_sequences} molecules with target properties...")

    generated_smiles_list = []
    with torch.no_grad():
        generated_smiles_list = my_model.generate(tokenizer=tokenizer,
                                                  max_length=max_length,
                                                  num_return_sequences=num_return_sequences,
                                                  start_token_id=start_token_id,
                                                  start_token_id_for_generation=start_token_id_for_generation,
                                                  end_token_id=end_token_id,
                                                  properties=generation_properties)

    # CVAE evaluation metrics
    metrics = calculate_generation_metrics(generated_smiles_list, test_smiles_set)
    print("\n--- Conditional Generation Metrics ---")
    print(f"Total Generated Count: {metrics['total_generated_count']}")
    print(f"Valid SMILES Count: {metrics['valid_smiles_count']}")
    print(f"1. Validity % : {metrics['valid_percentage']:.2f}%")
    print(f"2. Uniqueness % : {metrics['unique_percentage']:.2f}%")
    print(f"3. Novelty % : {metrics['novelty_percentage']:.2f}% (vs Test Set)")
    print(f"Example Generated SMILES: {generated_smiles_list[:5]}")
    print("--------------------------------------")

    return avg_test_loss, generated_smiles_list


# Training function with AMP
def train(my_model, train_loader, val_loader, reconstruction_loss_fn, optimizer, lr_scheduler, device,
          num_epochs, output_dir, tokenizer, model_name, filtered_num, random_pick_num, tokenizer_type, kl_annealing):
    scaler = GradScaler()
    my_model.train().to(device)
    print("=====> Starting Training")

    best_loss = float('inf')
    best_model_path = os.path.join(output_dir,
                                   f'{model_name}_model_len_{filtered_num}_num_{random_pick_num}_{tokenizer_type}.pt')
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx, (inputs, targets, lengths, _, properties) in enumerate(train_loader):
            inputs, targets, lengths, norm_properties = inputs.to(device), targets.to(device), lengths.to(
                device), properties.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs, z_mu, z_logvar = my_model(inputs, lengths, norm_properties)
                reconstruction_loss = reconstruction_loss_fn(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
                kl_loss = .5 * (torch.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar).sum(1).mean()
                kl_annealing_weight = kl_annealing(global_step)

                total_loss_re_kl = reconstruction_loss + kl_annealing_weight * kl_loss

            scaler.scale(total_loss_re_kl).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=50)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss_re_kl.item()
            global_step += 1  # this is set for KL loss Annealing Schedule

            if global_step % 10 == 0:
                print(
                    f'{epoch} Epochs | {batch_idx}/{len(train_loader)} | reconst_loss: {reconstruction_loss.item():.4g} | '
                    f'kl_loss: {kl_loss:.4g}, total_loss: {total_loss_re_kl:.4g}, '
                    f'kl_annealing: {kl_annealing(global_step - 1):.4g} ')

        avg_train_loss = epoch_loss / len(train_loader)

        # validation
        avg_val_loss = validate(my_model, val_loader, reconstruction_loss_fn, device)
        my_model.train()

        lr_scheduler.step(avg_val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4g} | Val Loss: {avg_val_loss:.4g}')

        # Check if the current model is the best one and save
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print(f'New best model found at epoch {epoch + 1} with Validation loss {best_loss:.4g}. Saving model.')

            # Use the refactored save_model function to save the best model
            save_model(model=my_model,
                       save_model_path=best_model_path,
                       epoch=epoch + 1,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler,
                       best_loss=best_loss)

        # # Generating smiles during training for model checking
        # if (epoch + 1) % 3 == 0:
        #     print(f'--- [Generation Check] at Epoch {epoch + 1} ---')
        #     my_model.eval()
        #     with torch.no_grad():
        #         batch_size_for_generate = 1  # how many smiles will be generated during training
        #
        #         generated_smiles = my_model.generate(tokenizer=tokenizer, max_length=50,
        #                                              num_return_sequences=batch_size_for_generate, )
        #         print(f"Generated SMILES: {generated_smiles}")
        #     my_model.train()

    print("=====> Training is complete")
    return best_loss


