import numpy as np
import yaml
import os
from Models.Diffusion import GSGM
import Models.Diffusion  as Diffusion
import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import utils
from torchinfo import summary


torch.manual_seed(1233)


def compute_loss(model, part, jet, cond, mask, device, index=0):
    B = cond.size(0)
        
    # For jets:
    random_t = torch.rand((B,1), device=device)

    _, alpha, sigma = Diffusion.get_logsnr_alpha_sigma(random_t, shape=(B,1,1))
    z = torch.randn_like(jet)
    perturbed_x = alpha*jet + z*sigma
    pred_jet = model.forward_jet(perturbed_x, random_t, cond)  # Similar to self.model_jet([perturbed_x, random_t, cond])
    v_jet = alpha*z - sigma*jet
    loss_jet = F.mse_loss(pred_jet, v_jet)



    # For particles:
    num_part = part.size(2)
    part_reshaped = part.reshape(-1, num_part, model.num_feat)  # [B*2, N, num_feat]
    mask_reshaped = mask.reshape(-1, num_part, 1)               # [B*2, N, 1]
    jet_reshaped = jet.reshape(-1, model.num_jet)               # [B*2, num_jet]

    cond_expanded = cond.unsqueeze(1)                           # [B,1,num_cond]
    cond_tiled = cond_expanded.repeat(1,2,1)          # [B,2,num_cond]
    cond_reshaped = cond_tiled.reshape(-1, 1)         # [B*2,1]

    random_t = torch.rand((cond_reshaped.size(0),1), device=device)
    _, alpha, sigma = Diffusion.get_logsnr_alpha_sigma(random_t, shape=(cond_reshaped.size(0),1,1))

    z = torch.randn_like(part_reshaped)*mask_reshaped
    perturbed_x = alpha*part_reshaped + z*sigma
    pred_part = model.forward_part(perturbed_x*mask_reshaped, random_t, jet_reshaped, cond_reshaped, mask_reshaped)
    v = alpha*z - sigma*part_reshaped
    losses = (pred_part - v)**2 * mask_reshaped
    loss_part = losses.mean()

    return loss_jet, loss_part


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_loss_part = 0
    total_loss_jet = 0

    with torch.no_grad():
        for batch in val_loader:
            part, jet, cond, mask = [x.to(device) for x in batch]

            # Jet loss:
            B = cond.size(0)
            random_t = torch.rand((B,1), device=device)
            _, alpha, sigma = Diffusion.get_logsnr_alpha_sigma(random_t, (B,1,1))

            z = torch.randn_like(jet)
            perturbed_x = alpha*jet + z*sigma
            pred = model.forward_jet(perturbed_x, random_t, cond)
            v = alpha*z - sigma*jet
            loss_jet = F.mse_loss(pred, v)

            # For particles, do the same reshaping as test_step in TF code:
            num_part = part.size(2)
            part_reshaped = part.reshape(-1, num_part, model.num_feat)
            mask_reshaped = mask.reshape(-1, num_part, 1)
            jet_reshaped = jet.reshape(-1, model.num_jet)

            cond_expanded = cond.unsqueeze(1)
            cond_tiled = cond_expanded.repeat(1,2,1)
            cond_reshaped = cond_tiled.reshape(-1,1)

            random_t = torch.rand((cond_reshaped.size(0),1), device=device)
            _, alpha, sigma = Diffusion.get_logsnr_alpha_sigma(random_t, (cond_reshaped.size(0),1,1))

            z = torch.randn_like(part_reshaped)*mask_reshaped
            perturbed_x = alpha*part_reshaped + z*sigma
            pred = model.forward_part(perturbed_x, random_t, jet_reshaped, cond_reshaped, mask_reshaped)
            v = alpha*z - sigma*part_reshaped
            losses = (pred - v)**2 * mask_reshaped
            loss_part = losses.mean()

            loss = loss_jet + loss_part
            total_loss += loss.item()
            total_loss_part += loss_part.item()
            total_loss_jet += loss_jet.item()

    return total_loss / len(val_loader), total_loss_jet / len(val_loader), total_loss_part / len(val_loader)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', default='configuration/config.yaml', help='Config file with training parameters')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--data_path', default='/pscratch/sd/d/dimathan/LHCO/Data', help='Path containing the training files')
    parser.add_argument('--load', action='store_true', default=False,help='Load trained model')
    parser.add_argument('--large', action='store_true', default=False,help='Train with a large model')
    parser.add_argument('--multi', action='store_true', default=False,help='Mutli-GPU training')


    flags = parser.parse_args()
    with open(flags.config, 'r') as stream:
        config = yaml.safe_load(stream)

    set_ddp = flags.multi
    local_rank = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if set_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank==0: print('Multi-GPU training')
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    batch_size = config['BATCH']
    if set_ddp:
        batch_size = batch_size // torch.cuda.device_count() # Adjust batch size for DDP
        if local_rank==0: print(f'Batch size per GPU: {batch_size}')

    data_size, train_loader, val_loader = utils.DataLoader(flags.data_path,
                                                            flags.file_name,
                                                            flags.npart,
                                                            ddp = set_ddp,
                                                            rank = local_rank,
                                                            size = torch.cuda.device_count(),
                                                            n_events=config['n_events'],
                                                            batch_size=batch_size,
                                                            norm=config['NORM'],)
    

    model = GSGM(config=config,npart=flags.npart, device=device)
    #print()
    #print(model)
    #print()
    total_params = sum(p.numel() for p in model.parameters())
    #print(f"Total number of parameters: {total_params}")
    #print()
    model_jet = model.model_jet
    mj_params = sum(p.numel() for p in model_jet.parameters())
    model_part = model.model_part
    mp_params = sum(p.numel() for p in model_part.parameters())
 
    #print(f'model_jet: {model_jet}')
    #print(f'mj_params: {mj_params}')
    #print()
    #print(f'model_part: {model_part}')
    #print(f'mp_params: {mp_params}')
    #print()


    # Count the total number of non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    #print(f"Total non-trainable parameters: {non_trainable_params}")

    # Count the total number of trainable parameters for verification
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Total trainable parameters: {trainable_params}")
    #print()

    model.to(device)
 
    model_name = config['MODEL_NAME']
    if flags.large:
        model_name+='_large'
    checkpoint_folder = 'checkpoints_{}/checkpoint'.format(model_name)
    if not os.path.exists('checkpoints_{}'.format(model_name)):
        os.makedirs('checkpoints_{}'.format(model_name))
        


    initial_lr = config['LR'] 
    optimizer = torch.optim.Adamax(model.parameters(), lr=initial_lr)

    steps_per_epoch = int(data_size / config['BATCH'])
    steps_per_epoch = max(steps_per_epoch, 1)
    total_steps = config['MAXEPOCH'] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Load weights if required
    if flags.load and os.path.exists(checkpoint_folder):
        checkpoint = torch.load(checkpoint_folder, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # Early stopping and checkpoint logic
    best_val_loss = float('inf')
    patience = config['EARLYSTOP']
    no_improve_count = 0



    for epoch in range(config['MAXEPOCH']):
        model.train()
        epoch_loss = 0
        epoch_loss_part = 0
        epoch_loss_jet = 0
        count = 0
        start_time = time.time()

        for param_group in optimizer.param_groups:
            l_rate = param_group['lr']
        
        for index, batch in enumerate(train_loader):
            part, jet, cond, mask = [x.to(device) for x in batch]
            optimizer.zero_grad()
            loss_jet, loss_part = compute_loss(model, part, jet, cond, mask, device, index=index)
            loss = loss_jet + loss_part
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update EMA weights for model_jet
            for p, q in zip(model.model_jet.parameters(), model.ema_jet.parameters()):
                q.data.mul_(model.ema).add_(p.data, alpha=1 - model.ema)

            # Update EMA weights for model_part
            for p, q in zip(model.model_part.parameters(), model.ema_part.parameters()):
                q.data.mul_(model.ema).add_(p.data, alpha=1 - model.ema)


            epoch_loss += loss.item() * part.size(0)
            epoch_loss_part += loss_part.item() 
            epoch_loss_jet += loss_jet.item()                # Multiply by 2 since we have 2 jets per event

            count += part.size(0)

            if local_rank == 0:
                scheduler.step()

        epoch_loss = epoch_loss / count
        epoch_loss_part = epoch_loss_part / len(train_loader)
        epoch_loss_jet = epoch_loss_jet / len(train_loader)

        # Average metrics across workers if needed:
        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
        epoch_loss = epoch_loss_tensor.item()
        epoch_loss_part_tensor = torch.tensor(epoch_loss_part, device=device)
        epoch_loss_part = epoch_loss_part_tensor.item()
        epoch_loss_jet_tensor = torch.tensor(epoch_loss_jet, device=device)
        epoch_loss_jet = epoch_loss_jet_tensor.item()

        # Validation
        val_loss, val_loss_jet, val_loss_part  = evaluate(model, val_loader, device)
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_loss = val_loss_tensor.item()
        val_loss_part_tensor = torch.tensor(val_loss_part, device=device)
        val_loss_part = val_loss_part_tensor.item()
        val_loss_jet_tensor = torch.tensor(val_loss_jet, device=device)
        val_loss_jet = val_loss_jet_tensor.item()

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{config['MAXEPOCH']}: Train Loss: {epoch_loss:.4f}, loss_part: {epoch_loss_part:.4f}, loss_jet: {epoch_loss_jet:.4f}, Val Loss: {val_loss:.4f}, val_loss_part: {val_loss_part:.4f}, val_loss_jet: {val_loss_jet:.4f}, lr: {1000*l_rate:.2f} 10^-3, Time: {time.time()-start_time:.1f}s")

        # Check for improvement
        if local_rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                # Save checkpoint
                torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'epoch': epoch
                    }, checkpoint_folder)
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print("Early stopping due to no improvement in validation loss")
                    break
