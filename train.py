import numpy as np
import yaml
import os
from Models.Diffusion import GSGM
import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import utils


torch.manual_seed(1233)


def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    # t: [B,1]
    logsnr_max = torch.tensor(logsnr_max, dtype=torch.float32)  # Convert to tensor
    logsnr_min = torch.tensor(logsnr_min, dtype=torch.float32)  # Convert to tensor

    b = torch.atan(torch.exp(-0.5 * logsnr_max))
    a = torch.atan(torch.exp(-0.5 * logsnr_min)) - b
    # torch.tan expects radians, so this matches TF
    return -2.0 * torch.log(torch.tan(a * t + b))


def inv_logsnr_schedule_cosine(logsnr, logsnr_min=-20., logsnr_max=20.):
    b = torch.atan(torch.exp(-0.5 * logsnr_max))
    a = torch.atan(torch.exp(-0.5 * logsnr_min)) - b
    return (torch.atan(torch.exp(-0.5 * logsnr)) / a) - (b/a)


def get_logsnr_alpha_sigma(time, shape=(-1,1,1)):
    # time: [B,1]
    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))

    logsnr = logsnr.view(shape)
    alpha = alpha.view(shape)
    sigma = sigma.view(shape)

    return logsnr, alpha, sigma





def compute_loss(model, part, jet, cond, mask, device):
    # part: [B,2,N,num_feat]
    # jet: [B,2,num_jet]
    # cond: [B,num_cond]
    # mask: [B,2,N,1]

    B = cond.size(0)

    # For jets:
    random_t = torch.rand((B,1), device=device)
    _, alpha, sigma = get_logsnr_alpha_sigma(random_t, shape=(B,1,1))
    z = torch.randn_like(jet)
    perturbed_x = alpha*jet + z*sigma
    pred_jet = model.forward_jet(perturbed_x, random_t, cond)  # Similar to self.model_jet([perturbed_x, random_t, cond])
    v_jet = alpha*z - sigma*jet
    loss_jet = F.mse_loss(pred_jet, v_jet)

    # For parts:
    # 1) Reshape part and mask:
    # num_part = N = part.shape[2]
    num_part = part.size(2)
    part_reshaped = part.reshape(-1, num_part, model.num_feat)  # [B*2, N, num_feat]
    mask_reshaped = mask.reshape(-1, num_part, 1)               # [B*2, N, 1]
    jet_reshaped = jet.reshape(-1, model.num_jet)               # [B*2, num_jet]

    cond_expanded = cond.unsqueeze(1)                           # [B,1,num_cond]
    # tf.tile(cond,(1,2)) and then reshape to (-1,1)
    # cond: [B,num_cond] -> expand_dims -> [B,1,num_cond] -> tile(1,2) -> [B,2,num_cond]
    # reshape to (-1,1) means [B*2, num_cond was 1??? The code had cond shaped (-1,1) at the end.
    # In the TF code, num_cond seems to become 1 eventually, ensure cond has dimension 1 at this stage.
    # If original code ended up with cond: [B*2,1], we must do similarly:
    # If num_cond was originally 1, this works directly. If not, we must adapt.
    # Assuming cond has shape [B,num_cond], the TF code reduces it to (-1,1).
    # Let's assume cond has shape [B,1] originally for simplicity. If not, adapt accordingly.
    cond_tiled = cond_expanded.repeat(1,2,1)          # [B,2,num_cond]
    cond_reshaped = cond_tiled.reshape(-1, 1)         # [B*2,1]

    random_t = torch.rand((cond_reshaped.size(0),1), device=device)
    _, alpha, sigma = get_logsnr_alpha_sigma(random_t, shape=(cond_reshaped.size(0),1,1))

    z = torch.randn_like(part_reshaped)*mask_reshaped
    perturbed_x = alpha*part_reshaped + z*sigma
    pred_part = model.forward_part(perturbed_x*mask_reshaped, random_t, jet_reshaped, cond_reshaped, mask_reshaped)
    v = alpha*z - sigma*part_reshaped
    # Compute masked MSE
    losses = (pred_part - v)**2 * mask_reshaped
    loss_part = losses.mean()

    return loss_jet, loss_part


def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            part, jet, cond, mask = [x.to(device) for x in batch]

            # Jet loss:
            B = cond.size(0)
            random_t = torch.rand((B,1), device=device)
            _, alpha, sigma = get_logsnr_alpha_sigma(random_t, (B,1,1))

            z = torch.randn_like(jet)
            perturbed_x = alpha*jet + z*sigma
            pred = model.forward_jet(perturbed_x, random_t, cond)
            v = alpha*z - sigma*jet
            loss_jet = F.mse_loss(pred, v)

            # For parts, do the same reshaping as test_step in TF code:
            num_part = part.size(2)
            part_reshaped = part.reshape(-1, num_part, model.num_feat)
            mask_reshaped = mask.reshape(-1, num_part, 1)
            jet_reshaped = jet.reshape(-1, model.num_jet)

            cond_expanded = cond.unsqueeze(1)
            cond_tiled = cond_expanded.repeat(1,2,1)
            cond_reshaped = cond_tiled.reshape(-1,1)

            random_t = torch.rand((cond_reshaped.size(0),1), device=device)
            _, alpha, sigma = get_logsnr_alpha_sigma(random_t, (cond_reshaped.size(0),1,1))

            z = torch.randn_like(part_reshaped)*mask_reshaped
            perturbed_x = alpha*part_reshaped + z*sigma
            pred = model.forward_part(perturbed_x, random_t, jet_reshaped, cond_reshaped, mask_reshaped)
            v = alpha*z - sigma*part_reshaped
            losses = (pred - v)**2 * mask_reshaped
            loss_part = losses.mean()

            loss = loss_jet + loss_part
            total_loss += loss.item()*B
            count += B

    return total_loss / count



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', default='configuration/config.yaml', help='Config file with training parameters')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--data_path', default='/pscratch/sd/d/dimathan/LHCO/Data', help='Path containing the training files')
    parser.add_argument('--load', action='store_true', default=False,help='Load trained model')
    parser.add_argument('--large', action='store_true', default=False,help='Train with a large model')

    flags = parser.parse_args()
    with open(flags.config, 'r') as stream:
        config = yaml.safe_load(stream)
      
    
    data_size, train_loader, test_loader = utils.DataLoader(
                                                            flags.data_path,
                                                            flags.file_name,
                                                            flags.npart,
                                                            n_events=config['n_events'],
                                                            batch_size=config['BATCH'],
                                                            norm=config['NORM']
                                                            )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GSGM(config=config,npart=flags.npart, device=device)
    model.to(device)
 
    model_name = config['MODEL_NAME']
    if flags.large:
        model_name+='_large'
    checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)
        
    initial_lr = config['LR'] 
    optimizer = torch.optim.Adamax(model.parameters(), lr=initial_lr)

    # Distributed optimizer
    #optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Learning rate scheduler (cosine decay)
    # steps_per_epoch = int(data_size/config['BATCH'])
    # PyTorch's CosineAnnealingLR needs a T_max (number of iterations)
    steps_per_epoch = int(data_size / config['BATCH'])
    total_steps = config['MAXEPOCH'] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Load weights if required
    if flags.load and os.path.exists(checkpoint_folder):
        checkpoint = torch.load(checkpoint_folder, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # Early stopping and checkpoint logic
    best_val_loss = float('inf')
    patience = 50
    no_improve_count = 0

    for epoch in range(config['MAXEPOCH']):
        model.train()
        epoch_loss = 0
        count = 0
        start_time = time.time()

        for index, batch in enumerate(train_loader):
            part, jet, cond, mask = [x.to(device) for x in batch]
            print(f'batch:{index}')
            print(f'part:{part.shape}')
            print(f'jet:{jet.shape}')
            print(f'cond:{cond.shape}')
            print(f'mask:{mask.shape}')
            print(f'=============================')
            optimizer.zero_grad()
            loss_jet, loss_part = compute_loss(model, part, jet, cond, mask, device)
            loss = loss_jet + loss_part
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * part.size(0)
            count += part.size(0)
            scheduler.step()

        epoch_loss = epoch_loss / count
        # Average metrics across workers if needed:
        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
        epoch_loss = epoch_loss_tensor.item()

        # Validation
        val_loss = evaluate(model, test_loader, device)
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_loss = val_loss_tensor.item()

        print(f"Epoch {epoch+1}/{config['MAXEPOCH']}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check for improvement
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
