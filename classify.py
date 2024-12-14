import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py as h5
import os
import sys
import yaml 
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from Models.deepsets_cond import DeepSetsClass
from torchinfo import summary

# Assume we have equivalent utilities in PyTorch:
import utils  # must be adjusted to PyTorch-compatible

from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import energyflow as ef


torch.manual_seed(1233)


def recover_jet(jet, part):
    new_j = np.copy(jet)
    new_p = np.copy(part)
    new_p[:,:,:,0]*=np.expand_dims(jet[:,:,0],-1)
    new_p[:,:,:,1]+=np.expand_dims(jet[:,:,1],-1)
    new_p[:,:,:,2]+=np.expand_dims(jet[:,:,2],-1)

    new_p[:,:,:,2] = np.clip(new_p[:,:,:,2], -np.pi, np.pi)

    mask = np.expand_dims(new_p[:,:,:,0]!=0,-1)
    new_p*=mask
    new_p = ef.p4s_from_ptyphims(new_p)
    jets = np.sum(new_p,2)
    jets = ef.ptyphims_from_p4s(jets)
    new_j[:,:,0] = jets[:,:,0]
    new_j[:,:,1] = ef.etas_from_p4s(np.sum(new_p,2))
    new_j[:,:,2] = np.clip(jets[:,:,2] - np.pi, -np.pi, np.pi)
    new_j[:,:,3] = jets[:,:,3]
    mjj = ef.ms_from_p4s(np.sum(new_p,(1,2)))
    return new_j.reshape((-1,jet.shape[-1])), mjj


def apply_mjj_cut(j,p,mjj,use_SR,mjjmin,mjjmax):
    mask = utils.get_mjj_mask(mjj,use_SR,mjjmin,mjjmax)
    return j[mask], p[mask], mjj[mask]


def combine_part_jet(jet, particle, mjj, npart, jet_from_cond=True):
    if jet_from_cond:
        new_j = np.copy(jet).reshape((-1, jet.shape[-1]))
    else:
        new_j,mjj = recover_jet(jet, particle)

    new_p = np.copy(particle).reshape((-1, particle.shape[-1]))
    mask = new_p[:,0]!=0

    mjj_tile = np.expand_dims(mjj,1)
    mjj_tile = np.reshape(np.tile(mjj_tile,(1,2)),(-1))
    new_j[:,0] = np.log(new_j[:,0]/mjj_tile)
    new_j[:,2] = np.clip(new_j[:,2] - np.pi, -np.pi, np.pi)
    new_j[:,3] = np.ma.log(new_j[:,3]/mjj_tile).filled(0)
    new_p[:,0] = np.ma.log(1.0 - new_p[:,0]).filled(0)

    data_dict = utils.LoadJson('preprocessing_{}.json'.format(npart))
    new_j = np.ma.divide(new_j-data_dict['mean_jet'],data_dict['std_jet']).filled(0)
    new_p = np.ma.divide(new_p-data_dict['mean_particle'],data_dict['std_particle']).filled(0)

    new_p *= np.expand_dims(mask,-1)
    new_j = np.reshape(new_j, jet.shape)
    new_p = np.reshape(new_p, particle.shape)
    return new_j, new_p, mjj


def class_loader(data_path,
                 file_name,
                 npart,
                 use_SR=False,
                 nsig=15000,
                 nbkg=60671,
                 mjjmin=2300,
                 mjjmax=5000):
    
    if not use_SR:
        nsig=0

    parts_bkg, jets_bkg, mjj_bkg = utils.SimpleLoader(data_path,file_name,
                                                      use_SR=use_SR,
                                                      npart=npart)

    parts_bkg = parts_bkg[:nbkg]
    mjj_bkg = mjj_bkg[:nbkg]
    jets_bkg = jets_bkg[:nbkg]

    if nsig>0:
        parts_sig,jets_sig,mjj_sig = utils.SimpleLoader(data_path,
                                                        'processed_data_signal_rel.h5',
                                                        use_SR=use_SR,
                                                        npart=npart)
        parts_sig = parts_sig[:nsig]
        mjj_sig = mjj_sig[:nsig]
        jets_sig = jets_sig[:nsig]

        labels = np.concatenate([np.zeros_like(mjj_bkg), np.ones_like(mjj_sig)])
        particles = np.concatenate([parts_bkg, parts_sig],0)
        jets = np.concatenate([jets_bkg, jets_sig],0)
        mjj = np.concatenate([mjj_bkg, mjj_sig],0)
    else:
        labels = np.zeros_like(mjj_bkg)
        particles = parts_bkg
        jets = jets_bkg
        mjj = mjj_bkg

    return jets, particles, mjj, labels



def model_train(model, SR, train_loader, val_loader, optimizer, criterion, MAX_EPOCH, data_j,data_p,labels,device):
    # Training loop
    best_auc = 0
    if SR:
        mask = data_p[:,:,:,0]!=0
        data_j = torch.tensor(data_j, dtype=torch.float32, device=device)
        data_p = torch.tensor(data_p, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)

        dataset = TensorDataset(data_j, data_p, mask_t, labels)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(MAX_EPOCH):
        model.train()
        total_loss = 0
        total_count = 0
        train_preds = []
        train_labs = []
        t_start = time.time()
        for index, batch_data in enumerate(train_loader):

            batch_data = [data.to(device) for data in batch_data]
            if SR:
                sample_j_b, sample_p_b, mask_b, labels_b, w_b = batch_data
                optimizer.zero_grad()
                outputs = model(sample_j_b, sample_p_b, mask_b)

                #if index == 0 and epoch == 0 :
                #    summary(model, input_data=[sample_j_b, sample_p_b, mask_b])

            else:
                sample_j_b, sample_p_b, mask_b, mjj_b, labels_b, w_b = batch_data
                optimizer.zero_grad()
                outputs = model(sample_j_b, sample_p_b, mask_b, mjj_b)

            loss_vals = criterion(outputs.squeeze(1), labels_b)
            train_preds.append(outputs.detach().cpu().numpy())
            train_labs.append(labels_b.cpu().numpy())

            loss = (loss_vals * w_b).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels_b.size(0)
            total_count += labels_b.size(0)

        avg_loss = total_loss/total_count
        train_preds = np.concatenate(train_preds)
        train_labs = np.concatenate(train_labs)
        fpr, tpr, _ = roc_curve(train_labs, train_preds, pos_label=1)
        auc_res = auc(fpr, tpr)

        # Validation
        model.eval()
        val_preds = []
        val_labs = []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = [data.to(device) for data in batch_data]
                if SR:
                    sample_j_b, sample_p_b, mask_b, labels_b, w_b = batch_data
                    outputs = model(sample_j_b, sample_p_b, mask_b)
                else:
                    sample_j_b, sample_p_b, mask_b, mjj_b, labels_b, w_b = batch_data
                    outputs = model(sample_j_b, sample_p_b, mask_b, mjj_b)
                val_loss = criterion(outputs.squeeze(1), labels_b)
                val_preds.append(outputs.cpu().numpy())
                val_labs.append(labels_b.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labs = np.concatenate(val_labs)
        fpr, tpr, _ = roc_curve(val_labs, val_preds, pos_label=1)
        auc_val = auc(fpr, tpr)
        
        print(f"Epoch {epoch+1}/{MAX_EPOCH}, Train Loss: {avg_loss:.4f}, Train AUC: {auc_res:.4f}, val AUC: {auc_val:.4f}, Time: {time.time()-t_start:.2f}s")


        # test on signal vs background from the data 
        if SR:
            tot_preds, tot_labs = [], []
            for batch_data in loader:
                batch_data = [data.to(device) for data in batch_data]
                sample_j_b, sample_p_b, mask_b, labels_b = batch_data
                pred = model(sample_j_b, sample_p_b, mask_b)
                pred = pred.detach().cpu().numpy()
                tot_preds.append(pred)
                tot_labs.append(labels_b.cpu().numpy())

            tot_preds = np.concatenate(tot_preds)
            tot_labs = np.concatenate(tot_labs)

            fpr, tpr, _ = roc_curve(tot_labs, tot_preds, pos_label=1)
            auc_res = auc(fpr, tpr)
            print(f"AUC on signal vs data background: {auc_res:.4f}")
            print(f'============================')
        if auc_res > best_auc:
            best_auc = auc_res
            torch.save(model.state_dict(), "checkpoint.pt")






if __name__ == "__main__":
    utils.SetStyle()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/pscratch/sd/d/dimathan/LHCO/Data/', help='Folder containing data and MC files')
    parser.add_argument('--plot_folder', default='/global/homes/d/dimathan/Diffusion-for-Anomaly-Detection-Pytorch/plots', help='Folder to save results')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--test', action='store_true', default=False,help='Test if inverse transform returns original data')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--config', default='configuration/config.yaml', help='Training parameters')    
    parser.add_argument('--SR', action='store_true', default=False,help='Load signal region background events')
    parser.add_argument('--hamb', action='store_true', default=False,help='Load Hamburg team dataset')
    parser.add_argument('--reweight', action='store_true', default=False,help='Apply mjj based reweighting to SR events')
    parser.add_argument('--nsig', type=int,default=2500,help='Number of injected signal events')
    parser.add_argument('--nbkg', type=int,default=100000,help='Number of injected signal events')
    parser.add_argument('--nid', type=int,default=0,help='Independent training ID')
    parser.add_argument('--large', action='store_true', default=False,help='Train with a large model')
    parser.add_argument('--data_file', default='', help='File to load')

    parser.add_argument('--LR', type=float,default=1e-4,help='learning rate')
    parser.add_argument('--MAX-EPOCH', type=int,default=1,help='maximum number of epochs for the training')
    parser.add_argument('--BATCH-SIZE', type=int,default=128,help='Batch size')
    flags = parser.parse_args()


    with open(flags.config, 'r') as stream:
        config = yaml.safe_load(stream)

    MAX_EPOCH = flags.MAX_EPOCH
    BATCH_SIZE = flags.BATCH_SIZE
    LR = flags.LR

    data_j, data_p, data_mjj, labels = class_loader(flags.data_folder,
                                                    flags.file_name,
                                                    npart=flags.npart,
                                                    use_SR=flags.SR,
                                                    nsig=flags.nsig,
                                                    nbkg=flags.nbkg,
                                                    mjjmax=config['MJJMAX'],
                                                    mjjmin=config['MJJMIN'])
    

    data_j, data_p, data_mjj = combine_part_jet(data_j, data_p, data_mjj, npart=flags.npart)

    sample_name = config['MODEL_NAME'] if not flags.hamb else 'Hamburg'
    if flags.large:
        sample_name += '_large'
    if flags.test:
        sample_name = 'supervised'
    if flags.SR:
        sample_name += '_SR'

    # Load background samples (generated or from Hamburg) similarly as done in TF code
    if flags.test:
        bkg_p,bkg_j,bkg_mjj = utils.SimpleLoader(flags.data_folder,flags.file_name,
                                                      use_SR=flags.SR,npart=flags.npart)

    elif flags.hamb:
        with h5.File(os.path.join(flags.data_folder,'generated_data_datacond_both_jets.h5'),"r") as h5f:
            bkg_p = np.stack([h5f['particle_data_rel_x'],h5f['particle_data_rel_y']],1)
            bkg_j = np.stack([h5f['jet_features_x'], h5f['jet_features_y']],1)
            npart = np.sum(bkg_p[:,:,:,0]>0,2)
            bkg_j[:,:,-1] = npart
            bkg_mjj = h5f['mjj'][:]
            
    else: # Load the generated background data from plot_jet.py 
        f = flags.data_file if flags.data_file else sample_name+'.h5'
        print(f'Loading {f}...')
        with h5.File(os.path.join(flags.data_folder,f),"r") as h5f:
            bkg_p = h5f['particle_features'][:]
            bkg_j = h5f['jet_features'][:]
            bkg_mjj = h5f['mjj'][:]
            

    data_size = int(bkg_j.shape[0] + data_j.shape[0])
    bkg_j,bkg_p,bkg_mjj = combine_part_jet(bkg_j,bkg_p,bkg_mjj,npart=flags.npart)
    #Using recalculated values of mjj, let's apply the sideband/signal region cuts again
    bkg_j,bkg_p,bkg_mjj = apply_mjj_cut(bkg_j,bkg_p,bkg_mjj,flags.SR,
                                        mjjmin=config['MJJMIN'],mjjmax=config['MJJMAX'])
    


    print("Loading {} generated samples and {} data samples".format(bkg_j.shape[0],data_j.shape[0]))
    print()

    # semi_labels = 0 for generated background, 1 for data (including background + signal)
    semi_labels = np.concatenate([np.zeros(bkg_j.shape[0]),np.ones(data_j.shape[0])],0)
    sample_j = np.concatenate([bkg_j,data_j],0)
    sample_p = np.concatenate([bkg_p,data_p],0)
    sample_mjj = np.concatenate([bkg_mjj,data_mjj],0)
    sample_mjj = utils.prep_mjj(sample_mjj,mjjmin=config['MJJMIN'],mjjmax=config['MJJMAX'])

    mask = sample_p[:,:,:,0]!=0    

    # TODO: For flags.reweight, apply mjj-based reweighting to SR events
    weights = np.ones(sample_j.shape[0])


    # Create datasets

    sample_j_t = torch.tensor(sample_j, dtype=torch.float32, device=device)
    sample_p_t = torch.tensor(sample_p, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    semi_labels_t = torch.tensor(semi_labels, dtype=torch.float32, device=device)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    mjj_t = torch.tensor(sample_mjj, dtype=torch.float32, device=device)


    if flags.SR:
        dataset = TensorDataset(sample_j_t, sample_p_t, mask_t, semi_labels_t, weights_t)
    else:
        dataset = TensorDataset(sample_j_t, sample_p_t, mask_t, mjj_t, semi_labels_t, weights_t)

    train_size = int(0.9*len(dataset))
    val_size = len(dataset)-train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = DeepSetsClass(num_heads=1, num_transformer=4, projection_dim=64, use_cond=(not flags.SR), num_part_features=config['NUM_FEAT'], num_jet_features=config['NUM_JET'], device=device)

    print()
    print(model)
    print()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print()


    model.to(device)

    optimizer = torch.optim.Adamax(model.parameters(), lr=LR)
    criterion = nn.BCELoss(reduction='none')  # We'll apply weights manually


    model_train(model, flags.SR, train_loader, val_loader, optimizer, criterion, MAX_EPOCH, data_j, data_p, labels, device)



