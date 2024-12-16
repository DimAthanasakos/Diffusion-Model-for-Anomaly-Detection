import numpy as np
import yaml
import matplotlib.pyplot as plt
import argparse
import h5py as h5
import os
import torch
import utils
import energyflow as ef
from plot_class import PlottingConfig
from Models.Diffusion import GSGM  # Assuming a PyTorch version of GSGM is available
import time
import torch.distributed as dist


def get_mjj(particle, jet):
    # Recover the particle information
    new_p = np.copy(particle)
    # Rescale particle features based on jet info
    new_p[:,:,:,0]*=np.expand_dims(jet[:,:,0],-1)
    new_p[:,:,:,1]+=np.expand_dims(jet[:,:,1],-1)
    new_p[:,:,:,2]+=np.expand_dims(jet[:,:,2],-1)

    # Mask zero particles
    mask = np.expand_dims(new_p[:,:,:,0]!=0,-1)
    new_p *= mask

    # Convert to four-momentum
    new_p = ef.p4s_from_ptyphims(new_p)
    mjj = ef.ms_from_p4s(np.sum(new_p,(1,2)))
    return mjj
    

def plot(jet1, jet2, nplots, title, plot_folder, rank = -1):
    for ivar in range(nplots):
        config = PlottingConfig(title, ivar)
                    
        feed_dict = {
            'true':jet1[:,ivar],
            'gen': jet2[:,ivar]
        }

        fig, gs, _ = utils.HistRoutine(feed_dict, xlabel=config.var,
                                       plot_ratio=True,
                                       reference_name='true',
                                       ylabel='Normalized entries', logy=config.logy, rank = rank)
        
        ax0 = plt.subplot(gs[0])     
        if not config.logy:
            yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((100,0))
            ax0.yaxis.set_major_formatter(yScalarFormatter)

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig('{}/{}_{}.pdf'.format(plot_folder, title, ivar), bbox_inches='tight')


def gather_data_across_gpus(data, device):
    """
    Gathers data from all GPUs and concatenates it on each GPU.
    """
    # Convert data to a tensor and move it to the current GPU
    data_tensor = torch.tensor(data, device=device)

    # Determine the size of the data on each rank
    local_size = torch.tensor([data_tensor.shape[0]], device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)

    # Create a tensor to hold the gathered data
    max_size = max(s.item() for s in sizes)
    padded_data = torch.zeros((max_size, *data_tensor.shape[1:]), device=device)
    padded_data[:local_size.item()] = data_tensor

    gathered_data = [torch.zeros_like(padded_data) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_data, padded_data)

    # Remove padding and concatenate the gathered data
    gathered_data = [gd[:size.item()].cpu().numpy() for gd, size in zip(gathered_data, sizes)]
    return np.concatenate(gathered_data, axis=0)



if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/pscratch/sd/d/dimathan/LHCO/Data', help='Folder containing data and MC files')    
    parser.add_argument('--plot_folder', default='/global/homes/d/dimathan/Diffusion-for-Anomaly-Detection-Pytorch/plots', help='Folder to save results')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--config', default='configuration/config.yaml', help='Training parameters')
    parser.add_argument('--nsplit', default=4, type=int, help='Number of batches to generate')
    parser.add_argument('--sample', action='store_true', default=False, help='Sample from the generative model')
    parser.add_argument('--test', action='store_true', default=False, help='Test if inverse transform returns original data')
    parser.add_argument('--SR', action='store_true', default=False, help='Load signal region background events')
    parser.add_argument('--hamb', action='store_true', default=False, help='Load hamburg team files')
    parser.add_argument('--large', action='store_true', default=False, help='Train with a large model')
    parser.add_argument('--multi', action='store_true', default=False,help='Mutli-GPU training')


    flags = parser.parse_args()

    with open(flags.config, 'r') as stream:
        config = yaml.safe_load(stream)
    
    n_events_sample = config['n_events_sample']

    set_ddp = flags.multi
    local_rank = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if set_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank==0: 
            print()
            print('===============================================')
            print('Running on multiple GPUs via torch.distributed')
            print('===============================================')
            print()
   
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        n_events_sample = n_events_sample // torch.cuda.device_count() # Split the number of events across GPUs
    
    else: 
        print()
        print('==================================')
        print(f'Running on a single {device}')
        print('==================================')
        print()

    if local_rank==0:
        if not os.path.exists(flags.plot_folder):
            os.makedirs(flags.plot_folder)
            print(f'Created folder {flags.plot_folder}')
        print(f'Each GPU will generate {n_events_sample} events')

    model_name = config['MODEL_NAME']
    if flags.large:
        model_name += '_large'

    sample_name = model_name
    if flags.SR:
        sample_name += '_SR'
    if flags.hamb:
        sample_name += '_Hamburg'


    # Load the actual data of LHCO as comparison to the generated data
    particles, jets, logmjj, _ = utils.DataLoader(flags.data_folder,
                                                  flags.file_name,
                                                  npart=flags.npart,              
                                                  ddp = set_ddp,
                                                  rank = local_rank,
                                                  size = torch.cuda.device_count(),
                                                  n_events=1000000,
                                                  n_events_sample=n_events_sample, 
                                                  norm=config['NORM'],
                                                  make_torch_data=False, use_SR=flags.SR)
    
    if local_rank==0: 
        print()
        print('After Loading')
        print(f'particles shape: {particles.shape}')
        print(f'jet shape: {jets.shape}')
        print()

    if flags.test:
        particles_gen, jets_gen, mjj_gen = utils.SimpleLoader(flags.data_folder,
                                                              flags.file_name,
                                                              use_SR=flags.SR,
                                                              npart=flags.npart)
    else:
        if flags.sample:
            # Load PyTorch model

            model = GSGM(config=config, npart=flags.npart, device=device)
            checkpoint_folder = f'checkpoints_{model_name}/checkpoint'
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_folder, map_location=device)

            # Load the model's state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
            #model.load_state_dict(torch.load(checkpoint_folder, map_location=device))
            model.eval()

            particles_gen = []
            jets_gen = []

            if flags.SR:
                logmjj_g = utils.LoadMjjFile(flags.data_folder, 'generated_data_all_200k.h5', use_SR=flags.SR)[:n_events_sample]
            else: 
                logmjj_g = logmjj

            # Generate data in splits
            
            # A single GPU can only ~store 100 events at a single split due to memory constraints
            n_splits = flags.nsplit
            if logmjj_g.size > 100 * n_splits or logmjj_g.size < 50 * n_splits: 
                n_splits = logmjj_g.size // 100
                if logmjj_g.size % 100 != 0: n_splits += 1 
                if local_rank==0:  print(f'Number of splits changed to {n_splits} to fit in memory and for speedup')
            
            for i,split in enumerate(np.array_split(logmjj_g, n_splits)):
                start = time.time()
                if split.size == 0:  break
                if local_rank==0: 
                    print(f'Generating split {i+1}/{n_splits}')
                with torch.no_grad():
                    p, j = model.generate(split)  # Assuming generate returns np arrays
                particles_gen.append(p)
                jets_gen.append(j)

                p_aux, j_aux = utils.ReversePrep(p, j, mjj=utils.revert_mjj(split), npart=flags.npart, norm=config['NORM'])
                mjj_aux = get_mjj(p_aux, j_aux)
                sr_events = np.sum((mjj_aux >= 3300) & (mjj_aux <= 3700))
                sb_events = np.sum(((mjj_aux < 3300) & (mjj_aux > 2300)) | ((mjj_aux > 3700) & (mjj_aux < 5000)))
                if local_rank==0: 
                    dt = time.time() - start
                    print(f"Time for sampling {split.shape[0]} events for rank: {local_rank}, is {dt:.2f} seconds")
                    print(f'# of events in the signal region: {sr_events}/{len(mjj_aux)}')
                    print(f'# of events in the side band: {sb_events}/{len(mjj_aux)}')
                    print()

            particles_gen = np.concatenate(particles_gen)
            jets_gen = np.concatenate(jets_gen)
            
            particles_gen, jets_gen = utils.ReversePrep(particles_gen,
                                                        jets_gen,
                                                        mjj=utils.revert_mjj(logmjj_g),
                                                        npart=flags.npart,
                                                        norm=config['NORM'])
            mjj_created = get_mjj(particles_gen, jets_gen)
            mjj_gen = utils.revert_mjj(logmjj_g)

            # Keep only sideband events if needed
            only_SB = False
            if only_SB:
                mask_region = utils.get_mjj_mask(mjj_created, flags.SR, mjjmin=config['MJJMIN'], mjjmax=config['MJJMAX'])
                passed = np.sum(mask_region)
                if local_rank==0: 
                    print('Keeping only SideBand Events')
                    print(f'# of Generated Events In the SideBand: {passed}/{len(mask_region)}')
                particles_gen = particles_gen[mask_region]
                jets_gen = jets_gen[mask_region]
                mjj_created = mjj_created[mask_region]

            # Save generated data
            with h5.File(os.path.join(flags.data_folder, sample_name+'.h5'), "w") as h5f:
                h5f.create_dataset("particle_features", data=particles_gen)
                h5f.create_dataset("jet_features", data=jets_gen)
                h5f.create_dataset("mjj", data=mjj_gen)

        elif flags.hamb:
            assert flags.SR, "ERROR: Hamburg files available only at SR"
            with h5.File(os.path.join(flags.data_folder, 'generated_data_datacond_both_jets.h5'),"r") as h5f:
                particles_gen = np.stack([
                    h5f['particle_data_rel_x'][:],
                    h5f['particle_data_rel_y'][:]],1)
                jets_gen = np.stack([
                    h5f['jet_features_x'][:],
                    h5f['jet_features_y'][:]],1)
                mjj_gen = h5f['mjj'][:]

        else:
            # Load from previously generated file
            with h5.File(os.path.join(flags.data_folder, sample_name+'.h5'),"r") as h5f:
                jets_gen = h5f['jet_features'][:]
                particles_gen = h5f['particle_features'][:]
                mjj_gen = h5f['mjj'][:]
    

    # Convert back to original space
    particles, jets = utils.ReversePrep(particles, jets, mjj=utils.revert_mjj(logmjj),
                                        npart=flags.npart, norm=config['NORM'])
    

    if set_ddp:
        # Gather particles and jets across all GPUs
        particles = gather_data_across_gpus(particles, device)
        jets = gather_data_across_gpus(jets, device)
        mjj = gather_data_across_gpus(utils.revert_mjj(logmjj), device)


        particles_gen = gather_data_across_gpus(particles_gen, device)
        jets_gen = gather_data_across_gpus(jets_gen, device)
        mjj_gen = gather_data_across_gpus(mjj_gen, device)

    # Execute the following only on rank 0
    if local_rank == 0:
        print(f"Gathered particles shape: {particles_gen.shape}")
        print(f"Gathered jets shape: {jets_gen.shape}")
        print(f"Gathered mjj shape: {mjj_gen.shape}")
        print()
        print(f'Plotting')


        feed_dict = {
            'true': get_mjj(particles, jets),
            'gen': get_mjj(particles_gen, jets_gen)
        }
        
        utils.SetStyle()
        fig, gs, _ = utils.HistRoutine(feed_dict, xlabel="mjj GeV",
                                    binning=np.linspace(2800, 4200, 50),
                                    plot_ratio=True,
                                    reference_name='true',
                                    ylabel='Normalized entries', logy=True, rank = local_rank)
            
        fig.savefig('{}/mjj_{}.pdf'.format(flags.plot_folder, sample_name), bbox_inches='tight')

        # Flatten and plot other features
        jets = jets.reshape(-1, config['NUM_JET'])
        jets_gen = jets_gen.reshape(-1, config['NUM_JET'])
        title = 'jet' if not flags.SR else 'jet_SR'
        if flags.hamb: title+='_Hamburg'
        
        plot(jets, jets_gen, title=title,
            nplots=config['NUM_JET'], plot_folder=flags.plot_folder, rank = local_rank)
        
        particles_gen=particles_gen.reshape((-1, config['NUM_FEAT']))
        mask_gen = particles_gen[:,0]!=0.
        particles_gen=particles_gen[mask_gen]
        particles=particles.reshape((-1, config['NUM_FEAT']))
        mask = particles[:,0]!=0.
        particles=particles[mask]
        title = 'part' if not flags.SR else 'part_SR'
        if flags.hamb: title+='_Hamburg'
        
        plot(particles, particles_gen,
            title=title,
            nplots=config['NUM_FEAT'],
            plot_folder=flags.plot_folder, rank = local_rank)

        print(f'Plots saved in {flags.plot_folder}')
        print()
        t_end = time.time()
        print(f'Total time taken: {t_end-t_start:.1f} seconds')
        print()
