# Diffusion Model for Anomaly Detection in Pytorch


Unofficial implementation in Pytorch of the Generative Diffusion Model in ["Full Phase Space Resonant Anomaly Detection"](https://arxiv.org/abs/2310.06897)

[Here](https://github.com/ViniciusMikuni/LHCO_diffusion) is the official Tensorflow imlementation.

The code can run as is in the Perlmutter cluster at LBL Berkeley. Minor changes might be necessary in other clusters. 

Most of the (hyper)parameters are set in the config.yaml file and/or via the flags.

For the list of requirements, check the init.sh script. 

To train the model: 
```bash
python train.py
```

To sample from the trained model you can run:
```bash
python plot_jet.py  --sample --SR
```
The ```--SR``` flags determines if the background sample is done in the signal region (with --SR flag) or in the side-band region (without the --SR flag).

Drop the ```--sample``` to reproduce the plots without having to regenerate new files.

A classifier used to separate generated samples from data can be trained using:

```bash
python classify.py --SR
```

Where the ```--SR``` flag again determines where the backgrounds are sampled from.

In case of multiple GPUs the code can run in parallel (all 3 scripts) via Pytorch's DDP via adding the arg --multi. For example, in Perlmutter we can use all the GPUs in a node via:

```bash
cmd="python train.py --multi"
srun -n 4 --cpus-per-task 32 --gpus-per-node 4 bash -c "source export_DDP_vars.sh && $cmd"
```

