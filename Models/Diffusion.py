import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
from deepsets_cond import DeepSetsAtt




######################################
# Helper Functions
######################################
def GaussianFourierProjection(num_embed, scale=16.0, device='cpu'):
    half_dim = num_embed // 2
    emb = torch.log(10000.0) / (half_dim - 1)
    freq = torch.arange(0, half_dim, dtype=torch.float32, device=device)
    freq = torch.exp(-emb * freq)
    return freq  # [half_dim]


def Embedding(inputs, projection, num_embed=64, activation=F.silu, device='cpu'):
    angle = inputs * projection.unsqueeze(0) * 1000.0  # broadcasting
    sin_angle = torch.sin(angle)        # Compute sine
    cos_angle = torch.cos(angle)        # Compute cosine
    embedding = torch.cat([sin_angle, cos_angle], dim=-1)  # Concatenate along the last dimension

    dense1 = nn.Linear(2*num_embed, 2*num_embed) # since we are concatenating sin and cos along the last dimension
    dense2 = nn.Linear(2*num_embed, num_embed)
    
    embedding = activation(dense1(embedding))
    embedding = activation(dense2(embedding))

    return embedding



def prior_sde(dimensions, device='cpu'):
    # returns normal noise given the dimensions
    return torch.randn(dimensions, dtype=torch.float32, device=device)



def FF(features, min_proj=4, max_proj=8, device='cpu'):
    # Gaussian features to the inputs
    freq = torch.arange(min_proj, max_proj, dtype=torch.float32, device=device)
    freq = (2.0**freq) * 2 * np.pi  # [num_freq]
    num_freq = max_proj - min_proj

    # features: [B, D]
    # Tile freq along features dimension:
    # In TF: freq = tf.tile(freq[None, :], (1, tf.shape(x)[-1]))
    # In PyTorch: we need to repeat freq so that it matches the number of features
    # freq: [num_freq]
    # We want freq expanded to [1, num_freq * D]? Actually, original code:
    # h = tf.repeat(x, max_proj-min_proj, axis=-1)
    # means we repeat the features along the last dimension num_freq times
    B, D = features.shape
    freq = freq.unsqueeze(0)  # [1,num_freq]
    # repeat features along last dimension num_freq times:
    h = features.repeat(1, num_freq)  # [B, D*num_freq]

    # Now we must ensure freq is matched with h:
    # freq should be repeated D times to match h's dimension:
    freq = freq.repeat(1, D)  # [1, num_freq*D]
    # broadcast angle computation
    angle = h * freq  # [B, D*num_freq]

    # sinusoidal expansions
    h = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # [B, 2*D*num_freq]
    return torch.cat([features, h], dim=-1)


def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    # t: [B,1]
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




class ModelJet(nn.Module):
    def __init__(self, num_jet, num_cond, num_embed, activation=F.silu):
        super().__init__()
        self.num_jet = num_jet
        self.num_cond = num_cond
        self.num_embed = num_embed
        self.activation = activation

        self.embed = GaussianFourierProjection(num_embed)
        self.dense1 = nn.Linear(2*num_jet, 2*num_jet)
        self.dense2 = nn.Linear(2*num_jet, num_jet)



class GSGM(nn.Module):
    def __init__(self, npart=100, config=None):
        super().__init__()
        if config is None:
            raise ValueError("Config file not given")
        
        self.config = config
        self.num_feat = config['NUM_FEAT']
        self.num_jet = config['NUM_JET']
        self.num_cond = config['NUM_COND']
        self.num_embed = config['EMBED']
        self.max_part = npart
        self.num_steps = config['MAX_STEPS']
        self.activation = F.silu

        self.model_jet = ModelJet(self.num_jet, self.num_cond, self.num_embed, self.activation)
        self.model_part = ModelPart(self.num_feat, self.num_jet, self.num_cond, self.num_embed, self.activation)



