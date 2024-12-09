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


def base_Embedding(inputs, projection):
    angle = inputs * projection.unsqueeze(0) * 1000.0  # broadcasting
    sin_angle = torch.sin(angle)        # Compute sine
    cos_angle = torch.cos(angle)        # Compute cosine
    embedding = torch.cat([sin_angle, cos_angle], dim=-1)  # Concatenate along the last dimension

    #dense1 = nn.Linear(2*num_embed, 2*num_embed) # since we are concatenating sin and cos along the last dimension
    #dense2 = nn.Linear(2*num_embed, num_embed)
    
    #embedding = activation(dense1(embedding))
    #embedding = activation(dense2(embedding))

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
    return torch.cat([features, h], dim=-1) # [B, D+2*D*num_freq]


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





class GSGM(nn.Module):
    def __init__(self, npart=100, config=None, device='cpu'):
        super().__init__()
        if config is None:
            raise ValueError("Config file not given")
        
        self.config = config
        self.activation = F.silu()
        self.num_feat = self.config['NUM_FEAT']
        self.num_jet = self.config['NUM_JET']
        self.num_cond = self.config['NUM_COND']
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.num_steps = self.config['MAX_STEPS']
        self.ema = 0.999
        self.device = device

        # parameters for FF(). We need to know the number of features before the forward pass
        self.min_proj = 4
        self.max_proj = 8

        # Projection for time embedding
        self.projection = GaussianFourierProjection(self.num_embed, scale=16.0, device=self.device)

        # Embedding linear layers (instead of creating them in the Embedding function)
        self.emb_fc1 = nn.Linear(2*self.num_embed, 2*self.num_embed)
        self.emb_fc2 = nn.Linear(2*self.num_embed, self.num_embed)

        # Layers equivalent to Keras Dense layers for conditional jet inputs
        self.jet_fc1 = nn.Linear(self.num_jet, 2*self.num_embed)
        self.jet_fc2 = nn.Linear(2*self.num_embed, self.num_embed)

        # FF layers for cond
        # We must know shape after FF; 
        cond_ff_dim = self.num_cond * (1 + 2*(self.max_proj-self.min_proj))

        self.cond_fc1 = nn.Linear(cond_ff_dim, 2*self.num_embed)
        self.cond_fc2 = nn.Linear(2*self.num_embed, self.num_embed)

        # graph_conditional and jet_conditional
        self.graph_fc = nn.Linear(3*self.num_embed, 3*self.num_embed) # this should probably be 2*self.num_embed + self.num_embed//2 due to GaussianFourierProjection
        self.jet_cond_fc = nn.Linear(2*self.num_embed, 2*self.num_embed) # similarly here, self.num_embed + self.num_embed//2
  
        # DeepSetsAtt
        self.model_part_att = DeepSetsAtt(
            num_feat=self.num_feat,
            time_embedding_dim=3*self.num_embed,
            num_heads=2,
            num_transformer=6,
            projection_dim=128,
            use_dist=False
        )

        self.model_jet_att = DeepSetsAtt(
            num_feat=self.num_jet,
            time_embedding_dim=2*self.num_embed,
            num_heads=2,
            num_transformer=6,
            projection_dim=128,
            use_dist=False
        )

        self.ema_jet = None
        self.ema_part = None



    def embed_time(self, t):
        # t: [B,1]
        emb = base_Embedding(t, self.projection) # [B, 2*num_embed] (since sin+cos)
        emb = self.activation(self.emb_fc1(emb))
        emb = self.activation(self.emb_fc2(emb)) # [B,num_embed]
        return emb



    def forward_conditional_embeddings(self, t, cond, jet):
        # t: [B,1], cond: [B,num_cond], jet: [B,num_jet]
        t_emb = self.embed_time(t)  # [B,num_embed]

        jet_dense = self.jet_fc1(jet)
        #jet_dense = self.activation(jet_dense)              # In the original code, this is not activated. Unintented ? 
        jet_dense = self.activation(self.jet_fc2(jet_dense)) # [B,num_embed]

        cond_ff = FF(cond) # [B, cond_ff_dim], we assumed cond_ff_dim = num_cond*9
        cond_dense = self.cond_fc1(cond_ff)
        #cond_dense = self.activation(cond_dense)              # In the original code, this is not activated. Unintented ?
        cond_dense = self.activation(self.cond_fc2(cond_dense)) # [B,num_embed]

        graph_concat = torch.cat([t_emb, jet_dense, cond_dense], dim=-1) # [B,3*num_embed]
        graph_conditional = self.activation(self.graph_fc(graph_concat)) # [B,3*num_embed]

        jet_concat = torch.cat([t_emb, cond_dense], dim=-1) # [B,2*num_embed]
        jet_conditional = self.activation(self.jet_cond_fc(jet_concat)) # [B,2*num_embed]

        return graph_conditional, jet_conditional



    def forward_part(self, part, t, jet, cond, mask):
        graph_conditional, jet_conditional = self.forward_conditional_embeddings(t, cond, jet)
        part_out = self.model_part_att(part, graph_conditional, mask=mask) # [B,N,num_feat]
        return part_out



    def forward_jet(self, jet, t, cond): 
        graph_conditional, jet_conditional = self.forward_conditional_embeddings(t, cond, jet)
        # jet: [B,num_jet] -> [B,1,num_jet]
        jet_inp = jet.unsqueeze(1)
        jet_out = self.model_jet_att(jet_inp, jet_conditional, mask=None) # [B,1,num_jet]
        jet_out = jet_out.squeeze(1) # [B,num_jet]
        return jet_out



    def forward(self, part, jet, cond, mask, t):
        part_pred = self.forward_part(part, t, jet, cond, mask)
        jet_pred = self.forward_jet(jet, t, cond)
        return jet_pred, part_pred


