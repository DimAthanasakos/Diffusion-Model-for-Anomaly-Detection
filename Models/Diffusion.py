import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
from Models.deepsets_cond import DeepSetsAtt
import time



torch.manual_seed(1233)

######################################
# Helper Functions
######################################
def GaussianFourierProjection(num_embed, scale=16.0, device='cpu'):
    half_dim = num_embed // 2
    emb = torch.log(torch.tensor(10000.0, dtype=torch.float32, device=device)) / (half_dim - 1)
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
    #print(f'FF: features shape: {features.shape}')
    # features is shape (batch_size,) when num_cond = 1, we need to transform it to (batch_size, 1)
    if len(features.shape) == 1:
        features = features.unsqueeze(1)
    features = features.to(dtype=torch.float32, device=device)
    #print(f'FF: features shape: {features.shape}')
    freq = torch.arange(min_proj, max_proj, dtype=torch.float32, device=device)
    freq = (2.0**freq) * 2 * np.pi  # [num_freq]
    num_freq = max_proj - min_proj

    # Tile freq along features dimension:
    # In TF: freq = tf.tile(freq[None, :], (1, tf.shape(x)[-1]))
    # In PyTorch: we need to repeat freq so that it matches the number of features
    # freq: [num_freq]
    
    d = features.shape[-1]
    freq = freq.unsqueeze(0)  # [1,num_freq]
    # repeat features along last dimension num_freq times:
    h = features.repeat(1, num_freq)  # [B, D*num_freq]

    # Now we must ensure freq is matched with h:
    # freq should be repeated d times to match h's dimension:
    freq = freq.repeat(1, d)  # [1, num_freq*D]
    # broadcast angle computation
    angle = h * freq  # [B, D*num_freq]

    # sinusoidal expansions
    h = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # [B, 2*D*num_freq]
    #print(f'FF: features shape: {features.shape}, h shape: {h.shape}')
    #print()

    return torch.cat([features, h], dim=-1) # [B, D+2*D*num_freq]





class GSGM(nn.Module):
    def __init__(self, npart=100, config=None, device='cpu'):
        super().__init__()
        if config is None:
            raise ValueError("Config file not given")
        
        self.config = config
        self.activation = nn.SiLU()
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
        self.emb_fc1 = nn.Linear(self.num_embed, 2*self.num_embed)
        self.emb_fc2 = nn.Linear(2*self.num_embed, self.num_embed)

        # Layers equivalent to Keras Dense layers for conditional jet inputs
        self.jet_fc1 = nn.Linear(self.num_jet, 2*self.num_embed)
        self.jet_fc2 = nn.Linear(2*self.num_embed, self.num_embed)

        # FF layers for cond
        # We must know shape after FF; 
        cond_ff_dim = self.num_cond * (1 + 2*(self.max_proj-self.min_proj))
        #print(f'cond_ff_dim: {cond_ff_dim}')
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
        # t: [B,1], cond: [B,num_cond], jet: [B,num_jet]\
        #print()
        #print(f'forward_conditional_embeddings: t shape: {t.shape}, cond shape: {cond.shape}, jet shape: {jet.shape}')
        #print()
        t_emb = self.embed_time(t)   # this is the graph_conditional = jet_conditional of the original code
        #print(f't_emb shape: {t_emb.shape}')

        jet_dense = self.jet_fc1(jet)
        #jet_dense = self.activation(jet_dense)              # In the original code, this is not activated. Unintended ? 
        jet_dense = self.activation(self.jet_fc2(jet_dense)) # [B,num_embed]

        cond_ff = FF(cond, device=self.device) # [B, cond_ff_dim], we assumed cond_ff_dim = num_cond*9
        cond_dense = self.cond_fc1(cond_ff)
        #cond_dense = self.activation(cond_dense)              # In the original code, this is not activated. Unintended ?
        cond_dense = self.activation(self.cond_fc2(cond_dense)) # [B,num_embed]

        #print(f't_emb shape: {t_emb.shape}, jet_dense shape: {jet_dense.shape}, cond_dense shape: {cond_dense.shape}')
        graph_concat = torch.cat([t_emb, jet_dense, cond_dense], dim=-1) # [B,3*num_embed]
        graph_conditional = self.activation(self.graph_fc(graph_concat)) # [B,3*num_embed]

        jet_concat = torch.cat([t_emb, cond_dense], dim=-1) # [B,2*num_embed]
        #print(f't_emb shape: {t_emb.shape}, cond_dense shape: {cond_dense.shape}')
        #print(f'jet_concat shape: {jet_concat.shape}')
        jet_conditional = self.activation(self.jet_cond_fc(jet_concat)) # [B,2*num_embed]

        return graph_conditional, jet_conditional



    def forward_part(self, part, t, jet, cond, mask):
        #print()
        #print(f'forward_part: part shape: {part.shape}, t shape: {t.shape}, jet shape: {jet.shape}, cond shape: {cond.shape}, mask shape: {mask.shape}')
        #print()
        # ensure that everything is on the same device 
        t = t.to(self.device)
        cond = cond.to(self.device)
        jet = jet.to(self.device)
        mask = mask.to(self.device)
        part = part.to(self.device)

        # we need to split the 2 jets for each event and concat them along the batch dimension. Do this for all input variables
        part = part.reshape(-1, self.max_part, self.num_feat) 
        jet = jet.reshape(-1, self.num_jet)  
        mask = mask.reshape(-1, self.max_part, 1)  
        # expand cond along the batch dimension

        #print(f'part shape: {part.shape}')
        #print(f'jet shape: {jet.shape}')
        #print(f'mask shape: {mask.shape}')
        #print(f'cond shape: {cond.shape}')
        #cond = cond.repeat(2, 1)  # Shape: (2B, num_cond)

        #cond = cond.unsqueeze(1)

        #print(f'cond shape: {cond.shape}')
        #print(f'self.num_cond = {self.num_cond}')
        #cond = cond.repeat(1,2).reshape(-1, self.num_cond) 

        t_emb = self.embed_time(t) 
        jet_dense = self.jet_fc1(jet)
        #jet_dense = self.activation(jet_dense)              # In the original code, this is not activated. Unintented ? 
        jet_dense = self.activation(self.jet_fc2(jet_dense)) # [B,num_embed]

        cond_ff = FF(cond, device=self.device) # [B, cond_ff_dim], we assumed cond_ff_dim = num_cond*9

        cond_dense = self.cond_fc1(cond_ff)

        #cond_dense = self.activation(cond_dense)              # In the original code, this is not activated. Unintented ?
        cond_dense = self.activation(self.cond_fc2(cond_dense)) # [B,num_embed]

        graph_concat = torch.cat([t_emb, jet_dense, cond_dense], dim=-1) # [B,3*num_embed]
        graph_conditional = self.activation(self.graph_fc(graph_concat)) # [B,3*num_embed]
        
        part_out = self.model_part_att(part, graph_conditional, mask=mask) # [B,N,num_feat]

        return part_out



    def forward_jet(self, jet, t, cond): 
        # ensure that everything is on the same device 
        t = t.to(self.device)
        cond = cond.to(self.device)
        jet = jet.to(self.device)
        
        t_emb = self.embed_time(t)   # this is the graph_conditional = jet_conditional of the original code

        cond_ff = FF(cond, device=self.device) # [B, cond_ff_dim], we assumed cond_ff_dim = num_cond*9
        cond_dense = self.cond_fc1(cond_ff)
        #cond_dense = self.activation(cond_dense)              # In the original code, this is not activated. Unintented ?
        cond_dense = self.activation(self.cond_fc2(cond_dense)) # [B,num_embed]

        jet_concat = torch.cat([t_emb, cond_dense], dim=-1) 

        jet_conditional = self.activation(self.jet_cond_fc(jet_concat)) # [2*B,2*num_embed]

        jet_out = self.model_jet_att(jet, jet_conditional, mask=None) # [B,1,num_jet]
        jet_out = jet_out.squeeze(1) # [B,num_jet]

        return jet_out



    def forward(self, part, jet, cond, mask, t):
        jet_pred = self.forward_jet(jet, t, cond)
        part_pred = self.forward_part(part, t, jet, cond, mask)
        return jet_pred, part_pred


