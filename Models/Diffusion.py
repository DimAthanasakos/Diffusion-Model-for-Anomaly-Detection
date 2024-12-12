import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
from Models.deepsets_cond import DeepSetsAtt
import time
import utils
import copy



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

    return torch.cat([features, h], dim=-1) # [B, D+2*D*num_freq]


def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    # t: [B,1]
    logsnr_max = torch.tensor(logsnr_max, dtype=torch.float32)  # Convert to tensor
    logsnr_min = torch.tensor(logsnr_min, dtype=torch.float32)  # Convert to tensor

    b = torch.atan(torch.exp(-0.5 * logsnr_max))
    a = torch.atan(torch.exp(-0.5 * logsnr_min)) - b
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
    def __init__(self, num_jet, num_cond, num_embed, projection, activation, device='cpu'):
        super().__init__()
        self.num_jet = num_jet
        self.num_cond = num_cond
        self.num_embed = num_embed
        self.device = device
        self.activation = activation
        self.projection = projection

        # Embedding layers for time
        self.emb_fc1 = nn.Linear(num_embed, 2*num_embed)
        self.emb_fc2 = nn.Linear(2*num_embed, num_embed)

        # Conditional layers for cond
        # cond_ff_dim = num_cond*(1+2*(max_proj-min_proj)) is handled outside or known:
        # Let's say min_proj=4, max_proj=8 => num_freq=4, 8 features per dim => cond_ff_dim = num_cond * 9
        self.cond_ff_dim = self.num_cond * 9
        self.cond_fc1 = nn.Linear(self.cond_ff_dim, 2*num_embed)
        self.cond_fc2 = nn.Linear(2*num_embed, num_embed)

        # jet conditional
        self.jet_cond_fc = nn.Linear(2*num_embed, 2*num_embed)

        # DeepSetsAtt for jets
        self.model_jet_att = DeepSetsAtt(
            num_feat=num_jet,
            time_embedding_dim=2*num_embed,
            num_heads=2,
            num_transformer=6,
            projection_dim=128,
            use_dist=False
        )

    def embed_time(self, t):
        # t: [B,1]
        emb = base_Embedding(t, self.projection) # [B,2*num_embed]
        emb = self.activation(self.emb_fc1(emb))
        emb = self.activation(self.emb_fc2(emb)) # [B,num_embed]
        return emb

    def forward(self, jet, t, cond):
        # jet: [B, num_jet], t: [B,1], cond: [B,num_cond]
        t = t.to(self.device)
        cond = cond.to(self.device)
        jet = jet.to(self.device)
        t_emb = self.embed_time(t)   # [B,num_embed]

        cond_ff = FF(cond, device=self.device) # [B, cond_ff_dim= num_cond*9]
        cond_dense = self.cond_fc1(cond_ff)
        # In original code no activation here after first fc?
        # Let's replicate TF logic exactly: 
        # The original code: cond_dense = layers.Dense(2*self.num_embed)(ff_cond)
        # and then cond_dense = self.activation(layers.Dense(self.num_embed)(cond_dense))
        # means we do have an activation after second fc only:
        cond_dense = self.activation(self.cond_fc2(cond_dense)) # [B,num_embed]

        jet_concat = torch.cat([t_emb, cond_dense], dim=-1) # [B,2*num_embed]
        jet_conditional = self.activation(self.jet_cond_fc(jet_concat)) # [B,2*num_embed]

        # jet: [B,num_jet], we need [B,N,D], N=1 for jets?
        # In TF code, model_jet presumably got inputs with shape [B,1,num_jet], then returned [B,1,num_jet].
        # Make sure to unsqueeze:
        #jet_inp = jet.unsqueeze(1) # [B,1,num_jet]

        jet_out = self.model_jet_att(jet, jet_conditional, mask=None) # [B,1,num_jet]
        jet_out = jet_out.squeeze(1) # [B,num_jet]

        return jet_out




class ModelPart(nn.Module):
    def __init__(self, num_feat, num_jet, num_cond, num_embed, max_part,
                 projection, activation, device='cpu'):
        super().__init__()
        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_cond = num_cond
        self.num_embed = num_embed
        self.max_part = max_part
        self.device = device
        self.activation = activation
        self.projection = projection

        # Time embedding layers
        self.emb_fc1 = nn.Linear(num_embed, 2*num_embed)
        self.emb_fc2 = nn.Linear(2*num_embed, num_embed)

        # Jet layers for part
        self.jet_fc1 = nn.Linear(self.num_jet, 2*self.num_embed)
        self.jet_fc2 = nn.Linear(2*self.num_embed, self.num_embed)

        self.cond_ff_dim = self.num_cond * 9  # same logic as above
        self.cond_fc1 = nn.Linear(self.cond_ff_dim, 2*self.num_embed)
        self.cond_fc2 = nn.Linear(2*self.num_embed, self.num_embed)

        self.graph_fc = nn.Linear(3*self.num_embed, 3*self.num_embed)

        # model_part_att
        self.model_part_att = DeepSetsAtt(
            num_feat=self.num_feat,
            time_embedding_dim=3*self.num_embed,
            num_heads=2,
            num_transformer=6,
            projection_dim=128,
            use_dist=False
        )

    def embed_time(self, t):
        emb = base_Embedding(t, self.projection) # [B,2*num_embed]
        emb = self.activation(self.emb_fc1(emb))
        emb = self.activation(self.emb_fc2(emb)) # [B,num_embed]
        return emb

    def forward(self, part, t, jet, cond, mask):
        # part: [B*2, N, num_feat] after reshaping outside this model or done inside?
        # The original code reshaped outside. Let's assume we get them pre-reshaped:
        # Actually, in TF code it was done inside. Let's keep consistent with original:
        # but we want a separate model that just expects the final shapes. 
        # For clarity, let's assume part, jet, cond, mask are already in the correct form.
        # If not, do the reshaping logic outside this model.
        
        t = t.to(self.device)
        cond = cond.to(self.device)
        jet = jet.to(self.device)
        mask = mask.to(self.device)
        part = part.to(self.device)

        t_emb = self.embed_time(t) # [B*2,num_embed]

        jet_dense = self.jet_fc1(jet)
        jet_dense = self.activation(self.jet_fc2(jet_dense)) # [B*2,num_embed]

        cond_ff = FF(cond, device=self.device) # [B*2, cond_ff_dim]
        cond_dense = self.cond_fc1(cond_ff)
        cond_dense = self.activation(self.cond_fc2(cond_dense)) # [B*2,num_embed]

        graph_concat = torch.cat([t_emb, jet_dense, cond_dense], dim=-1) # [B*2,3*num_embed]
        graph_conditional = self.activation(self.graph_fc(graph_concat)) # [B*2,3*num_embed]

        part_out = self.model_part_att(part, graph_conditional, mask=mask) # [B*2,N,num_feat]
        return part_out




class GSGM(nn.Module):
    def __init__(self, npart=100, config=None, device='cpu'):
        super().__init__()
        if config is None:
            raise ValueError("Config file not given")

        self.config = config
        self.activation = nn.SiLU()
        self.num_feat = config['NUM_FEAT']
        self.num_jet = config['NUM_JET']
        self.num_cond = config['NUM_COND']
        self.num_embed = config['EMBED']
        self.max_part = npart
        self.num_steps = config['MAX_STEPS']
        self.ema = 0.999
        self.shape=(-1,1,1)
        self.device = device

        self.projection = GaussianFourierProjection(self.num_embed, scale=16.0, device=self.device)

        # Provide FF and all needed functions to submodules
        self.model_jet = ModelJet(num_jet=self.num_jet,
                                  num_cond=self.num_cond,
                                  num_embed=self.num_embed,
                                  projection=self.projection,
                                  activation=self.activation,
                                  device=self.device)

        self.model_part = ModelPart(num_feat=self.num_feat,
                                    num_jet=self.num_jet,
                                    num_cond=self.num_cond,
                                    num_embed=self.num_embed,
                                    max_part=self.max_part,
                                    projection=self.projection,
                                    activation=self.activation,
                                    device=self.device)

        # EMA copies
        self.ema_jet = copy.deepcopy(self.model_jet)
        self.ema_part = copy.deepcopy(self.model_part)
        for p in self.ema_jet.parameters():  p.requires_grad = False
        for p in self.ema_part.parameters(): p.requires_grad = False


    def forward_jet(self, jet, t, cond):
        return self.model_jet(jet, t, cond)


    def forward_part(self, part, t, jet, cond, mask):
        return self.model_part(part, t, jet, cond, mask)


    def forward(self, part, jet, cond, mask, t): # NOTE: This forward requires the same t for both jet and part, which is not necessarily true during training
        jet_pred = self.forward_jet(jet, t, cond)
        part_pred = self.forward_part(part, t, jet, cond, mask)
        return jet_pred, part_pred


#----------------------------------------------------------------------------------------------

    def generate(self, cond):
        start = time.time()
        # move cond to torch if not already
        cond = torch.tensor(cond, dtype=torch.float32, device=self.device)
        self.ema_jet = self.ema_jet.to(self.device)
        self.ema_part = self.ema_part.to(self.device)
        #print(f'cond.device={cond.device}')

        # cond: torch.Tensor of shape [B, num_cond]
        # data_shape=[cond.shape[0],2,self.num_jet]
        # Use the DDPMSampler to generate jets
        with torch.no_grad():
            #print(f'self.ema_jet.device={self.ema_jet.device}')
            #self.ema_jet = self.ema_jet.to(self.device)
            #for name, param in self.ema_jet.named_parameters():
            #    print(f"{name}: {param.device}")
            jets = self.DDPMSampler(cond, self.ema_jet,
                                    data_shape=[cond.shape[0], 2, self.num_jet],
                                    const_shape=self.shape)
        jets = jets.cpu().numpy()
        
        particles = []
        for ijet in range(2):
            jet_info = jets[:, ijet]  # shape: [B, num_jet]
            nparts = np.expand_dims(
                np.clip(
                    utils.revert_npart(jet_info[:, -1], self.max_part, norm=self.config['NORM']),
                    1, self.max_part
                ), -1
            )

            # Create mask
            # mask: [B, max_part, 1]
            # np.tile to replicate np.arange(self.max_part) for each sample
            indices = np.arange(self.max_part)
            mask_bool = indices < nparts  # broadcasting comparison
            mask = np.expand_dims(mask_bool, -1)  # [B, max_part, 1]

            # Check assertion
            assert np.sum(np.sum(mask.reshape(mask.shape[0], -1), -1, keepdims=True) - nparts) == 0, \
                'ERROR: Particle mask does not match the expected number of particles'

            # Convert cond, jet_info, mask to torch for sampling
            cond_torch = cond if isinstance(cond, torch.Tensor) else torch.tensor(cond, dtype=torch.float32, device=self.device)
            jet_torch = torch.tensor(jet_info, dtype=torch.float32, device=self.device)
            mask_torch = torch.tensor(mask, dtype=torch.float32, device=self.device)

            # data_shape=[B,max_part,num_feat]
            with torch.no_grad():
                parts = self.DDPMSampler(
                    cond_torch,
                    self.ema_part,
                    data_shape=[cond.shape[0], self.max_part, self.num_feat],
                    jet=jet_torch,
                    const_shape=self.shape,
                    mask=mask_torch
                )
            parts = parts.cpu().numpy()
            particles.append(parts * mask)  # Apply mask

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0], end - start))

        # Stack along the jet dimension: result shape [B,2,max_part,num_feat]
        return np.stack(particles, 1), jets



    def second_order_correction(self, time_step, x, pred_images, pred_noises,
                                alphas, sigmas,
                                cond, model, jet=None, mask=None,
                                second_order_alpha=0.5):
        # time_step, alphas, sigmas are torch.Tensors
        # pred_images, pred_noises, x, cond are torch.Tensors

        step_size = 1.0 / self.num_steps
        with torch.no_grad():
            logsnr, alpha_signal_rates, alpha_noise_rates = get_logsnr_alpha_sigma(
                time_step - second_order_alpha * step_size
            )
        
        alpha_noisy_images = alpha_signal_rates * pred_images + alpha_noise_rates * pred_noises

        with torch.no_grad():
            if jet is None:
                score = model(alpha_noisy_images, time_step - second_order_alpha * step_size, cond)
            else:
                alpha_noisy_images = alpha_noisy_images * mask
                score = model(alpha_noisy_images, time_step - second_order_alpha * step_size, jet, cond, mask)
                score = score * mask if mask is not None else score

        alpha_pred_noises = alpha_noise_rates * alpha_noisy_images + alpha_signal_rates * score

        # Linearly combine the two noise estimates
        pred_noises = (1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises + \
                      (1.0 / (2.0 * second_order_alpha)) * alpha_pred_noises

        mean = (x - sigmas * pred_noises) / alphas
        eps = pred_noises

        return mean, eps


    def DDPMSampler(self, cond, model, data_shape=None, const_shape=None, jet=None, mask=None, clip=False, second_order=True):
        # cond: torch.Tensor [B, cond_dim]
        device = self.device
        batch_size = cond.shape[0]

        x = prior_sde(data_shape).to(device)

        with torch.no_grad():
            for time_step in range(self.num_steps, 0, -1):
                # create random_t: [B,1]
                random_t = torch.ones((batch_size, 1), dtype=torch.float32, device=device) * (time_step / self.num_steps)
                logsnr, alpha, sigma = get_logsnr_alpha_sigma(random_t)
                logsnr_, alpha_, sigma_ = get_logsnr_alpha_sigma(
                    torch.ones((batch_size, 1), dtype=torch.float32, device=device) * ((time_step - 1) / self.num_steps)
                )

                if jet is None:
                    score = model(x, random_t, cond)  # score shape: [B,N,D]
                else:
                    x = x * mask
                    score = model(x, random_t, jet, cond, mask) * mask

                # compute mean and eps
                mean = alpha * x - sigma * score
                eps = sigma * x + alpha * score

                if second_order:
                    mean, eps = self.second_order_correction(
                        random_t, x, mean, eps,
                        alpha, sigma,
                        cond, model, jet, mask
                    )

                x = alpha_ * mean + sigma_ * eps

            # The last step does not include noise, so x is final
            return x
