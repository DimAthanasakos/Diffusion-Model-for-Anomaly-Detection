import torch
import torch.nn as nn
import torch.nn.functional
import time
import sys
import torch.nn.functional as F

torch.manual_seed(1233)

class DeepSetsAtt(nn.Module):
    def __init__(self,
                 num_feat,
                 time_embedding_dim,
                 num_heads=4,
                 num_transformer=4,
                 projection_dim=32,
                 use_dist=False):
        super().__init__()
        
        self.num_feat = num_feat
        self.num_heads = num_heads
        self.num_transformer = num_transformer
        self.projection_dim = projection_dim
        self.use_dist = use_dist

        # Layers equivalent to TimeDistributed(Dense(...)) + LeakyReLU
        # First steps (masked_features)
        self.input_fc = nn.Linear(num_feat, projection_dim)
        self.input_act = nn.LeakyReLU(0.01)

        # Time embedding layers
        self.time_fc = nn.Linear(time_embedding_dim, projection_dim)
        self.time_act = nn.LeakyReLU(0.01)

        # Combined dense after concatenating features+time
        self.concat_fc1 = nn.Linear(projection_dim*2, projection_dim)
        self.concat_act1 = nn.LeakyReLU(0.01)
        self.concat_fc2 = nn.Linear(projection_dim, projection_dim)

        # Transformer blocks
        # Define a series of ModuleLists where each index corresponds to a particular transformer layer
        self.transformer_norms_1 = nn.ModuleList([nn.LayerNorm(projection_dim) for _ in range(num_transformer)])
        self.transformer_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, batch_first=True) for _ in range(num_transformer)])
        self.transformer_norms_2 = nn.ModuleList([nn.LayerNorm(projection_dim) for _ in range(num_transformer)])
        self.transformer_fc1 = nn.ModuleList([nn.Linear(projection_dim, 4*projection_dim) for _ in range(num_transformer)])
        self.transformer_fc2 = nn.ModuleList([nn.Linear(4*projection_dim, projection_dim) for _ in range(num_transformer)])

        # Final representation
        self.final_norm = nn.LayerNorm(projection_dim)
        self.final_concat = nn.Linear(projection_dim + projection_dim, 2*projection_dim)
        self.final_act = nn.LeakyReLU(0.01)
        self.output_fc = nn.Linear(2*projection_dim, num_feat)


    def forward(self, inputs, time_embedding, mask=None):
        """
        inputs: [B, N, num_feat]
        time_embedding: [B, time_embedding_dim] (embedding vector for time)
        mask: [B, N, 1] binary mask where 1 means valid, 0 means padded
        """
        B, N, F = inputs.shape
       
        # Apply initial projection to inputs
        x = self.input_fc(inputs)  # [B, N, projection_dim]
        x = self.input_act(x)

        # Process time embedding
        t = self.time_fc(time_embedding) # [B, projection_dim]
        t = self.time_act(t)

        # Repeat time along N dimension
        t = t.unsqueeze(1).repeat(1, N, 1)  # [B, N, projection_dim]

        # Concat masked_features and time
        concat = torch.cat([x, t], dim=-1)  # [B, N, 2*projection_dim]
        concat = self.concat_fc1(concat)
        concat = self.concat_act1(concat)
        tdd = self.concat_fc2(concat)  # [B, N, projection_dim]

        # encoded_patches starts as tdd
        encoded_patches = tdd

        # Create mask matrix [B, N, N]
        # Original code: mask_matrix = mask * mask.transpose
        # mask: [B,N,1], mask.transpose: [B,1,N], product: [B,N,N]
        if mask is not None:
            mask_matrix = torch.matmul(mask, mask.transpose(1,2))  # [B,N,N]
            # Pytorch MultiheadAttention expects attn_mask to be of shape [batch_size * num_heads, ...]
            # so we repeat the mask_matrix num_heads times
            mask_matrix = mask_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask_matrix = mask_matrix.view(-1, mask_matrix.size(-2), mask_matrix.size(-1))

            # Convert mask_matrix to attention mask
            attn_mask = mask_matrix.clone()
            attn_mask[attn_mask != 0] = float('-inf')
            attn_mask[attn_mask == 0] = 0.0
            
        else:
            attn_mask = None

        # Transformer layers
        for i in range(self.num_transformer):
            x1 = self.transformer_norms_1[i](encoded_patches)  # [B, N, projection_dim]

            # Multi-head Attention
            attn_output, _ = self.transformer_attn[i](query=x1, key=x1, value=x1, attn_mask=attn_mask)

            # Skip connection
            x2 = encoded_patches + attn_output

            # Second norm
            x3 = self.transformer_norms_2[i](x2)
            
            # MLP
            x3 = self.transformer_fc1[i](x3)
            
            x3 = torch.nn.functional.gelu(x3) 
            x3 = self.transformer_fc2[i](x3)
            x3 = torch.nn.functional.gelu(x3)

            # Another skip connection
            encoded_patches = x2 + x3

        # Final representation
        representation = self.final_norm(encoded_patches)

        # concat with tdd (like original code)
        add = torch.cat([tdd, representation], dim=-1) # [B,N,projection_dim+projection_dim]
        representation = self.final_concat(add)
        representation = self.final_act(representation)
        outputs = self.output_fc(representation)  # [B, N, num_feat]

        return outputs



def make_patches(inputs, projection_dim):
    l1 = nn.Linear(inputs.size(-1), projection_dim).to(inputs.device)
    l2 = nn.Linear(projection_dim, projection_dim).to(inputs.device)
    tdd = l1(inputs)
    tdd = F.leaky_relu(tdd, negative_slope=0.01)
    encoded_patches = l2(tdd)
    return encoded_patches


def encode(inputs, projection_dim):
    l = nn.Linear(inputs.size(-1), projection_dim).to(inputs.device)
    masked_features = l(inputs)
    masked_features = F.leaky_relu(masked_features, negative_slope=0.01)
    return masked_features


def transformer(encoded_patches, num_transformer, num_heads, projection_dim, mask_matrix=None):
    for _ in range(num_transformer):
        # Layer normalization 1
        norm1 = nn.LayerNorm(encoded_patches.size(-1)).to(encoded_patches.device) 
        x1 = norm1(encoded_patches)

        # Multi-head attention
        attn_layer = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, dropout=0.1, batch_first=True).to(encoded_patches.device)
        attention_output, _ = attn_layer(x1, x1, x1, attn_mask=mask_matrix)

        # Skip connection 1
        x2 = x1 + attention_output

        # Layer normalization 2
        norm2 = nn.LayerNorm(x2.size(-1)).to(encoded_patches.device)
        x3 = norm2(x2)
        l1 = nn.Linear(projection_dim, 4 * projection_dim).to(encoded_patches.device)
        x3 = l1(x3)
        x3 = F.gelu(x3)
        l2 = nn.Linear(4 * projection_dim, projection_dim).to(encoded_patches.device)
        x3 = l2(x3)
        x3 = F.gelu(x3)

        # Skip connection 2
        encoded_patches = x2 + x3
    last_norm = nn.LayerNorm(encoded_patches.size(-1)).to(encoded_patches.device)
    representation = last_norm(encoded_patches)
    return representation


class DeepSetsClass(nn.Module):
    def __init__(self, num_heads=1, num_transformer=8, projection_dim=256, use_cond=False, device='cpu'):
        super(DeepSetsClass, self).__init__()
        self.num_heads = num_heads
        self.num_transformer = num_transformer
        self.projection_dim = projection_dim
        self.use_cond = use_cond
        self.device = device

        # Conditional embedding layer (if use_cond is True)
        if use_cond:
            self.cond_fc = nn.Linear(projection_dim, projection_dim)

        # Output layers
        self.merged_fc1 = nn.Linear(projection_dim, 2*projection_dim)     # change 1 to 2*projection_dim
        self.merged_fc2 = nn.Linear(2 * projection_dim, projection_dim)
        self.merged_fc3 = nn.Linear(projection_dim, 1)


    def forward(self, inputs_jet, inputs_particle, mask=None, cond_embedding=None):
        inputs_jet = inputs_jet.to(self.device)
        inputs_particle = inputs_particle.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        
        npart = inputs_particle.size(2)

        # Reshape inputs and mask
        inputs_reshape = inputs_particle.view(-1, npart, inputs_particle.size(-1))

        if mask is not None:
            mask_matrix = torch.matmul(mask, mask.transpose(1,2))  # [B,N,N]
            # Pytorch MultiheadAttention expects attn_mask to be of shape [batch_size * num_heads, ...]
            # so we repeat the mask_matrix num_heads times
            mask_matrix = mask_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask_matrix = mask_matrix.view(-1, mask_matrix.size(-2), mask_matrix.size(-1))

            # Convert mask_matrix to attention mask
            attn_matrix = mask_matrix.clone()
            attn_matrix[attn_matrix != 0] = float('-inf')
            attn_matrix[attn_matrix == 0] = 0.0
        else:
            attn_matrix = None

        mask_reshape = mask.view(-1, npart, 1) if mask is not None else None
        # Encode masked features and jet features
        masked_features = encode(inputs_reshape, self.projection_dim)
        jet_features = encode(inputs_jet, self.projection_dim)


        # Create mask matrix for attention
        if mask_reshape is not None:
            mask_matrix = torch.matmul(mask_reshape, mask_reshape.transpose(1, 2))
            mask_matrix = mask_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask_matrix = mask_matrix.view(-1, mask_matrix.size(-2), mask_matrix.size(-1))
            # Convert mask_matrix to attention mask
            attn_matrix = mask_matrix.clone()
            attn_matrix[attn_matrix != 0] = float('-inf')
            attn_matrix[attn_matrix == 0] = 0.0
        else:
            attn_matrix = None


        # Add conditional embedding if use_cond is True
        if self.use_cond and cond_embedding is not None:
            cond = self.cond_fc(cond_embedding)
            cond = F.leaky_relu(cond, negative_slope=0.01)
            cond = cond.unsqueeze(1).repeat(1, npart, 1)
            masked_features = torch.cat([masked_features, cond], dim=-1)

        # Transformer encoding for particles
        encoded_patches = make_patches(masked_features, self.projection_dim)
        representation = transformer(encoded_patches, self.num_transformer, self.num_heads, self.projection_dim, attn_matrix)

        # right now representation is [B*num_particles, n_jets = 2, projection_dim]
        # we want to reshape it to [B, num_particles*n_jets, projection_dim]
        representation = representation.view(-1, 2*npart, self.projection_dim)


        # Transformer encoding for jets
        encoded_patches_jet = make_patches(jet_features, self.projection_dim)
        representation_jet = transformer(encoded_patches_jet, self.num_transformer, self.num_heads, self.projection_dim)
        # reshape representation and representation_jet ?? 

        # Merge representations
        merged = torch.cat([representation, representation_jet], dim=-2)

        merged = self.merged_fc1(merged)
        merged = F.dropout(merged, p=0.1, training=self.training) # self.training is a boolean flag toggled by model.train() and model.eval()
        merged = F.leaky_relu(merged, negative_slope=0.01)
        merged = torch.mean(merged, dim=1)  # Global average pooling

        # Final output
        merged = self.merged_fc2(merged)
        merged = F.leaky_relu(merged, negative_slope=0.01)
        merged = F.dropout(merged, p=0.1, training=self.training)
        
        outputs = torch.sigmoid(self.merged_fc3(merged))

        return outputs
