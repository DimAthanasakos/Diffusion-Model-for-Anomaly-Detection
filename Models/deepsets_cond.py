import torch
import torch.nn as nn
import torch.nn.functional
import time
import sys

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
        #print(f'defining transformer layers')
        #print(f'num_feat={num_feat}, time_embedding_dim={time_embedding_dim}, projection_dim={projection_dim}')
        #print(f'num_heads={num_heads}, num_transformer={num_transformer}')
        #print()
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
            #print(f'mask.shape={mask.shape}')
            #print()
            #print(f'mask[0,:55]: {mask[0, :55]}')
            mask_matrix = torch.matmul(mask, mask.transpose(1,2))  # [B,N,N]
            #print(f'mask_matrix.shape={mask_matrix.shape}')
            #print()
            #print(f'mask_matrix[0, :55, :55]: {mask_matrix[0, :55, :55]}')
            # Pytorch MultiheadAttention expects attn_mask to be of shape [batch_size * num_heads, ...]
            # so we repeat the mask_matrix num_heads times
            mask_matrix = mask_matrix.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask_matrix = mask_matrix.view(-1, mask_matrix.size(-2), mask_matrix.size(-1))
            #print(f'mask_matrix.shape={mask_matrix.shape}')            
            #print()
            #print(f'mask_matrix[0, :55, :55]: {mask_matrix[0, :55, :55]}')

            # Convert mask_matrix to attention mask
            attn_mask = mask_matrix.clone()
            attn_mask[attn_mask != 0] = float('-inf')
            attn_mask[attn_mask == 0] = 0.0

            
        else:
            attn_mask = None

        #print(f'Forward DeepSetsAtt: attn_mask.shape={attn_mask.shape}')

        # Transformer layers
        for i in range(self.num_transformer):
            # Norm
            #print(f'=================')
            #print(f'transformer layer {i}')
            #print(f'encoded_patches.shape={encoded_patches.shape}')
            #print(f'encoded_patches[:1, :5, :5]: {encoded_patches[:1, :5, :5]}')
            #print()
            x1 = self.transformer_norms_1[i](encoded_patches)  # [B, N, projection_dim]
            #print(f'x1.shape={x1.shape}')
            #print(f'x1[:1, :5, :5]: {x1[:1, :5, :5]}')

            # Multi-head Attention
            attn_output, _ = self.transformer_attn[i](query=x1, key=x1, value=x1, attn_mask=attn_mask)

            if False and torch.isnan(attn_output).any():
                print()
                print('WARNING')
                print(f'attn_output has NaNs')
                print()            
                # Create a boolean mask where True indicates a NaN
                nan_mask = torch.isnan(attn_output)

                # Get the indices of NaN values
                nan_indices = nan_mask.nonzero(as_tuple=False)

                # Print the number of NaNs found
                num_nans = nan_indices.size(0)
                print(f"Total NaN values found in attn_output: {num_nans}")
                print(f'total number of elements in attn_output: {attn_output.numel()}')
                print()
                print(f"Indices of NaN values:")
                print(nan_indices)
                nan_mask0 = torch.isnan(attn_output[0])

                # Get the indices of NaN values
                nan_indices = nan_mask0.nonzero(as_tuple=False)

                # Print the number of NaNs found
                num_nans = nan_indices.size(0)
                print(f"Total NaN values found in attn_output[0]: {num_nans}")
                print(f'total number of elements in attn_output[0]: {attn_output[0].numel()}')
                print(f'indices of NaN values in attn_output[0]:')
                print(nan_indices)

                # how many False values in attn_mask, print the indices 
                attn_mask0 = attn_mask[0]
                print(f'attn_mask0.shape={attn_mask0.shape}')
                n_false = torch.sum(attn_mask0 == False)
                n_true = torch.sum(attn_mask0 == True)
                print(f'number of False values in attn_mask[0]: {n_false}')
                print(f'number of True values in attn_mask[0]: {n_true}')
                print()
                
                true_indices = (attn_mask0 == True).nonzero(as_tuple=False)
                print(f'indices of True values in attn_mask[0]:')
                print(true_indices)


                print()
                print()
                nan_mask1 = torch.isnan(attn_output[1])

                # Get the indices of NaN values
                nan_indices = nan_mask1.nonzero(as_tuple=False)

                # Print the number of NaNs found
                num_nans = nan_indices.size(0)
                print(f"Total NaN values found in attn_output[0]: {num_nans}")
                print(f'total number of elements in attn_output[0]: {attn_output[1].numel()}')
                print(f'indices of NaN values in attn_output[0]:')
                print(nan_indices)

                # how many False values in attn_mask, print the indices 
                attn_mask1 = attn_mask[1]
                print(f'attn_mask0.shape={attn_mask1.shape}')
                n_false = torch.sum(attn_mask1 == False)
                n_true = torch.sum(attn_mask1 == True)
                print(f'number of False values in attn_mask[0]: {n_false}')
                print(f'number of True values in attn_mask[0]: {n_true}')
                print()
                
                true_indices = (attn_mask1 == True).nonzero(as_tuple=False)
                print(f'indices of True values in attn_mask[1]]:')
                print(true_indices)

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
