import torch
import torch.nn as nn
import torch.nn.functional
import time


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
       
        print()
        print(f'===================================================')
        print(f'Forward DeepSetsAtt: inputs.shape={inputs.shape}')
        print()
        
    
        # Apply initial projection to inputs
        x = self.input_fc(inputs)  # [B, N, projection_dim]
        x = self.input_act(x)
        print(f'Forward DeepSetsAtt: x.shape={x.shape}')
        print(f'x[:1, :5, :5]: {x[:1,  :5, :5]}')

        # Process time embedding
        t = self.time_fc(time_embedding) # [B, projection_dim]
        t = self.time_act(t)
        print(f'Forward DeepSetsAtt: t.shape={t.shape}')
        print(f't[:1, :5]: {t[:1, :5]}')

        # Repeat time along N dimension
        t = t.unsqueeze(1).repeat(1, N, 1)  # [B, N, projection_dim]

        # Concat masked_features and time
        concat = torch.cat([x, t], dim=-1)  # [B, N, 2*projection_dim]
        concat = self.concat_fc1(concat)
        concat = self.concat_act1(concat)
        tdd = self.concat_fc2(concat)  # [B, N, projection_dim]

        print(f'Forward DeepSetsAtt: tdd.shape={tdd.shape}')
        print(f'tdd[:1, :5,  :5]: {tdd[:1, :5,  :5 ]}')
        print()

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
            attn_mask = (mask_matrix == 0)  # True where we cannot attend
            print(f'Forward DeepSetsAtt: attn_mask.shape={attn_mask.shape}')
            print(f'attn_mask[:1, :5, :5]: {attn_mask[:1, :5, :5]}')
            print()
            # check if attn__mask has any true values 
            if attn_mask.any():
                print()
                print('WARNING')
                print(f'attn_mask has True values')
                print()
            if torch.isnan(attn_mask).any():
                print()
                print('WARNING')
                print(f'attn_mask has NaNs')
                print()
            
        else:
            attn_mask = None

        #print(f'Forward DeepSetsAtt: attn_mask.shape={attn_mask.shape}')

        # Transformer layers
        for i in range(self.num_transformer):
            # Norm
            print(f'=================')
            print(f'transformer layer {i}')
            print(f'encoded_patches.shape={encoded_patches.shape}')
            print(f'encoded_patches[:1, :5, :5]: {encoded_patches[:1, :5, :5]}')
            print()
            x1 = self.transformer_norms_1[i](encoded_patches)  # [B, N, projection_dim]
            print(f'x1.shape={x1.shape}')
            print(f'x1[:1, :5, :5]: {x1[:1, :5, :5]}')
            if torch.isnan(x1).any():
                print()
                print('WARNING')
                print(f'x1 has NaNs')
                print()
            if torch.isinf(x1).any():
                print()
                print('WARNING')
                print(f'x1 has Infs')
                print()

            # Multi-head Attention
            attn_output, _ = self.transformer_attn[i](query=x1, key=x1, value=x1, attn_mask=attn_mask)
            print()
            print(f'attn_output.shape={attn_output.shape}')
            print(f'attn_output[:1, :5, :5]: {attn_output[:1, :5, :5]}')

            if torch.isnan(attn_output).any():
                print()
                print('WARNING')
                print(f'attn_output has NaNs')
                print()

            # Skip connection
            x2 = encoded_patches + attn_output

            # Second norm
            x3 = self.transformer_norms_2[i](x2)
            print()
            print(f'x3.shape={x3.shape}')
            print(f'x3[:1, :5, :5]: {x3[:1, :5, :5]}')

            if torch.isnan(x3).any():
                print()
                print('WARNING')
                print(f'x3 has NaNs')
                print()

            # MLP
            x3 = self.transformer_fc1[i](x3)
            print()
            print(f'x3.shape={x3.shape}')
            print(f'x3[:1, :5, :5]: {x3[:1, :5, :5]}')


            x3 = torch.nn.functional.gelu(x3) 
            x3 = self.transformer_fc2[i](x3)
            x3 = torch.nn.functional.gelu(x3)
            print()
            print(f'x3.shape={x3.shape}')
            print(f'x3[:1, :5, :5]: {x3[:1, :5, :5]}')
            if torch.isnan(x3).any():
                print()
                print('WARNING')
                print(f'x3 has NaNs')
                print()
            
            #if i == 0:
            #    time.sleep(5)

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
