import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.transformer_attn = nn.ModuleList([nn.MultiheadAttention(projection_dim, num_heads, batch_first=True) for _ in range(num_transformer)])
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

        if mask is None:
            mask = torch.ones(B, N, 1, dtype=inputs.dtype, device=inputs.device)

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
        mask_matrix = torch.matmul(mask, mask.transpose(1,2))  # [B,N,N]

        # Convert mask_matrix to attention mask
        # In TensorFlow, attention_mask=True means attend, False means don't.
        # In PyTorch MultiheadAttention, attn_mask is usually additive or boolean:
        # According to docs, attn_mask can be a bool tensor where True means NOT allowed.
        # We have mask_matrix=1 where allowed. We want True where disallowed: mask_out = mask_matrix==0
        # We'll use bool mask and rely on PyTorch 1.12+ that supports bool attn_mask with batch_first=True
        attn_mask = (mask_matrix == 0)  # True where we cannot attend

        # Transformer layers
        for i in range(self.num_transformer):
            # Norm
            x1 = self.transformer_norms_1[i](encoded_patches)  # [B, N, projection_dim]

            # Multi-head Attention
            # MultiheadAttention in PyTorch expects shape [B, N, D] if batch_first=True
            # attn_mask shape [B, N, N], True means no attention
            attn_output, _ = self.transformer_attn[i](x1, x1, x1, attn_mask=attn_mask)

            # Skip connection
            x2 = encoded_patches + attn_output

            # Second norm
            x3 = self.transformer_norms_2[i](x2)
            # MLP
            x3 = self.transformer_fc1[i](x3)
            x3 = F.gelu(x3)
            x3 = self.transformer_fc2[i](x3)
            x3 = F.gelu(x3)

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
