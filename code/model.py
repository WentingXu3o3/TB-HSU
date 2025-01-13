# MIT License

# Copyright (c) 2025 Wenting Xu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PositionalEmbedding import OBB
# from clip import clip
# from clip.model import CLIP
from einops import rearrange, repeat

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("dropout", nn.Dropout(dropout))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.attn_weight = None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # attn_outputs, attn_weight = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask) 
        # print(attn_weight.shape) #torch.Size([batchsize, 121, 121]) #attn_weight only not nonetype when need_weights = True 
        # print("attention_outputs ", attn_outputs.shape) # [121,batchsize,512]
        # attn_weight = attn_weight.squeeze(0).cpu().detach().numpy() # 121,121
        # print("attention_outputs[0] ", att_outputs[0].shape) # [121,512]
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        # return attn_outputs, attn_weight
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
        #if need_weights is True [1] will not be nonetype
     
    def forward(self, x: torch.Tensor):
        x = x.to(dtype=torch.float32)  # Example: Convert to float32
        # print(x.shape) #torch.Size([batchsize, 121, 512])
        attn_outputs, self.attn_weight = self.attention(self.ln_1(x))
        x = x + attn_outputs
        # self.attention(self.ln_1(x)) #size(batchsize, num_objects, num_objects)?
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])
        # self.attention_matrices = []
    def forward(self, x: torch.Tensor):
        # print(self.resblocks)
        # return self.resblocks(x), None
        attention_matrices = []
        for layer in self.resblocks:
            x = layer(x)
            # self.attention_matrices.append(layer.attn_weight.squeeze(0).cpu().detach().numpy())
            attention_matrices.append(layer.attn_weight)
        ##### you shouldn't do the self.attention_matirces append. it will save all layers attn_matrix till the end of the training.
        return x, attention_matrices

class H3DSG_model(nn.Module):
    def __init__(self, config,  num_classes,num_areas= None,
                  embed_dim = 512, 
                  transformer_width = 512, transformer_layers=4, 
                #   transformer_heads=8, 
                    num_objects = 77,
                  text_embedding_context_length = 77,
                  text_embedding_vocab_size = 49408,
                  text_embedding_transformer_width = 512,
                #   text_embedding_transformer_heads = text_embedding_transformer_width/64,
                  text_embedding_layers = 3,
                  positional_embedding_in_channel = 24,
                  pool = 'mean',
                  semantic_embedding_is = True,
                  positional_embedding_is = True,
                  baseline_is = False,
                  zero_shot = False,
                  dropout = 0,
                  baseline_model = 'CNN'
        ):
        super(H3DSG_model, self).__init__()

        self.config = config
        if semantic_embedding_is:
            if config.SE_type == 'nnEmbedding':
                self.semantic_embedding = nn.Embedding(text_embedding_vocab_size, transformer_width)

        if positional_embedding_is:
            self.positional_embedding = OBB(in_channel=1 ,out_channel = transformer_width, num_objects = num_objects, pre_layer_norm = False, config=config)
        
        self.transformer = Transformer(
            width = transformer_width,
            layers = transformer_layers,
            heads = int(transformer_width/64),
            dropout = dropout
            # attn_mask=self.build_attention_mask()
        )
        
        self.ln_final = nn.LayerNorm(transformer_width)

        self.context_length = text_embedding_context_length
        self.objects_projection = nn.Parameter(torch.empty(transformer_width, embed_dim)) # 768, 768
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(transformer_width, num_classes)
        self.semantic_embedding_is = semantic_embedding_is
        self.positional_embedding_is = positional_embedding_is
        self.zero_shot = zero_shot
        if self.zero_shot:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.num_areas = num_areas

        if self.num_areas != None:
            self.mlp_head_area = nn.Linear(transformer_width, num_areas)

        self.position_projection = nn.Linear(transformer_width, transformer_width)
        self.position_projection_nopadding = nn.Parameter(torch.empty(transformer_width, embed_dim)) 

        self.dropout = nn.Dropout(0.1)
        self.layernorm = LayerNorm(embed_dim)

    
    def dtype(self):
        return torch.float16 # Return the valid data type object
        
    def forward(self, len_obj=None, roomtype = None,room_obb=None, obj_points = None, obj_obbs= None, obj_token = None, room_points = None, obj_area_tokenized = None, cat=False, train_mode = False):
        if self.semantic_embedding_is:
            if self.config.SE_type == 'nnEmbedding':
                semantic_embedding = self.semantic_embedding(obj_token)

        if self.positional_embedding_is:
            positional_embedding = self.positional_embedding(room_obb)


        if self.semantic_embedding_is:
            if self.positional_embedding_is:
                if cat:
                    objects = torch.cat((semantic_embedding, positional_embedding), dim=-1)  # Concatenate along the feature dimension
                else:
                    objects = semantic_embedding + positional_embedding
                    objects = objects / 2
            else:
                objects = semantic_embedding
                # no positonal information, bag of words
        elif self.positional_embedding_is:
            objects = positional_embedding
        
        '''cls token'''
        b,n,_ = objects.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        objects = torch.cat((cls_tokens, objects), dim=1)


      
        objects = objects.permute(1,0,2) #num_objects+1, batchsize, embedded_dim_size
        # objects = self.dropout(objects)
        x, attn_matrix = self.transformer(objects)
        x = x.permute(1,0,2) #batchsize, num_objects+1, embedded_dim_size
        x = self.ln_final(x)


        if not self.zero_shot :    
            x = self.to_latent(x) # batchsize, num_objects+1, transformer_width
           
            x_roomtype = self.mlp_head(x[:,0,:]) #[batchsize num_objects+1, room_types_num])
          
            if self.num_areas != None:
                x_areatype = self.mlp_head_area(x[:,1:,:])
            else:
                x_areatype = None
           
            return x_roomtype, x_areatype, attn_matrix
