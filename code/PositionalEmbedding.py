# MIT License

# Copyright (c) [2025] [Wenting Xu]

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

import torch.nn as nn
import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np
# from collections import OrderedDict


device = "cuda" if torch.cuda.is_available() else "cpu"
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class OBB(nn.Module):
    def __init__(self,in_channel, out_channel,num_objects=77, batch_norm = True,pre_layer_norm = True, dropout = 0.,config=None):
        super(OBB, self).__init__()
        self.config = config
        self.fc = nn.Linear(1, out_channel)  

        self.pre_layer_norm = pre_layer_norm
        if self.pre_layer_norm:
            self.layernorm = nn.LayerNorm(3)

    def forward(self,x):
        if self.config.relative_distance:
            x = x[:,:,-1]
            return (self.fc(x.unsqueeze(-1)))
        else:
            x = x[:,:,:3] # comment to consider volumn

            if self.pre_layer_norm:
              
                x = self.layernorm(x)
               
            return self.layernorm_2(torch.mean(self.mlp(x.unsqueeze(3)),dim=2))
