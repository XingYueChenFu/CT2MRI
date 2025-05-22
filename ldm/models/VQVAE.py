import torch
from torch import nn

from ldm.modules.diffusionmodules.model import Encoder, Decoder

class VQVAE(nn.Module):
    def __init__(self, 
                 embed_dim,
                 n_embed,
                 ddconfig,
                 **kwargs):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # 不需要quantizer
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
    
class LPIPSWithDiscriminator(nn.Module):
    pass