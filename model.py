"""
TransformerMIWB model. From "Revisiting Image-Fusion for Multi-Illuminant White-Balance Correction. In ICCV, 2025.
Based on the Restoermer (CVPR 2022) implementation.

David Serrano (dserrano@cvc.uab.cat)
July 2025
"""

import torch
import torch.nn as nn
from einops import rearrange
import numbers
import torch.nn.functional as F

class TransformerWB(nn.Module):
    def __init__(self, device, P=15):
        """ Define the layers """
        super(TransformerWB, self).__init__()
        self.device = device
        self.P = P
        
        self.att = TransformerBlock(inp_dim=15, feat_dim=P, num_heads=5, ffn_expansion_factor=4, bias=True, LayerNorm_type='BiasFree')

    def forward(self, x):
        """ Define the forward pass
        :param x: dictionary of input images. The dictionary has to have the following format:
        {'daylight': pytorch tensor [batch_size, 3, width, height],
         'shade': pytorch tensor [batch_size, 3, width, height],
         'cloudy': pytorch tensor [batch_size, 3, width, height],
         'tungsten': pytorch tensor [batch_size, 3, width, height],
         'fluorescent': pytorch tensor [batch_size, 3, width, height]}
        :return: pytorch tensor [batch_size, 3, width, height]
        """
        
        x_cat = torch.cat((x['daylight'], x['shade'], x['cloudy'], x['tungsten'], x['fluorescent']), dim=1).to(self.device)
            
        x = self.att(x_cat)
        return torch.clamp(x, 0, 1)     

class TransformerBlock(nn.Module):
    def __init__(self, inp_dim, feat_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        """
        Transformer block.
        Args:
            inp_dim: input dimension (number of channels - usually 15 - RGB)
            feat_dim: feature dimension
            num_heads: number of attention heads
            ffn_expansion_factor: expansion factor for the feed forward network
            bias: whether to use bias in the layers
            LayerNorm_type: type of layer normalization
        """
        self.conv1 = nn.Conv2d(inp_dim, feat_dim, kernel_size=3, stride=1, padding=1, bias=bias) # 15 -> P
        self.conv2 = nn.Conv2d(feat_dim, 3, kernel_size=3, stride=1, padding=1, bias=bias) # P -> 3

        self.norm1 = LayerNorm(feat_dim, LayerNorm_type)
        self.attn = Attention(feat_dim, num_heads, bias)
        self.norm2 = LayerNorm(feat_dim, LayerNorm_type)
        self.ffn = FeedForward(feat_dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = self.conv2(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# NOT USED IN THE FINAL MODEL (just for the ablation studies)
class SpatialTransformerBlock(nn.Module):
    def __init__(self, inp_dim, feat_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SpatialTransformerBlock, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, feat_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(feat_dim, 3, kernel_size=3, stride=1, padding=1, bias=bias)

        self.norm1 = LayerNorm(feat_dim, LayerNorm_type)
        self.attn = SpatialAttention(dim=feat_dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm(feat_dim, LayerNorm_type)
        self.ffn = FeedForward(feat_dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = self.conv2(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SpatialAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.max_pool_ksize = 4

        self.qkv = nn.Sequential(
            nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias),
            nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias),
            nn.MaxPool2d(self.max_pool_ksize, self.max_pool_ksize),
            nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias),
            nn.MaxPool2d(self.max_pool_ksize, self.max_pool_ksize)
        )

        self.project_out = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.ReLU(inplace=True),
        )   

    def forward(self, x):

        original_size = x.size()[-2:]

        qkv = self.qkv(x)
        q,k,v = torch.split(qkv, self.dim, dim=1)   
        
        b,c,h,w = q.shape

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = attn.softmax(dim=-1)

        #out = (attn @ v)
        out = (v @ attn)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        return out