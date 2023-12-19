import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

########################################### CBAM #######################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial

        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        if not (type(x) == torch.Tensor):
            x = x[0]
        
        B, _, _, _ = x.size()
        attn_c = self.ChannelGate(x)
        x_out = x * attn_c

        if not self.no_spatial:
            attn_s = self.SpatialGate(x_out)
            x_out = x_out * attn_s

        else:
            attn_s = None

        return x_out, [attn_c[:,:,0,0], attn_s]


############ Squeeze and Excitation Attention Module ###############
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if not (type(x) == torch.Tensor):
            x = x[0]

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x), [y, None]




############ Global Attention ###################
class stem_attention(nn.Module):
    def __init__(self, global_attention=False):
        super(stem_attention, self).__init__()
        if global_attention:
            self.stem = AttentionConv2d(2, dk=4, dv=8, num_heads=1, rel_encoding=False, height=56, width=56)
            out_channel = 8
        else:
            self.stem = None
            out_channel = 2
        
        self.compress = ChannelPool()
        kernel_size = 7
        self.spatial = BasicConv(out_channel, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        
        
    def forward(self, x):
        x = self.compress(x)
        
        if self.stem is not None:
            # Global Attention
            x = self.stem(x)
            
        # First Local Attention
        x_out = self.spatial(x)
        attn = F.sigmoid(x_out) # broadcasting
        return attn

class AttentionConv2d(nn.Module):
    def __init__(self, input_dim, dk, dv, num_heads, rel_encoding=False, height=None, width=None):
        super(AttentionConv2d, self).__init__()
        self.input_dim = input_dim

        self.dk = dk
        self.dv = dv
        self.num_heads = num_heads

        self.dkh = self.dk // self.num_heads
        if rel_encoding and not height:
            raise("Cannot use relative encoding without specifying input's height and width")
        self.H = height
        self.W = width

        self.conv_qkv = nn.Conv2d(input_dim, 2*dk + dv, 1)
        self.conv_attn = nn.Conv2d(dv, dv, 1)
        self.softmax = nn.Softmax(dim=-1)
        
        self.relative_encoding = rel_encoding
        if rel_encoding: 
            self.key_rel_w = nn.Parameter(self.dkh**-0.5 + torch.rand(2*width-1, self.dkh), requires_grad=True)
            self.key_rel_h = nn.Parameter(self.dkh**-0.5 + torch.rand(2*height-1, self.dkh), requires_grad=True)
            

    def forward(self, input):
        qkv = self.conv_qkv(input)    # batch_size, 2*dk+dv, H, W
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        
        batch_size, _, H, W = q.size()

        q = q.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        k = k.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        v = v.view([batch_size, self.num_heads, self.dv // self.num_heads, H*W])

        q *= self.dkh ** -0.5
        logits = einsum('ijkl, ijkm -> ijlm', q, k)
        
        if self.relative_encoding:
            h_rel_logits, w_rel_logits = self._relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = self.softmax(logits)
        attn_out = einsum('ijkl, ijfl -> ijfk', weights, v)
        attn_out = attn_out.view(batch_size, self.dv, H, W)
        attn_out = self.conv_attn(attn_out)
        return attn_out


    def _relative_logits(self, q):
        b, nh, dkh, _ = q.size()
        q = q.view(b, nh, dkh, self.H, self.W)

        rel_logits_w = self._relative_logits1d(q, self.key_rel_w, self.H, self.W, nh, [0, 1, 2, 4, 3, 5])
        rel_logits_h = self._relative_logits1d(q.permute(0, 1, 2, 4, 3), self.key_rel_h, self.W, self.H, nh, [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w

    def _relative_logits1d(self, q, rel_k, H, W, Nh, transpose_mask):
        rel_logits = einsum('bhdxy, md -> bhxym', q, rel_k)

        rel_logits = rel_logits.view([-1, Nh*H, W, 2*W-1])
        rel_logits = self._rel_to_abs(rel_logits)
        rel_logits = rel_logits.view([-1, Nh, H, W, W]).unsqueeze(dim=3).repeat([1,1,1,H,1,1])
        rel_logits = rel_logits.permute(*transpose_mask)
        rel_logits = rel_logits.contiguous().view(-1, Nh, H*W, H*W)
        return rel_logits

    def _rel_to_abs(self, x):
        b, nh, l, _ = x.size()

        x = F.pad(x, (0,1), 'constant', 0)
        flat_x = x.view([b, nh, l*(2*l)]);
        flat_x_padded = F.pad(flat_x, (0, l-1), 'constant', 0)

        final_x = flat_x_padded.view([b, nh, l+1, 2*l-1])
        final_x = final_x[:, :, :l, l-1:]

        return final_x
    

if __name__=='__main__':
    dk, dv= 8, 12
    num_heads = 4
    
    
    model = AttentionConv2d(64, dk, dv, num_heads, rel_encoding=False, height=224, width=224)
    x = torch.ones([1, 64, 58, 58])
    out = model(x)
    print(out.size())