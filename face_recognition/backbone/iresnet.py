import torch
from torch import nn
from .attention import *


__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


def at(x):
    return F.normalize(x.pow(2).mean(1))


class HintLayer(nn.Module):
    def __init__(self, inplane, planes, bn=False):
        super(HintLayer, self).__init__()
        self.hint = nn.Conv2d(inplane, planes, stride=1, kernel_size=1, padding=0, bias=False)

        if bn:
            self.bn = nn.BatchNorm2d(planes, eps=1e-05,)
        else:
            self.bn = None
        
    def forward(self, x):
        out = self.hint(x)
        if self.bn is not None:
            out = self.bn(out)
        return out
    
    

class target(nn.Module): 
    def __init__(self, feat_type='feature'):
        super(target, self).__init__()
        self.feat_type = feat_type
        
    def forward(self, x):
        if self.feat_type == 'feature':
            return x
        elif self.feat_type == 'attention':
            return x
        else:
            raise('Select Proper Feat Type')


class identity(nn.Module):
    def forward(self, out):
        return out, []
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, attention_type='ir'):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride
        
        if attention_type == 'cbam' or attention_type == 'se':
            self.attention_target_all = target(feat_type='attention')
        else:
            self.attention_target_all = None
            
        self.attention_type = attention_type
        if attention_type == 'ir':
            self.attention = identity()
        elif attention_type == 'cbam':
            self.attention = CBAM(planes, 16)
        elif attention_type == 'se':
            self.attention = SELayer(planes, 16)
        else:
            raise('Select Proper Attention Type')
        
    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Calculation Attention
        out, attn = self.attention(out)
        if self.attention_target_all is not None:
            _ = self.attention_target_all(attn)

        out = out + identity
        return out

    def forward(self, x):
        return self.forward_impl(x)


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, pooling, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, attention_type='ir', qualnet=False, student=False, hint_bn=True):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.inplanes = 64
        self.dilation = 1
        self.attention_type = attention_type
        self.qualnet = qualnet        
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.student = student
        if self.student:
            self.hint_layer1 = HintLayer(self.layer1[0].conv1.out_channels, self.layer1[0].conv1.out_channels, bn=hint_bn)
            self.hint_layer2 = HintLayer(self.layer2[0].conv1.out_channels, self.layer2[0].conv1.out_channels, bn=hint_bn)
            self.hint_layer3 = HintLayer(self.layer3[0].conv1.out_channels, self.layer3[0].conv1.out_channels, bn=hint_bn)
            self.hint_layer4 = HintLayer(self.layer4[0].conv1.out_channels, self.layer4[0].conv1.out_channels, bn=hint_bn)
        else:
            self.hint_layer1, self.hint_layer2, self.hint_layer3, self.hint_layer4 = None, None, None, None
        
        self.feature_target_block = target()
        self.pooling = pooling
        
        # Bridge
        if self.qualnet:
            self.conv_bridge = nn.Conv2d(512, 3072, kernel_size=3, stride=1, padding=1)
        
        # Option A
        if self.pooling == 'A':
            self.gp = nn.AdaptiveAvgPool2d((1,1))
        elif self.pooling == 'E':
            # Option E
            self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
            self.dropout = nn.Dropout(p=dropout, inplace=True)
            self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
            self.features = nn.BatchNorm1d(num_features, eps=1e-05)
            nn.init.constant_(self.features.weight, 1.0)
            self.features.weight.requires_grad = False
        else:
            raise('Select Proper Pooling')
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, attention_type=self.attention_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation, attention_type=self.attention_type))

        return nn.Sequential(*layers)


    def forward(self, x, extract_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        
        if self.hint_layer1 is None:
            x1 = x
        else:
            x1 = self.hint_layer1(x)
            
            
        x = self.layer2(x)
        if self.hint_layer2 is None:
            x2 = x
        else:
            x2 = self.hint_layer2(x)
            
        
        x = self.layer3(x)
        if self.hint_layer3 is None:
            x3 = x
        else:
            x3 = self.hint_layer3(x)
            
                    
        x = self.layer4(x)
        if self.hint_layer4 is None:
            x4 = x
        else:
            x4 = self.hint_layer4(x)
            
        
        if self.qualnet:
            x_recon = self.conv_bridge(x)
        
        if self.pooling == 'A':
            # Option-A
            x = self.gp(x)
            x = torch.flatten(x, 1)
            
        elif self.pooling == 'E':
            # Option-E
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            x = self.features(x)
        else:
            raise('Select Proper Pooling')
        
        if self.qualnet:
            if extract_feature:
                return x, x_recon, [x1,x2,x3,x4]
            else:
                return x, x_recon
        else:
            if extract_feature:
                return x, [x1,x2,x3,x4]
            else:
                return x          
              
            
def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)

    
                            
if __name__=='__main__':
    model = iresnet100(attention_type='ir')
    x = torch.ones([4, 3, 128, 128])
    out = model(x)
    
    
    




