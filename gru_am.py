import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock
from Bigru_Attention4 import Bigru_Attention


class VGGVox2(nn.Module):

    def __init__(self, block, layers, emb_dim,
                 zero_init_residual=False):
        super(VGGVox2, self).__init__()
        self.embedding_size = emb_dim
        self.num_layers=2
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_layer = nn.Linear(512 * block.expansion, self.embedding_size)
        self.model_gru = Bigru_Attention(input_size=self.embedding_size,input_embeding=self.embedding_size)               

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.xavier_uniform_(self.embedding_layer.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with z
        # eros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to 
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        
    
        out = out.view(out.size(0),-1)
        #out = self.embedding_layer(out)
        out= out.unsqueeze(dim=1)  

        #print('pre_gru:{}'.format(out.shape))#(32,1,512)
        out=self.model_gru(out)
        
        return out
    

if __name__ == '__main__':
    net = VGGVox2(BasicBlock, [2, 2, 2, 2],emb_dim=512)
    inputdata = torch.randn(32, 192, 251)
    result = net(inputdata)
    print('result:{}'.format(result.shape))