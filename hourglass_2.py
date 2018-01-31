'''
Hourglass network inserted in the pre-activated Resnet 
Use lr=0.01 for current version
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

# from .preresnet import BasicBlock, Bottleneck


__all__ = ['HourglassNet', 'hg']

class linear_model(nn.Module):


    def __init__(self,linear_size,predict_14):

        super(linear_model, self).__init__()

        self.HUMAN_2D_SIZE = 16 * 2
        self.HUMAN_3D_SIZE = 16 * 3 if predict_14 else 16 * 3
        self.input_size  = self.HUMAN_2D_SIZE
        self.output_size = self.HUMAN_3D_SIZE

        self.linear_size   = linear_size

        self.l_initial = nn.Linear(self.input_size,linear_size)

        self.relu = nn.ReLU(inplace = True)

        self.dropout = nn.Dropout(p=0.5)

    def linear_block(self,x, input_size,linear_size):

        l = nn.Linear(input_size,linear_size)
        y = l(x)
        y = self.relu(y)
        y = self.dropout(y)

        l = nn.Linear(linear_size,linear_size)

        y = l(x)
        y = self.relu(y)
        y = self.dropout(y)

        return x + y

    def forward(self,x):

        x = self.l_initial(x)
        x = self.relu(x)
        x = self.dropout(x)

        input_l1_size = x.size()
        
        for i in range(2):
            x = self.linear_block(x,int(input_l1_size[1]),self.linear_size)
        x = self.relu(x)
        x = self.dropout(x)

        l2 = nn.Linear(self.linear_size, self.HUMAN_3D_SIZE)
        x = l2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)
        self.soft = nn.Softmax()
        self.linear = linear_model(1024,True)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def get_preds(self,scores):
        ''' get predictions from score maps in torch Tensor
            return type: torch.LongTensor
        '''
        assert scores.dim() == 4, 'Score maps should be 4-dim'
        maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

        maxval = maxval.view(scores.size(0), scores.size(1), 1)
        idx = idx.view(scores.size(0), scores.size(1), 1) + 1

        preds = idx.repeat(1, 1, 2).float()

        preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
        preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        preds *= pred_mask
        return preds

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)  

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        last_heatmap = out[-1]

        print(last_heatmap.size())
        preds = self.get_preds(last_heatmap)

        batch = preds.size()[0]

        temp = Variable(torch.FloatTensor(batch,32))


        for i in range(batch):

            temp[i,:] = torch.cat((preds[i,:,0],preds[i,:,1]))

        print(temp.size())

        final_out = self.linear(temp)

        print(final_out.size())

        return final_out






def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model
