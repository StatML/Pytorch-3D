import os
import time
import datetime

import torch
import torch.utils.data
from opts import opts
import ref
from models.hourglass_2 import hg
from models.hg_3d import HourglassNet3D
from utils.utils import adjust_learning_rate
from datasets.fusion import Fusion
from datasets.h36m import H36M
from datasets.mpii import MPII
from utils.logger import Logger
from train import train, val

from utils.utils import AverageMeter
from utils.eval import Accuracy, getPreds, MPJPE
from utils.debugger import Debugger
from models.layers.FusionCriterion import FusionCriterion
import cv2
import ref
from progress.bar import Bar

##Change the auto_grad varibles to .cuda()
##Change the criterion to .cuda()

opt = opts().parse()

train_loader = torch.utils.data.DataLoader(
      Fusion(opt, 'train'), 
      batch_size = 6)

model = hg(num_stacks = 3, num_blocks = 1,num_classes = 16)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), opt.LR, 
                                  alpha = ref.alpha, 
                                  eps = ref.epsilon, 
                                  weight_decay = ref.weightDecay, 
                                  momentum = ref.momentum)

Loss, Acc, Mpjpe, Loss3D = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
nIters = len(train_loader)
bar = Bar('==>', max=nIters)

model.train()

num_epoch = 1

for epoch in range(num_epoch):

	for i, (input1, target2D, target3D, meta) in enumerate(train_loader):

		##
		input_var = torch.autograd.Variable(input1).float()
		target2D_var = torch.autograd.Variable(target2D).float()
		x = target3D[:,:,0]
		y = target3D[:,:,1]
		z = target3D[:,:,2]

		xyz = torch.cat((x,y,z),1)

		target3D_var = torch.autograd.Variable(xyz).float()
		output = model(input_var)

		reg = output[-1]

		loss = criterion(reg, target3D_var)
		Loss3D.update(loss.data[0], input1.size(0))

		print(len(output))
		for k in range(0,3):
			loss += criterion(output[k], target2D_var)

		Loss.update(loss.data[0], input1.size(0))
		Acc.update(Accuracy((output[-2].data).cpu().numpy(), (target2D_var.data).numpy()))
		mpjpe, num3D = MPJPE((output[-2].data).cpu().numpy(), (reg[:,32:].data).numpy(), meta)
		if num3D > 0:
			Mpjpe.update(mpjpe, num3D)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		Bar.suffix = 'Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Loss3D {loss3d.avg:.6f} | Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f}'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, loss3d = Loss3D,Mpjpe=Mpjpe)
		bar.next()
	
	bar.finish()





