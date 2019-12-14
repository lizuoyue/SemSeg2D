import os, glob, time
import torch, torchvision
import torch.nn as nn
import numpy as np
from PIL import Image
import networks
import random
import tqdm



img_transform = torchvision.transforms.Compose([
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

lbl_transform = torchvision.transforms.ToTensor()

t = time.time()

def create_data_loader(batch_size, mode, init_idx=0, seed=7):
	imgs = sorted(glob.glob('/local/zoli/SemSeg2D/datasets/scannet_v2/%s/images/*/*.jpg' % mode))
	lbls = sorted(glob.glob('/local/zoli/SemSeg2D/datasets/scannet_v2/%s/labels/*/*.png' % mode))
	assert(len(imgs) == len(lbls))
	for img, lbl in tqdm.tqdm(list(zip(imgs, lbls))):
		assert(os.path.basename(img).replace('.jpg', '') == os.path.basename(lbl).replace('.png', ''))
	pairs = list(zip(imgs, lbls))
	random.seed(seed)
	random.shuffle(pairs)

	idx = init_idx
	while True:
		img_li, lbl_li = [], []
		for i in range(batch_size):
			img_li.append(img_transform(Image.open(pairs[idx % len(pairs)][0])))
			lbl_li.append(lbl_transform(Image.open(pairs[idx % len(pairs)][1])))
			idx += 1
		yield idx, torch.stack(img_li, dim=0), torch.stack(lbl_li, dim=0)



if __name__ == '__main__':

	batch_size = 4
	train_data_loader = create_data_loader(batch_size=batch_size, mode='train')
	val_data_loader = create_data_loader(batch_size=batch_size, mode='val')

	netG = networks.define_G(input_nc=3, output_nc=256, nz=0, ngf=256, netG='unet_16', norm='batch', nl='relu',
		use_dropout=True, init_type='xavier', init_gain=0.02, gpu_ids=[0], where_add='input', upsample='basic')
	linear = nn.Linear(256, 40).cuda()
	criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=255, reduce=None, reduction='mean')

	optimizer = torch.optim.Adam(list(netG.parameters()) + list(linear.parameters()), lr=1e-4)

	if False:
		netG.load_state_dict(torch.load('./netG_latest.pth'))
		linear.load_state_dict(torch.load('./linear_latest.pth'))
		netG.train()
		linear.train()

	while True:
		optimizer.zero_grad()
		it, imgs, lbls = next(train_data_loader)
		features = netG(imgs.cuda())
		print(features.shape)
		logits = linear(features)
		loss = criterion(logits, lbls)

		loss.backward()
		optimizer.step()

		pred = torch.argmax(logits, dim=-1).cpu().numpy()
		lbls = lbls.cpu().numpy()
		acc = (pred == lbls)[lbls < 255]

		print('train', it, loss.item(), acc.mean(), flush=True)

		if it % 5000 == batch_size:
			torch.save(netG.state_dict(), './netG_%d.pth' % it)
			torch.save(netG.state_dict(), './netG_latest.pth')
			torch.save(linear.state_dict(), './linear_%d.pth' % it)
			torch.save(linear.state_dict(), './linear_latest.pth')

			_, imgs, lbls = next(val_data_loader)
			features = netG(imgs.cuda())
			logits = linear(features)
			loss = criterion(logits, lbls)
			pred = torch.argmax(logits, dim=-1).cpu().numpy()
			lbls = lbls.cpu().numpy()
			acc = (pred == lbls)[lbls < 255]

			print('val', it, loss.item(), acc.mean(), flush=True)






