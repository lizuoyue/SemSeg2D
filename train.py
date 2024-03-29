import os, glob, time, tqdm, random
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn

import networks

def create_data_loader(batch_size, mode, device='cuda:0', init_idx=0, seed=7):

	img_transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])

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
			lbl = np.array(Image.open(pairs[idx % len(pairs)][1])).astype(np.int32)
			assert(lbl.min() >= 0 and lbl.max() <= 40)
			lbl_li.append(torch.from_numpy(lbl - 1))
			idx += 1
		yield idx, torch.stack(img_li, dim=0).to(device), torch.stack(lbl_li, dim=0).long().to(device)


def miou(gt, dt, num_classes):
	res = []
	for i in range(num_classes):
		gt_mask = gt == i
		dt_mask = dt == i
		sum_i = (gt_mask & dt_mask).sum()
		sum_u = (gt_mask | dt_mask).sum()
		if sum_u != 0:
			res.append(sum_i / sum_u)
	return np.array(res).mean()


if __name__ == '__main__':

	if torch.cuda.is_available():
		device = 'cuda:0'
	else:
		device = 'cpu'

	batch_size = 4
	feature_dim = 256
	num_classes = 40

	train_data_loader = create_data_loader(batch_size=batch_size, mode='train', device=device)
	val_data_loader = create_data_loader(batch_size=batch_size, mode='val', device=device)

	netG = networks.define_G(input_nc=3, output_nc=feature_dim, nz=0, ngf=256, netG='unet_16', norm='batch', nl='relu',
		use_dropout=True, init_type='xavier', init_gain=0.02, gpu_ids=[0], where_add='input', upsample='basic')
	linear = nn.Linear(feature_dim, num_classes).cuda()
	criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-1, reduce=None, reduction='mean')
	optimizer = torch.optim.Adam(list(netG.parameters()) + list(linear.parameters()), lr=1e-4)

	if False:
		netG.load_state_dict(torch.load('./netG_latest.pth'))
		linear.load_state_dict(torch.load('./linear_latest.pth'))
		netG.train()
		linear.train()

	while True:
		optimizer.zero_grad()
		it, imgs, lbls = next(train_data_loader)

		features = netG(imgs.cuda()).permute(0, 2, 3, 1)
		logits = linear(features.reshape(-1, feature_dim))
		lbls = lbls.reshape(-1)
		loss = criterion(logits, lbls)

		loss.backward()
		optimizer.step()

		pred = torch.argmax(logits, dim=-1).cpu().numpy()
		lbls = lbls.cpu().numpy()
		eq = (pred == lbls)
		acc = eq[lbls > -1]

		print('train', it, loss.item(), acc.mean(), miou(lbls, pred, num_classes), flush=True)

		if it % 5000 == 0:
			torch.save(netG.state_dict(), './netG_%d.pth' % it)
			torch.save(netG.state_dict(), './netG_latest.pth')
			torch.save(linear.state_dict(), './linear_%d.pth' % it)
			torch.save(linear.state_dict(), './linear_latest.pth')

		if it % 500 == 0:
			_, imgs, lbls = next(val_data_loader)
			features = netG(imgs.cuda()).permute(0, 2, 3, 1)
			logits = linear(features.reshape(-1, feature_dim))
			lbls = lbls.reshape(-1)
			loss = criterion(logits, lbls)
			
			pred = torch.argmax(logits, dim=-1).cpu().numpy()
			lbls = lbls.cpu().numpy()
			eq = (pred == lbls)
			acc = eq[lbls > -1]

			print('val', it, loss.item(), acc.mean(), miou(lbls, pred, num_classes), flush=True)



