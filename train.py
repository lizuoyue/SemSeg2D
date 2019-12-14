import os, glob, time
import torch, torchvision
import torch.nn as nn
import numpy as np
from PIL import Image
import networks



input_transform = torchvision.transforms.Compose([
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

input_to_tensor = torchvision.transforms.ToTensor()

t = time.time()

def create_data_loader(batch_size, num_frames, patch_size, init_i = 0):
	num = train_patch_idxs.shape[0]
	i = init_i
	while True:
		imgs, labels, mappings, patch_masks = [], [], [], []
		pc, pc_color, pc_label = [], [], []
		for _ in range(batch_size):
			# print(i)
			scene_id = train_patch_scene_idxs[i]
			scene_name = train_scene_names[scene_id]
			coords, colors, all_labels = train_scenes[scene_id]
			pc.append(coords)
			pc_color.append(colors)
			pc_label.append(all_labels.astype(np.int32))
			# print('num of points in pc:', coords.shape)

			pid = train_patch_idxs[i]
			center_p = coords[pid]
			upper = center_p + patch_size / 2
			lower = center_p - patch_size / 2
			patch_mask = np.ones((coords.shape[0]), np.bool)
			for j in range(3):
				patch_mask = np.logical_and(patch_mask, coords[:, j] <= upper[j])
				patch_mask = np.logical_and(patch_mask, lower[j] <= coords[:, j])
			patch_masks.append(patch_mask)

			fids = selected_frames[scene_name][pid][:num_frames]
			imgs.extend([input_transform(Image.open('/local/zoli/3DMV/data/scannetv2_images/%s/color/%d.jpg'
				% (scene_name, fid * 20))) for fid in fids])
			# labels.extend([input_to_tensor(Image.open('/local/zoli/3DMV/data/scannetv2_images/%s/label/%d.png'
			# 	% (scene_name, fid * 20))) for fid in fids])

			mapping = np.load('../compute_coverage_pc/mapping/%s_img2pc_mapping.npy' % scene_name, allow_pickle=True).item()
			# mapping = np.load('../compute_coverage_pc/mapping/%s_pc2img_mapping.npy' % scene_name, allow_pickle=True).item()
			mapping = inverse_img2pc_mapping(np.stack([mapping[fid * 20] for fid in fids]), coords.shape[0])
			mappings.append(mapping)

			assert(train_patch_labels[i] == all_labels[pid])
			
			i += 1
			if i >= num:
				i = 0
		imgs = torch.stack(imgs, dim=0)
		yield i, imgs, mappings, patch_masks, pc, pc_color, pc_label
	return None


def backproject(num_points, features, mappings):
	# num_points: scalar
	# features: (num_frames, num_features, height, width) torch.Tensor
	# mappings: (num_frames, num_points)                  numpy.array

	# output  : (num_frames, num_points, num_features)
	# valid   : (num_frames, num_points)

	# Reshape
	num_frames, num_features = features.shape[:2]
	fts = features.view(num_frames, num_features, -1).transpose(1, 2)
	pad = 0 * fts[:,:1,:]
	val_fts = 0 * fts[..., :1] + 1
	val_pad = 0 * val_fts[:,:1,:]
	fts = torch.cat([fts, pad], dim=1)
	val_fts = torch.cat([val_fts, val_pad], dim=1).squeeze()
	# fts: (num_frames, num_pixels+1, num_features)

	# Tile
	mps = np.repeat(mappings[..., np.newaxis], num_features, axis=-1)
	mps = torch.from_numpy(mps).long().cuda()
	# mps: (num_frames, num_points, num_features)

	output = torch.gather(input=fts, dim=1, index=mps)
	valid = torch.gather(input=val_fts, dim=1, index=mps[...,0])
	return output, valid


if __name__ == '__main__':

	batch_size = 2
	num_frames = 5
	data_loader = create_data_loader(batch_size=batch_size, num_frames=num_frames, patch_size=2.56, init_i=110000)
	netG = networks.define_G(input_nc=3, output_nc=128, nz=0, ngf=128, netG='unet_16', norm='batch', nl='relu',
		use_dropout=True, init_type='xavier', init_gain=0.02, gpu_ids=[0], where_add='input', upsample='basic')
	linear = nn.Linear(128, 20).cuda()
	criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

	optimizer = torch.optim.Adam(list(netG.parameters()) + list(linear.parameters()), lr=1e-4)

	if True:
		netG.load_state_dict(torch.load('./netG_latest.pth'))
		linear.load_state_dict(torch.load('./linear_latest.pth'))
		netG.train()
		linear.train()

	while True:
		try:
			optimizer.zero_grad()
			it, imgs, mappings, patch_masks, pcs, pcs_color, pcs_label = next(data_loader)
			features = netG(imgs.cuda())

			# print(imgs.shape, type(imgs))
			# print(features.shape, type(features))
			# print(len(mappings), len(patch_masks), len(pcs), len(pcs_color), len(pcs_label))
			loss = 0
			accs = []
			for i, (mapping, patch_mask, pc, pc_color, pc_label) in enumerate(zip(mappings, patch_masks, pcs, pcs_color, pcs_label)):
				# print(mapping.shape, type(mapping), mapping.dtype)
				# print(patch_mask.shape, type(patch_mask), patch_mask.dtype)
				# print(pc.shape, type(pc), pc.dtype)
				# print(pc_color.shape, type(pc_color), pc_color.dtype)
				# print(pc_label.shape, type(pc_label), pc_label.dtype, pc_label.max(), pc_label.min())

				num_points = pc.shape[0]
				out_ft, val = backproject(num_points, features[i*num_frames:(i+1)*num_frames], mapping)
				cnt = torch.sum(val, dim=0)
				out_sum = torch.sum(out_ft, dim=0)

				# print(out.shape, type(out), out.dtype)
				# print(val.shape, type(val), val.dtype)
				# print(torch.max(cnt).item(), torch.min(cnt).item())

				val_mask = (torch.from_numpy(patch_mask).cuda().int() * (cnt.int() > 0).int()) > 0
				# print(torch.sum(cnt > 0).item(), patch_mask.sum(), torch.sum(val_mask).item())

				cnt = cnt[val_mask]
				out_sum = out_sum[val_mask]
				lbl = torch.from_numpy(pc_label).long().cuda()[val_mask]
				# print(out_sum.shape, cnt.shape, lbl.shape)
				out_mean = (out_sum.t() / cnt).t()
				logits = linear(out_mean)
				# print(logits.shape)
				loss += criterion(logits, lbl)

				pred = torch.argmax(logits, dim=-1)[lbl >= 0]
				lbll = lbl[lbl >= 0]
				# print(pred.shape, pred.dtype, pred.min().item(), pred.max().item())
				# print(lbll.shape, lbll.dtype, lbll.min().item(), lbll.max().item())
				acc = torch.eq(pred, lbll)
				accs.append(acc)
				# print(acc.shape, acc.dtype, acc.min().item(), acc.max().item())
				# print(acc)

			loss.backward()
			optimizer.step()

			acc = torch.cat(accs, dim=0)
			# print(acc.shape, acc.max(), acc.min())
			# print(acc.sum())
			print(it, loss.item(), acc.sum().item() / acc.shape[0], flush=True)

			if it % 5000 == 0:
				torch.save(netG.state_dict(), './netG_%d.pth' % it)
				torch.save(netG.state_dict(), './netG_latest.pth')
				torch.save(linear.state_dict(), './linear_%d.pth' % it)
				torch.save(linear.state_dict(), './linear_latest.pth')
		except:
			pass



