import numpy as np
import os, glob, tqdm, time
from PIL import Image
import cv2

valid_label = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39}
mapping = []
v = 1
for i in range(41):
	if i in valid_label:
		mapping.append(v)
		v += 1
	else:
		mapping.append(0)

tab20_list = [
	[0, 0, 0],
	[31, 119, 180],
	[174, 199, 232],
	[255, 127, 14],
	[255, 187, 120],
	[44, 160, 44],
	[152, 223, 138],
	[214, 39, 40],
	[255, 152, 150],
	[148, 103, 189],
	[197, 176, 213],
	[140, 86, 75],
	[196, 156, 148],
	[227, 119, 194],
	[247, 182, 210],
	[127, 127, 127],
	[199, 199, 199],
	[188, 189, 34],
	[219, 219, 141],
	[23, 190, 207],
	[158, 218, 229],
]
tab20_palette = [i for l in tab20_list for i in l]

train_scene_names = [os.path.basename(x).replace('_vh_clean_2.pth', '')
	for x in sorted(glob.glob('/media/root/data/ScanNet_v2_data/train/*_vh_clean_2.pth'))]
val_scene_names = [os.path.basename(x).replace('_vh_clean_2.pth', '')
	for x in sorted(glob.glob('/media/root/data/ScanNet_v2_data/val/*_vh_clean_2.pth'))]

src_image_path = '/local/zoli/3DMV/data/scannetv2_images/%s/color/'
src_depth_path = '/local/zoli/3DMV/data/scannetv2_images/%s/depth_filled/'
src_label_path = '/local/zoli/3DMV/data/scannetv2_images/%s/label/'
src_li = [src_image_path, src_depth_path, src_label_path]
dst_image_path = '/local/zoli/SemSeg2D/datasets/scannet_v2_ssma/%s/image/%s'
dst_depth_path = '/local/zoli/SemSeg2D/datasets/scannet_v2_ssma/%s/depth/%s'
dst_label_path = '/local/zoli/SemSeg2D/datasets/scannet_v2_ssma/%s/label/%s'
dst_li = [dst_image_path, dst_depth_path, dst_label_path]

os.makedirs('/local/zoli/SemSeg2D/datasets/scannet_v2_ssma/', exist_ok=True)
for tv, scene_names in zip(['val', 'train'], [val_scene_names, train_scene_names]):
	with open('/local/zoli/SemSeg2D/datasets/scannet_v2_ssma/%s.txt' % tv, 'w') as f:
		for scene_name in tqdm.tqdm(scene_names):
			for path in dst_li:
				os.makedirs(path % (tv, scene_name), exist_ok=True)
			li_files = [sorted(glob.glob(path % scene_name + '*')) for path in src_li]
			assert(len(li_files[0]) == len(li_files[2]))
			assert(len(li_files[1]) == len(li_files[2]))
			for image_file, depth_file, label_file in list(zip(li_files[0], li_files[1], li_files[2])):
				li = []
				image_basename = os.path.basename(image_file).replace('.jpg', '.png')
				# image = Image.open(image_file).resize((768, 384), resample=Image.BILINEAR)
				li.append(dst_image_path % (tv, scene_name) + '/' + image_basename)
				# image.save(li[-1])

				depth_basename = os.path.basename(depth_file)
				depth = Image.open(depth_file).resize((768, 384), resample=Image.BILINEAR)
				depth = np.clip(np.array(depth, np.float) / 7000.0 * 255.0, 0, 255).astype(np.uint8)
				depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
				li.append(dst_depth_path % (tv, scene_name) + '/' + depth_basename)
				cv2.imwrite(li[-1], depth)
				
				label_basename = os.path.basename(label_file)
				# label = np.array(Image.open(label_file)).astype(np.int32)
				# label = np.take(mapping, label.reshape(-1)).reshape(label.shape)
				# label = Image.fromarray(label.astype(np.uint8)).resize((768, 384), resample=Image.NEAREST)
				# label.putpalette(tab20_palette)
				li.append(dst_label_path % (tv, scene_name) + '/' + label_basename)
				# label.save(li[-1])
				f.write('%s %s %s\n' % tuple(li))


