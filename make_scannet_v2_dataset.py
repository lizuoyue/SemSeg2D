import numpy as np
import os, glob, tqdm, time
from PIL import Image

mapping = {
	1: 0,
	2: 1,
	3: 2,
	4: 3,
	5: 4,
	6: 5,
	7: 6,
	8: 7,
	9: 8,
	10: 9,
	11: 10,
	12: 11,
	14: 12,
	16: 13,
	24: 14,
	28: 15,
	33: 16,
	34: 17,
	36: 18,
	39: 19
}

def f(x):
	if x in mapping:
		y = mapping[x]
	else:
		y = 255
	return y

vf = np.vectorize(f)

tab20_list = [
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
tab20_palette = []
for item in tab20_list:
	tab20_palette = tab20_palette + item

train_scene_names = [os.path.basename(x).replace('_vh_clean_2.pth', '')
	for x in sorted(glob.glob('/media/root/data/ScanNet_v2_data/train/*_vh_clean_2.pth'))]
val_scene_names = [os.path.basename(x).replace('_vh_clean_2.pth', '')
	for x in sorted(glob.glob('/media/root/data/ScanNet_v2_data/val/*_vh_clean_2.pth'))]

src_image_path = '/local/zoli/3DMV/data/scannetv2_images/%s/color/'
src_label_path = '/local/zoli/3DMV/data/scannetv2_images/%s/label/'
dst_image_path = '/local/zoli/SemSeg2D/datasets/scannet_v2/%s/images/%s'
dst_label_path = '/local/zoli/SemSeg2D/datasets/scannet_v2/%s/labels/%s'

for tv, scene_names in zip(['train', 'val'], [train_scene_names, val_scene_names]):
	os.popen('mkdir -p /local/zoli/SemSeg2D/datasets/scannet_v2/%s/images' % tv)
	for scene_name in tqdm.tqdm(scene_names):
		os.popen('cp -r %s %s' % (src_image_path % scene_name, dst_image_path % (tv, scene_name)))
		label_files = sorted(glob.glob(src_label_path % scene_name + '*.png'))
		os.popen('mkdir -p ' + dst_label_path % (tv, scene_name))
		time.sleep(0.2)
		for label_file in label_files:
			basename = os.path.basename(label_file)
			label = np.array(Image.open(label_file)).astype(np.int32)
			label = Image.fromarray(label.astype(np.uint8))
			label.putpalette(tab20_palette)
			tab20_palette.save(dst_label_path % (tv, scene_name) + '/' + basename)


