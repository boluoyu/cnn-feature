import sys
import os
import cv2

paramFile = './params/All.txt'

# create directories
folders = []
with open(paramFile) as f:
	for line in f:
		terms = line.split('\t')
		if len(terms) > 2:
			folders.append(terms[1])

print 'folders:',folders

crop_root = './crop_img'
if not os.path.exists(crop_root):
	os.mkdir(crop_root)

for folder in folders:
	path = os.path.join(crop_root, folder)
	if not os.path.exists(path):
		os.mkdir(path)

bbox_root = './bbox/hand'
bbox_files = [os.path.join(bbox_root, f) for f in os.listdir(bbox_root) if f.endswith('txt')]
print '{0} bbox files'.format(len(bbox_files))

bf = bbox_files[0]
with open(bf) as f:
	line = f.readline()
	print line
	print line.split('\t')

img_root = './img'

for bf in bbox_files:
	terms = bf.split('_')
	folder = terms[1]
	img_index = int(terms[2].split('.')[0])
	img_file = os.path.join(img_root, folder, str(img_index) + '.jpg')
	with open(bf) as f:
		for line in f:
			terms = line.split('\t')
			frame = terms[1]
			index = terms[2]
			x = int(terms[3])
			y = int(terms[4])
			w = int(terms[5])
			h = int(terms[6])

			I = cv2.imread(img_file)
			Ic = I[y : y + h, x : x + w, :]

			crop_img = os.path.join(crop_root, folder, str(img_index) + '_' + str(index) + '.jpg')
			print 'saving:',crop_img
			cv2.imwrite(crop_img, Ic)
			'''
			cv2.imshow('raw', I)
			cv2.imshow('hand', Ic)
			cv2.waitKey(0)
			'''