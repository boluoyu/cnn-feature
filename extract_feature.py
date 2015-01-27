import numpy as np
import os
import sys
import multiprocessing as mp
import signal
import matplotlib.pyplot as plt

caffe_root = '/home/minghuam/Documents/Dev/caffe'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

def convertToCvXML(feature, featData):
	rows = featData.shape[0]
	cols = featData.shape[1]
	dataStr = '\n'
	length = 0
	for r in range(rows):
		for c in range(cols):
			s = ' {0:.8f}'.format(featData[r, c])
			dataStr += s
			length += len(s)
			if length > 70:
				length = 0
				dataStr += '\n'
	xml = '<?xml version="1.0"?>\n<opencv_storage>\n<{0} type_id="opencv-matrix">\n' \
  		'<rows>{1}</rows>\n<cols>{2}</cols>\n<dt>f</dt>\n<data>{3}\n</data>\n</{0}>\n</opencv_storage>' \
  		.format(feature, rows, cols, dataStr)
  	return xml

def extractFeatures(net, feature, imgDir, outDir):
	imgList = [img for img in os.listdir(imgDir) if img.endswith('.jpg')]
	
	imgs = dict()
	# some images might have multiple hands
	for img in imgList:
		terms = img.split('_')
		frame = int(terms[0])
		if not frame in imgs:
			imgs[frame] = 0
		imgs[frame] += 1

	for frame in imgs:
		featData = None
		for index in range(imgs[frame]):
			img = os.path.join(imgDir, str(frame) + '_' + str(index) + '.jpg')
			print img
			net.predict([caffe.io.load_image(img)])
			f = net.blobs[feature].data[0].ravel()
			if featData is None:
				featData = np.zeros((imgs[frame], f.shape[0]))
			featData[index, :] = f
		xml = convertToCvXML(feature, featData)
		xmlFile = os.path.join(outDir, 'bbox_HAND_feat_{0}_{1:08d}.xml'.format(feature, frame))
		print xmlFile
		with open(xmlFile, 'w') as f:
			f.write(xml)

def doJob(jobID, net, feature, imgDir, featureDir):
	print 'JOB[{0}] started: {1}, {2}'.format(jobID, imgDir, featureDir)
	extractFeatures(net, feature, imgDir, featureDir)

def sigint_handler(signal, frame):
	print 'SIGINT: PID = {0}'.format(os.getpid())
	sys.exit(0)

def sigterm_handler(signal, frame):
	print 'SIGTERM: PID = {0}'.format(os.getpid())
	sys.exit(0)

def main():
	rootDir = './'
	paramFile = os.path.join(rootDir, 'params', 'All.txt')
	feature = 'pool5'
	featureFolder = 'bbox_HAND_feat_' + feature

	# folders: 201, 202...
	folders = []
	with open(paramFile) as f:
		for line in f:
			terms = line.split('\t')
			if len(terms) > 2:
				folders.append(terms[1])

	# create directories
	for folder in folders:
		folderDir = os.path.join(rootDir, 'features', folder)
		if not os.path.exists(folderDir):
			continue
		featureDir = os.path.join(folderDir, featureFolder)
		if not os.path.exists(featureDir):
			os.mkdir(featureDir)

	# load cnn net
	net = caffe.Classifier(os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet', 'deploy.prototxt'),
	                       os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet', 'bvlc_reference_caffenet.caffemodel'))

	# ImageNet mean
	net.set_mean('data', np.load(os.path.join(caffe_root, 'python', 'caffe', 'imagenet', 'ilsvrc_2012_mean.npy')))

	# the reference model operates on images in [0,255] range instead of [0,1]
	net.set_raw_scale('data', 255)

	# the reference model has channels in BGR order instead of RGB
	net.set_channel_swap('data', (2,1,0))

	'''
	# extract and save features, single process
	for folder in folders:
		imgDir = os.path.join(rootDir, 'crop_img', folder)
		featureDir = os.path.join(rootDir, 'features', folder, featureFolder)
		extractFeatures(net, feature, imgDir, featureDir)
	'''

	# dispatch job to maximum 8 processes
	# signal handler
	print 'main pid = ',os.getpid()
	signal.signal(signal.SIGINT, sigint_handler)
	signal.signal(signal.SIGTERM, sigterm_handler)

	cpus = mp.cpu_count()
	if cpus > 8:
		cpus = 8
	nfolders = len(folders)

	nJobsDone = 0
	while nJobsDone < nfolders:
		nworkers = nfolders - nJobsDone
		if nworkers > cpus:
			nworkers = cpus
		workers = []
		for i in range(nworkers):
			jobID = nJobsDone + i
			imgDir = os.path.join(rootDir, 'crop_img', folders[jobID])
			featureDir = os.path.join(rootDir, 'features', folders[jobID], featureFolder)
			args = (jobID, net, feature, imgDir, featureDir)
			p = mp.Process(target = doJob, args = args)
			workers.append(p)
			p.start()

		for i in range(nworkers):
			workers[i].join()
			jobID = nJobsDone + i
			print str(jobID) + " joined."

		nJobsDone += nworkers

if __name__ == '__main__':
	main()