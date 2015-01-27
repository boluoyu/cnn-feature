import os
import shutil

def main():
	rootDir = './'
	paramFile = os.path.join(rootDir, 'params', 'All.txt')
	feature = 'pool5'
	featureFolder = 'bbox_HAND_feat_' + feature
	hogFeature = 'HANDHOG'
	hogFeatureFolder = 'bbox_HAND_feat_' + hogFeature

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
		featureDir = os.path.join(folderDir, featureFolder)
		featureFiles = [f for f in os.listdir(featureDir) if f.endswith('.xml')]

		hogFeatureDir = os.path.join(folderDir, hogFeatureFolder)
		hogFeatureFiles = [f.replace(hogFeature, feature) for f in os.listdir(hogFeatureDir) if f.endswith('.xml')]

		count = 0
		for f in featureFiles:
			if f not in hogFeatureFiles:
				f = os.path.join(featureDir, f)
				print f
				os.remove(f)

if __name__ == '__main__':
	main()