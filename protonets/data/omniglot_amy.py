import os
import sys
import glob

from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

OMNIGLOT_DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data/omniglot')
OMNIGLOT_CACHE = { }

def load_image_path(key, out_field, d):
	d[out_field] = Image.open(d[key])
	return d

def convert_tensor(key, d):
	d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
	return d

def rotate_image(key, rot, d):
	d[key] = d[key].rotate(rot)
	return d

def scale_image(key, height, width, d):
	d[key] = d[key].resize((height, width))
	return d

omniglot_amy_root = '/home/xamyzhao/datasets/omniglot/images_background/'
vinyals_split_files = ['../../data/omniglot/splits/vinyals/train.txt']

def load_class_images(d, filter_by_files=None):

	if d['class'] not in OMNIGLOT_CACHE:
		if len(d['class'].split('/')) == 3:
			alphabet, character, rot = d['class'].split('/') #, rot = d['class'].split('/')
		else:
			# if rotation info is not included, just assume no rotation
			alphabet, character = d['class'].split('/') #, rot = d['class'].split('/')
			rot = 'rot0'  

		image_dir = os.path.join(omniglot_amy_root, alphabet, character)

		class_images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
		if len(class_images) == 0:
			raise Exception("No images found for omniglot class {} at {}. Did you run download_omniglot.sh first?".format(d['class'], image_dir))

		if filter_by_files is not None:
			class_images = [ci for ci in class_images if ci in filter_by_files]

		print('Loaded {} images for class {}!'.format(len(class_images), d['class']))

		image_ds = TransformDataset(ListDataset(class_images),
									compose([partial(convert_dict, 'file_name'),
											 partial(load_image_path, 'file_name', 'data'),
											 partial(rotate_image, 'data', float(rot[3:])),
											 partial(scale_image, 'data', 28, 28),
											 partial(convert_tensor, 'data')]))

		loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

		for sample in loader:
			OMNIGLOT_CACHE[d['class']] = sample['data']
			break # only need one sample because batch size equal to dataset length

	return { 'class': d['class'], 'data': OMNIGLOT_CACHE[d['class']] }

def extract_episode(n_support, n_query, d):
	# data: N x C x H x W
	n_examples = d['data'].size(0)

	if n_query == -1:
		n_query = n_examples - n_support

	example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
	support_inds = example_inds[:n_support]
	query_inds = example_inds[n_support:]

	xs = d['data'][support_inds]
	xq = d['data'][query_inds]
	#xs = X[support_inds]

	return {
		'class': d['class'],
		'class_s': [d['class']] * len(support_inds),  # we will concat across classes later, so keep track
		'xs': xs,
		'class_q': [d['class']] * len(query_inds),
		'xq': xq
	}

def extract_validation_episode(d, n_query, xs, support_classes):
	# xs should be all of the support examples from training
	n_examples = d['data'].size(0)
	query_inds = torch.randperm(n_examples)[:(n_query)]

	xq = d['data'][query_inds]
	curr_class_idxs = [i for i in range(len(support_classes)) if support_classes[i] == d['class']]

	return {
		'class': d['class'],
		'class_s': [d['class']] * len(curr_class_idxs),
		'xs': xs[curr_class_idxs],
		'class_q': [d['class']] * len(query_inds),
		'xq': xq
	}

	

def load(opt, splits):
	'''

	ret = { }
	for split in splits:
		if split in ['val', 'test'] and opt['data.test_way'] != 0:
			n_way = opt['data.test_way']
		else:
			n_way = opt['data.way']

		if split in ['val', 'test'] and opt['data.test_shot'] != 0:
			n_support = opt['data.test_shot']
		else:
			n_support = opt['data.shot']

		if split in ['val', 'test'] and opt['data.test_query'] != 0:
			n_query = opt['data.test_query']
		else:
			n_query = opt['data.query']

		if split in ['val', 'test']:
			n_episodes = opt['data.test_episodes']
		else:
			n_episodes = opt['data.train_episodes']
	'''
	# import my own omniglot loader
	import sys
	sys.path.append('../LPAT')
	from dataset_utils import omniglot_loader
	import classification_utils
	data_params = {
		'img_shape': (28, 28, 1),  # default
		'n_old_train': 20,  # just load all examples into X_old_train
		'n_old_validation': 0,
		'n_shot': 1,
		'n_new_validation': 10,
		'split_id': None,
		'use_langs': opt['data.languages']
	}
	(X_old_train, Y_old_train, old_train_files), \
	(X_old_valid, Y_old_valid, old_valid_files), \
	(X_new_train, Y_new_train, new_train_files), \
	(X_new_valid, Y_new_valid, new_valid_files), label_mapping \
		= omniglot_loader.load_dataset(data_params)	
	train_files = list(set(old_train_files + new_train_files))
	valid_files = list(set(new_valid_files))

	train_classes = list(set([os.path.dirname(f).replace(omniglot_amy_root, '') for f in old_train_files + new_train_files]))
	valid_classes = list(set([os.path.dirname(f).replace(omniglot_amy_root, '') for f in new_valid_files]))


	# load class lists from predefined split .txt files so that we can incorporate rotation aug
	# as in the original alg
	split_dir = os.path.join(OMNIGLOT_DATA_DIR, 'splits', 'vinyals')
	presplit_class_names = []
	with open(os.path.join(split_dir, "{:s}.txt".format('train')), 'r') as f:
		for class_name in f.readlines():
			presplit_class_names.append(class_name.rstrip('\n'))

	train_classes_with_rot = []#train_classes
	for tc in train_classes:
		# find corresponding entries in vinyals split so that we know what rotation augs to apply
		matching_presplit_class_names = [pcn for pcn in presplit_class_names if '/'.join(pcn.split('/')[:-1]) in tc]
		train_classes_with_rot += matching_presplit_class_names
	valid_classes_with_rot = [vc + '/rot000' for vc in valid_classes]
	class_rot_mapping = train_classes_with_rot + valid_classes_with_rot
	print('Total of {} classes with rotation'.format(len(class_rot_mapping)))


	#all_classes = list(set(train_classes + valid_classes))



	ret = {}
	for split in splits:
		if split == 'val' or split == 'test':
			n_episodes = 1
			sample_from_classes = class_rot_mapping

			print('Validating on classes {}'.format(valid_classes))
			print('Validating on ims {}'.format(valid_files))
			
			n_query = opt['data.query']
			support_ims = np.transpose(np.concatenate([X_old_train, X_new_train], axis=0), (0, 3, 1, 2)) 
			print('Using {} support examples, {} query'.format(support_ims.shape[0], n_query))
			support_ims = torch.tensor(support_ims)
			support_classes=[os.path.dirname(f).replace(omniglot_amy_root, '') for f in old_train_files + new_train_files]
	
			transforms = [partial(convert_dict, 'class'),
						  partial(load_class_images, filter_by_files=valid_files),  
						  partial(extract_validation_episode, 
							n_query=n_query, 
							xs=support_ims,
							support_classes=support_classes
							)]
			'''
			transforms = [
					# of our old class examples, split into "train" and "validation" for each episode
					partial(extract_episode
				]
			'''
			

		else:  # training
			n_episodes = opt['data.train_episodes']
			n_support = opt['data.shot']
			n_query = opt['data.query']

			sample_from_classes = class_rot_mapping

			print('Training on classes {}'.format(sample_from_classes))
			print('Extracting {} support examples, {} query'.format(n_support, n_query))
			transforms = [partial(convert_dict, 'class'),
						  partial(load_class_images, filter_by_files=train_files), 
						  partial(extract_episode, n_support, n_query)]

		if opt['data.cuda']:
			transforms.append(CudaTransform())

		transforms = compose(transforms)

		# always train on all-way classification

		n_way = len(sample_from_classes)
		print('N-way classification: {}'.format(n_way))

		ds = TransformDataset(ListDataset(sample_from_classes), transforms)
	
		# selects a random n_way number of classes from the class names in ds	
		sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

		'''
		if opt['data.sequential']:
			sampler = SequentialBatchSampler(len(ds))
		else:
			sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)
		'''

		# use num_workers=0, otherwise may receive duplicate episodes
		ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0, 
			collate_fn=partial(collate_support_query, class_mapping=class_rot_mapping))

	return ret

def collate_support_query(datapoint_list, class_mapping):
	batch = {}
	batch['xs'] = torch.cat([d['xs'] for d in datapoint_list], dim=0)
	batch['xq'] = torch.cat([d['xq'] for d in datapoint_list], dim=0)
	batch['class_s'] = []
	for d in datapoint_list:
		batch['class_s'] += d['class_s']  # concatenate all lists end to end

	batch['class_q'] = []
	for d in datapoint_list:
		batch['class_q'] += d['class_q']  # concatenate all lists end to end

	batch['class'] = [d['class'] for d in datapoint_list]
	batch['class_mapping'] = class_mapping
	return batch
	
	

