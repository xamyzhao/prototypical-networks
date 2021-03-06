import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class Protonet(nn.Module):
	def __init__(self, encoder):
		super(Protonet, self).__init__()
		
		self.encoder = encoder

	def loss(self, sample):
		xs = Variable(sample['xs']).view(-1, 1, 28, 28) # support
		xq = Variable(sample['xq']).view(-1, 1, 28, 28) # query
		support_classes = sample['class_s']
		query_classes = sample['class_q']
		n_support = xs.size(0)
		n_query = xq.size(0)

		if 'class_mapping' in sample.keys():
			class_mapping = sample['class_mapping']
		else:
			class_mapping = sorted(list(set(support_classes + query_classes)))
		# we would like 1 at these class indices for each query example
		target_inds = [class_mapping.index(c) for c in query_classes] 
		target_inds = torch.tensor(target_inds).view(n_query, 1).long()
		target_inds = Variable(target_inds, requires_grad=False)

		if xq.is_cuda:
			target_inds = target_inds.cuda()

		x = torch.cat([xs.view(n_support, *xs.size()[1:]),
					   xq.view(n_query, *xq.size()[1:])], 0)

		#z = self.encoder.forward(x)
		zq = self.encoder.forward(xq)
		zs = self.encoder.forward(xs)
		z_dim = zq.size(-1)
		
		# get prototypes for each class. Order classes in the the same order as class_mapping
		z_protos = []
		for c in class_mapping:
			curr_class_ex_idxs = [i for i in range(len(support_classes)) if support_classes[i] == c]
			z_protos.append(zs[curr_class_ex_idxs].mean(0).view(1, -1))
		z_proto = torch.cat(z_protos, 0)
		#zq = z[n_support:]

		dists = euclidean_dist(zq, z_proto)
		# softmax over classes?
		log_p_y = F.log_softmax(-dists, dim=1).view(n_query, -1)

		# want to maximize the value at the true class for each query example
		loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()
		#loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

		# predicted class is the one with highest probability
		_, y_hat = log_p_y.max(1)

		acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

		return loss_val, {
			'loss': loss_val.item(),
			'acc': acc_val.item()
		}

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
	x_dim = kwargs['x_dim']
	hid_dim = kwargs['hid_dim']
	z_dim = kwargs['z_dim']

	def conv_block(in_channels, out_channels):
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

	encoder = nn.Sequential(
		conv_block(x_dim[0], hid_dim),
		conv_block(hid_dim, hid_dim),
		conv_block(hid_dim, hid_dim),
		conv_block(hid_dim, z_dim),
		Flatten()
	)

	return Protonet(encoder)
