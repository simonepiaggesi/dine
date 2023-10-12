import torch
from torch import nn

EPS = torch.tensor(1e-15)

class DINEModel(torch.nn.Module):

	def __init__(self, params):
		super(DINEModel, self).__init__()
		
		# params
		self.inp_dim = params['inp_dim']
		self.hdim = params['emb_dim']
		self.noise_level = params['noise_level']
		self.lsize = params['lambda_size']
		self.lorth = params['lambda_orth']

        # autoencoder
		print("Building model ")
		self.linear1 = nn.Linear(self.inp_dim, self.hdim)
		self.linear2 = nn.Linear(self.hdim, self.inp_dim)
		

	def forward(self, batch_x, batch_y):
		
		#forward
		batch_size = batch_x.size(0)
		linear1_out = self.linear1(batch_x)
		h = linear1_out.sigmoid()
		out = self.linear2(h)
                
		partitions = (h * h.sum(axis=0))

		# different terms of the loss
		reconstruction_loss = self._getReconstructionLoss(out, batch_y) # reconstruction loss 
		size_loss = self._getSizeLoss(h.T) # size loss
		orth_loss = self._getOrthogonalityLoss(partitions.T) # orthogonality loss
		total_loss = reconstruction_loss + self.lsize * size_loss + self.lorth * orth_loss 
		
		return out, h, total_loss, [reconstruction_loss, size_loss, orth_loss]

	def _getReconstructionLoss(self, x, y):
		loss = nn.MSELoss(reduction='mean')
		return loss(x, y)
        
	def _getSizeLoss(self, mask):
		axs = torch.arange(mask.dim())        
		mask_size = torch.sum(mask, axis=tuple(axs[1:]))
		mask_norm = mask_size / torch.sum(mask_size, axis=0)
		mask_ent = torch.sum(- mask_norm * torch.log(mask_norm + EPS), axis=0)

		max_ent = torch.log(torch.tensor(mask.shape[0], dtype=torch.float32))
		if torch.cuda.is_available():
			max_ent = max_ent.cuda()
		return max_ent - torch.mean(mask_ent)
    
	def _getOrthogonalityLoss(self, P):
		O = P.matmul(P.T)
		I = torch.eye(O.shape[0])

		if torch.cuda.is_available():
			I = I.cuda()
		return self._getReconstructionLoss(O/torch.norm(O), I/torch.norm(I))


