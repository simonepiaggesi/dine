import torch
from torch import nn
import argparse
import utils
from utils import DataHandler
from model import DINEModel
import numpy as np

#########################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input', dest='input', required=True, 
					help='Input embedding file')

parser.add_argument('--output', dest='output', required=True, 
					help='Output embedding file')

parser.add_argument('--emb-dim', dest='emb_dim', type=int, default=32,
                    help='Embedding size')

parser.add_argument('--denoising', dest='denoising',
					default=False,
					action='store_true',
                    help='Denoising auto-encoder')

parser.add_argument('--noise-level', dest='noise_level', type=float,
					default=0.2,
                    help='Noise amount for denoising auto-encoder')

parser.add_argument('--num-epochs', dest='num_epochs', type=int,
					default=2000,
                    help='Number of epochs')

parser.add_argument('--lambda-size', dest='lambda_size', type=float,
					default=1.,
                    help='Size regularization coeff.')

parser.add_argument('--lambda-orth', dest='lambda_orth', type=float,
					default=1.,
                    help='Orthogonality regularization coeff.')

parser.add_argument('--learning-rate', dest='learning_rate', type=float,
					default=0.1,
                    help='Learning rate')

parser.add_argument('--seed', default=42, type=int,
                  help='Seed')

#########################################################

class Solver:

	def __init__(self, params):

		# Build data handler
		self.data_handler = DataHandler()
		self.data_handler.loadData(params['input'])
		params['inp_dim'] = self.data_handler.getDataShape()[1]
		params['batch_size'] = self.data_handler.getDataShape()[0]

		# Build model
		self.model = DINEModel(params)
		self.dtype = torch.FloatTensor
		use_cuda = torch.cuda.is_available()
		if use_cuda:
			self.model.cuda()
			self.dtype = torch.cuda.FloatTensor
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])


	def train(self, params):
		num_epochs, batch_size = params['num_epochs'], params['batch_size'],
        
		optimizer = self.optimizer
		dtype = self.dtype

		for iteration in range(num_epochs+1):
			self.data_handler.shuffleTrain()
			num_batches = self.data_handler.getNumberOfBatches(batch_size)
			epoch_losses = np.zeros(4)
			for batch_idx in range(num_batches):
				# Zero the gradients
				optimizer.zero_grad()
                
				# Forward and backward propagation
				batch_x, batch_y = self.data_handler.getBatch(batch_idx, batch_size, params['noise_level'], params['denoising'] )
				batch_x = torch.from_numpy(batch_x).type(dtype)
				batch_y = torch.from_numpy(batch_y).type(dtype)
				out, h, loss, loss_terms = self.model(batch_x, batch_y)
				nlosses = len(loss_terms)
				loss.backward()
				optimizer.step()
				for idx, loss_term in enumerate(loss_terms):
					epoch_losses[idx] += loss_term.item()
				epoch_losses[idx+1] = loss.item()                
            
			# Show progress
			if iteration % 500 == 0:
				print("After epoch %i, Rec. Loss = %.5f, Size Loss = %.5f, Orth. Loss = %.5f, and Total = %.5f"
						%(iteration, epoch_losses[0], epoch_losses[1], epoch_losses[2], epoch_losses[3]) )

	def getDineEmbeddings(self, batch_size, params):
		ret = []
		self.data_handler.resetDataOrder()
		num_batches = self.data_handler.getNumberOfBatches(batch_size)
		for batch_idx in range(num_batches):
			batch_x, batch_y = self.data_handler.getBatch(batch_idx, batch_size, params['noise_level'], params['denoising'] )
			batch_x = torch.from_numpy(batch_x).type(self.dtype)
			batch_y = torch.from_numpy(batch_y).type(self.dtype)
			_, h, _, _ = self.model(batch_x, batch_y)
			ret.extend(h.cpu().data.numpy())
		return np.array(ret)

	def getNodesList(self):
		return self.data_handler.getNodesList()


#########################################################

def main():

	params = vars(parser.parse_args())
	torch.manual_seed(params['seed'])
	solver = Solver(params)
	solver.train(params)
		
	# dumping the final vectors
	print("Dumping the DINE embeddings")
	output_path = params['output'] 
	final_batch_size = 512
	dine_embeddings = solver.getDineEmbeddings(final_batch_size, params)
	utils.dump_embs(dine_embeddings, output_path, solver.getNodesList())


if __name__ == '__main__':
	main()
