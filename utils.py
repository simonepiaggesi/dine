import numpy as np
import torch
import gzip
from sklearn.datasets import make_blobs

############################################

class DataHandler:

	def __init__(self):
		pass

	def loadData(self, filename):
		print("Loading data from " + filename )
		lines = gzip.open(filename, 'rt').readlines()
		self.data = []
		self.nodes = []
		for line in lines:
			tokens = line.strip().split()
			self.nodes.append(tokens[0])
			self.data.append([float(i) for i in tokens[1:]])
		self.data = np.array(self.data)
		print("Loaded data. #shape = " + str(self.data.shape) )
		print(" #nodes = %d " %(len(self.nodes)) )
		self.data_size = self.data.shape[0]
		self.inp_dim = self.data.shape[1]
		self.original_data = self.data[:]
		self.data = self.data[np.argsort(np.array(self.nodes).astype(int)), :]
		
	def getNodesList(self):
		return self.nodes

	def getDataShape(self):
		return self.data.shape

	def resetDataOrder(self):
		self.data = self.original_data[:]

	def getNumberOfBatches(self, batch_size):
		return int(( self.data_size + batch_size - 1 ) / batch_size)
	
	def getBatch(self, i, batch_size, noise_level, denoising):
		batch_y = self.data[i*batch_size:min((i+1)*batch_size, self.data_size)]
		batch_x = batch_y
		if denoising:
			batch_x = batch_y + get_noise_features(batch_y.shape[0], self.inp_dim, noise_level)
		return batch_x, batch_y

	def shuffleTrain(self):
		indices = np.arange(self.data_size)
		np.random.shuffle(indices)
		self.data = self.data[indices]

############################################

def dump_embs(X, outfile, nodes):
	print ("shape", X.shape)
	assert len(X) == len(nodes)
	fw = gzip.open(outfile, 'wt')
	for i in range(len(nodes)):
		fw.write(nodes[i] + " ")
		for j in X[i]:
			fw.write(str(j) + " ")
		fw.write("\n")
	fw.close()

def load_embs(infile):
	emb_file = gzip.open(infile, 'rt')
	Lines = emb_file.readlines()
	keys = []
	X = []
	for line in Lines:
		keys.append(line.strip().split()[0])
		X.append(np.array([float(x) for x in line.strip().split()[1:]]).astype(np.float32))
	return(dict(zip(keys, X)))

def get_noise_features(n_samples, n_features, noise_amount):
	noise_x,  _ =  make_blobs(n_samples=n_samples, n_features=n_features, 
			cluster_std=noise_amount,
			centers=np.array([np.zeros(n_features)]))
	return noise_x

############################################

from functools import partial
from multiprocessing import Pool
from collections import Counter

def interpret_single_dim(edge_emb, edge_com, edge_scores_all, max_dims, dim):

	'''
	Function to compute individual interpretability scores from a given k-dim embedding of the graph with N nodes and E edges.
	Input:
	    edge_emb: numpy array with the shape (E, k) containing edge-level embeddings.
	    			Tipically it is obtained with element-wise product of node embeddings.
	    edge_com: numpy array with the shape (E,) containing edge community labels.
	    edge_scores_all: numpy array with the shape (E,) containing average edge product scores over all the dimensions.
	    max_dims: number of dimensions k of the embedding.
	    dim: the dimension index to interpreted.
	Output:
	    dim: the dimension index.
	    mask: numpy array with the shape (E,) containing the edge utility scores for the single dimension.
		label: the label of the community with maximum dimension matching.
		iscore: community-aware interpretability score for the single dimension.
		escore: sparsity-aware interpretability score for the single dimension.
	'''

	THRESH = 1e-9

	dims_slice = [d for d in range(max_dims) if d!=dim]
	edge_scores_removed = np.mean(edge_emb[:, dims_slice], axis=1)

	mask = np.maximum(edge_scores_all - edge_scores_removed, THRESH)

	edge_labels = edge_com[mask>THRESH]
	edge_importance = mask[mask>THRESH]

	if len(edge_importance)==0:
		return dim, mask, None, 0. , 1.

	M = ((mask>THRESH).astype(float) + 1e-15)
	Z = M.sum(axis=0)
	E = -np.sum((M/Z) *np.log(M/Z), axis=0)
	escore = E/np.log(M.shape[0])
		
	community_counts = dict(Counter(edge_com))
	ccc = Counter(edge_labels)

	label_scores = []

	for label in np.unique(edge_com):

		if label in ccc:
			precision = ccc[label]/sum([n for c,n in ccc.most_common()])
			recall = ccc[label]/community_counts[label]
			interpretability_score = 2*precision*recall/(precision+recall)
		else:
			interpretability_score = 0.

		label_scores.append((interpretability_score, label))

	iscore, label = sorted(label_scores, reverse=True)[0]
	
	return dim, mask, label, iscore, escore


def edge_interpretability_parallel(graph, node_emb, edge_com, workers=12):

	'''
	Function to compute global interpretability scores from a given k-dim embedding of the graph with N nodes and E edges.
	Input:
		graph: networkx graph (unweighted and undirected).
	    node_emb: numpy array with the shape (N, k) containing node embeddings.
	    edge_com: numpy array with the shape (E,) containing edge community labels.
	    workers: number of parallel processes.
	Output:
	    idims: numpy array of dimensions' indices.
	    imasks: numpy array with the shape (k, E) containing the per-dimension edge utility scores.
		ilabels: numpy array with labels of the communities with maximum per-dimension matching.
		iscores: numpy array of community-aware interpretability scores.
		ientropies: numpy array of sparsity-aware interpretability scores.
	'''
	
	node_name = np.array(list(graph.nodes()))
	node_dict = dict(zip(node_name, np.arange(len(node_name))))
	
	edge_idx = np.array([[node_dict[i], node_dict[j]] for (i,j) in graph.edges()])
	edge_emb = node_emb[edge_idx[:,0]] * node_emb[edge_idx[:,1]]
	
	dimensions = node_emb.shape[1]
	
	with Pool(processes=workers) as pool:  
		ilist = pool.map(partial(interpret_single_dim, 
								edge_emb, edge_com, np.mean(edge_emb, axis=1), dimensions), 
								range(dimensions))
		pool.close()
		pool.join() 
			
	if len(ilist)>0:
		idims, imasks, ilabels, iscores, ientropies = zip(*ilist)
		return np.array(idims), np.array(imasks), np.array(ilabels), np.array(iscores), np.array(ientropies)
	else:
		return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])