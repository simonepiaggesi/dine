import argparse
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
import gzip


#########################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input', required=True, 
                  help='Input edgelist file')

parser.add_argument('--output', required=True,
                  help='Output embedding file')

parser.add_argument('--num-walks', default=20, type=int,
                  help='Number of random walks to start at each node')

parser.add_argument('--emb-dim', default=128, type=int,
                  help='Number of latent dimensions to learn for each node')

parser.add_argument('--walk-length', default=10, type=int,
                  help='Length of the random walk started at each node')

parser.add_argument('--window-size', default=5, type=int,
                  help='Window size of skipgram model')

parser.add_argument('--workers', default=1, type=int,
                  help='Number of parallel processes')

parser.add_argument('--seed', default=42, type=int,
                  help='Seed for random walk generator')

#########################################################

def main():

	params = vars(parser.parse_args())

	graph = nx.read_weighted_edgelist(params['input'])
	graph.remove_edges_from(nx.selfloop_edges(graph))
	node_name = np.array(list(graph.nodes()))
	
	node2vec = Node2Vec(graph, dimensions=params['emb_dim'], 
						walk_length=params['walk_length'], num_walks=params['num_walks'], 
						seed=params['seed'])
	print('Training')
	model = Word2Vec(node2vec.walks, vector_size=params['emb_dim'], 
					window=params['window_size'], min_count=0, sg=1, 
					workers=params['workers'], seed=params['seed'])

	emb_X = np.array([model.wv[n] for n in node_name])

	print("Dumping the DeepWalk embeddings")
	with gzip.open(params['output'], 'wt') as f:
	    for i,n in enumerate(node_name):
	        f.write(str(n))
	        for d in emb_X[i]:
	            f.write(' ')
	            f.write('%.8f'%d)
	        f.write('\n')
	    f.close()

if __name__ == '__main__':
	main()
