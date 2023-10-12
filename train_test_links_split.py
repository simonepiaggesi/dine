import argparse
import numpy as np
import networkx as nx
import os, gzip
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from stellargraph.data import EdgeSplitter


#########################################################

parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input', default='data/cora.edgelist.gz',
				  help='Input graph edgelist file')

parser.add_argument('--graph-output', default='predict/cora.edgelist.gz',
				  help='Output graph edgelist file')

parser.add_argument('--test-output', default='predict/cora.test.npz',
				  help='Test edges output file')

parser.add_argument('--test-frac', default=0.1, type=float,
				  help='Fraction of test edges')

parser.add_argument('--seed', default=42, type=int,
				  help='Seed for edge sampling')

#########################################################

def main():

	params = vars(parser.parse_args())

	graph = nx.read_weighted_edgelist(params['input'])
	graph.remove_edges_from(nx.selfloop_edges(graph))
	node_name = np.array([str(n) for n in graph.nodes()])

	print('Sampling edges')
	edge_splitter_test = EdgeSplitter(graph)
	graph_train, edges_test, labels_test = edge_splitter_test.train_test_split(
			p=params['test_frac'], method="global", keep_connected=True, seed=params['seed'])

	print('Writing files')
	nx.set_edge_attributes(graph_train, 1., 'weight')
	nx.write_weighted_edgelist(graph_train, params['graph_output'])
	np.savez(params['test_output'], np.concatenate((edges_test, labels_test[:, np.newaxis]), axis=1))	

if __name__ == '__main__':
	main()
