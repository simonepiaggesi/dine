# DINE: Dimensional Interpretability of Node Embeddings
In this repository you will find code to run experiments of the paper [DINE: Dimensional Interpretability of Node Embeddings](https://ieeexplore.ieee.org/abstract/document/10591463). The code has been tested with Python 3.6.13.

If you use the code in this repository, please consider citing us:
```bibtex
@article{piaggesi2024dine,
  title={Dine: Dimensional interpretability of node embeddings},
  author={Piaggesi, Simone and Khosla, Megha and Panisson, Andr{\'e} and Anand, Avishek},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```

## Repository organization

### Python scripts

To generate DINE embedding, run the following command:

```bash
python3 dine.py --input input_embedding \
                --output output_embedding \
                --emb-dim size \
                --noise-level 0.2 \
                --num-epochs 2000 \
                --learning-rate 0.1 \
                --lambda-size l_size \
                --lambda-orth l_orth \
```
The input and output embedding are compressed text files where each line contains the node index and embedding entries in a space-separated format:

```
node0 0.21 0.14 0.48 ...
node1 0.35 0.56 0.12 ...
node2 ...
```

Input embeddings can be generated by any user-defined method. Please refer to `deepwalk.py` to write your own embedding script. 

### Jupyter notebooks
- `DeepWalk-Subgraphs-KarateClub.ipynb`
It shows how to draw pictures of utility-induced subgraphs on the Karate Club graph. 

- `DeepWalk+DINE-Interpretability-Scores-SBM.ipynb`
It shows how to compare interpretability metrics on DeepWalk embeddings before and after the application of DINE, on synthetic graph generated with the Stochastic Block Model.

- `DeepWalk+DINE-Link-Prediction-Cora.ipynb`
It shows the link prediction performance of DeepWalk embeddings before and after the application of DINE on the Cora graph.
