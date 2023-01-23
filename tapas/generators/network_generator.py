from random import choices

import networkx as nx
import pandas as pd

from ..datasets import TUDataset
from .generator import Generator


class GNP(Generator):
    """A Gnp (E-R model) synthetic network generator."""
    def __init__(self):
        super().__init__()

    def fit(self, dataset):
        self.dataset = dataset
        self.trained = True

    def generate(self, num_samples = None):
        """
        Generate a list of Gnp random graphs using the parameters calculated from input datasets

        Parameters
        ---------
        num_samples: # of Gnp samples to be generated

        Returns
        -------
            TUDataset
                A list (Pandas.Series) of Gnp graph objects
        """
        if self.trained:
            if num_samples is None:
                return self.dataset

            data = self.dataset.data
            graph_nodes = [data.iloc[i].number_of_nodes() for i in range(len(data))]
            graph_edges = [data.iloc[i].number_of_edges() for i in range(len(data))]

            param_dict = {}
            for i in range(len(data)):
                # compute parameters n, p for each input graph
                n = graph_nodes[i]
                p = float(graph_edges[i])/((n-1)*n/2)
                param = [n, p, 1]

                if n not in param_dict.keys():
                    param_dict[n] = param
                else:
                    # if n is present in param_dict
                    # update parameters based on repeat times
                    count = param_dict[n][-1]
                    param_dict[n] = [(param_dict[n][i] * count + param[i]) / (count + 1)
                                     for i in range(len(param_dict[n]))]
                    param_dict[n][-1] = count + 1

            # choose *num_samples* parameter pairs from the pool and
            keys = choices(list(param_dict), k=num_samples)
            graphs = []
            for i in keys:
                nodes = graph_nodes[i]
                n = int(param_dict[nodes][0])
                p = param_dict[nodes][1]
                graph = nx.fast_gnp_random_graph(n, p)
                # initialise node labels used for graph kernel
                labels = dict(zip(graph, [0]*len(graph)))
                nx.set_node_attributes(graph, labels, "label")

                graphs.append(graph)

            return TUDataset(pd.Series(graphs), self.dataset.description)
        else:
            raise RuntimeError("No dataset provided to generator")
        
    def __call__(self, dataset, num_samples, random_state = None):
        self.fit(dataset)
        return self.generate(num_samples)

    @property
    def label(self):
        return "Gnp"


