"""
Attacks designed for network data.

The attacks presented in this file are extensions of TAPAS attacks for tabular
data to graph data. As such, they extend the objects defined in tapas.attacks.

"""

# Type checking stuff
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset, TUDataset

from .shadow_modelling import ShadowModellingAttack
from .set_classifiers import SetClassifier, FeatureBasedSetClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

import numpy as np
import networkx as nx
from grakel import WeisfeilerLehman, VertexHistogram
from grakel.kernels import Kernel
from grakel.utils import graph_from_networkx

from .set_classifiers import SetClassifier, SetFeature


# First, we implement a custom set classifier for datasets containing multiple
# graphs, where all the graphs are composed as one large dataset.


class ComposedGraphClassifier(SetClassifier):
    """
    A classifier for a set of graphs, where all graphs in the dataset are
    composed together in a large, disconnected graph, from which [graph]
    features are then extracted.

    """

    def _relabel_graphs(self, graphs):
        """
        Relabel nodes in a list of graphs such that all graphs have non
        overlapping node sets. The relabeled graphs can be combined together
        in one large graph.

        For instance, the graphs G1 = [0, 1, 2], G2 = [0, 1] would be relabeled
        as the graphs G1' = [0, 1, 2], G2' = [3, 4].

        """
        idx = 0
        relabeled_graphs = []
        for graph in graphs:
            end = idx + len(graph) + 1
            mapping = dict(zip(range(idx, end), range(idx, end)))

            # Sorting graph nodes.
            sorted_graph = nx.Graph()
            sorted_graph.add_nodes_from(sorted(graph.nodes(data=True)))
            sorted_graph.add_edges_from(graph.edges(data=True))

            g = nx.relabel_nodes(sorted_graph, mapping)
            nx.set_node_attributes(g, mapping, "label")

            relabeled_graphs.append(g)
            idx += len(graph)

        return relabeled_graphs

    def _compose_datasets(self, datasets: list[TUDataset]):
        """
        Compose all the graphs in each TUDataset as one large graph with
        non-overlapping nodes for classification tasks.

        Parameters
        ----------
        datasets : list[TUDataset]
            List of TUDataset

        """
        composed_datasets = []
        for ds in datasets:
            relabeled_graphs = self._relabel_graphs(ds.data)
            composed_graph = nx.compose_all(relabeled_graphs)
            composed_datasets.append(composed_graph)

        return composed_datasets

    def __init__(
        self,
        classifier: ClassifierMixin = SVC(kernel="precomputed", probability=True),
        kernel: Kernel = WeisfeilerLehman(
            n_iter=4, base_graph_kernel=VertexHistogram, normalize=True
        ),
        label: str = None,
    ):
        """
        NetworkMIA aims to infer whether a given record (graph) is in the training
        dataset. The attacker uses auxiliary information, and train a classifier
        to make the prediction.

        Parameters
        ----------
        classifier: sklearn.base.ClassifierMixin (default SVM)
        kernel: grakel.kernels.Kernel (default ShortestPath)
        label: str (default None)
            An optional label to refer to the attack in reports.

        """
        self.kernel = kernel
        self.classifier = classifier
        self._label = label or "NetworkMIA"

    def fit(self, datasets: list[Dataset], labels: list[int]):
        """
        Train the attack classifier on a labelled set of datasets. The datasets
        will either be generated from threat_model or need to be provided.

        Parameters
        ----------
        threat_model : ThreatModel
            Threat model to use to generate training samples if synthetic_datasets
            or labels are not given.
        num_samples : int, optional
            Number of datasets to generate using threat_model if
            synthetic_datasets or labels are not given. The default is 100.

        """
        # Compose each dataset of graphs as a large graph.
        cds = self._compose_datasets(datasets)
        transformed_graphs = graph_from_networkx(cds, node_labels_tag="label")

        # Fit the (internal) classifier to the data.
        kernel_features = self.kernel.fit_transform(transformed_graphs)
        self.classifier.fit(kernel_features, labels)

    def predict(self, datasets: list[Dataset]):
        """
        Make a guess about the label of each dataset in the input list.

        Parameters
        ----------
        datasets : list[Dataset]
            List of (synthetic) datasets to make a guess for.

        Returns
        -------
        list[int]
            Binary guesses for each dataset.

        """

        # Compose each dataset of graphs as a large graph and transform to Grakel objects.
        cds = self._compose_datasets(datasets)
        transformed_graphs = graph_from_networkx(cds, node_labels_tag="label")

        # Extract the kernel features and feed them to the classifier.
        kernel_features = self.kernel.transform(transformed_graphs)
        return self.classifier.predict(kernel_features)

    def predict_proba(self, datasets: list[Dataset]):
        """
        Calculate classifier's raw probability about the presence of the target.
        Output is a probability in [0, 1].

        Parameters
        ----------
        datasets : list[Dataset]
            List of (synthetic) datasets to make a guess for.

        Returns
        -------
        list[float]
            List of probabilities corresponding to attacker's guess about the truth.

        """

        # Compose each dataset of graphs as a large graph and transform to Grakel objects.
        cds = self._compose_datasets(datasets)
        transformed_graphs = graph_from_networkx(cds, node_labels_tag="label")

        # Extract the kernel features and feed them to the classifier.
        kernel_features = self.kernel.transform(transformed_graphs)
        return self.classifier.predict_proba(kernel_features)

    # Map __call__ to predict.
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @property
    def label(self):
        return "ComposedGraphClassifier"


# Second, we implement graph features that can be used to produce features
# for a dataset containing multiple graphs.


class BasicNetworkFeature(SetFeature):
    """
    Extracts basic features from a graph. These basic features include the
    average degree of a node in any graph and the average clustering.

    """

    # TODO: implement better features! e.g. a graph kernel.

    def _average_degree(self, tu_dataset):
        """Return the average degree of all nodes in the whole graph."""
        return np.mean([np.mean([d for _, d in g.degree()]) for g in tu_dataset.data])

    def extract(self, datasets: list[TUDataset]) -> np.array:
        """Extract a vector of features for each dataset in the list."""
        return np.stack(
            np.concatenate(
                [
                    [np.mean([nx.average_clustering(g) for g in tu_dataset.data])],
                    [self._average_degree(tu_dataset)],
                ]
            )
            for tu_dataset in datasets
        )

    @property
    def label(self):
        return "F_Network"
