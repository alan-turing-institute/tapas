"""
Class for lunching a network-based membership inference attack on the output
of a generative model
"""
# Type checking stuff
from __future__ import annotations
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from ..datasets import Dataset

from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from grakel.kernels import ShortestPath
from grakel.kernels import Kernel
from grakel.utils import graph_from_networkx

from .base_classes import Attack
from ..threat_models import LabelInferenceThreatModel


def _relabel_graphs(graphs):
    """
    Relabel the nodes of the graphs in consecutive integers
    e.g, G1 = [0, 1, 2], G2 = [0, 1] ->
    G1 = [0, 1, 2], G2 = [3, 4]
    """
    idx = 0
    relabeled_graphs = []
    for graph in graphs:
        mapping = dict(zip(graph, range(idx, idx + len(graph))))
        g = nx.relabel_nodes(graph, mapping)
        nx.set_node_attributes(g, mapping, 'label')
        relabeled_graphs.append(g)
        idx += len(graph)

    return relabeled_graphs


def _compose_datasets(datasets):
    """
    Compose all the graphs for each TUDataset
    """
    composed_datasets = []
    for ds in datasets:
        relabeled_graphs = _relabel_graphs(ds.data)
        composed_graph = nx.compose_all(relabeled_graphs)
        composed_datasets.append(composed_graph)

    return composed_datasets


class NetworkMIA(Attack):

    def __init__(self, classifier: ClassifierMixin = SVC(kernel="precomputed", probability=True),
                 kernel: Kernel = ShortestPath(normalize=True), label: str = None):
        """
        NetworkMIA aims to infer whether a given network (graph) is in the training
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
        self.trained = False
        self._label = label or "NetworkMIA"

    def train(self, threat_model: LabelInferenceThreatModel = None, num_samples: int = 100):

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

        assert isinstance(
            threat_model, LabelInferenceThreatModel
        ), "Network attacks require a label-inference threat model."

        self.threat_model = threat_model

        # Generate data from threat model if no data is provided
        synthetic_datasets, labels = threat_model.generate_training_samples(num_samples)
        # Compose all the graphs in each synthetic dataset and
        # transform the composed datasets to Grakel objects
        ds = _compose_datasets(synthetic_datasets)
        transformed_graphs = graph_from_networkx(ds, node_labels_tag='label')

        # Fit the classifier to the data
        k_train = self.kernel.fit_transform(transformed_graphs)
        self.classifier.fit(k_train, labels)
        self.trained = True

    def attack(self, datasets: list[Dataset]) -> list[int]:
        """
        Make a guess about the target's membership in the training data that was
        used to produce each dataset in datasets.

        Parameters
        ----------
        datasets : list[Dataset]
            List of (synthetic) datasets to make a guess for.

        Returns
        -------
        list[int]
            Binary guesses for each dataset. A guess of 1 at index i indicates
            that the attack believes that the target was present in dataset i.

        """
        assert self.trained, "Attack must first be trained."

        # Compose all the graphs in each synthetic dataset and
        # transform the composed datasets to Grakel objects
        ds = _compose_datasets(datasets)
        transformed_graphs = graph_from_networkx(ds, node_labels_tag='label')
        k_test = self.kernel.transform(transformed_graphs)

        return self.classifier.predict(k_test)

    def attack_score(self, datasets: list[Dataset]) -> list[float]:
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
        ds = _compose_datasets(datasets)
        transformed_graphs = graph_from_networkx(ds, node_labels_tag='label')
        k_test = self.kernel.transform(transformed_graphs)

        scores = self.classifier.predict_proba(k_test)

        # If there are only two possible values, output the score for the positive label.
        if scores.shape[1] == 2:
            return scores[:,1]
        return scores

    @property
    def label(self):
        return self._label
