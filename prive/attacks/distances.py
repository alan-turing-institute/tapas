"""
Distance metrics for closest-distance attacks.

Distances are callable objects that return an array of real number for pairs of
datasets (either records, or datasets of same lengths). We here implement a
range of simple distances, and easy methods to combine them.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset, DataDescription

from abc import ABC, abstractmethod
from ..datasets import TabularDataset
import numpy as np


class DistanceMetric:
    """
    Distance metric between datasets. This is a callable of two datasets that
    returns an array of pairwise distances, with a label for attack labelling.

    """

    @abstractmethod
    def __call__(self, x: Dataset, y: Dataset):
        """
        Compute the distance between all records in x with records in y.

        Parameters
        ----------
        x, y: two Dataset of same description.

        Returns
        -------
        distances: np.array of size len(x) x len(y).

        """
        return np.full((len(x), len(y)), np.inf)

    ## Interface to easily combine distance metrics.
    # Sum of distances.
    def __add__(self, d: DistanceMetric):
        return SumOfDistances([self, d])

    # Multiply this distance metric by scaling factor.
    def __mul__(self, factor: float):
        return ScaledDistance(self, factor)

    def __rmul__(self, factor: float):
        return self.__mul__(factor)

    @property
    def label(self):
        return self._label


class SumOfDistances(DistanceMetric):
    """(internal) class for __sum__."""

    def __init__(self, distances: list[DistanceMetric]):
        self.distances = distances
        self._label = "+".join([d.label for d in self.distances])

    def __call__(self, x: Dataset, y: Dataset):
        dists = [d(x, y) for d in self.distances]
        return sum(dists[1:], start=dists[0])


class ScaledDistance(DistanceMetric):
    """(internal) class for __prod__."""

    def __init__(self, distance: DistanceMetric, factor: float):
        self.distance = distance
        self.factor = factor
        self._label = f"{factor}*{distance.label}"

    def __call__(self, x: Dataset, y: Dataset):
        return self.factor * self.distance(x, y)


## We implement a few common distances.


class HammingDistance(DistanceMetric):
    """
    Hamming distance ("L_0"): counts the number of attributes that are
    identical between two records. While this is mainly for categorical
    attributes, it also works with continuous values.

    """

    def __init__(self, columns: list[str] = None):
        """
        Parameters
        ----------
        columns: list of column names, optional (None)
            List of the columns on which to compute the distance. If this is
            not provided, all columns are used. Only for tabular datasets.

        """
        self.columns = columns
        self._label = "Hamming"
        if columns:
            self._label += "(" + ", ".join(columns) + ")"

    def __call__(self, x: Dataset, y: Dataset):
        # Check that the datasets have the same description.
        assert (
            x.description == y.description
        ), "Input datasets must have same description."
        # The distance is implemented differently for different data types.
        if isinstance(x, TabularDataset):
            dists = np.zeros((len(x), len(y)))
            for i, (_, row) in enumerate(x.data.iterrows()):
                dists[i] = (y.data != row).sum(axis=1).values
            return dists
        else:
            raise Exception("Unsupported dataset type.")


class LpDistance(DistanceMetric):
    """
    L_p distance between two datasets (typically, tabular datasets).
    This 1-hot encodes categorical attributes.

    """

    def __init__(self, p: float = 2, weights: np.array = None):
        """
        Parameters
        ----------
        p: float
            Order of the distance (default 2, Euclidean distance). p must be
            a positive number.
        weights: real-valued numpy array
            Weighting to apply to individual entries in the 1-hot encoded
            dataset. The distance between records x and y (of length k) is
            computed as (sum_i weights_i * abs(x_i - y_i)^p )^(1/p).

        Use the weights to restrict this distance to a specific subset of
        variables (e.g., to exclude 1-hot encoded columns).

        """
        self.p = p
        assert self.p > 0, "Order p must be positive."
        self.weights = weights
        self._label = f"L_{p}"

    def __call__(self, x: Dataset, y: Dataset):
        assert (
            x.description == y.description
        ), "Input datasets must have same description."
        if isinstance(x, TabularDataset):
            # 1-hot encode both datasets.
            x_1hot = x.as_numeric
            y_1hot = y.as_numeric
            # Set uniform weights (1) if none are provided.
            weights = self.weights if self.weights is not None else np.ones(x_1hot.shape[1])
            dists = np.zeros((len(x), len(y)))
            for i in range(x_1hot.shape[0]):
                # The distance is defined as (sum_i weights_i * abs(x_i - y_i)^p)^(1/p).
                dists[i] = (np.abs(y_1hot - x_1hot[i,:])**self.p).dot(weights)**(1/self.p)
            return dists
        else:
            raise Exception("Unsupported dataset type.")
