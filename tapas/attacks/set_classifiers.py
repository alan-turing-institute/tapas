# Imports for type annotations.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset, DataDescription, TabularDataset, TabularRecord
    from ..threat_models import LabelInferenceThreatModel
    from sklearn.base import ClassifierMixin

import numpy as np

from abc import ABC, abstractmethod


class SetClassifier(ABC):
    """
    Abstract base class for classifiers over set-valued data.

    """

    @abstractmethod
    def fit(self, datasets: list[Dataset], labels: list[int]):
        """
        Fit classifier to datasets-data.

        """

    @abstractmethod
    def predict(self, datasets: list[Dataset]):
        """
        Predict labels of datasets.

        """

    @abstractmethod
    def predict_proba(self, datasets: list[Dataset]):
        """
        Predict a score over labels for each dataset.

        """

    # Map __call__ to predict.
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @property
    def label(self):
        return "Unknown classifier"


## Typical setup: first, features are extracted from the dataset, then a
## classifier is trained on the output features.


class SetFeature(ABC):
    """
    Represents a set of features that can be extracted from a dataset, as a
    np.array vector. This is a callable that can optionally have some
    configuration parameters passed via __init__.

    By default, this is memoryless and static, but it can be extended to be
    trainable from datasets.

    """

    @abstractmethod
    def extract(self, datasets: list[Dataset]) -> np.array:
        """
        Extract features from each dataset in a list.

        Parameters
        ----------
            datasets: list of Datasets.
                Datasets to extract features from.
        Returns
        -------
            features: np.array
                Array of size len(datasets) x k, where the number of features
                k can be estimated by self.size(dataset.description).
                Each row is a dataset, and each column a different feature.
        """

    # Map __call__ to get_representation.
    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)

    # Additional tool: set features can be combined (concatenated) using the
    # addition operator (overloaded).
    def __add__(self, set_feature_2):
        return CombinedSetFeatures(self, set_feature_2)

    @property
    def label(self):
        return "Unknown SetFeature"


class CombinedSetFeatures(SetFeature):
    """
    Combined (concatenated) set features extracted from datasets.
    While you may use this directly, it is mostly as part of __add__.

    """

    def __init__(self, *features):
        self.features = features

    def extract(self, dataset: Dataset) -> np.array:
        return np.concatenate([f.extract(dataset) for f in self.features], axis=1)

    @property
    def label(self):
        return "+".join([f.label for f in self.features])


class FeatureBasedSetClassifier(SetClassifier):
    """
    Classifier that first computes a representation of the dataset and then
    uses 'traditional' (vector-based) classification techniques.

    """

    def __init__(
        self, features: SetFeature, classifier: ClassifierMixin, label: str = None
    ):
        """
        Parameters
        ----------
        features: SetFeature
            A static (non-trainable) SetFeature object that is used to
            extract features from a dataset.
        classifier: sklearn.base.ClassifierMixin
            A (sklearn) classifier trained to predict a label based on
            the features extracted from an input dataset.
        label: str (optional)
            Label to represent this classifier in reports.
        """
        self.features = features
        self.classifier = classifier
        self._label = (
            label or f"Classifier({self.features.label}, {str(self.classifier)})"
        )

    def fit(self, datasets: list[Dataset], labels: list[int]):
        self.classifier.fit(self.features(datasets), labels)

    def predict(self, datasets: list[Dataset]):
        return self.classifier.predict(self.features(datasets))

    def predict_proba(self, datasets: list[Dataset]):
        return self.classifier.predict_proba(self.features(datasets))

    @property
    def label(self):
        return self._label


## We here propose a few possible SetFeature that can be used for attacks.

## We first implement features from Stadler et al. :
# As feature extractors, we implemented a naive feature set with simple summary
# statistics FNaive, a histogram feature set that contains the marginal
# frequency counts of each data attribute FHist, and a correlations feature set
# that encodes pairwise attribute correlations FCorr.


class NaiveSetFeature(SetFeature):
    """
    Naive set feature F_Naive from Stadler et al. Mean, median, and variance of
    each column is computed.

    """

    def extract(self, datasets: list[TabularDataset]) -> np.array:
        np_data = [dataset.as_numeric for dataset in datasets]
        return np.stack(
            [
                np.concatenate(
                    [
                        np.nanmean(data, axis=0),
                        np.nanmedian(data, axis=0),
                        np.nanvar(data, axis=0),
                    ]
                )
                for data in np_data
            ]
        )

    @property
    def label(self):
        return "F_Naive"


class HistSetFeature(SetFeature):
    """
    F_Hist set feature from Stadler et al. Compute a histogram of each column,
    with binning for continuous variables.

    """

    def __init__(self, num_bins: int = 10, bounds: tuple[float, float] = (0, 1)):
        """
        Parameters
        ----------
        num_bins: int (default, 10)
            Number of bins to use when categorising continuous columns, for
            the computation of histograms.
        bounds: (float, float) (default, (0,1))
            Bounds on continuous attributes, within which the histograms are
            computed.

        """
        self.num_bins = num_bins
        self.bounds = bounds

    def extract(self, datasets: list[TabularDataset]) -> np.array:
        dataset_description = datasets[0].description
        np_data = [dataset.as_numeric for dataset in datasets]
        features = []
        # Index of the current column in Numpy form.
        cidx = 0
        for column_descriptor in dataset_description:
            ctype = column_descriptor["type"]

            if ctype.startswith("finite"):
                # Categorical variables.
                # First, retrieve the number of values.
                if isinstance(column_descriptor["representation"], int):
                    num_values = column_descriptor["representation"]
                else:
                    num_values = len(column_descriptor["representation"])
                # Then, compute the means of the next num_values 1-hot encoded columns.
                features.append(
                    np.stack(
                        [
                            data[:, cidx : cidx + num_values].mean(axis=0)
                            for data in np_data
                        ]
                    )
                )
                cidx += num_values

            elif ctype.startswith("real") or ctype == "interval":
                # Continuous variables.
                cmin, cmax = (0, 1) if ctype == "interval" else self.bounds
                bins = np.linspace(cmin, cmax, self.num_bins + 1)
                features.append(
                    np.stack(
                        [
                            np.histogram(data[:, cidx], bins)[0] / data.shape[0]
                            for data in np_data
                        ]
                    )
                )
                cidx += 1

            else:
                # This type is not supported by this attack, at least
                # at the moment, and will be ignored.
                pass

        return np.concatenate(features, axis=1)

    @property
    def label(self):
        return f"F_Hist({self.num_bins})"


class CorrSetFeature(SetFeature):
    """
    F_Corr set feature from Stadler et al. Compute linear correlation between
    features. For categorical attributes, do this after 1-hot encoding.

    """

    def _corr(self, array):
        """
        Compute a flattened correlation matrix. This also filters NaNs and
        removes symmetrical elements.

        """
        corr_matrix = np.corrcoef(array.T)
        # Remove redundant entries from the symmetrical matrix.
        above_diagonal = np.triu_indices(corr_matrix.shape[0], 1)
        array = corr_matrix[above_diagonal]
        # Fill in NaNs with 0.
        array[np.isnan(array)] = 0
        return array

    def extract(self, datasets: list[TabularDataset]) -> np.array:
        """
        Compute correlations between numeric attributes and 1-hot encoded
        categorical attributes.

        """
        np_data = [dataset.as_numeric for dataset in datasets]
        return np.stack([self._corr(dataset.as_numeric) for dataset in datasets])

    @property
    def label(self):
        return "F_Corr"


# New custom features.

import itertools
from scipy.special import comb


class RandomTargetedQueryFeature(SetFeature):
    """
    Features that computes random targeted queries that include the user.

    These queries take the form SUM_{x in D} AND_{a in S} I{x_s = target_s},
    counting all records that match the target record in all attributes in S.
    The set S is sampled randomly from all subsets of attributes of a fixed
    length (order).

    This is equivalent to using entries of the order-way contingency
    table that count the target user as features. In particular, for 
    order=1, this is a subset of HistSetFeature().

    """

    def __init__(self, target: TabularRecord, order: int, number: int, num_bins = None):
        """
        Parameters
        ----------
        target: TabularRecord
            The record of the target user to attack.
        order: int
            The number of conditions in each query.
        number: int
            The number of queries to use.
        num_bins: int
            How to bin continuous attributes (currently unsupported).

        """
        # Target user.
        self.target = target
        self.target_values = target.data.values
        # Order (number of conditions) of the queries.
        self.order = order
        # Choose the queries.
        self.queries = []
        # Do something with num_bins:
        if num_bins is not None:
            raise Exception("Only categorical variables are currently allowed.")
            # TODO: manage continuous variables.
        # First, check the total number of combinations. If this number if low
        # enough, we generate all possible queries and sample uniformly from
        # the full set. If the number is too high, we instead randomly sample
        # each query independently, which may create duplicates.
        num_columns = self.target_values.shape[1]
        total_num_queries = comb(num_columns, order)
        if total_num_queries < int(1e6):
            # We can just enumerate everything, and sample uniformly.
            all_combinations = list(
                itertools.combinations(range(num_columns), self.order)
            )
            self.number = min(number, int(total_num_queries))
            indices = np.random.choice(
                len(all_combinations), replace=False, size=(self.number,)
            )
            # A query is represented by a np.array of column indices, which
            # represents all the columns used in the query.
            self.queries = [all_combinations[idx] for idx in indices]
        else:
            # TODO: test this code.
            self.number = number
            self.queries = [
                np.random.choice(num_columns, replace=False, size=(self.order,))
                for _ in range(self.number)
            ]
            # TODO: duplicate detection+removal?

    def extract(self, datasets):
        """Compute queries on each dataset."""
        features = []
        for dataset in datasets:
            ft = []
            for columns in self.queries:
                ft.append(
                    (dataset.data.values[:, columns] == self.target_values[0, columns])
                    .prod(axis=1)
                    .sum(axis=0)
                )
            features.append(np.array(ft))
        return features

    @property
    def label(self):
        return f"RandomQueries({self.order}, {self.number})"
