# Imports for type annotations.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset, DataDescription, TabularDataset
    from ..threat_models import LabelInferenceThreatModel
    from .set_classifiers import SetClassifier
    from sklearn.base import ClassifierMixin

import numpy as np

from abc import ABC, abstractmethod

from prive.utils.data import encode_data, get_num_features, one_hot


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

    # This only applies to tabular datasets, which have a DataDescription.
    # @abstractmethod
    # def size(self, data_description: DataDescription) -> int:
    #     """
    #     Compute the number of features extracted by this FeatureSet.

    #     Parameters
    #     ----------
    #     data_description: DataDescription
    #         Description of the dataset whose features will be computed.

    #     """
    #     return 0

    # Map __call__ to get_representation.
    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)

    # Additional tool: set features can be combined (concatenated) using the
    # addition operator (overloaded).
    def __add__(self, set_feature_2):
        return CombinedSetFeatures(self, set_feature_2)


class CombinedSetFeatures(SetFeature):
    """
    Combined (concatenated) set features extracted from datasets.
    While you may use this directly, it is mostly as part of __add__.

    """

    def __init__(self, *features):
        self.features = features

    def extract(self, dataset: Dataset) -> np.array:
        return np.concatenate([f.extract(dataset) for f in self.features], axis=1)

    # def size(self, data_description: DataDescription) -> int:
    #     return sum([f.size(data_description) for f in self.features])


class FeatureBasedSetClassifier(SetClassifier):
    """
    Classifier that first computes a representation of the dataset and then
    uses 'traditional' (vector-based) classification techniques.

    """

    def __init__(self, features: SetFeature, classifier: ClassifierMixin):
        """
        Parameters
        ----------
        features: SetFeature
            A static (non-trainable) SetFeature object that is used to
            extract features from a dataset.
        classifiers: sklearn.base.ClassifierMixin
            A (sklearn) classifier trained to predict a label based on
            the features extracted from an input dataset.
        """
        self.features = features
        self.classifier = classifier

    def fit(self, datasets: list[Dataset], labels: list[int]):
        self.classifier.fit(self.features(datasets), labels)

    def predict(self, datasets: list[Dataset]):
        return self.classifier.predict(self.features(datasets))

    def predict_proba(self, datasets: list[Dataset]):
        return self.classifier.predict_proba(self.features(datasets))


## We here propose a few possible SetFeature that can be used for attacks.

## We first implement features from Stadler et al. :
# As feature extractors, we implemented a naive feature set with simple summary
# statistics FNaive, a histogram feature set that contains the marginal
# frequency counts of each data attribute FHist, and a correlations feature set
# that encodes pairwise attribute correlations FCorr.


class NaiveSetFeature(SetFeature):
    """
    Naive feature set F_Naive from Stadler et al. Mean, median, and variance of
    each column is computed.

    """

    def extract(self, datasets: list[TabularDataset]) -> np.array:
        np_data = [encode_data(dataset) for dataset in datasets]
        representation = np.stack(
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
        return representation

    # def size(self, data_description: DataDescription) -> int:
    #     return data_description.encoded_dim


class HistSetFeature(SetFeature):
    """
    F_Hist feature set from Stadler et al. Compute a histogram of each column,
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
        features = []
        for column_descriptor in dataset_description:
            ctype = column_descriptor["type"]

            if ctype.startswith("finite"):
                # Categorical variables.
                # First, retrieve all values.
                if isinstance(column_descriptor["representation"], int):
                    values = list(range(column_descriptor["representation"]))
                else:
                    values = sorted(column_descriptor["representation"])
                # Then, 1-hot encode all these columns and compute the means.
                features.append(
                    np.stack(
                        [
                            one_hot(dataset.data[column_descriptor["name"]], values).mean(axis=0)
                            for dataset in datasets
                        ]
                    )
                )

            elif ctype.startswith("real") or ctype == "interval":
                # Continuous variables
                cmin, cmax = (0, 1) if ctype == "interval" else self.bounds
                bins = np.linspace(cmin, cmax, self.num_bins + 1)
                features.append(
                    np.stack(
                        [
                            np.histogram(ds.data[column_descriptor["name"]], bins)[0]
                            / len(ds)
                            for ds in datasets
                        ]
                    )
                )

            else:
                # This type is not supported by this attack, at least
                # at the moment, and will be ignored.
                pass

        print(features)
        return np.concatenate(features, axis=1)


class CorrSetFeature(SetFeature):
    pass

    def extract(self, datasets: list[TabularDataset]) -> np.array:
        todo
