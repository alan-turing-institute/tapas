from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression

from privE.utils.data import encode_data, get_num_features

class SetClassifier(ABC):
    """Abstract base class for set classifiers"""

    @abstractmethod
    def fit(datasets, labels, *args, **kwargs):
        """Fit classifier to datasets-data"""
        pass

    @abstractmethod
    def predict(datasets, *args, **kwargs):
        """Predict labels of datasets"""
        pass

    # Map __call__ to predict
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class SetReprClassifier(SetClassifier):
    """
    Classifier that first computes a representation of the dataset and then
    uses 'traditional' (vector-based) classification techniques
    """
    def __init__(self, SetRep, Classifier, data_description):
        self.SetRep = SetRep(data_description)
        self.Classifier = Classifier(self.SetRep.data_description_out)
        self.data_description = data_description

    def fit(datasets, labels):
        representations = self.SetRep(datasets)
        self.Classifier.fit(representations, labels)

    def predict(datasets):
        representations = self.SetRep(datasets)
        return self.Classifier.predict(representations)


class LRClassifier:
    """Logistic regresison classifier"""
    def __init__(self, data_description):
        self.num_features = data_description['num_features']
        self.Classifier = LogisticRegression()

    def fit(self, X, y):
        assert X.shape[1] == self.num_features, f'Data has {X.shape[1]} features, expected {self.num_features}'
        self.Classifier.fit(X, y)

    def predict(self, X):
        return self.Classifier.predict(X)


class NaiveRep: # TODO: Write tests
    """
    Naive feature set from Stadler et al. paper. Mean, median, and variance 
    of each column is computed.
    """
    def __init__(self, data_description):
        self.data_description = data_description
        self.data_description_out = {'num_features': 3*get_num_features(self.data_description)}

    def get_representations(self, datasets):
        np_data = [encode_data(dataset, self.data_description) for dataset in datasets]
        representations = np.stack(
            [np.concatenate(
                [np.nanmean(data, axis=0), np.nanmedian(data, axis=0), np.nanvar(data, axis=0)]
            ) for data in np_data]
        )
        return representations