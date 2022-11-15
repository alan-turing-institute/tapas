"""Attacks based on inference models trained on synthetic data.

This groups attacks that follow the following approximate structure:
 1. Train a statistical model on the synthetic data.
 2. Use this model to infer something about the real data.

The second step often involves applying the model to a target record.

"""

# Imports for type annotations.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset
    from ..threat_models import ThreatModel
    from sklearn.base import ClassifierMixin

from abc import ABC, abstractmethod

import numpy as np

from .base_classes import Attack, TrainableThresholdAttack
from ..threat_models import TargetedMIA, TargetedAIA
from ..datasets import TabularDataset


class DensityEstimator(ABC):
    """
    Density estimator for records in a dataset.

    """

    @abstractmethod
    def fit(self, dataset: Dataset):
        pass

    @abstractmethod
    def score(self, dataset: Dataset):
        pass

    @property
    def label(self):
        return self._label


class sklearnDensityEstimator(DensityEstimator):
    """
    Extends DensityEstimator to run a sklearn model on a 1-hot encoding of datasets.
    This is mostly intended as an internal wrapper for Tabular datasets.

    """

    def __init__(self, model, label=None):
        self.model = model
        self._label = label or str(model)

    def fit(self, dataset: Dataset):
        self._data_description = dataset.description
        self.model.fit(dataset.as_numeric)

    def score(self, dataset: Dataset):
        assert (
            dataset.description == self._data_description
        ), "Incompatible data description!"
        return self.model.score_samples(dataset.as_numeric)


class ProbabilityEstimationAttack(TrainableThresholdAttack):
    """
    Membership Inference Attack that first estimates a statistical model p_x
    of the distribution of records in the *synthetic* data, and then uses
    p_x(target_record) as score. The intuition is that the distribution of the
    synthetic data, which is defined by the generator trained on the real data,
    is more likely to be high for records in the real data. This works best on
    overfitted models.

    """

    def __init__(
        self, estimator: DensityEstimator, criterion: tuple, label: str = None
    ):
        """
        Create an inference-on-synthetic attack.

        Parameters
        ----------
        estimator: DensityEstimator
            The estimator, as a DensityEstimator object with .fit and .score.
            If an object of another type is passed, this object is assumed to
            be a sklearn model, and is fed into sklearnDensityEstimator.
        criterion: str or tuple
            How to select the threshold (see TrainableThresholdAttack).
        label: str (optional)
            String to represent this attack.

        """
        TrainableThresholdAttack.__init__(self, criterion)
        if not isinstance(estimator, DensityEstimator):
            estimator = sklearnDensityEstimator(estimator)
        self.estimator = estimator
        self._label = label or f"ProbabilityEstimation({estimator.label})"

    def attack_score(self, datasets: list[Dataset]):
        """
        Perform the attack on each dataset in a list, but return a confidence
        score (specifically for classification tasks).

        """
        assert isinstance(
            self.threat_model, TargetedMIA
        ), "This attack can only applied to targeted MIAs."
        # Treat each dataset individually.
        scores = []
        for dataset in datasets:
            self.estimator.fit(dataset)
            scores.append(self.estimator.score(self.threat_model.target_record)[0])
        return np.array(scores)

    @property
    def label(self):
        return self._label


class SyntheticPredictorAttack(TrainableThresholdAttack):
    """
    Attribute Inference Attack that first trains a classifier C on the
    synthetic data to predict the sensitive value v of a record x, then uses
    C(target_record) as prediction for the target record.

    This is a common baseline, linked to CAP (Correct Attribution Probability),
    although whether it constitutes a privacy violation is controversial, since
    correlations in the data could reveal the sensitive attribute even if the
    user does not contribute their data. TAPAS circumvents this issue by
    randomising the sensitive attribute independently from all others. As such,
    this attack mostly aims at detecting overfitted models.

    This attack is implemented exclusively for tabular data.

    """

    def __init__(self, estimator: ClassifierMixin, criterion: tuple, label=None):
        TrainableThresholdAttack.__init__(self, criterion)
        self.estimator = estimator
        self._label = label or f"SyntheticPredictor({estimator})"

    def attack_score(self, datasets: list[Dataset]):
        assert isinstance(
            self.threat_model, TargetedAIA
        ), "This attack can only be applied to targeted AIAs."
        scores = []
        target_record_x = self.threat_model.target_record.view(
            exclude_columns=[self.threat_model.sensitive_attribute]
        )
        for dataset in datasets:
            assert isinstance(
                dataset, TabularDataset
            ), "This attack can only be applied to TabularDatasets."
            # Train a classifier to predict the sensitive attribute from the rest.
            X = dataset.view(exclude_columns=[self.threat_model.sensitive_attribute])
            y = dataset.view(columns=[self.threat_model.sensitive_attribute])
            self.estimator.fit(X.as_numeric, y.data.values.ravel())
            # Apply this model to the target record.
            score = self.estimator.predict_proba(target_record_x.as_numeric)[0]
            # If there are only two classes, set the second label as positive.
            if len(score) == 2:
                score = score[1]
            scores.append(score)
        return np.array(scores)

    @property
    def label(self):
        return self._label
