"""Closest-distance attacks """

# Imports for type annotations.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from ..datasets import Dataset, DataDescription
    from ..threat_models import ThreatModel

import numpy as np
from sklearn.metrics import roc_curve

from .base_classes import Attack, TrainableThresholdAttack
from .distances import DistanceMetric
from ..threat_models import TargetedMIA


class ClosestDistanceAttack(TrainableThresholdAttack):
    """
    Attack that looks for the closest record to a given target in the
    synthetic data to determine whether the target was in the
    training dataset.

    This attack predicts that a target record is in the training dataset
     iff:  min_{x in synth_data} distance(x, target) <= threshold.
    The threshold can either be specified, or selected automatically.

    """

    def __init__(
        self,
        distance_function: DistanceMetric = None,
        criterion: dict = "accuracy",
        metric_name="default",
    ):
        """
        Create the attack with chosen parameters.

        Parameters
        ----------
        distance_function (callable): maps (record1, record2) to a positive
            float. If left None (default), this is self._default_distance.
        criterion: tuple
            Criterion to select the threshold (see TrainableThresholdAttack for details)
        metric_name (optional): name of the distance metric used.

        Exactly one of threshold, fpr or tpr must be not None.

        """
        TrainableThresholdAttack.__init__(self, criterion)
        self.distance_function = distance_function or self._default_distance
        self.__name__ = f"ClosestDistance({metric_name}, {criterion})"

    def attack_score(self, datasets: list[Dataset]):
        """
        Compute the decision score for this attack.

        The target score is the minimum distance between the target record x
        and records in the dataset.
        
        Parameters
        ----------
        datasets: a list of synthetic datasets.

        Returns
        -------
        scores: array of (nonnegative) record distances.

        """
        target_record = self.threat_model.target_record
        scores = []
        for ds in datasets:
            # Use the __iter__ function of Dataset to iterate over records.
            distances = [self.distance_function(record, target_record) for record in ds]
            # The attack score is the NEGATIVE min distance to the target record.
            # This is because larger scores are associated with higher probability
            # of membership, whereas distances do the opposite.
            scores.append(-np.min(distances))
        return np.array(scores)

    def _default_distance(self, x: Dataset, y: Dataset):
        """
        Hamming distance between two records.

        Parameters
        ----------
        x, y: the records to compare.

        Returns
        -------
        distance between x and y.

        """
        # TODO: check the dataset description.
        return (x.data.values != y.data.values).mean()
