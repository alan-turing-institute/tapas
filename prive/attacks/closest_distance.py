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
from .distances import DistanceMetric, HammingDistance
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
        distance: DistanceMetric = HammingDistance(),
        criterion: tuple = "accuracy",
        label: str = None,
    ):
        """
        Create the attack with chosen parameters.

        Parameters
        ----------
        distance: DistanceMetric
            Distance to use between records for the attack.
        criterion: tuple
            Criterion to select the threshold (see TrainableThresholdAttack for details).
        label (optional): name of this attack, for reporting.

        """
        TrainableThresholdAttack.__init__(self, criterion)
        self.distance = distance
        self._label = label or f"ClosestDistance({distance.label}, {criterion})"

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
            # This returns an np.array of size (1, len(ds)) with pairwise distances
            # between target_record and records in ds.
            distances = self.distance(target_record, ds)[0]
            # The attack score is the NEGATIVE min distance to the target record.
            # This is because larger scores are associated with higher probability
            # of membership, whereas distances do the opposite.
            scores.append(-np.min(distances))
        return np.array(scores)

    @property
    def label(self):
        return self._label
