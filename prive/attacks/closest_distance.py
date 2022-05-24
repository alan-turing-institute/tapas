"""Closest-distance attacks """

# Imports for type annotations.
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Callable
    from ..datasets import Dataset, DataDescription
    from ..threat_models import ThreatModel # for typing

import numpy as np
from sklearn.metrics import roc_curve

from .base_classes import Attack
from ..threat_models import TargetedMIA


class ClosestDistanceAttack(Attack):
    """Attack that looks for the closest record to a given target in the
        synthetic data to determine whether the target was in the
        training dataset.

       This attack predicts that a target record is in the training dataset
        iff:  min_{x in synth_data} distance(x, target) <= threshold.
       The threshold can either be specified, or selected automatically.

    """

    def __init__(self, distance_function: Callable[[Dataset,Dataset], float] = None,
                 threshold: float = None, fpr: float = None, tpr: float = None,
                 metric_name='default'):
        """
        Create the attack with chosen parameters.

        Parameters
        ----------
        distance_function (callable): maps (record1, record2) to a positive
            float. If left None (default), this is self._default_distance.
        threshold: decision threshold for the classifier. If lest None (default),
            the threshold is learned from training data.
        fpr/tpr: the target false/true positive rate for the threshold selection.
        metric_name (optional): name of the distance metric used.

        Exactly one of threshold, fpr or tpr must be not None.

        """
        self.distance_function = distance_function or self._default_distance
        # Check that at least one of threshold, tpr and fpr is set.
        self.threshold = threshold
        self.fpr, self.tpr = fpr, tpr
        assert ((fpr is None) + (tpr is None) + (threshold is None)) == 2,\
            'Exactly one of threshold, fpr or tpr must be specified.'

        self.trained = False

        self.__name__ = f'ClosestDistance({metric_name}, '
        if fpr is not None:
            self.__name__ += f'fpr={self.fpr:.2f})'
        elif tpr is not None:
            self.__name__ += f'tpr={self.tpr:.2f})'
        else:
            self.__name__ += f'threshold={self.threshold})'


    def train(self, threat_model: TargetedMIA, num_samples: int = 5):
        """
        Train the attack for a specific target.

        Select a target for the attack, and -- if needed -- use samples
         (synthetic datasets) to select a threshold that results in
         approximately the target fpr/tpr.

        Parameters
        ----------
        threat_model: the threat model of this attack, must be a TargetedMIA.
        num_samples: number of training  datasets to generate (default: 100).

        """
        assert isinstance(threat_model, TargetedMIA), \
             "Incompatible attack model: needs targeted MIA."
        self.threat_model = threat_model
        self.target_record = self.threat_model.target_record
        if self.threshold is not None:
            self.trained = True
            return  # No training required.
        # If the threshold is not specified, train to get the desired tpr or fpr.
        synthetic_datasets, labels = self.threat_model.generate_training_samples(num_samples)
        # Compute the roc curve with - threshold, since the decision we use is
        #  score <= threshold, but roc_curve uses score >= threshold.
        fpr_all, tpr_all, thresholds = roc_curve(labels, - self.attack_score(synthetic_datasets))
        # Select the threshold such that the fpr (or tpr) is closest to target.
        if self.fpr is not None:
            index = np.argmin(np.abs(fpr_all - self.fpr))
        else:
            index = np.argmin(np.abs(tpr_all - self.tpr))
        self.threshold = - thresholds[index]
        self.trained = True


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
        scores = []
        for ds in datasets:
            # Use the __iter__ function of Dataset to iterate over records.
            distances = [self.distance_function(record, self.target_record) for record in ds]
            scores.append(np.min(distances))
        return np.array(scores)


    def attack(self, datasets: list[Dataset]):
        """
        Make a prediction of membership for the record in each dataset.

        This computes attack_score for each dataset, then decides that the
        target user is in the training dataset if and only if the closest
        record in the synthetic data is at a distance <= self.threshold to
        the target dataset.

        Parameters
        ----------
        datasets: a list of synthetic datasets.

        Returns
        -------
        predictions: array of booleans.

        """
        if not self.trained:
            raise Exception('Please train this attack.')
        attack_scores = self.attack_score(datasets)
        return attack_scores <= self.threshold


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
