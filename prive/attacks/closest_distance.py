"""Closest-distance attacks.

These attacks use the local neighbourhood of real records in the synthetic data
to make inferences about those real records.

"""

# Imports for type annotations.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset, DataDescription
    from ..threat_models import ThreatModel


import numpy as np
from sklearn.metrics import roc_curve

from .base_classes import Attack, TrainableThresholdAttack
from .distances import DistanceMetric, HammingDistance

from ..datasets import TabularDataset
from ..threat_models import TargetedMIA, TargetedAIA


# TODO: k-nearest neighbours?
class ClosestDistanceMIA(TrainableThresholdAttack):
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


class ClosestDistanceAIA(ClosestDistanceMIA):
    """
    Attack that finds the closest-record to the target record, and uses the
    value of the sensitive attribute of that closest-record as answer to the
    attribute-inference attack.

    This attack is a bit more flexible: for each value v, it returns a score
    equal to  - distance(r(v), D) / sum_v' distance(r(v'), D), where r(v) is
    the target record with v as value for its sensitive attribute.

    This is a TrainableThresholdAttack, and thus is able to automatically
    select the threshold for .attack from .attack_score. To set the behaviour
    to be "choose v minimising distance(r(v), D)", set:
    criterion = ("threshold", -0.5) in the constructor of this object.

    """

    # Same __init__ as ClosestDistanceMIA.

    def attack_score(self, datasets: list[Dataset]):
        """
        Compute the decision score for this attack.

        The target score is the minimal distance between the target record with
        a given value v for the sensitive attribute and records in the
        synthetic dataset, weighted such that the sum of scores is 1 for each
        dataset. If the distance is 0 for all values, this returns 1/num_values
        for each value.
        
        Parameters
        ----------
        datasets: a list of synthetic datasets.

        Returns
        -------
        scores: array of size len(datasets) or len(datasets) x k
            If the number of possible values (threat_model.attribute_values) is
            two, then the scores array is 1-dimensional, and each entry
            contains the score for the second ("positive") value.
            Otherwise, this returns a score per value, and k is the number of
            possible values (k = len(self.threat_model.attribute_values).

        """
        # Modify the records to have different target values.
        modified_records = {}
        for value in self.threat_model.attribute_values:
            r = self.threat_model.target_record.copy()
            r.set_value(self.threat_model.sensitive_attribute, value)
            modified_records[value] = r
        # For each dataset, compute the min distance from the modified record
        # to records in the synthetic datasets.
        # (This is somewhat of a trick to use general distance metrics).
        scores = []
        for dataset in datasets:
            # Compute distance(r(v), D) for each record.
            distances = np.array(
                [
                    self.distance(modified_records[v], dataset)[0].min()
                    for v in self.threat_model.attribute_values
                ]
            )
            # Compute the relative scores.
            if distances.sum() > 1e-16:
                s = distances / distances.sum()
            else:
                # If all distances are 0, use uninformative scores.
                s = np.full(distances.shape, 1 / len(distances))
            # For binary choices, only return the second element (score for 1).
            if len(s) == 2:
                s = s[1]
            scores.append(-s)
        return np.array(scores)


class LocalNeighbourhoodAttack(TrainableThresholdAttack):
    """
    Attack that makes a decision based on records similar to the target record,
    specifically all records within a sphere of a given radius, for a specific
    choice of distance.

    For membership inference attacks, the score is the fraction of all records
    that are within distance `radius` of the target record.

    For attribute inference attacks, the score is the fraction of all records
    within the sphere which have a given value for the sensitive attribute.

    """

    def __init__(
        self,
        distance: DistanceMetric = HammingDistance(),
        radius: float = 1,
        criterion="accuracy",
        label: str = None,
    ):
        """
        Create the attack with chosen parameters.

        Parameters
        ----------
        distance: DistanceMetric
            Distance to use between records for the attack.
        radius: float
        criterion: tuple
            Criterion to select the threshold (see TrainableThresholdAttack for details).
        label (optional): name of this attack, for reporting.
        """
        TrainableThresholdAttack.__init__(self, criterion)
        self.distance = distance
        self.radius = radius
        self._label = (
            label or f"LocalNeighbourhood({distance.label}, {radius}, {criterion})"
        )

    def attack_score(self, datasets: list[Dataset]):
        """
        Compute the decision score for this attack.

        Parameters
        ----------
        datasets: a list of synthetic datasets.

        Returns
        -------
        scores: a np.array of scores.
            For attribute inference attacks, if the number k of possible values
            is > 2, this is of size len(datasets) x k, where entry (i,j) is for
            dataset i and value j. Otherwise, this is an array of len(dataset)
            scores, with a score per dataset.

        """
        # First, check that the attack model is compatible.
        if isinstance(self.threat_model, TargetedMIA):
            mia = True
        elif isinstance(self.threat_model, TargetedAIA):
            mia = False
        else:
            raise Exception("Unsupported threat model.")
        # Compute local spheres and decision based on that.
        scores = []
        for dataset in datasets:
            # Compute the sphere around the record.
            distances = self.distance(self.threat_model.target_record, dataset)[0]
            in_sphere = distances <= self.radius
            # Make a decision based on this.
            if mia:
                # MIA: return the fraction of all records that are in sphere.
                scores.append(np.mean(in_sphere))
            else:
                # AIA: compute a histogram of the attribute values.
                if isinstance(dataset, TabularDataset):
                    attr_values = self.threat_model.attribute_values
                    if in_sphere.sum() == 0:
                        # If there are no entries, make the score be 1/k
                        k = len(attr_values)
                        s = np.full((k,), 1 / k)
                    else:
                        s = np.zeros((len(attr_values),))
                        # We here use the internal representation of the dataset
                        # as a Pandas DataFrame.
                        values_in_sphere = dataset.data[
                            self.threat_model.sensitive_attribute
                        ].values[in_sphere]
                        # Compute a score for each value.
                        for i, v in enumerate(attr_values):
                            s[i] = np.mean(values_in_sphere == v)
                    # If there are only two values, return the score for the
                    # second ("positive") value.
                    if len(attr_values) == 2:
                        s = s[1]
                    scores.append(s)
                else:
                    raise Exception("Unsupported dataset type.")
        return np.array(scores)

    @property
    def label(self):
        return self._label
