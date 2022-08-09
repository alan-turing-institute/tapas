"""
Abstract base classes for various privacy attacks.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from ..datasets import Dataset
    from ..threat_models import ThreatModel

from ..threat_models import LabelInferenceThreatModel

from abc import ABC, abstractmethod


class Attack(ABC):
    """
    Abstract base class for all privacy attacks.

    This class defines (only) three common elements of attacks:

    * a .train method (that can be left empty), that selects parameters for
      the attack to make decisions.
    * a .attack method, that makes a binary decision for a (list of) dataset(s).
    * a .attack_score method that can be ignored if not meaningful, but can be
      useful for deeper analysis of attacks.

    """

    @abstractmethod
    def train(self, threat_model: ThreatModel):
        """
        Train parameters of the attack.

        """
        pass

    @abstractmethod
    def attack(self, datasets: list[Dataset]):
        """
        Perform the attack on each dataset in a list and return a (discrete) decision.

        """
        pass

    @abstractmethod
    def attack_score(self, datasets: list[Dataset]):
        """
        Perform the attack on each dataset in a list, but return a confidence
        score (specifically for classification tasks).

        """
        pass

    @property
    def label(self):
        """
        A label to describe this attack in reports.

        """
        return "Unnamed Attack"

    def __str__(self):
        return self.label


# Many attacks are based around an `attack_score` (potentially trainable), with
# the decision threshold then tailored for a specific task, e.g., max accuracy,
# matching a given true positive rate or true negative rate. We here implement
# a generic class that implements threshold selection for a generic score.

import numpy as np
from sklearn.metrics import roc_curve


# TODO: extend to multi-class labels. Currently assumes binary labels.
class TrainableThresholdAttack(Attack):
    """
    Generic class to represent attacks that rely on a score, combined with a
    threshold that is chosen according to some (fairly generic) criterion.
    Many attacks should fall under this 

    """

    # To implement a specific attack from this generic class, you must
    # instantiate the `attack_score` function from Attack. Additionally,
    # you may implement _train_attack_score if relevant.
    # Also, implement the .label property.

    def __init__(self, criterion: tuple):
        """
        Initialise this attack with a given threshold-selection criterion.

        The criterion is a tuple with at least one entry. The first entry,
        criterion[0], is the target criterion (accuracy/tp/fp/threshold).
        Further entries give additional information on the target.

        Acceptable criterions are:
         - ("accuracy",): choose the threshold that yields maximum accuracy.
         - ("tp", float): choose the threshold that yields as close as possible
            to a given true positive ("tp") rate.
         - ("fp", float): similarly, for the false positive rate ("fp").
         - ("threshold", float): manually specify the threshold.

        """
        if not isinstance(criterion, tuple):
            # We actually allow for criterion = "accuracy", as it is cleaner.
            assert criterion == "accuracy", "Criterion should be a tuple."
            criterion = ("accuracy",)
        # Parse the criterion into target_criterion and target_value.
        self.target_criterion = criterion[0]
        if self.target_criterion != "accuracy":
            assert len(criterion) == 2, "Missing second argument to criterion."
            self.target_value = criterion[1]
        # Initialise the threshold.
        self._threshold = None
        if self.target_criterion == "threshold":
            self._threshold = self.target_value

    def train(
        self,
        threat_model: LabelInferenceThreatModel,
        num_samples: int = 100,
        **attack_score_kwargs
    ):
        """
        Train this attack: train the score, then choose a threshold meeting
        the target criterion.

        Parameters
        ----------
        threat_model: LabelInferenceThreatModel
            The threat model from which to generate labelled samples.
        num_samples: int (default, 100).
            Number of training samples to generate to select the threshold.
        (optionally), additional keyword arguments, passed to _train_attack_score.

        """
        # First, optionally train the score.
        assert isinstance(
            threat_model, LabelInferenceThreatModel
        ), "Threat model must be a LabelInferenceThreatModel."
        self.threat_model = threat_model
        self._train_attack_score(threat_model, num_samples, **attack_score_kwargs)
        # Then, compute the scores for a number of training datasets.
        if self._threshold is not None:
            return  # No training required.
        # If the threshold is not specified, train to get the desired tpr or fpr.
        synthetic_datasets, labels = self.threat_model.generate_training_samples(
            num_samples
        )
        # Finally, compute the ROC curve and select the threshold from it.
        fpr_all, tpr_all, thresholds = roc_curve(
            labels, self.attack_score(synthetic_datasets)
        )
        # Select the threshold such that the criterion is matched.
        if self.target_criterion == "fpr":
            index = np.argmin(np.abs(fpr_all - self.target_value))
        elif self.target_criterion == "tpr":
            index = np.argmin(np.abs(tpr_all - self.target_value))
        elif self.target_criterion == "accuracy":
            # Assuming classes are balanced, we have that:
            #  Accuracy(t) = 1/2 (TPR + TNR) = 1/2 (TPR + 1 - FPR).
            index = np.argmax(tpr_all - fpr_all)
        self._threshold = thresholds[index]

    def attack(self, datasets: list[Dataset]):
        """
        Make a prediction for each dataset.

        This computes attack_score for each dataset, then decides that the
        target user is in the training dataset if and only if the score is
        higher than self._threshold.

        Parameters
        ----------
        datasets: a list of synthetic datasets.

        Returns
        -------
        predictions: np.array of booleans.

        """
        if self._threshold is None:
            raise Exception("Attack has not been trained (threshold is None).")
        scores = self.attack_score(datasets)
        return scores >= self._threshold

    # Implement this if needed.
    def _train_attack_score(
        self, threat_model: LabelInferenceThreatModel, num_samples: int = 100, **kwargs
    ):
        """
        Train the attack score function (optional). By default, this does nothing.

        Interface to implement
        ----------------------
        threat_model: LabelInferenceThreatModel
            The threat model from which to generate training samples.
        num_samples: int (default 100)
            Number of samples to generate to train the attack score.
        optional keyword arguments: passed from .train(**kwargs).

        """
        pass
