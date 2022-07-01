"""
Parent class for launching a membership inference attack on the output of a 
generative model.
"""
# Type checking stuff
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset
    from .set_classifiers import SetClassifier

# Real imports
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .base_classes import Attack
from ..threat_models import LabelInferenceThreatModel


class GroundhogAttack(Attack):
    """
    Implementation of Stadler et al. (2022) attack.

    Parent class for membership inference attack on the output of a 
    generative model using a classifier.

    Attributes
    ----------
    classifier : SetClassifier
        Instance of a SetClassifier that will be used as the classification
        model for this attack.
    trained : bool
        Indicates whether or not the attack has been trained on some data.

    """

    def __init__(self, classifier: SetClassifier):
        """
        Initialise a Groundhog attack from a threat model and classifier.

        Parameters
        ----------
        classifier : SetClassifier
            SetClassifier to set for attack.

        """
        self.classifier = classifier

        self.trained = False

        self.__name__ = f"{self.classifier.__class__.__name__}Groundhog"

    def train(
        self, threat_model: LabelInferenceThreatModel = None, num_samples: int = 100,
    ):
        """
        Train the attack classifier on a labelled set of datasets. The datasets
        will either be generated from threat_model or need to be provided.

        Parameters
        ----------
        threat_model : ThreatModel
            Threat model to use to generate training samples if synthetic_datasets
            or labels are not given.
        num_samples : int, optional
            Number of datasets to generate using threat_model if
            synthetic_datasets or labels are not given. The default is 100.

        """

        assert isinstance(
            threat_model, LabelInferenceThreatModel
        ), "The Groundhog attack requires a label-inference threat model."

        # Generate data from threat model if no data is provided
        synthetic_datasets, labels = threat_model.generate_training_samples(num_samples)

        # Fit the classifier to the data.
        self.classifier.fit(synthetic_datasets, labels)
        self.trained = True

    def attack(self, datasets: list[Dataset]) -> list[int]:
        """
        Make a guess about the target's membership in the training data that was
        used to produce each dataset in datasets.

        Parameters
        ----------
        datasets : list[Dataset]
            List of (synthetic) datasets to make a guess for.

        Returns
        -------
        list[int]
            Binary guesses for each dataset. A guess of 1 at index i indicates
            that the attack believes that the target was present in dataset i.

        """
        assert self.trained, "Attack must first be trained."

        return self.classifier.predict(datasets)

    def attack_score(self, synT: list[Dataset]) -> list[float]:
        """
        Calculate classifier's raw probability about the presence of the target.
        Output is a probability in [0, 1].

        Parameters
        ----------
        synT : list[Dataset]
            List of (synthetic) datasets to make a guess for.

        Returns
        -------
        list[float]
            List of probabilities corresponding to attacker's guess about the truth.

        """
        # None of this will work
        assert self.trained, "Attack must first be trained."

        return self.classifier.predict_proba(datasets)
