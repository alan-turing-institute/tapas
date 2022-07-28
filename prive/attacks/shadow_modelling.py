"""
Parent class for launching a membership inference attack on the output of a 
generative model.
"""
# Type checking stuff
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset

# Real imports
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .base_classes import Attack
from .set_classifiers import SetClassifier
from ..threat_models import LabelInferenceThreatModel


class ShadowModellingAttack(Attack):
    """
    Shadow-modelling attacks simulate the dataset generation process, using
    auxiliary information available to the attacker, and train a classifier
    (here, a set classifier) to predict a property of the training dataset
    from the synthetic dataset. This class implements the logic of shadow
    modelling attacks, and takes as argument a SetClassifier.

    Attributes
    ----------
    classifier : SetClassifier
        Instance of a SetClassifier that will be used as the classification
        model for this attack.
    trained : bool
        Indicates whether or not the attack has been trained on some data.

    """

    def __init__(self, classifier: SetClassifier, label: str = None):
        """
        Initialise a Groundhog attack from a threat model and classifier.

        Parameters
        ----------
        classifier : SetClassifier
            SetClassifier to set for attack.
        label: str (optional)
            A label to reference this attack in reports.

        """
        self.classifier = classifier

        self.trained = False

        self._label = label or f"ShadowModelling({self.classifier.label})"

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
        ), "Shadow-modelling attacks require a label-inference threat model."

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
        assert self.trained, "Attack must first be trained."

        return self.classifier.predict_proba(datasets)

    @property
    def label(self):
        return self._label
