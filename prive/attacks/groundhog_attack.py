"""
Parent class for launching a membership inference attack on the output of a 
generative model
"""
import numpy as np
import pandas as pd

from attack_models.base_classes import MIAttack

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)


class Groundhog(MIAttack):
    """
    Parent class for membership inference attack on the output of a 
    generative model using a classifier. A single instance of this class
    corresponds to a specific target data point.

    Args:
        Classifier : Classifier to use to distinguish synthetic datasets
        metadata (dict) : Metadata dictionary describing data
        quids (list) : List of column names to be regarded as quasi-identifiers.
            This list makers the Attack class aware of which columns that would
            usually be continuous are going to be categorical (binned).
    """
    def __init__(self,
                 Classifier,
                 data_description,
                 quids=None
                 ):
        self.Classifier = Classifier
        self.data_description = data_description

        self.trained = False

        self.__name__ = f'{self.Classifier.__class__.__name__}Groundhog'

    def train(self, datasets, labels):
        """
        Train the attack classifier on a labelled training set

        Args:
            datasets (List[pd.DataFrame]): List of synthetic datasets to use
                as training data
            labels (np.ndarray): Labels for datasets indicating whether or not
                the target was present in the training data that produced each
                synthetic dataset
        """
        # Fit the classifier to the data
        self.Classifier.fit(synthetic_datasets, labels)

        self.trained = True

    def attack(self, datasets):
        """
        Make a guess about the target's membership in the training data of
        each of the generative models that produced each of the synthetic
        input datasets

        Args:
            datasets (List[pd.DataFrame]) : List of synthetic datasets to attack
            attemptLinkage (bool) : If True, search for Target explicitly in
                each dataset before using the classifier
            target : Target to find if attemptLinkage=True

        Returns:
            guesses (List[int]) : List of guesses. The ith entry corresponds
                to the guess for the ith dataset in datasets. Guess of 0
                corresponds to target not present.
        """
        assert self.trained, 'Attack must first be trained.'

        guesses = []
        for df in datasets:
            guess = self._make_guess(df)
            guesses.append(guess)
        return guesses

    def _make_guess(self, df):
        """
        Make a guess for a single dataset about the presence of the target in
        the training data that generated the dataset
        """
        return np.round(self.Distinguisher.predict(f), 0).astype(int)[0]

    def get_confidence(self, synT, secret):
        """
        Calculate classifier's raw probability about the presence of the target.
        Output is a probability in [0, 1].

        Args:
            synT (List[pd.DataFrame]) : List of dataframes to predict
            secret (List[int]) : List indicating the truth of target's presence
                in (training data of) corresponding dataset in synT

        Returns:
            List of probabilities corresponding to attacker's guess about the truth
        """
        assert self.trained, 'Attack must first be trained.'
        if self.FeatureSet is not None:
            synT = np.stack([self.FeatureSet.extract(s) for s in synT])
        else:
            if isinstance(synT[0], pd.DataFrame):
                synT = np.stack([convert_df_to_array(s, self.metadata).flatten() for s in synT])
            else:
                synT = np.stack([s.flatten() for s in synT])

        probs = self.Classifier.predict_proba(synT)

        return [p[s] for p,s in zip(probs, secret)]
