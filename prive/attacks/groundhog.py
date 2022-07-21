"""
This file implements the Groundhog attack, as introduced in:
    Stadler, T., Oprisanu, B. and Troncoso, C., 2021. Synthetic data–anonymisation
    groundhog day. In 31st USENIX Security Symposium (USENIX Security 22).

The Groundhog attack is a black-box attack based on shadow modelling that
uses a combination of three set features (Naive, Hist and Corr) and a random
forest as learner.

"""

from .shadow_modelling import ShadowModellingAttack
from .set_classifiers import (
    FeatureBasedSetClassifier,
    NaiveSetFeature,
    HistSetFeature,
    CorrSetFeature,
)

from sklearn.ensemble import RandomForestClassifier


class GroundhogAttack(ShadowModellingAttack):
    """
    The attack introduced by Stadler et al.

    """

    def __init__(
        self, use_naive=True, use_hist=True, use_corr=True, model=None, label=None
    ):
        """
        Parameters
        ----------
        use_naive: bool (default True)
            Whether to use F_naive as a feature.
        use_hist: bool (default True)
            Whether to use F_hist as a feature.
        use_corr: bool (default True)
            Whether to use F_corr as a feature.
        model: sklearn.base.ClassifierMixin (default None)
            If specified, the binary classifier to use for the attack. If None,
            the default (random forest with 100 learners) is used.
        label: str (default None)
            An optional label to refer to the attack in reports.
        """
        # Parse the selection of features.
        features = None
        if use_naive:
            features = NaiveSetFeature()
        if use_hist:
            h = HistSetFeature()
            features = h if features is None else features + h
        if use_corr:
            c = CorrSetFeature()
            features = c if features is None else features + c
        assert features is not None, "At least one feature must be specified."

        ShadowModellingAttack.__init__(
            self,
            # The attack uses a feature-based set classifier, which first
            # extracts fixed (non-trainable) features from the set, then
            # fits a machine learning classifier from these features.
            FeatureBasedSetClassifier(
                # Use the default features: Naive, Hist and/or Corr.
                features,
                # If a classifier is specified, use it. Otherwise, use a random
                # forest with 100 trees and default parameters.
                model or RandomForestClassifier(n_estimators=100),
            ),
            label=label or "Groundhog",
        )
