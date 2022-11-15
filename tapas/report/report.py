"""
Classes to produce reports summarising the success of attacks.

Reports combine several attack summaries together to produce a more
comprehensive picture of the robustness of a generator against attacks.

"""

from abc import ABC, abstractmethod
import numpy as np
import os
import pandas as pd
from scipy.stats import binomtest

from .attack_summary import MIAttackSummary
from .utils import metric_comparison_plots, plot_roc_curve


class Report(ABC):
    """
    A report groups together the outputs of a range of attacks, potentially
    in different threat models (dataset etc), and summarises the output of
    these attacks in a concise and useful way.

    """

    @abstractmethod
    def publish(self, filepath):
        """
        Compare the outcome of attacks, potentially on different threat models.

        """
        pass


class BinaryLabelAttackReport(Report):
    """
    Report and visualise the results of a series of label-inference attacks
    with binary target.

    """

    # List of all metrics that can be used in a report.
    ALL_METRICS = [
        "accuracy",
        "true_positive_rate",
        "false_positive_rate",
        "mia_advantage",
        "privacy_gain",
        "auc",
        "effective_epsilon",
    ]

    def __init__(self, summaries, metrics=None):
        """
        Parameters
        ----------
        summaries: dataframe
            Dataframe where each row is the result of a given attack as obtained from 
            he BinaryLabelInferenceAttackSummary class. The dataframe must have the
            following structure:
            Index:
                RangeIndex
            Columns:
                dataset: str
                target_id: str
                generator: str
                attack: str
                accuracy: float
                true_positive_rate: float
                false_positive_rate: float
                mia_advantage: float
                privacy_gain: float
                auc: float
                effective_epsilon: float

            Alternatively, this can be passed as an iterable of BinaryLabelInferenceAttackSummary
            objects, in which case .get_metrics() is called on each object, and the results are
            concatenated as one DataFrame.

        metrics: list[str]
            List of metrics to be used in the report, these can be any of the following:
            "accuracy", "true_positive_rate", "false_positive_rate", "mia_advantage",
            "privacy_gain", and "auc". If left as None, all metrics are used.

        """
        if not isinstance(summaries, pd.DataFrame):
            summaries = pd.concat([s.get_metrics() for s in summaries])
        self.attacks_data = summaries
        self.metrics = metrics or MIAttackReport.ALL_METRICS

    def compare(self, comparison_column, fixed_pair_columns, marker_column, filepath):
        """
        For a fixed pair of datasets-attacks-generators-target available in the data make a figure comparing
        performance between metrics. Options configure which dimension to compare against. Figures are saved to disk.

        Parameters
        ----------
        comparison_column: str
            Column in dataframe that be used to make point plot comparison in the x axis. It can be either: 'generator',
            'dataset', 'attack' or 'target_id'.
        fixed_pair_columns: list[str]
             Columns in dataframe to fix for a given figure in order to make meaningful comparisons. It can be any pair
             of the following:'generator', 'dataset', 'attack' or 'target_id'.
        marker_column: str
            Column in dataframe that be used to as marker in a point plot comparison. It can be either: 'generator',
            'attack' or 'target_id'.
        filepath: str
            Path where the figure is to be saved.

        Returns
        -------
        None

        """

        metric_comparison_plots(
            data=self.attacks_data,
            comparison_label=comparison_column,
            fixed_pair_label=fixed_pair_columns,
            metrics=self.metrics,
            marker_label=marker_column,
            output_path=filepath,
        )

        return None

    def publish(self, filepath):
        """
        Make all comparison plots and save them to disk.

        Parameters
        ----------
        filepath: str
            Path where the figure is to be saved.

        Returns
        -------
        None

        """

        # compare generators and target ids for fixed dataset-atacks
        self.compare("generator", ["dataset", "attack"], "target_id", filepath)

        # compare attacks and target ids for fixed dataset-generators
        self.compare("attack", ["dataset", "generator"], "target_id", filepath)

        # compare datasets and target ids for fixed attacks-generators
        self.compare("dataset", ["attack", "generator"], "target_id", filepath)

        # compare targets and generators ids for fixed attacks-dataset
        self.compare("target_id", ["dataset", "attack"], "generator", filepath)

        # compare targets and attacks ids for fixed dataset-generators
        self.compare("target_id", ["dataset", "generator"], "attack", filepath)

        # compare attacks and generators for fixed dataset-targets.
        self.compare("generator", ["dataset", "target_id"], "attack", filepath)

        print(f"All figures saved to directory {filepath}")


class MIAttackReport(BinaryLabelAttackReport):
    """
    Report for a Membership Inference Attack.

    """

    @classmethod
    def load_summary_statistics(cls, attacks, metrics=None):
        """
        Load attacks data, calculate summary statistics, merge into a single dataframe and initialise object.

        Parameters
        ----------
        attacks: list[dict]
            List of dictionaries with results of attacks. Each dictionary should contain the following keys:

            dict:
            labels: list[int]
                List with true labels of the target membership in the dataset.
            predictions: list[int]
                List with the predicted labels of the target membership in the dataset.
            generator_info: str
                Metadata with information about the method used to generate the dataset.
            attack_info: str
                Metadata with information about the attacked used to infer membership of the target on the dataset.
            dataset_info: str
                Metadata with information about the original raw dataset.
            target_id: str
                Metadata with information about the target record used on the attack.

        metrics: list[str] List of metrics to be included in the report, these can be any of the following:
        "accuracy", "true_positive_rate", "false_positive_rate", "mia_advantage", "privacy_gain", "auc".

        Returns
        -------
        MIAReport class

        """

        metrics = metrics or MIAttackReport.ALL_METRICS
        df_list = []

        for attack in attacks:
            mia_summary = MIAttackSummary(
                attack["labels"],
                attack["predictions"],
                attack["scores"],
                attack["generator"],
                attack["attack"],
                attack["dataset"],
                attack["target_id"],
            ).get_metrics()

            df_list.append(mia_summary)

        return cls(pd.concat(df_list), metrics)


class BinaryAIAttackReport(MIAttackReport):
    """
    Report for an Attribute Inference Attack for binary attributes.

    """

    # This is functionally identical to MIAttackReport.


class ROCReport(Report):
    """
    Report the Receiver Operating Characteristic curves for several attacks.

    """

    def __init__(self, attack_summaries, suffix="", eff_epsilon=None, zooms=[1]):
        """
        Parameters
        ----------
        attack_summaries: list[BinaryLabelInferenceAttackSummary]
            The output of binary label-inference attacks. These can have
            been applied to different threat models (including datasets).
        suffix: str
            Text to display in the title, and filename.
        eff_epsilon: float or None
            If not None, the value of effective epsilon used to plot the
            TP/FP and TN/FN curves. If None, no curves are plotted.
        zooms: list of floats, default [1]
            List of zooms of ROC curves to produce. A zoom is a restriction of
            the ROC curve to [0,zoom_in] x [0,zoom_in]. This allows to visualise
            the TPR at low FPR for privacy analysis.

        """
        self.summaries = attack_summaries
        self.suffix = suffix
        self.eff_epsilon = eff_epsilon
        self.zooms = zooms

    def publish(self, filepath):
        """
        Plot the ROC curves and save them to disk.

        """
        for zoom_in in self.zooms:
            for low_corner in [True, False] if zoom_in < 1 else [True]:
                plot_roc_curve(
                    [(s.labels, s.scores) for s in self.summaries],
                    [s.attack for s in self.summaries],
                    f"Comparison of ROC curves ({self.suffix})",
                    filepath,
                    self.suffix
                    + (
                        f"_zoom={'' if low_corner else '-'}{zoom_in}"
                        if zoom_in < 1
                        else ""
                    ),
                    eff_epsilon=self.eff_epsilon,
                    zoom_in=zoom_in,
                    low_corner=low_corner,
                )


class EffectiveEpsilonReport(Report):
    """
    Estimate the effective epsilon of a *generator* from a library of attacks.

    This first selects an attack (score) and a threshold (tau) that are likely
    to lead to the highest effective epsilon, then compute Clopper-Pearson
    bounds for the selected attack. The two parts are performed on disjoint
    subsets of the results (validation and test) to avoid bias.

    This analysis is based on "Jagielski, M., Ullman, J. and Oprea, A., 2020.
    Auditing differentially private machine learning: How private is private sgd?.
    Advances in Neural Information Processing Systems, 33, pp.22205-22216.""

    Recommendations:
    - Use ExactDataKnowledge as auxiliary data knowledge.
    - Have a large enough number of test samples.

    """

    def __init__(
        self,
        attack_summaries,
        validation_split=0.1,
        confidence_levels=(0.9, 0.95, 0.99),
        heuristic="cp",
    ):
        """
        Parameters
        ----------
        attack_summaries: list[BinaryLabelInferenceAttackSummary]
            The summaries of Attacks that have been applied against the same
            threat model. One of these attacks (the one with highest TP/FP) will
            be used to estimate the effective epsilon.
        validation_split: float in (0,1)
            Fraction of each summary to use as validation, to select the attack
            and threshold to use in the estimation.
        confidence_levels: list[float in (0,1)], or a float.
            The confidence levels for which to compute the estimate.
        heuristic: str
            Which heuristic to use to choose the attack and threshold used for
            estimation of Clopper-Pearson bounds. Acceptable values are "cp"
            (max effeps with Clopper-Pearson) and "ratio" (max TP/FP).

        """
        self.summaries = attack_summaries
        self.split = validation_split
        assert 0 < validation_split < 1, "validation_split should be in (0,1)."
        if isinstance(confidence_levels, float):
            confidence_levels = [confidence_levels]
        self.confidence_levels = confidence_levels
        self.heuristic = heuristic
        assert heuristic in ("cp", "ratio"), "Unsupported heuristic."
        self._select_attack = {
            "cp": self._select_attack_cp,
            "ratio": self._select_attack_ratio,
        }[heuristic]

    def publish(self, filepath):
        """
        Returns a Pandas DataFrame with the Clopper-Pearson estimate of the
        effective epsilon for a range of confidence levels.

        Parameters
        ----------
        filepath: str
            Path of the folder where the DataFrame is to be saved.

        """
        # First, select an attack and threshold.
        index, threshold, inverse = self._select_attack()
        summary = self.summaries[index]
        print(
            f"Using attack {summary.attack} with threshold {threshold} and {inverse}."
        )
        # Compute the effective epsilon for each confidence level.
        split_index = int(self.split * len(summary.scores))
        epsilons = [
            (c,)
            + self._estimate_effective_epsilon(
                summary.scores[split_index:],
                summary.labels[split_index:],
                threshold,
                c,
                inverse,
            )
            for c in self.confidence_levels
        ]
        # Compile these in one single DataFrame.
        df_epsilons = pd.DataFrame(
            epsilons, columns=["confidence", "epsilon_low", "epsilon_high"]
        )
        df_epsilons.to_csv(os.path.join(filepath, "effective_epsilon.csv"))
        return df_epsilons

    def _select_attack_ratio(self):
        """
        Select an attack and threshold from the summaries. This uses the ratio TP/FP
        of the attack as selection criterion. If FP=0 for any attack/threshold, then
        the attack/threshold with highest TP for FP=0 is selected. In order to avoid
        the case where FP = 1 by chance, we exclude the first and last10 values of
        threshold for each attack (or first and last 10% of threshold values,
        whichever is lowest).

        This heuristic privileges "worst-case" setups where TP>0 for FP=0. Since such
        situations are forbidden by DP, this is where the highest potential epsilon
        can be found. This is however quite sensitive to randomness.

        """
        # Best effective epsilon found so far, and the corresponding (index, threshold).
        best_eps = -1
        best_selection = None
        for index, summary in enumerate(self.summaries):
            # Compute the validation scores and labels.
            split_index = int(self.split * len(summary.scores))
            s = summary.scores[:split_index]
            l = summary.labels[:split_index]
            positive_count = np.sum(l == True)
            negative_count = np.sum(l == False)
            # We adapt in case the min count is too small.
            min_count = min(10, 1 + int(len(s) * 0.1))
            # The minimum value of numerator if 
            min_count_for_num = max(10, int(len(s) * 0.05))
            # Do not consider the first and last 10 thresholds, since we allow FP=0.
            for threshold in np.unique(np.sort(s)[min_count:-min_count]):
                # Compute the ratio TP/FP or TN/FN.
                true_positives = np.sum(s[l == True] >= threshold)
                false_positives = np.sum(s[l == False] >= threshold)
                for inverse in [False, True]:
                    if not inverse:
                        # Normal scenario (compute TP/FP)
                        num = true_positives / positive_count
                        denom = false_positives / negative_count
                    else:
                        # Inverse (compute TN/FN).
                        num = (negative_count - false_positives) / negative_count
                        denom = (positive_count - true_positives) / positive_count
                    # print(f"{summary.attack}\t{threshold:.3f}\t{num:.3f}/{denom:.3f}")
                    # If the denominator is 0, only consider this if the numerator is large.
                    if denom == 0:
                        if num >= min_count_for_num:
                            # In this case, this is the star candidate.
                            best_eps = np.inf
                            min_count_for_num = num
                            best_selection = (index, threshold, inverse)
                    # if fp is not 0, then eps is finite.
                    elif num / denom > best_eps:
                        best_eps = num / denom
                        best_selection = (index, threshold, inverse)
        return best_selection

    def _select_attack_cp(self, conf_level=0.9):
        """
        Select an attack and threshold from the summaries. This uses the CP
        bounds to estimate effective epsilon with relatively low confidence
        (to allow for smaller sample size).

        This heuristic privileges safe values, and tends to consistently find
        high-ish values of epsilon, but not necessarily the most successful
        attack (i.e. it is conservative).

        """
        best_eps = -1
        best_selection = None
        for index, summary in enumerate(self.summaries):
            # Compute the validation scores and labels.
            split_index = int(self.split * len(summary.scores))
            s = summary.scores[:split_index]
            l = summary.labels[:split_index]
            for threshold in np.sort(np.unique(s)):
                # Estimate effective epsilon for this threshold, using the CP procedure.
                for inverse in [False, True]:
                    eps_bounds = self._estimate_effective_epsilon(
                        s, l, threshold, conf_level, inverse
                    )
                    # If the lower bound is higher than previous estimations, memorise.
                    eps_low = eps_bounds[0]
                    if not np.isnan(eps_low) and eps_low > best_eps:
                        best_eps = eps_low
                        best_selection = (index, threshold, inverse)
        return best_selection

    def _estimate_effective_epsilon(
        self, scores, labels, threshold, confidence_level, inverse=False
    ):
        """
        Use the Clopper-Pearson confidence interval over the true and false positive
        rates (with confidence 1 - (1-confidence_level)/2) to obtain a high confidence
        lower bound on TPR and upper bound on FPR.

        If inverse is True, swap positives and negatives: instead of using TP/FP for
        the estimation, use TN/FN.

        """
        num_samples = len(scores)
        confidence_level_half = 1 - (1 - confidence_level) / 2
        # By default, positive label is True and the threshold increases with P[True].
        positive_label = not inverse
        test = (lambda x: x <= threshold) if inverse else (lambda x: x >= threshold)
        # Compute the number of true and false positives out of these samples.
        positive_count = np.sum(labels == positive_label)
        negative_count = np.sum(labels != positive_label)
        true_positives = np.sum(test(scores[labels == positive_label]))
        false_positives = np.sum(test(scores[labels != positive_label]))
        # Use binomial tests to determine Clopper-Pearson bounds.
        bi_tpr = binomtest(
            k=true_positives, n=positive_count, p=true_positives / positive_count
        )
        ci_tpr = bi_tpr.proportion_ci(confidence_level_half)
        bi_fpr = binomtest(
            k=false_positives, n=negative_count, p=false_positives / negative_count
        )
        ci_fpr = bi_fpr.proportion_ci(confidence_level_half)
        # Effective epsilon is estimated as log(tpr/fpr), and this is thus the confidence interval.
        low_bound = max(0, np.log(ci_tpr.low / ci_fpr.high))
        high_bound = np.log(ci_tpr.high / ci_fpr.low) if ci_fpr.low > 0 else np.inf
        return low_bound, high_bound
