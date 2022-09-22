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

from prive.report import MIAttackSummary
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

    def __init__(self, attack_summaries, suffix="", eff_epsilon=None):
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

        """
        self.summaries = attack_summaries
        self.suffix = suffix
        self.eff_epsilon = eff_epsilon

    def publish(self, filepath):
        """
        Plot the ROC curves and save them to disk.

        """
        plot_roc_curve(
            [(s.labels, s.scores) for s in self.summaries],
            [s.attack for s in self.summaries],
            f"Comparison of ROC curves ({self.suffix})",
            filepath,
            self.suffix,
            eff_epsilon = self.eff_epsilon
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

        """
        self.summaries = attack_summaries
        self.split = validation_split
        assert 0 < validation_split < 1, "validation_split should be in (0,1)."
        if isinstance(confidence_levels, float):
            confidence_levels = [confidence_levels]
        self.confidence_levels = confidence_levels

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
        index, threshold = self._select_attack()
        summary = self.summaries[index]
        # Compute the effective epsilon for each confidence level.
        split_index = int(self.split * len(summary.scores))
        epsilons = [
            (c,)
            + self._estimate_effective_epsilon(
                summary.scores[split_index:],
                summary.labels[split_index:],
                threshold,
                c,
            )
            for c in self.confidence_levels
        ]
        # Compile these in one single DataFrame.
        df_epsilons = pd.DataFrame(
            epsilons, columns=["confidence", "epsilon_low", "epsilon_high"]
        )
        df_epsilons.to_csv(os.path.join(filepath, "effective_epsilon.csv"))
        return df_epsilons

    def _select_attack(self):
        """
        Select an attack and threshold from the summaries. This uses the CP
        bounds to estimate effective epsilon with relatively low confidence
        (to allow for smaller sample size).

        """
        # Trying out a new heuristic.
        min_count = 10
        # Best effective epsilon found so far, and the corresponding (index, threshold).
        best_eps = -1
        best_tp_if_fp0 = 0
        best_selection = None
        for index, summary in enumerate(self.summaries):
            # Compute the validation scores and labels.
            split_index = int(self.split * len(summary.scores))
            s = summary.scores[:split_index]
            l = summary.labels[:split_index]
            # NEW: do not consider the first and last 10 thresholds, since we allow FP=0.
            for threshold in np.unique(
                np.sort(s)[min_count:-min_count]
            ):  # np.sort(np.unique(s))[min_count:-min_count]:
                # Estimate effective epsilon for this threshold, using the CP procedure.
                # eps = self._estimate_effective_epsilon(s, l, threshold, conf_level)
                tp = np.sum(s[l == True] >= threshold)
                fp = np.sum(s[l == False] >= threshold)
                eps = tp / fp
                # If fp is 0,
                if fp == 0:
                    if tp == 0:
                        continue
                    if best_tp_if_fp0 is not None:
                        if tp >= best_tp_if_fp0:
                            best_tp_if_fp0 = tp
                            best_selection = (index, threshold)
                    else:
                        best_tp_if_fp0 = tp
                        best_selection = (index, threshold)
                # else
                elif not np.isnan(eps) and eps > best_eps:
                    best_selection = (index, threshold)
        return best_selection

    def _estimate_effective_epsilon(self, scores, labels, threshold, confidence_level):
        """
        Use the Clopper-Pearson confidence interval over the true and false positive
        rates (with confidence 1 - (1-confidence_level)/2) to obtain a high confidence
        lower bound on TPR and upper bound on FPR.

        """
        num_samples = len(scores)
        confidence_level_half = 1 - (1 - confidence_level) / 2
        # Compute the number of true and false positives out of these samples.
        tp = np.sum(scores[labels == True] >= threshold)
        fp = np.sum(scores[labels == False] >= threshold)
        # Use binomial tests to determine Clopper-Pearson bounds.
        bi_tpr = binomtest(k=tp, n=num_samples, p=tp / num_samples)
        ci_tpr = bi_tpr.proportion_ci(confidence_level_half)
        bi_fpr = binomtest(k=fp, n=num_samples, p=fp / num_samples)
        ci_fpr = bi_fpr.proportion_ci(confidence_level_half)
        # Effective epsilon is estimated as log(tpr/fpr).
        return np.log(ci_tpr.low / ci_fpr.high), np.log(ci_tpr.high / ci_fpr.low)
