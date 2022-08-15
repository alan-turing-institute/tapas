"""
Classes to summarise the output of an attack in terms of a range of metrics.

AttackSummary are produced by ThreatModel.test calls, and typically contain all
relevant outputs of the test. They provide an interface to access and interpret
these outputs.

"""

from abc import ABC, abstractmethod
import numpy as np
import os
import pandas as pd


class AttackSummary(ABC):
    """Summarise the results of an attack in a specific threat model."""

    @abstractmethod
    def get_metrics(self):
        """
        Calculate metrics relevant for an attack.

        """
        pass

    @abstractmethod
    def get_metric_filename(self, postfix=""):
        """
        Returns the name of the file to save to.

        Parameters
        ----------
        postfix: str
            An optional string to append to the filename.

        """

    def write_metrics(self, output_path, postfix=""):
        """
        Write metrics to file.

        Parameters
        ----------
        output_path: str
            The prefix of the path where the metrics should be saved.
        postfix: str
            An optional string to append to the filename

        """
        file_name = self.get_metric_filename(postfix)
        self.get_metrics().to_csv(os.path.join(output_path, file_name), index=False)


class LabelInferenceAttackSummary(AttackSummary):
    """
    Class summarising main performance metrics of a label-inference attack.

    """

    def __init__(self, labels, predictions):
        """
        Parameters
        ----------
        labels: list[int]
            List with true labels of the target membership in the dataset.
        predictions: list[int]
            List with the predicted labels of the target membership in the dataset.

        """
        self.labels = np.array(labels)
        self.predictions = np.array(predictions)

    @property
    def accuracy(self):
        """
        Accuracy of the attacks based on the rate of correct predictions.

        Returns
        -------
        float

        """
        return np.mean(self.predictions == self.labels)

    def get_metrics(self):
        """
        Calculates all MIA relevant metrics and returns it as a dataframe.

        Returns
        -------
        A dataframe
            A dataframe with attack info and metrics.  The dataframe has the following structure.
            Index:
                RangeIndex
            Columns:
                accuracy: float

        """

        return pd.DataFrame([[self.accuracy,]], columns=["accuracy",],)

    def get_metric_filename(self, postfix=""):
        return f"result_labelIA_{postfix}.csv"


class BinaryLabelInferenceAttackSummary(LabelInferenceAttackSummary):
    """
    LabelInferenceAttackSummary, where the label is binary.

    """

    def __init__(self, labels, predictions, positive_label=1):
        """
        Parameters
        ----------
        labels: list[int]
            List with true labels of the target membership in the dataset.
        predictions: list[int]
            List with the predicted labels of the target membership in the dataset.
        positive_label: int
            Value to associate with the positive label (1). All other values are
            considered to be negative (0).

        """
        # Modifier for the positive value, that transforms the labels as binary {0,1}.
        transform = lambda x: (np.array(x) == positive_label).astype(int)
        LabelInferenceAttackSummary.__init__(
            self, transform(labels), transform(predictions)
        )

    @property
    def tp(self):
        """
        True positives based on rate of attacks where the target is correctly inferred
        as being in the sample.

        Returns
        -------
        float

        """
        targetin = np.where(self.labels == 1)[0]
        return np.sum(self.predictions[targetin] == 1) / len(targetin)

    @property
    def fp(self):
        """
        False positives based on rate of attacks where the target is incorrectly inferred
        as being in the sample.

        Returns
        -------
        float

        """
        targetout = np.where(self.labels == 0)[0]
        return np.sum(self.predictions[targetout] == 1) / len(targetout)

    @property
    def mia_advantage(self):
        """
        MIA attack advantage as defined by Stadler et al.

        Returns
        -------
        float

        """
        return self.tp - self.fp

    @property
    def privacy_gain(self):
        """
        Privacy gain as defined by Stadler et al.

        Returns
        -------
        float

        """
        return 1 - self.mia_advantage

    def get_metric_filename(self, postfix=""):
        return f"BinaryLIAttack_result_{postfix}.csv"

    def get_metrics(self):
        return pd.DataFrame(
            [[self.accuracy, self.tp, self.fp, self.mia_advantage, self.privacy_gain]],
            columns=[
                "accuracy",
                "true_positive_rate",
                "false_positive_rate",
                "mia_advantage",
                "privacy_gain",
            ],
        )


class MIAttackSummary(BinaryLabelInferenceAttackSummary):
    """
    Class summarising main performance metrics of a membership inference attack.

    """

    def __init__(
        self, labels, predictions, generator_info, attack_info, dataset_info, target_id
    ):
        """
        Parameters
        ----------
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

        """
        BinaryLabelInferenceAttackSummary.__init__(
            self, labels, predictions, positive_label=1
        )
        self.generator = generator_info
        self.attack = attack_info
        self.dataset = dataset_info
        self.target_id = target_id

    def get_metric_filename(self, postfix=""):
        """
        Returns the file name to which results should be saved.

        """
        return f"result_mia_{self.dataset}_{self.attack}_{self.generator}_Target{self.target_id}_{postfix}.csv"

    def get_metrics(self):
        """
        Calculates all MIA relevant metrics and returns them as a dataframe.

        Returns
        -------
        A dataframe
            A dataframe with attack info and metrics.  The dataframe has the following structure.
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

        """
        return pd.concat(
            [
                pd.DataFrame(
                    [[self.dataset, self.target_id, self.generator, self.attack,]],
                    columns=["dataset", "target_id", "generator", "attack"],
                ),
                BinaryLabelInferenceAttackSummary.get_metrics(self),
            ]
        )


class BinaryAIAttackSummary(BinaryLabelInferenceAttackSummary):
    def __init__(
        self,
        labels,
        predictions,
        generator_info,
        attack_info,
        dataset_info,
        target_id,
        sensitive_attribute,
        positive_value=1,
    ):
        """
        Parameters
        ----------
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
        sensitive_attribute: str
            The name of the sensitive attribute that the attack aims to infer.
        positive_value: int (default 1)
            The value of the sensitive attribute to mark as positive.

        """
        BinaryLabelInferenceAttackSummary.__init__(
            self, labels, predictions, positive_label=positive_value
        )
        self.generator = generator_info
        self.attack = attack_info
        self.dataset = dataset_info
        self.target_id = target_id
        self.sensitive_attribute = sensitive_attribute

    def get_metric_filename(self, postfix=""):
        """
        Returns the file name to which results should be saved.

        """
        return f"result_aia_{self.dataset}_{self.attack}_{self.generator}_Target{self.target_id}_{self.sensitive_attribute}_{postfix}.csv"

    def get_metrics(self):
        """
        Calculates all AIA relevant metrics and returns them as a dataframe.

        Returns
        -------
        A dataframe
            A dataframe with attack info and metrics.  The dataframe has the following structure.
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

        """
        return pd.concat(
            [
                pd.DataFrame(
                    [
                        [
                            self.dataset,
                            self.target_id,
                            self.generator,
                            self.attack,
                            self.sensitive_attribute,
                        ]
                    ],
                    columns=[
                        "dataset",
                        "target_id",
                        "generator",
                        "attack",
                        "sensitive_attribute",
                    ],
                ),
                BinaryLabelInferenceAttackSummary.get_metrics(self),
            ]
        )
