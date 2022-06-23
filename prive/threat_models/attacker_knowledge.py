"""
This file implements common examples of attacker knowledge of both the
private dataset and the generator. This specifies elements (2.) and (3.)
of threat models (see .base_classes.py).

Knowledge of the data is represented by AttackerKnowledgeOnData objects,
which defines methods to sample training and testing datasets.

Knowledge of the generator is represented by AttackerKnowledgeOnGenerator
objects, which is primarily a wrapper of generator.__call__ with a given
number of synthetic records, but can be extended to more.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..attacks import Attack  # for typing
    from ..datasets import Dataset  # for typing
    from ..generators import Generator  # for typing

from abc import ABC, abstractmethod


class AttackerKnowledgeOnData:
    """
    Abstract base class that represents the knowledge that attackers have
    on real datasets, in the form of a prior from which they can sample
    training datasets for the attack. This class also requires a way to
    generate "testing" datasets from (potentially) another distribution,
    which will be used to test an attack trained on "training" datasets.

    Note that these methods generate _real_ datasets -- a generator still
    needs to be applied to produce synthetic datasets.

    """

    @abstractmethod
    def generate_datasets(
        self, num_samples: int, training: bool = True
    ) -> list[Dataset]:
        """
        Generate `num_samples` training or testing datasets.

        """
        abstract


class AuxiliaryDataKnowledge(AttackerKnowledgeOnData):
    """
    This attacker knowledge assumes access to some auxiliary dataset from which
    training datasets are sampled, as random subset of this auxiliary data.
    A distinct testing dataset, sampled from the same distribution, is also
    used to generate testing samples.

    """

    def __init__(
        self,
        dataset: Dataset,
        aux_data: Dataset = None,
        sample_real_frac: float = 0.0,
        num_training_records: int = 1000,
    ):
        """
        Initialise threat model with a given dataset. This dataset is either
        split between auxiliary and test (if sample_real_frac>0), or is treated
        as test dataset (if aux_data is not None). 

        Parameters
        ----------
        dataset : Dataset
            Real dataset to use to generate test synthetic datasets from.
        aux_data : Dataset, optional
            Dataset that the adversary is assumed to have access to. This or
            sample_real_frac must be provided. The default is None.
        sample_real_frac : float, optional
            Fraction of real data to sample and assume adversary has access to.
            Must be in [0, 1]. This must be > 0 or aux_data provided.
            The default is 0.
        num_training_records : int, optional
            Number of training records to use to train each copy of shadow_model,
            when generating synthetic training datasets for the attack.
            The default is 1000.
        """
        assert (aux_data is not None) or (
            sample_real_frac != 0.0
        ), "At least one of aux_data or sample_real_frac must be given"
        assert (
            0 <= sample_real_frac <= 1
        ), f"sample_real_frac must be in [0, 1], got {sample_real_frac}"
        if aux_data:
            assert (
                aux_data.description == dataset.description
            ), "aux_data does not match the description of dataset"

        self.dataset = dataset
        self.num_training_records = num_training_records

        # Define the attacker's knowledge.
        self._adv_data = {
            "aux": (aux_data or self.dataset.empty()),
            "real": self.dataset.sample(frac=sample_real_frac),
        }

    @property
    def adv_data(self):
        """
        Dataset: The data the adversary has access to.

        """
        return self._adv_data["aux"] + self._adv_data["real"]

    def generate_datasets(
        self, num_samples: int, training: bool = True
    ) -> list[Dataset]:
        """
        Generate training/testing "real" datasets.

        Parameters
        ----------
        num_samples : int
            Number of training dataset *pairs* to generate.
        training : bool, optional
            If True, D's will be sampled from the adversary's data (self.adv_data).
            Otherwise, D's will be sampled from the real data (self.datasets).
            The default is True.

        Returns
        -------
        tuple(list[Dataset], np.ndarray)
            List of generated synthetic datasets. List of labels.

        """
        # If training, sample datasets from the adversary's data. Otherwise,
        # sample datasets from the real dataset.
        dataset = self.adv_data if training else self.dataset

        # Split the data into subsets.
        return dataset.create_subsets(num_samples, self.num_training_records)


class ExactDataKnowledge(AuxiliaryDataKnowledge):
    """
    Also called worst-case attack, this assumes that the attacker knows the
    exact dataset used to generate 

    """

    def __init__(self, training_dataset: Dataset):
        self.training_dataset = training_dataset

    def generate_datasets(
        self, num_samples: int, training: bool = True
    ) -> list[Dataset]:
        return [self.training_dataset] * num_samples


class AttackerKnowledgeOnGenerator:
    """
    Abstract base class that represents the knowledge that attachers have on
    the generator used to produce the synthetic datasets. This typically just
    requires a Generator objct and a choice of length for generated synthetic
    datasets.

    """

    @abstractmethod
    def generate(self, training_dataset: Dataset):
        """Generate a synthetic dataset from a given training dataset."""
        pass

    # Equivalently, you can just call this object:
    def __call__(self, training_dataset: Dataset):
        return self.generate(training_dataset)


class BlackBoxKnowledge(AttackerKnowledgeOnGenerator):
    """
    The attacker has access to the generator method with access to the
    generator has an exact black-box. The attacker can call the generator
    with the same parameters as were used to produce the real dataset.

    """

    def __init__(self, generator: Generator, num_synthetic_records: int):
        self.generator = generator
        self.num_synthetic_records = num_synthetic_records

    def generate(self, training_dataset: Dataset):
        return self.generator(training_dataset, self.num_synthetic_records)
