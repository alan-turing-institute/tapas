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
    from ..attacks import Attack
    from ..datasets import Dataset
    from ..generators import Generator
    from collections.abc import Iterable

from abc import ABC, abstractmethod
from .base_classes import TrainableThreatModel


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

    @property
    def label(self):
        """
        A string to represent this knowledge.

        """
        return self._label


class AttackerKnowledgeWithLabel(AttackerKnowledgeOnData):
    """
    Abstract base class that builds on AttackerKnowledgeOnData that adds the
    functionality of labeling the datasets. Such labels can be represent, e.g.,
    whether a specific user is part of the dataset. This is used to define
    membership/attribute inference attacks.
    """

    @abstractmethod
    def generate_datasets_with_label(
        self, num_samples: int, training: bool = True
    ) -> tuple[list[Dataset], list[int]]:
        """
        Generate `num_samples` training or testing datasets with corresponding
        labels (arbitrary ints or bools).

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
        dataset: Dataset = None,
        auxiliary_split: float = 0.5,
        aux_data: Dataset = None,
        test_data: Dataset = None,
        num_training_records: int = 1000,
    ):
        """
        Initialise this threat model with a given dataset. This threat model
        requires an auxiliary dataset available to the attacker, and a test
        dataset to evaluate attacks. These can either be specified by giving
        a dataset (`dataset`) to split between the two, with a fraction of
        `auxiliary_split` giving the relative size of the auxiliary dataset;
        and/or by explicitly specifying aux_data and test_data. If both are
        given together, the resulting auxiliary/test datasets are obtained by
        concatenating the other two.

        Parameters
        ----------
        dataset : Dataset (None)
            Dataset to split between test and auxiliary data.
        auxiliary_split: float in [0,1], optional.
            Fraction of `dataset` to use as auxiliary dataset. The rest of the
            dataset is used as test dataset.
        aux_data : Dataset, optional
            Dataset that the adversary is assumed to have access to. This or
            auxiliary_split must be provided. The default is None.
        test_data: Dataset, optional
            Dataset used to generate test datasets to evaluate the attack. The
            default is None.
        num_training_records : int, optional (default 1000).
            Number of training records to use to train each copy of shadow_model,
            when generating synthetic training datasets for the attack.
            The default is 1000.
        """
        assert (aux_data is not None) or (
            auxiliary_split > 0.0 and dataset is not None
        ), "No auxiliary data given."
        assert (test_data is not None) or (
            auxiliary_split < 1 and dataset is not None
        ), "No test data given."
        assert (
            0 <= auxiliary_split <= 1
        ), f"sample_real_frac must be in [0, 1], got {sample_real_frac}"
        # Check that the descriptions match for all pairs of two datasets.
        for d1, d2 in [
            (dataset, aux_data),
            (dataset, test_data),
            (aux_data, test_data),
        ]:
            if d1 is not None and d2 is not None:
                assert (
                    d1.description == d2.description
                ), "Dataset descriptions do not match"

        # If provided, split the dataset.
        if dataset is not None:
            aux_size = int(auxiliary_split * len(dataset))
            self.test_data = dataset.copy()
            self.aux_data = self.test_data.create_subsets(
                n=1, sample_size=aux_size, drop_records=True
            ) [0]
            # Add other specified datasets, if any.
            if aux_data is not None:
                self.aux_data = self.aux_data + aux_data
            if test_data is not None:
                self.test_data = self.test_data + test_data

        # Otherwise, just use the datasets specified by the users.
        else:
            self.aux_data = aux_data
            self.test_data = test_data

        self.num_training_records = num_training_records

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
        dataset = self.aux_data if training else self.test_data

        # Split the data into subsets.
        return dataset.create_subsets(num_samples, self.num_training_records)

    @property
    def label(self):
        return self.aux_data.label + " (AUX)"


class ExactDataKnowledge(AttackerKnowledgeOnData):
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

    @property
    def label(self):
        return self.training_dataset.label + " (EXACT)"


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

    @property
    def label(self):
        """
        A string to represent this knowledge.

        """
        return self._label


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

    @property
    def label(self):
        return self.generator.label


# With the tools developed in this module, we can define a generic threat model
# where the attacker aims to infer the "label" of the private dataset. The
# label is defined by the attacker's knowledge being AttackerKnowledgeWithLabel.

# TODO: add output class.
class LabelInferenceThreatModel(TrainableThreatModel):
    def __init__(
        self,
        attacker_knowledge_data: AttackerKnowledgeWithLabel,
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
        memorise_datasets=True,
        iterator_tracker: Callable[[list], Iterable] = None,
    ):
        """
        Generate a Label-Inference Threat Model.

        Parameters
        ----------
        attacker_knowledge_data: AttackerKnowledgeWithLabel
            The knowledge on data available to the attacker, which includes a
            label that the attack aims to predict.
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator
            The knowledge on the generator available to the attacker.
        memorise_datasets: True
            Whether to memoise the synthetic datasets generated,
        iterator_tracker: Callable list L -> Iterable over L.
            A callable used to track iterations. The method __next__ is called
            whenever a dataset needs to be generated. This can be used to track
            progress, e.g. with tqdm. Default is lambda x: x (silent).
            Note that this iterator is only called for synthetic data generation,
            which is often the bottleneck, and not training data generation.

        """
        self.atk_know_data = attacker_knowledge_data
        self.atk_know_gen = attacker_knowledge_generator
        # Also, handle the memoisation to prevent recomputing datasets.
        self.memorise_datasets = memorise_datasets
        self.iterator_tracker = iterator_tracker or (lambda x: x)
        # maps training = True/False -> list of datasets, list of labels.
        self._memory = {True: ([], []), False: ([], [])}

    def _generate_samples(
        self, num_samples: int, training: bool = True, ignore_memory: bool = False,
    ) -> tuple[list[Dataset], list[bool]]:
        """
        Internal method to generate samples for training or testing. This outputs 
        two lists, the first of synthetic datasets and the second of labels (1 if
        the target is in the training dataset used to produce the corresponding
        dataset, and 0 otherwise).

        Parameters
        ----------
        num_samples: int
            The number of synthetic datasets to generate.
        training: bool (default, True)
            whether to generate samples from the training or test distribution.
        ignore_memory: bool, default False
            Whether to ignore the memoised datasets.

        """
        # Retrieve memoized samples (if needed).
        use_memory = (not ignore_memory) and self.memorise_datasets
        if use_memory:
            mem_datasets, mem_labels = self._memory[training]
            if num_samples <= len(mem_datasets):
                # No samples are needed! Return what is in memory:
                return mem_datasets[:num_samples], mem_labels[:num_samples]
            # Decrease the number of samples (= number of samples to generate).
            num_samples -= len(mem_datasets)
        else:
            mem_datasets = []
            mem_labels = []
        # Generate sample: first, produce the original datasets with labels.
        training_datasets, gen_labels = self.atk_know_data.generate_datasets_with_label(
            num_samples, training=training
        )
        # Then, generate synthetic data from each original dataset.
        gen_datasets = [
            self.atk_know_gen(ds) for ds in self.iterator_tracker(training_datasets)
        ]
        # Add the entries generated to the memory.
        if use_memory:
            self._memory[training] = (
                mem_datasets + gen_datasets,
                mem_labels + gen_labels,
            )
        # Combine results from the memory with generated results.
        return mem_datasets + gen_datasets, mem_labels + gen_labels

    def generate_training_samples(
        self, num_samples: int, ignore_memory: bool = False,
    ) -> tuple[list[Dataset], list[bool]]:
        """
        Generate samples to train an attack.

        Parameters
        ----------
        num_samples: int
            The number of synthetic datasets to generate.
        ignore_memory: bool, default False
            Whether to use the memoized datasets, or ignore them.
        """
        return self._generate_samples(num_samples, True, ignore_memory)

    def test(
        self, attack: Attack, num_samples: int = 100, ignore_memory: bool = False,
    ) -> tuple[list[int], list[int]]:
        """
        Test an attack against this threat model. This samples `num_samples`
        testing synthetic datasets along with labels. It then runs the attack
        on each synthetic dataset, to estimate a label on each. The true and
        predicted labels are returned.

        Parameters
        ----------
        attack : Attack
            Attack to test.
        num_samples : int
            Number of test datasets to generate and test against.
        ignore_memory: bool, default False
            Whether to use the memoized datasets, or ignore them.

        Returns
        -------
        tuple(list(int), list(int))
            Tuple of (true_labels, pred_labels), where true_labels indicates
            the true label of the original datasets and pred_labels are the
            labels predicted by the attack from the synthetic datasets.
        """
        test_datasets, truth_labels = self._generate_samples(
            num_samples, False, ignore_memory
        )
        pred_labels = attack.attack(test_datasets)
        return pred_labels, truth_labels
