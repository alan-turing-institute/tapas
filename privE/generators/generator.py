"""Abstract base classes to represent synthetic data generators."""
from abc import ABC, abstractmethod

class Generator(ABC):
    """Base class for generators"""
    def __init__(self):
        self.trained = False

    @abstractmethod
    def generate(self, num_samples):
        """Given an input dataset, output a synthetic dataset with given number of samples."""
        pass

    @abstractmethod
    def fit(self, dataset, *args, **kwargs):
        """Fit generator to real data."""
        pass

    # For convenience, we also map the __call__ function to .generate.
    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


# We can implement some generators that extend this file.

class ReturnRaw(Generator):
    """This generator simply samples from the real data."""
    def __init__(self):
        super().__init__()

    def fit(self, dataset):
        self.dataset = dataset
        self.trained = True

    def generate(self, num_samples):
        return self.dataset.sample(num_samples)


# And importantly, import generators from disk executables.

class GeneratorFromExecutable(Generator):
	"""This class interfaces with a generator as executable on disk."""