"""Abstract base classes to represent synthetic data generators."""
from abc import ABC, abstractmethod
import shutil
import os
from io import StringIO
import subprocess
from subprocess import PIPE
from prive.datasets import TabularDataset

class Generator(ABC):
    """Base class for generators"""
    def __init__(self):
        self.trained = False

    @abstractmethod
    def generate(self, num_samples, random_state):
        """Given an input dataset, output a synthetic dataset with given number of samples."""
        pass

    @abstractmethod
    def fit(self, dataset, *args, **kwargs):
        """Fit generator to real data."""
        pass

    # The call method should map a dataset to a synthetic dataset.
    def __call__(self, dataset, num_samples):
        pass
        #return self.generate(*args, **kwargs)


# We can implement some generators that extend this file.

class Raw(Generator):
    """This generator simply samples from the real data."""
    def __init__(self):
        super().__init__()

    def fit(self, dataset):
        self.dataset = dataset
        self.trained = True

    def generate(self, num_samples = None, random_state = None):
        if self.trained: 
            if num_samples is None:
                return self.dataset
            return self.dataset.sample(num_samples, random_state = random_state)
        else:
            raise RuntimeError("No dataset provided to generator")
        
    def __call__(self, dataset, num_samples, random_state = None):
        self.fit(dataset)
        return self.generate(num_samples, random_state = random_state)


# And importantly, import generators from disk executables.

class GeneratorFromExecutable(Generator):
    """
    A class which wraps an external executable as a generator. Currently supports
    only tabular datasets.
    """
    def __init__(self, exe):
        """
        Parameters
        ----------
        exe : The path to the executable as a string.
        """
        actual_exe = shutil.which(exe)
        if actual_exe is not None:
            self.exe = actual_exe
        else: 
            actual_exe = shutil.which(exe, path = os.getcwd())
            if actual_exe is not None:
                self.exe = actual_exe
            else:
                raise RuntimeError("Can't find user-supplied executable")
        super().__init__()

    def fit(self, dataset):
        self.dataset = dataset
        self.trained = True

    def generate(self, num_samples):
        if self.trained:
            proc = subprocess.Popen([self.exe, f"{num_samples}"], stdin = PIPE, stdout = PIPE)
            input = bytes(self.dataset.write_to_string(), 'utf-8')
            output = proc.communicate(input = input)[0].decode()

            return TabularDataset.read_from_string(output, self.dataset.description)
        else:
            raise RuntimeError("No dataset provided to generator")

    def __call__(self, dataset, num_samples):
        self.fit(dataset)
        return self.generate(num_samples)
