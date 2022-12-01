"""Abstract base classes to represent synthetic data generators."""
from abc import ABC, abstractmethod
import shutil
import os
import json
from io import StringIO
import subprocess
from subprocess import PIPE
from ..datasets import TabularDataset


class Generator(ABC):
    """Base class for generators"""
    def __init__(self):
        self.trained = False

    @abstractmethod
    def generate(self, num_samples, random_state=None):
        """Given an input dataset, output a synthetic dataset with given number of samples."""
        pass

    @abstractmethod
    def fit(self, dataset, *args, **kwargs):
        """Fit generator to real data."""
        pass

    # The call method should map a dataset to a synthetic dataset.
    def __call__(self, dataset, num_samples):
        self.fit(dataset)
        return self.generate(num_samples)

    @property
    def label(self):
        return "Unnamed Generator"

    def __str__(self):
        return self.label
    


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

    @property
    def label(self):
        return "Raw"


# And importantly, import generators from disk executables.

class GeneratorFromExecutable(Generator):
    """
    A class which wraps an external executable as a generator. Currently supports
    only tabular datasets.
    """
    def __init__(self, exe, label = None):
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
        self._label = label or exe
        super().__init__()

    def fit(self, dataset):
        assert isinstance(dataset, TabularDataset)
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

    @property
    def label(self):
        return self._label
    

# TODO: This doesn't work on recent versions of reprosyn.
class ReprosynGeneratorFromCLI(Generator):
    """
    A class which wraps an external executable as a generator. Currently supports
    only tabular datasets.
    """
    def __init__(self, exe='rsyn', method='mst', config = {}, verbose=True, label = None):
        """
        Parameters
        ----------
        exe : The path to the executable as a string. defaults to rsyn
        method : the reprosyn generator
        config: dictionary stating method parameters. 
        verbose: whether to display the stderr from reprosyn.
        label: string to represent this generator in reports.
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
        self.method = method
        self.config = config
        self.verbose = verbose

        self._label = label or method
        
        super().__init__()

    def fit(self, dataset):
        assert isinstance(dataset, TabularDataset), 'dataset must be of class TabularDataset'
        self.dataset = dataset
        self.trained = True
        
    def get_default_config(self):
        out = subprocess.run([self.exe, "--generateconfig", self.method], capture_output=True)
        return json.load(StringIO(out.stdout.decode()))

    def generate(self, num_samples):
        if self.trained:
            proc = subprocess.Popen([self.exe, "--size", f"{num_samples}", "--configstring", f"{json.dumps(self.config)}", self.method], stdin = PIPE, stdout = PIPE, stderr = PIPE)
            input = bytes(self.dataset.write_to_string(), 'utf-8') 
            
            output = proc.communicate(input = input)
            if self.verbose:
                print('stderr: ', output[1])
            
            return TabularDataset.read_from_string(output[0].decode(), self.dataset.description)
        else:
            raise RuntimeError("No dataset provided to generator")

    def __call__(self, dataset, num_samples):
        self.fit(dataset)
        return self.generate(num_samples)

    @property
    def label(self):
        return self._label


class ReprosynGenerator(Generator):
    """A wrapper for reprosyn objects. This is better than the CLI, which
       fetches theh config JSON file from the GitHub repo (?)."""

    def __init__(self, reprosyn_class, label=None, **kwargs):
        self.reprosyn_class = reprosyn_class
        self.generator_kwargs = kwargs
        self.trained = False
        self._label = label or str(reprosyn_class)

    def fit(self, dataset):
        """Fitting does nothing, as we don't yet know the output size."""
        assert isinstance(dataset, TabularDataset), 'dataset must be of class TabularDataset'
        self.dataset = dataset
        self.trained = True

    def generate(self, num_samples):
        """Instantiate a reprosyn model, run it, and return output."""
        assert self.trained, "No dataset provided to generator."
        model = self.reprosyn_class(
            dataset=self.dataset.data,
            metadata=self.dataset.description.schema,
            size=num_samples,
            **self.generator_kwargs,
        )
        model.run()
        return TabularDataset(model.output, self.dataset.description)

    @property
    def label(self):
        """Cherry on top."""
        return self._label
