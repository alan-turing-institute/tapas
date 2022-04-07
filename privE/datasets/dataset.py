"""Abstract base classes to represent the data object"""
from abc import ABC, abstractmethod


class Dataset(ABC):

    """Base class for data objects """
    def __init__(self):
        self.description = None
        self.dataset = None

    @abstractmethod
    def read(self, input_path):
        pass

    @abstractmethod
    def write(self, output_path):
        pass

    @abstractmethod
    def split(self, *args, **kwargs):
        pass

    # For convenience, we also map the __call__ function to .read.
    def __call__(self, *args, **kwargs):
        return self.read(*args, **kwargs)



