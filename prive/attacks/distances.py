"""
Distance metrics for closest-distance attacks.

Distances are callable objects that return an array of real number for pairs of
datasets (either records, or datasets of same lengths). We here implement a
range of simple distances, and easy methods to combine them.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset, DataDescription

from abc import ABC, abstractmethod


class DistanceMetric:
    """"""

    @abstractmethod
    def __call__(self, x: Dataset, y: Dataset):
        return float("inf")

    # Interface to easily combine distance metrics.
    def __sum__(self, d: DistanceMetric):
        pass

    def __prod__(self, factor: float):
        pass
