"""
A direct linkage attack looks up the target record in the synthetic data.

"""

from .base_classes import Attack


class DirectLinkage(Attack):
    """
    Attack that checks only whether or not the target is in the generated synthetic
    dataset or not.

    """
    def __init__(self, threat_model):
        self.target_record = target_record

    def attack(self, datasets):
        return [(dataset.data==self.target_record).all(1).any() for dataset in datasets]

    def attack_score(self, datasets):
        return self.attack(datasets)

    def train(self):
        pass