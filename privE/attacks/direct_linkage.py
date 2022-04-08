"""A direct linkage attack looks up the target record in the synthetic data."""

from attacks.base_classes import MIAttack


class DirectLinkage(MIAttack):
    """
    Attack that checks only whether or not the target is in the generated synthetic
     dataset or not.
    """
    def __init__(self, target_record):
        self.target = target_record

    def attack(self, datasets):
        guesses = [(df==self.target_record).all(1).any() for df in datasets]
        return guesses

    def attack_score(self, datasets):
        return self.attack(datasets)