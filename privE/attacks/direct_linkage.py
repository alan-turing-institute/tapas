"""A direct linkage attack looks up the target record in the synthetic data."""

from attacks.base_classes import Attack


class DirectLinkage(Attack):
    """
    Attack that checks only whether or not the target is in the generated synthetic
     dataset or not.
    """
    def __init__(self, threat_model):
        Attack.__init__(self, threat_model)
        self.target_record = target_record

    def attack(self, datasets):
        return [(df==self.target_record).all(1).any() for df in datasets]

    def attack_score(self, datasets):
        return self.attack(datasets)