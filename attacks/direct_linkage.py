from utils.data import read_meta

class DirectLinkage(MIAttack):
    """
    Attack that checks only whether or not the target is in the generated synthetic
    dataset or not
    """
    def __init__(self, target):
        self.target = target

    def train(self):
        pass

    def attack(self, datasets):
        target_row = self.target.iloc[0].to_numpy()
        guesses = [(df==target_row).all(1).any() for df in datasets]
        return guesses