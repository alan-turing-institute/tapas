"""
This short example shows how to use TAPAS in setups where the generator is
not (completely) known. It also illustrates how to implement a (very simple)
generator from scratch.

"""

import numpy as np
import pandas as pd

from tapas.datasets import TabularDataset, DataDescription
from tapas.generators import Generator
from tapas.threat_models import (
    NoBoxKnowledge,
    UncertainBoxKnowledge,
    ExactDataKnowledge,
    BlackBoxKnowledge,
    TargetedMIA,
)
from tapas.attacks import ClosestDistanceMIA, GroundhogAttack
from tapas.report import MIAttackReport

# We define a simple generator that computes histograms and adds Gaussian noise.
class NoisyHistogram(Generator):
    def fit(self, dataset, sigma=1):
        self.histograms = {}
        self.description = dataset.description
        for variable in dataset.description:
            data = dataset.data[variable["name"]]
            histogram = np.array(
                [(data == v).sum() for v in range(variable["representation"])]
            )
            histogram = histogram + np.random.normal(loc=0, scale=sigma, size=histogram.shape)
            histogram = np.maximum(histogram, 0)
            self.histograms[variable["name"]] = histogram / histogram.sum()

    def generate(self, num_samples):
        data = np.zeros((num_samples, len(self.description.schema)))
        columns = []
        for i, variable in enumerate(self.description):
            data[:, i] = np.random.choice(
                variable["representation"],
                replace=True,
                size=(num_samples,),
                p=self.histograms[variable["name"]],
            )
            columns.append(variable["name"])
        return TabularDataset(pd.DataFrame(data, columns=columns), self.description)

    @property
    def label(self):
        return "NoisyHistogram"
    


# We create a simplistic one-dimensional dataset.
num_categories = 100
description = DataDescription(
    [{"name": "a", "type": "finite", "representation": num_categories}],
    label="1-D"
)
dataset = TabularDataset(
    pd.DataFrame(np.random.randint(num_categories - 1, size=(1000,)), columns=["a"]),
    description
)
target = TabularDataset(
    pd.DataFrame([[num_categories - 1]], columns=["a"]), description,
)

# We assume a worst-case attacker.
atk_data_knowledge = ExactDataKnowledge(dataset)

# Three different threat-models, in increasing order of knowledge on the generator.
generator = NoisyHistogram()

atk_generator_knowledge = [
    NoBoxKnowledge(generator, 1000),
    UncertainBoxKnowledge(
        generator,
        1000,
        # Here is the uncertainty that the attacker has on sigma (here, U[0,2]).
        (lambda: {"sigma": 2 * np.random.random()}),
        # And here is the real value.
        {"sigma": 1},
    ),
    BlackBoxKnowledge(generator, 1000),
]

summaries = []

for atk_gen in atk_generator_knowledge:
    threat_model = TargetedMIA(
        attacker_knowledge_data=atk_data_knowledge,
        target_record=target,
        attacker_knowledge_generator=atk_gen,
    )
    attacks = [ClosestDistanceMIA(criterion=("threshold", 0))]
    if not isinstance(atk_gen, NoBoxKnowledge):
        attacks.append(GroundhogAttack(use_naive=False, use_corr=False))
    for attack in attacks:
        attack.train(threat_model, num_samples=1000)
        summaries.append(threat_model.test(attack, num_samples=1000))

# Finally, group together the summaries as a report.
print("Publishing a report.")
report = MIAttackReport(summaries, metrics=["accuracy", "auc"])
report.publish("uncertain_box")