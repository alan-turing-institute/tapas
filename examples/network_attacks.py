"""
In this file, we apply the MIA attack to synthetic networks. This simple example
assumes that a record is an entire graph, and is based on groundhog_census.py.

"""

import numpy as np
np.random.seed(1312)

import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks

import tqdm

print("Loading dataset...")

# Download and load a dataset from TU.
tu_dataset = tapas.datasets.TUDataset.download_and_read("DD")

## Alternatively, if you already have the files:
# tu_dataset = tapas.datasets.TUDataset.read("DD", "DD")

## Here are some other possibilities.
# tu_dataset = tapas.datasets.TUDataset.download_and_read("ENZYMES")
# tu_dataset = tapas.datasets.TUDataset.download_and_read("MUTAG")


# Choose your generator here.

# Simple generator based on an Erdos-Renyi model.
# generator = tapas.generators.network_generator.GNP()

# Raw generator: the attack should be able to succeed here!
generator = tapas.generators.Raw()


# The knowledge of the attacker can be described with the same objects as for
# tabular datasets.

data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    tu_dataset, auxiliary_split=0.5, num_training_records=100,
)

sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
    generator, num_synthetic_records=100,
)

threat_model = tapas.threat_models.TargetedMIA(
    attacker_knowledge_data=data_knowledge,
    target_record=tu_dataset.get_records([0]),
    attacker_knowledge_generator=sdg_knowledge,
    generate_pairs=True,
    replace_target=True,
    iterator_tracker=tqdm.tqdm,
)

# The second change we need to make is in the attacks that we use.

# A first idea is to use Shadow Modelling with basic network features.
# This extracts (very) basic features from each graph, then aggregates them
# (mean) for each dataset of graph to obtain simple features.

# from sklearn.ensemble import RandomForestClassifier
# attacker = tapas.attacks.ShadowModellingAttack(
#     tapas.attacks.FeatureBasedSetClassifier(
#         features = tapas.attacks.BasicNetworkFeature(),
#         classifier = RandomForestClassifier(),
#     ),
#     label = "NetworkGroundhog"
# )

# A second possibility is to use a specifically designed set classifier that
# groups all individual graphs in the dataset in one disconnected graph, then
# extract graph kernel features from this larger graph.

attacker = tapas.attacks.ShadowModellingAttack(
    tapas.attacks.ComposedGraphClassifier(),
    label = 'ComposedAttack'
)


# Finally, the rest of the code remains unchanged!
print("Training the attack...")
attacker.train(threat_model, num_samples=200)

print("Testing the attack...")
attack_summary = threat_model.test(attacker, num_samples=100)

metrics = attack_summary.get_metrics()
print("Results:\n", metrics.head())

for report in [
    tapas.report.MIAttackReport([attack_summary]),
    tapas.report.ROCReport([attack_summary]),
]:
    report.publish("networks_mia")
